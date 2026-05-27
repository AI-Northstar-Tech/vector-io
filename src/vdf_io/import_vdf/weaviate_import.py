import math
from typing import Any, Dict, List
from uuid import UUID

import numpy as np
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.data import DataObject
from weaviate.util import generate_uuid5

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
)


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
        )
        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate Cloud")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--host", type=str, default="localhost", help="Local Weaviate host"
        )
        parser_weaviate.add_argument(
            "--port", type=int, default=8080, help="Local Weaviate HTTP port"
        )
        parser_weaviate.add_argument(
            "--grpc_port", type=int, default=50051, help="Local Weaviate gRPC port"
        )

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the Weaviate Cloud URL (leave empty for local Weaviate): ",
            str,
            "",
        )
        if args.get("url"):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )
        else:
            set_arg_from_input(
                args, "host", "Enter the Weaviate host: ", str, "localhost"
            )
            set_arg_from_input(
                args, "port", "Enter the Weaviate HTTP port: ", int, 8080
            )
            set_arg_from_input(
                args, "grpc_port", "Enter the Weaviate gRPC port: ", int, 50051
            )
        weaviate_import = ImportWeaviate(args)
        try:
            weaviate_import.upsert_data()
        finally:
            weaviate_import.client.close()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = self._connect()

    def _connect(self):
        api_key = self.args.get("api_key")
        auth_credentials = weaviate.auth.AuthApiKey(api_key) if api_key else None
        if self.args.get("url"):
            return weaviate.connect_to_wcs(
                cluster_url=self.args["url"],
                auth_credentials=auth_credentials,
                skip_init_checks=True,
            )
        return weaviate.connect_to_local(
            host=self.args.get("host") or "localhost",
            port=self.args.get("port") or 8080,
            grpc_port=self.args.get("grpc_port") or 50051,
            auth_credentials=auth_credentials,
            skip_init_checks=True,
        )

    def get_all_index_names(self):
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [getattr(collection, "name", collection) for collection in collections]

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        collections = self.get_all_index_names()
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                new_collection_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_collection_name = self.create_new_name(
                    new_collection_name, collections
                )
                vector_column_names, _ = self.get_vector_column_name(
                    index_name, namespace_meta, multi_vector_supported=True
                )
                if new_collection_name not in collections:
                    self._create_collection(new_collection_name, vector_column_names)
                    collections.append(new_collection_name)

                collection = self.client.collections.get(new_collection_name)
                previous_count = self._get_total_count(collection)

                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(
                        file_path,
                        max_num_rows=(
                            (self.args.get("max_num_rows") or INT_MAX)
                            - self.total_imported_count
                        ),
                    )
                    df = cleanup_df(df)
                    batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                    for batch in tqdm(
                        divide_into_batches(df, batch_size),
                        desc="Importing batches",
                        total=len(df) // batch_size,
                    ):
                        objects = self._build_objects(batch, vector_column_names)
                        if self.total_imported_count + len(objects) >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            max_hit = True
                            objects = objects[
                                : (self.args.get("max_num_rows") or INT_MAX)
                                - self.total_imported_count
                            ]
                            tqdm.write("Truncating data to limit to max rows")
                        if len(objects) == 0:
                            continue
                        response = collection.data.insert_many(objects)
                        if getattr(response, "has_errors", False):
                            tqdm.write(
                                "Some Weaviate objects failed to import: "
                                f"{response.errors}"
                            )
                        self.total_imported_count += len(objects)
                        if max_hit:
                            break
                    if max_hit:
                        break

                current_count = self._get_total_count(collection)
                tqdm.write(
                    "Imported "
                    f"{self.total_imported_count} rows into {new_collection_name}"
                )
                if previous_count >= 0 and current_count >= 0:
                    tqdm.write(
                        "Collection grew from "
                        f"{previous_count} to {current_count} objects"
                    )
                if max_hit:
                    break
            if max_hit:
                break
        self.args["imported_count"] = self.total_imported_count
        tqdm.write("Data imported successfully")

    def _create_collection(self, collection_name, vector_column_names):
        vector_config = self._get_vector_config(vector_column_names)
        self.client.collections.create(
            collection_name,
            vector_config=vector_config,
        )

    def _get_vector_config(self, vector_column_names):
        if len(vector_column_names) == 1 and vector_column_names[0] == "vector":
            return Configure.Vectors.self_provided()
        return [
            Configure.Vectors.self_provided(name=vector_column_name)
            for vector_column_name in vector_column_names
        ]

    def _build_objects(self, batch, vector_column_names):
        objects = []
        skipped = 0
        for _, row in batch.iterrows():
            vector = self._get_vector_payload(row, vector_column_names)
            if vector is None:
                skipped += 1
                continue
            original_id = str(row[self.id_column]) if self.id_column in row else ""
            uuid = (
                original_id
                if self._is_uuid(original_id)
                else generate_uuid5(original_id)
            )
            properties = self._get_properties(
                row, vector_column_names, original_id, uuid
            )
            objects.append(
                DataObject(
                    properties=properties,
                    uuid=uuid,
                    vector=vector,
                )
            )
        if skipped:
            tqdm.write(f"Skipped {skipped} rows with empty vector columns")
        return objects

    def _get_vector_payload(self, row, vector_column_names):
        vectors = {}
        for vector_column_name in vector_column_names:
            if vector_column_name not in row:
                continue
            vector = self.extract_vector(row[vector_column_name])
            if vector is None or len(vector) == 0:
                continue
            vectors[vector_column_name] = vector
        if len(vectors) == 0:
            return None
        if len(vectors) == 1 and vector_column_names[0] == "vector":
            return next(iter(vectors.values()))
        return vectors

    def _get_properties(self, row, vector_column_names, original_id, uuid):
        properties = {}
        for key, value in row.to_dict().items():
            if key in vector_column_names or key == self.id_column:
                continue
            value = self._to_weaviate_value(value)
            if value is not None:
                properties[key] = value
        if original_id != uuid:
            properties["_vdf_id"] = original_id
        return properties

    def _to_weaviate_value(self, value):
        if value is None:
            return None
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            return None
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, dict):
            converted = {}
            for k, v in value.items():
                converted_value = self._to_weaviate_value(v)
                if converted_value is not None:
                    converted[str(k)] = converted_value
            return converted
        if isinstance(value, (list, tuple)):
            return [
                converted
                for converted in (self._to_weaviate_value(v) for v in value)
                if converted is not None
            ]
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def _get_total_count(self, collection):
        try:
            return collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            return -1

    def _is_uuid(self, value):
        try:
            UUID(value)
            return True
        except (TypeError, ValueError):
            return False
