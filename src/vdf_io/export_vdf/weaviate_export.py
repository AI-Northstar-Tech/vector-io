import json
import os
import sys
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vdf_io.constants import DISK_SPACE_LIMIT, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.weaviate_util import (
    DEFAULT_WEAVIATE_GRPC_PORT,
    DEFAULT_WEAVIATE_URL,
    connect_weaviate,
    get_weaviate_collection,
    is_local_weaviate_url,
    list_weaviate_collection_names,
    serializable_weaviate_config,
)


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        parser_weaviate.add_argument(
            "--url",
            type=str,
            help="URL of a local or cloud Weaviate instance",
            default=DEFAULT_WEAVIATE_URL,
        )
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--classes", type=str, help="Collections/classes to export (comma-separated)"
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help=f"Weaviate gRPC port (default: {DEFAULT_WEAVIATE_GRPC_PORT})",
            default=DEFAULT_WEAVIATE_GRPC_PORT,
        )
        parser_weaviate.add_argument(
            "--skip_init_checks",
            type=bool,
            help="Skip Weaviate client startup checks (default: True)",
            default=True,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            f"Enter the URL of the Weaviate instance (default: '{DEFAULT_WEAVIATE_URL}'): ",
            str,
            DEFAULT_WEAVIATE_URL,
        )
        set_arg_from_input(
            args,
            "grpc_port",
            f"Enter the Weaviate gRPC port (default: {DEFAULT_WEAVIATE_GRPC_PORT}): ",
            int,
            DEFAULT_WEAVIATE_GRPC_PORT,
        )
        if not is_local_weaviate_url(args["url"]):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = weaviate_export.get_all_index_names()
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the classes to export (comma-separated, hit return to export all): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate(args)

    def get_all_index_names(self) -> List[str]:
        return list_weaviate_collection_names(self.client)

    def get_index_names(self) -> List[str]:
        if self.args.get("classes") is None:
            return self.get_all_index_names()

        all_classes = getattr(self, "all_classes", self.get_all_index_names())
        input_classes = self.args["classes"].split(",")
        missing_classes = set(input_classes) - set(all_classes)
        if missing_classes:
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {missing_classes}"
            )
        return [class_name for class_name in all_classes if class_name in input_classes]

    def get_data(self):
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for collection_name in tqdm(self.get_index_names(), desc="Exporting classes"):
            index_metas[collection_name] = self.get_data_for_collection(collection_name)

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def get_data_for_collection(self, collection_name):
        collection = get_weaviate_collection(self.client, collection_name)
        vectors_directory = self.create_vec_dir(collection_name)
        index_config = self.get_collection_config(collection)

        vectors_by_column = {}
        metadata = {}
        vector_columns = []
        num_vectors_exported = 0

        for item in tqdm(
            self.get_collection_iterator(collection),
            total=self.get_collection_count(collection),
            desc=f"Exporting {collection_name}",
        ):
            object_id = str(item.uuid)
            object_vectors = self.get_object_vectors(getattr(item, "vector", None))
            if not object_vectors:
                continue
            for vector_column, vector in object_vectors.items():
                if vector_column not in vector_columns:
                    vector_columns.append(vector_column)
                vectors_by_column.setdefault(vector_column, {})[object_id] = vector
            metadata[object_id] = dict(getattr(item, "properties", {}) or {})

            if sys.getsizeof(vectors_by_column) + sys.getsizeof(metadata) > DISK_SPACE_LIMIT:
                num_vectors_exported += self.save_weaviate_vectors_to_parquet(
                    vectors_by_column,
                    metadata,
                    vectors_directory,
                    vector_columns,
                )
                vectors_by_column = {}
                metadata = {}

        if vectors_by_column:
            num_vectors_exported += self.save_weaviate_vectors_to_parquet(
                vectors_by_column,
                metadata,
                vectors_directory,
                vector_columns,
            )

        dim = self.get_first_vector_dim(vectors_by_column)
        if dim == -1 and hasattr(self, "_last_vector_columns_dim"):
            dim = self._last_vector_columns_dim

        namespace_meta = self.get_namespace_meta(
            collection_name,
            vectors_directory,
            total=self.get_collection_count(collection),
            num_vectors_exported=num_vectors_exported,
            dim=dim,
            index_config=index_config,
            vector_columns=vector_columns or ["vector"],
            distance=self.get_collection_distance(index_config),
        )
        self.args["exported_count"] += num_vectors_exported
        return [namespace_meta]

    def get_collection_iterator(self, collection):
        try:
            return collection.iterator(include_vector=True)
        except TypeError:
            return collection.iterator()

    def get_collection_count(self, collection):
        try:
            return len(collection)
        except TypeError:
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count

    def get_collection_config(self, collection):
        if hasattr(collection, "config"):
            try:
                return serializable_weaviate_config(collection.config.get())
            except Exception:
                return None
        return None

    def get_collection_distance(self, index_config):
        def find_distance(value):
            if isinstance(value, dict):
                if "distance" in value and value["distance"]:
                    return value["distance"]
                for nested_value in value.values():
                    distance = find_distance(nested_value)
                    if distance:
                        return distance
            if isinstance(value, list):
                for item in value:
                    distance = find_distance(item)
                    if distance:
                        return distance
            return None

        return find_distance(index_config) or "cosine"

    def get_object_vectors(self, vector):
        if vector is None:
            return {}
        if hasattr(vector, "to_dict"):
            vector = vector.to_dict()
        if isinstance(vector, dict):
            if set(vector.keys()) == {"default"}:
                return {"vector": vector["default"]}
            return {
                str(vector_name): vector_value
                for vector_name, vector_value in vector.items()
                if vector_value is not None
            }
        return {"vector": vector}

    def get_first_vector_dim(self, vectors_by_column):
        for vectors in vectors_by_column.values():
            for vector in vectors.values():
                if vector is not None:
                    try:
                        return len(vector)
                    except TypeError:
                        return -1
        return -1

    def save_weaviate_vectors_to_parquet(
        self, vectors_by_column, metadata, vectors_directory, vector_columns
    ):
        rows_by_id = {}
        for vector_column, vectors in vectors_by_column.items():
            for object_id, vector in vectors.items():
                rows_by_id.setdefault(object_id, {ID_COLUMN: object_id})[
                    vector_column
                ] = vector
                if vector is not None:
                    self._last_vector_columns_dim = len(vector)

        for object_id, object_metadata in metadata.items():
            row = rows_by_id.setdefault(object_id, {ID_COLUMN: object_id})
            for key, value in object_metadata.items():
                column_name = key
                if column_name in vector_columns or column_name == ID_COLUMN:
                    column_name = f"metadata_{column_name}"
                row[column_name] = value

        if not rows_by_id:
            return 0

        df = pd.DataFrame.from_records(list(rows_by_id.values()))
        parquet_file = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(parquet_file)
        if not hasattr(self, "parquet_schema"):
            self.parquet_schema = pq.read_schema(parquet_file)
        else:
            self.parquet_schema = pa.unify_schemas(
                [self.parquet_schema, pq.read_schema(parquet_file)]
            )
        self.file_structure.append(parquet_file)
        self.file_ctr += 1
        return len(df)
