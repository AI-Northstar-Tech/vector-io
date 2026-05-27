import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import weaviate

from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
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
        parser_weaviate.add_argument(
            "--collections",
            type=str,
            help="Collection names to export (comma-separated)",
        )
        parser_weaviate.add_argument(
            "--classes",
            type=str,
            help="Deprecated alias for --collections",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for writing parquet files",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
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
        set_arg_from_input(
            args,
            "batch_size",
            (
                "Enter the batch size for exporting data "
                f"(default: {DEFAULT_BATCH_SIZE}): "
            ),
            int,
            DEFAULT_BATCH_SIZE,
        )
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_collections = weaviate_export.get_all_index_names()
        set_arg_from_input(
            args,
            "collections",
            (
                "Enter collection(s) to export "
                "(comma-separated, hit return to export all): "
            ),
            str,
            choices=weaviate_export.all_collections,
        )
        weaviate_export.get_data()
        return weaviate_export

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
        return [
            getattr(collection, "name", collection)
            for collection in collections
        ]

    def get_index_names(self):
        requested = self.args.get("collections") or self.args.get("classes")
        if not requested:
            return self.get_all_index_names()
        all_collections = set(self.get_all_index_names())
        input_collections = [
            name.strip() for name in requested.split(",") if name.strip()
        ]
        missing = set(input_collections) - all_collections
        if missing:
            tqdm.write(
                f"These collections are not present in the Weaviate instance: {missing}"
            )
        return [name for name in input_collections if name in all_collections]

    def get_data(self):
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        index_metas = {}
        for collection_name in tqdm(
            self.get_index_names(), desc="Exporting collections"
        ):
            collection = self.client.collections.get(collection_name)
            vectors_directory = self.create_vec_dir(collection_name)
            index_config = self._to_jsonable(collection.config.get(simple=False))
            distance = self._find_distance(index_config) or "cosine"
            total_count = self._get_total_count(collection)
            rows: List[Dict[str, Any]] = []
            exported_count = 0
            dim = -1
            vector_columns = set()

            for item in tqdm(
                collection.iterator(include_vector=True),
                desc=f"Exporting {collection_name} collection",
                total=total_count if total_count >= 0 else None,
            ):
                row, row_vector_columns, row_dim = self._object_to_row(item)
                rows.append(row)
                vector_columns.update(row_vector_columns)
                if dim == -1 and row_dim != -1:
                    dim = row_dim
                if len(rows) >= batch_size:
                    exported_count += self._save_rows_to_parquet(
                        rows, vectors_directory
                    )
                    rows = []

            if rows:
                exported_count += self._save_rows_to_parquet(rows, vectors_directory)

            namespace_metas = [
                self.get_namespace_meta(
                    collection_name,
                    vectors_directory,
                    total=total_count if total_count >= 0 else exported_count,
                    num_vectors_exported=exported_count,
                    dim=dim,
                    index_config=index_config,
                    vector_columns=sorted(vector_columns) or ["vector"],
                    distance=distance,
                )
            ]
            index_metas[collection_name] = namespace_metas

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4, default=str)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        self.client.close()
        return True

    def _object_to_row(self, item) -> Tuple[Dict[str, Any], Iterable[str], int]:
        row = {ID_COLUMN: str(getattr(item, "uuid", ""))}
        properties = getattr(item, "properties", None) or {}
        vector = getattr(item, "vector", None)
        vector_columns = []
        dim = -1

        if isinstance(vector, dict):
            for name, value in vector.items():
                vector_column = "vector" if name in ("default", "") else str(name)
                row[vector_column] = self._vector_to_list(value)
                vector_columns.append(vector_column)
        elif vector is not None:
            row["vector"] = self._vector_to_list(vector)
            vector_columns.append("vector")

        for key, value in properties.items():
            key = str(key)
            column = f"metadata_{key}" if key in row else key
            row[column] = self._to_jsonable(value)

        for vector_column in vector_columns:
            value = row.get(vector_column)
            if value is not None:
                dim = len(value)
                break
        return row, vector_columns, dim

    def _save_rows_to_parquet(self, rows, vectors_directory):
        df = pd.DataFrame.from_records(rows)
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

    def _get_total_count(self, collection):
        try:
            return collection.aggregate.over_all(total_count=True).total_count
        except Exception as exc:
            tqdm.write(f"Could not fetch Weaviate collection count: {exc}")
            return -1

    def _find_distance(self, value):
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key in ("distance", "distance_metric"):
                    if hasattr(nested_value, "value"):
                        return str(nested_value.value)
                    return str(nested_value)
                found = self._find_distance(nested_value)
                if found:
                    return found
        elif isinstance(value, list):
            for nested_value in value:
                found = self._find_distance(nested_value)
                if found:
                    return found
        return None

    def _vector_to_list(self, vector):
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            return vector.tolist()
        return list(vector)

    def _to_jsonable(self, value):
        if hasattr(value, "model_dump"):
            return self._to_jsonable(value.model_dump())
        if hasattr(value, "__dict__") and not isinstance(value, type):
            return self._to_jsonable(vars(value))
        if isinstance(value, dict):
            return {str(k): self._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_jsonable(v) for v in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
