import json
import os
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import weaviate

from vdf_io.constants import ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.weaviate_helpers import (
    find_distance_metric,
    first_vector_dimension,
    make_jsonable_config,
    split_weaviate_object,
)

MAX_EXPORT_BATCH_SIZE = 1_000


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate instance")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--classes", type=str, help="Classes to export (comma-separated)"
        )
        parser_weaviate.add_argument(
            "--connection_type",
            type=str,
            choices=["auto", "cloud", "local", "custom"],
            default="auto",
            help="Weaviate connection type (default: auto)",
        )
        parser_weaviate.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="Host for local/custom Weaviate connections",
        )
        parser_weaviate.add_argument(
            "--http_port",
            type=int,
            default=8080,
            help="HTTP port for local/custom Weaviate connections",
        )
        parser_weaviate.add_argument(
            "--grpc_host",
            type=str,
            help="gRPC host for custom Weaviate connections",
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            default=50051,
            help="gRPC port for local/custom Weaviate connections",
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance: ",
            str,
        )
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
            "Enter the name of the classes to export (comma-separated, all will be exported by default): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args):
        super().__init__(args)
        self.client = self._connect()

    def _api_key_auth(self):
        if not self.args.get("api_key"):
            return None
        try:
            from weaviate.classes.init import Auth

            return Auth.api_key(self.args["api_key"])
        except Exception:
            return weaviate.auth.AuthApiKey(self.args["api_key"])

    def _connect(self):
        connection_type = self.args.get("connection_type", "auto") or "auto"
        url = self.args.get("url")
        auth_credentials = self._api_key_auth()

        if connection_type == "auto":
            if url and ("weaviate.cloud" in url or "weaviate.network" in url):
                connection_type = "cloud"
            elif url:
                connection_type = "custom"
            else:
                connection_type = "local"

        if connection_type == "cloud":
            connector = getattr(weaviate, "connect_to_weaviate_cloud", None)
            if connector is None:
                connector = getattr(weaviate, "connect_to_wcs")
            return connector(
                cluster_url=url,
                auth_credentials=auth_credentials,
                skip_init_checks=True,
            )

        if connection_type == "local":
            kwargs = {
                "host": self.args.get("host", "localhost"),
                "port": self.args.get("http_port", 8080),
                "grpc_port": self.args.get("grpc_port", 50051),
            }
            if auth_credentials is not None:
                kwargs["auth_credentials"] = auth_credentials
            return weaviate.connect_to_local(**kwargs)

        from urllib.parse import urlparse

        parsed_url = urlparse(url if "://" in url else f"http://{url}")
        http_secure = parsed_url.scheme == "https"
        http_port = parsed_url.port or (
            443 if http_secure else self.args.get("http_port", 8080)
        )
        grpc_host = self.args.get("grpc_host") or parsed_url.hostname
        grpc_port = self.args.get("grpc_port") or (443 if http_secure else 50051)
        kwargs = {
            "http_host": parsed_url.hostname,
            "http_port": http_port,
            "http_secure": http_secure,
            "grpc_host": grpc_host,
            "grpc_port": grpc_port,
            "grpc_secure": http_secure,
        }
        if auth_credentials is not None:
            kwargs["auth_credentials"] = auth_credentials
        return weaviate.connect_to_custom(**kwargs)

    def get_all_index_names(self) -> List[str]:
        all_collections = self.client.collections.list_all()
        if isinstance(all_collections, dict):
            return list(all_collections.keys())
        return [
            getattr(collection, "name", str(collection))
            for collection in all_collections
        ]

    def get_index_names(self) -> List[str]:
        if not hasattr(self, "all_classes"):
            self.all_classes = self.get_all_index_names()
        if self.args.get("classes") is None:
            return self.all_classes
        else:
            input_classes = self.args["classes"].split(",")
            if set(input_classes) - set(self.all_classes):
                tqdm.write(
                    f"These classes are not present in the Weaviate instance: {set(input_classes) - set(self.all_classes)}"
                )
            return [c for c in self.all_classes if c in input_classes]

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for class_name in index_names:
            index_metas[class_name] = self.get_data_for_collection(class_name)

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def _get_collection(self, class_name):
        if hasattr(self.client.collections, "use"):
            return self.client.collections.use(class_name)
        return self.client.collections.get(class_name)

    def _collection_total(self, collection):
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count

    def get_data_for_collection(self, class_name) -> List[NamespaceMeta]:
        vectors_directory = self.create_vec_dir(class_name)
        collection = self._get_collection(class_name)
        total = self._collection_total(collection)
        config = collection.config.get()
        vector_columns: List[str] = []
        dim = -1
        exported_count = 0
        rows = []

        pbar = tqdm(total=total, desc=f"Exporting {class_name}")
        for obj in collection.iterator(include_vector=True):
            object_id, vectors, metadata = split_weaviate_object(obj)
            if not vectors:
                pbar.update(1)
                continue
            for vector_column in vectors:
                if vector_column not in vector_columns:
                    vector_columns.append(vector_column)
            if dim == -1:
                dim = first_vector_dimension(vectors)
            rows.append(self._make_vdf_row(object_id, vectors, metadata))
            if len(rows) >= MAX_EXPORT_BATCH_SIZE:
                exported_count += self.save_rows_to_parquet(rows, vectors_directory)
                rows = []
            pbar.update(1)

        if rows:
            exported_count += self.save_rows_to_parquet(rows, vectors_directory)

        index_config = make_jsonable_config(config)
        namespace_meta = self.get_namespace_meta(
            class_name,
            vectors_directory,
            total,
            exported_count,
            dim,
            index_config=index_config,
            vector_columns=vector_columns or ["vector"],
            distance=find_distance_metric(index_config),
        )
        self.args["exported_count"] += exported_count
        return [namespace_meta]

    def _make_vdf_row(self, object_id, vectors, metadata):
        row = {ID_COLUMN: object_id}
        for key, value in metadata.items():
            if key == ID_COLUMN:
                continue
            row[f"metadata_{key}" if key in vectors else key] = value
        for vector_column, vector in vectors.items():
            row[vector_column] = vector
        return row

    def save_rows_to_parquet(self, rows, vectors_directory):
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
