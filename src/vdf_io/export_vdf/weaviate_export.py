import argparse
import json
import os
from typing import Dict, List
from urllib.parse import urlparse

import pandas as pd
from tqdm import tqdm
import weaviate

from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password


def connect_weaviate_client(args):
    url = (
        args.get("url")
        or os.getenv("WEAVIATE_URL")
        or os.getenv("YOUR_WCS_URL")
        or "http://localhost:8080"
    )
    api_key = (
        args.get("api_key")
        or os.getenv("WEAVIATE_API_KEY")
        or os.getenv("YOUR_WCS_API_KEY")
    )
    use_local = args.get("local", False) or "localhost" in url or "127.0.0.1" in url

    if use_local and not api_key:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        host = parsed.hostname or "localhost"
        port = parsed.port or 8080
        grpc_port = int(args.get("grpc_port") or 50051)
        return weaviate.connect_to_local(
            host=host, port=port, grpc_port=grpc_port, skip_init_checks=True
        )

    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
    return weaviate.connect_to_wcs(
        cluster_url=url,
        auth_credentials=auth,
        skip_init_checks=True,
    )


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )
        parser_weaviate.add_argument(
            "-u", "--url", type=str, help="URL of Weaviate instance (local or WCS)"
        )
        parser_weaviate.add_argument(
            "--api_key",
            type=str,
            help="Weaviate API key (for WCS/authenticated instances)",
        )
        parser_weaviate.add_argument(
            "-c",
            "--classes",
            type=str,
            help="Classes/collections to export (comma-separated)",
        )
        parser_weaviate.add_argument(
            "--local",
            type=bool,
            help="Connect to local Weaviate instance (default: auto-detected from URL)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help="gRPC port for local Weaviate instance (default: 50051)",
            default=50051,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: 'http://localhost:8080'): ",
            str,
            default_value="http://localhost:8080",
            env_var="WEAVIATE_URL",
        )
        if "localhost" not in str(args.get("url", "")) and "127.0.0.1" not in str(
            args.get("url", "")
        ):
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
        self.client = connect_weaviate_client(self.args)
        self.all_classes = []

    def get_all_index_names(self) -> List[str]:
        return list(self.client.collections.list_all().keys())

    def get_index_names(self) -> List[str]:
        if not self.all_classes:
            self.all_classes = self.get_all_index_names()
        if not self.args.get("classes"):
            return self.all_classes
        input_classes = [
            c.strip() for c in self.args["classes"].split(",") if c.strip()
        ]
        missing = set(input_classes) - set(self.all_classes)
        if missing:
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {missing}"
            )
        return [c for c in self.all_classes if c in input_classes]

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for class_name in tqdm(index_names, desc="Exporting Weaviate classes"):
            index_metas[class_name] = self.get_data_for_class(class_name)

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def get_data_for_class(self, class_name: str) -> List[NamespaceMeta]:
        vectors_directory = self.create_vec_dir(class_name)
        collection = self.client.collections.get(class_name)
        agg = collection.aggregate.over_all(total_count=True)
        total = getattr(agg, "total_count", 0) or 0

        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        rows = []
        vector_columns: List[str] = []
        dim = 0
        num_vectors_exported = 0

        for item in tqdm(
            collection.iterator(include_vector=True),
            total=total,
            desc=f"Exporting {class_name}",
        ):
            row = {ID_COLUMN: str(item.uuid)}
            if item.properties:
                for k, v in item.properties.items():
                    row[k if k != ID_COLUMN else f"metadata_{k}"] = v

            vec = item.vector
            if isinstance(vec, dict):
                for vec_name, vec_val in vec.items():
                    col_name = vec_name if vec_name != "default" else "vector"
                    if col_name not in vector_columns:
                        vector_columns.append(col_name)
                    row[col_name] = vec_val
                    if not dim and isinstance(vec_val, list):
                        dim = len(vec_val)
            elif isinstance(vec, list):
                if "vector" not in vector_columns:
                    vector_columns.append("vector")
                row["vector"] = vec
                if not dim:
                    dim = len(vec)

            rows.append(row)
            if len(rows) >= batch_size:
                num_vectors_exported += self._flush_rows_to_parquet(
                    rows, vectors_directory
                )
                rows = []

        if rows:
            num_vectors_exported += self._flush_rows_to_parquet(rows, vectors_directory)

        if not vector_columns:
            vector_columns = ["vector"]

        distance_metric = "cosine"
        try:
            cfg = collection.config.get()
            if hasattr(cfg, "vector_index_config") and hasattr(
                cfg.vector_index_config, "distance_metric"
            ):
                distance_metric = str(cfg.vector_index_config.distance_metric.value)
        except Exception:
            pass

        namespace_meta = self.get_namespace_meta(
            class_name,
            vectors_directory,
            total=total or num_vectors_exported,
            num_vectors_exported=num_vectors_exported,
            dim=dim,
            vector_columns=vector_columns,
            distance=distance_metric,
        )
        self.args["exported_count"] += num_vectors_exported
        return [namespace_meta]

    def _flush_rows_to_parquet(self, rows: List[Dict], vectors_directory: str) -> int:
        df = pd.DataFrame.from_records(rows)
        parquet_file = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(parquet_file)
        self.file_structure.append(parquet_file)
        self.file_ctr += 1
        return len(df)
