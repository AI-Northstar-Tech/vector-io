import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import weaviate
except ImportError:
    weaviate = None

from vdf_io.constants import DEFAULT_BATCH_SIZE, DISK_SPACE_LIMIT, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    create_weaviate_index_class,
    set_arg_from_input,
    set_arg_from_password,
)


def connect_weaviate_client(args: Dict[str, Any]):
    """
    Connect to either a local, embedded, custom, or Weaviate Cloud Services (WCS) instance.
    """
    if weaviate is None:
        raise ImportError("The 'weaviate-client' package is required to use Weaviate.")

    if args.get("embedded"):
        return weaviate.connect_to_embedded()

    url = args.get("url") or "http://localhost:8080"
    api_key = args.get("api_key")
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    port = args.get("port") or parsed.port or (443 if parsed.scheme == "https" else 8080)
    grpc_port = args.get("grpc_port") or 50051

    is_local = args.get("local") or host in ("localhost", "127.0.0.1", "0.0.0.0")

    if is_local and not api_key:
        return weaviate.connect_to_local(
            host=host,
            port=int(port),
            grpc_port=int(grpc_port),
            skip_init_checks=True,
        )

    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
    if hasattr(weaviate, "connect_to_wcs"):
        return weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=auth,
            skip_init_checks=True,
        )
    return weaviate.connect_to_weaviate_cloud(
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
            "-u",
            "--url",
            type=str,
            help="URL of Weaviate instance (default: 'http://localhost:8080')",
        )
        parser_weaviate.add_argument(
            "--api_key",
            type=str,
            help="Weaviate API key (optional for local instances)",
        )
        parser_weaviate.add_argument(
            "-c",
            "--classes",
            type=str,
            help="Weaviate classes/collections to export (comma-separated). Exports all by default.",
        )
        parser_weaviate.add_argument(
            "--local",
            help="Connect to a local Weaviate instance (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--embedded",
            help="Connect to an embedded Weaviate instance (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--port",
            type=int,
            help="HTTP port for local Weaviate instance (default: 8080)",
            default=8080,
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help="gRPC port for local Weaviate instance (default: 50051)",
            default=50051,
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting objects (default: 10000)",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: 'http://localhost:8080'): ",
            str,
            "http://localhost:8080",
            env_var="YOUR_WCS_URL",
        )
        set_arg_from_input(
            args,
            "local",
            "Is this a local Weaviate instance? (default: False): ",
            bool,
            False,
        )
        if not args.get("local") and not args.get("embedded"):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key (press Enter if none): ",
                "WEAVIATE_API_KEY",
            )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter batch size for export (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )

        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = weaviate_export.get_all_index_names()
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the name of the classes to export (comma-separated, hit return to export all): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args, client=None):
        super().__init__(args)
        self.client = client if client is not None else connect_weaviate_client(self.args)
        self.generated_classes: Dict[str, type] = {}

    def get_all_index_names(self) -> List[str]:
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [str(c) for c in collections]

    def get_index_names(self) -> List[str]:
        if not hasattr(self, "all_classes") or self.all_classes is None:
            self.all_classes = self.get_all_index_names()
        if not self.args.get("classes"):
            return self.all_classes
        input_classes = [c.strip() for c in self.args["classes"].split(",") if c.strip()]
        missing = set(input_classes) - set(self.all_classes)
        if missing:
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {missing}"
            )
        return [c for c in self.all_classes if c in input_classes]

    def save_multi_vectors_to_parquet(
        self,
        records: List[Dict[str, Any]],
        vectors_directory: str,
    ) -> int:
        if not records:
            return 0
        df = pd.DataFrame.from_records(records)
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

    def _extract_collection_schema_and_metric(self, collection) -> Tuple[Dict[str, str], Optional[str], Optional[Dict[str, Any]]]:
        schema_map: Dict[str, str] = {}
        distance_metric: Optional[str] = "cosine"
        raw_config_dict: Optional[Dict[str, Any]] = None
        try:
            cfg = collection.config.get()
            if hasattr(cfg, "to_dict"):
                raw_config_dict = cfg.to_dict()
            if hasattr(cfg, "properties") and cfg.properties:
                for prop in cfg.properties:
                    dtype = str(getattr(prop, "data_type", "text")).lower()
                    schema_map[prop.name] = dtype
            if hasattr(cfg, "vector_index_config") and cfg.vector_index_config:
                dist = getattr(cfg.vector_index_config, "distance_metric", None)
                if dist is not None:
                    distance_metric = str(getattr(dist, "value", dist)).lower()
        except Exception:
            pass
        return schema_map, distance_metric, raw_config_dict

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE

        for class_name in tqdm(index_names, desc="Fetching Weaviate classes"):
            collection = self.client.collections.get(class_name)
            vectors_directory = self.create_vec_dir(class_name)

            schema_map, distance_metric, raw_config_dict = (
                self._extract_collection_schema_and_metric(collection)
            )
            # Automatically create Python class representing the exported index schema
            self.generated_classes[class_name] = create_weaviate_index_class(
                class_name, schema_map
            )

            try:
                agg_resp = collection.aggregate.over_all(total_count=True)
                total_count = int(getattr(agg_resp, "total_count", 0) or 0)
            except Exception:
                total_count = 0

            records_buffer: List[Dict[str, Any]] = []
            num_vectors_exported = 0
            detected_vector_columns: List[str] = []
            detected_dim: int = -1

            pbar = tqdm(total=total_count or None, desc=f"Exporting {class_name}")
            for item in collection.iterator(include_vector=True):
                obj_id = str(getattr(item, "uuid", getattr(item, "id", "")))
                props = dict(getattr(item, "properties", {}) or {})
                raw_vec = getattr(item, "vector", None)

                row: Dict[str, Any] = {ID_COLUMN: obj_id}
                # Rename any property colliding with ID_COLUMN
                for k, v in props.items():
                    col_key = f"metadata_{k}" if k == ID_COLUMN else k
                    row[col_key] = v

                # Handle both single vector and multiple named vectors per row
                if isinstance(raw_vec, dict):
                    if list(raw_vec.keys()) == ["default"]:
                        vec_list = raw_vec["default"]
                        col_name = "vector"
                        row[col_name] = vec_list
                        if col_name not in detected_vector_columns:
                            detected_vector_columns.append(col_name)
                        if detected_dim == -1 and isinstance(vec_list, list):
                            detected_dim = len(vec_list)
                    else:
                        for vec_name, vec_list in raw_vec.items():
                            row[vec_name] = vec_list
                            if vec_name not in detected_vector_columns:
                                detected_vector_columns.append(vec_name)
                            if detected_dim == -1 and isinstance(vec_list, list):
                                detected_dim = len(vec_list)
                elif isinstance(raw_vec, list):
                    row["vector"] = raw_vec
                    if "vector" not in detected_vector_columns:
                        detected_vector_columns.append("vector")
                    if detected_dim == -1:
                        detected_dim = len(raw_vec)

                records_buffer.append(row)
                pbar.update(1)

                if (
                    len(records_buffer) >= batch_size
                    or sys.getsizeof(records_buffer) > DISK_SPACE_LIMIT
                ):
                    num_vectors_exported += self.save_multi_vectors_to_parquet(
                        records_buffer, vectors_directory
                    )
                    records_buffer = []

            if records_buffer:
                num_vectors_exported += self.save_multi_vectors_to_parquet(
                    records_buffer, vectors_directory
                )
            pbar.close()

            if not detected_vector_columns:
                detected_vector_columns = ["vector"]
            if total_count == 0:
                total_count = num_vectors_exported

            namespace_meta = self.get_namespace_meta(
                class_name,
                vectors_directory.replace("\\", "/"),
                total=total_count,
                num_vectors_exported=num_vectors_exported,
                dim=detected_dim,
                index_config=raw_config_dict,
                vector_columns=detected_vector_columns,
                distance=distance_metric,
            )
            self.args["exported_count"] += num_vectors_exported
            index_metas[class_name] = [namespace_meta]

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True
