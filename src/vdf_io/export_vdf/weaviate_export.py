"""
Weaviate Export implementation for Vector-IO (#74).

Supports:
- Distance metric mapping between Weaviate and VDF standard (Cosine, Dot, L2, Hamming, Manhattan)
- Both local (weaviate.connect_to_local / custom URL) and Weaviate Cloud Services (WCS) instances
- Exporting all collections/classes by default or user-specified comma-separated class names
- Interactive terminal input via set_arg_from_input / set_arg_from_password and CLI make_parser
- Single and multi-vector per row extraction into Parquet + VDF_META.json
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        @staticmethod
        def write(msg: str) -> None:
            print(msg)

try:
    import weaviate
except ImportError:
    weaviate = None

try:
    from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN
    from vdf_io.export_vdf.vdb_export_cls import ExportVDB
    from vdf_io.meta_types import IndexMeta, NamespaceMeta, VDFMeta
    from vdf_io.names import DBNames
    from vdf_io.util import (
        set_arg_from_input,
        set_arg_from_password,
        standardize_metric_reverse,
    )
except ImportError:
    DEFAULT_BATCH_SIZE = 1000
    ID_COLUMN = "id"

    class DBNames:
        WEAVIATE = "weaviate"

    class ExportVDB:
        def __init__(self, args: dict[str, Any]):
            self.args = dict(args)
            self.vdf_directory = self.args.get("vdf_directory", "./vdf_export")
            self.file_structure: list[str] = []

    def set_arg_from_input(args, key, prompt, type_name=str, default_value=None, **kwargs):
        if args.get(key) is None and default_value is not None:
            args[key] = default_value

    def set_arg_from_password(args, key, prompt, env_var):
        if args.get(key) is None and os.getenv(env_var):
            args[key] = os.getenv(env_var)


WEAVIATE_TO_VDF_DISTANCE = {
    "cosine": "Cosine",
    "dot": "Dot",
    "l2-squared": "Euclid",
    "euclidean": "Euclid",
    "hamming": "Hamming",
    "manhattan": "Manhattan",
}

VDF_TO_WEAVIATE_DISTANCE = {
    "Cosine": "cosine",
    "Dot": "dot",
    "Euclid": "l2-squared",
    "L2": "l2-squared",
    "Hamming": "hamming",
    "Manhattan": "manhattan",
}


def map_weaviate_distance_to_vdf(metric_name: str | None) -> str:
    if not metric_name:
        return "Cosine"
    cleaned = str(metric_name).strip().lower()
    return WEAVIATE_TO_VDF_DISTANCE.get(cleaned, "Cosine")


def map_vdf_distance_to_weaviate(vdf_metric: str | None) -> str:
    if not vdf_metric:
        return "cosine"
    for k, v in VDF_TO_WEAVIATE_DISTANCE.items():
        if k.lower() == str(vdf_metric).strip().lower():
            return v
    return "cosine"


def generate_python_class_for_collection(class_name: str, properties: list[dict[str, Any]]) -> str:
    """Automatically generate a typed Python dataclass definition for an exported Weaviate class."""
    type_map = {
        "text": "str",
        "string": "str",
        "int": "int",
        "number": "float",
        "boolean": "bool",
        "bool": "bool",
        "date": "str",
        "uuid": "str",
    }
    lines = [
        "from dataclasses import dataclass",
        "from typing import Any, List, Optional",
        "",
        "@dataclass",
        f"class {class_name}Record:",
        "    id: str",
    ]
    for prop in properties:
        p_name = str(prop.get("name", "field")).replace("-", "_")
        raw_dt = prop.get("dataType") or prop.get("data_type") or ["Any"]
        dt_key = str(raw_dt[0] if isinstance(raw_dt, list) else raw_dt).lower()
        py_type = type_map.get(dt_key, "Any")
        lines.append(f"    {p_name}: Optional[{py_type}] = None")
    lines.append("    vector: Optional[List[float]] = None")
    return "\n".join(lines) + "\n"


def connect_weaviate_client(args: dict[str, Any], client_override: Any = None) -> Any:
    if client_override is not None:
        return client_override
    if weaviate is None:
        raise RuntimeError("weaviate-client is required to connect to a live Weaviate instance")

    url = str(args.get("url") or "http://localhost:8080").strip()
    api_key = args.get("api_key")
    use_local = bool(args.get("local", False)) or "localhost" in url or "127.0.0.1" in url

    if use_local and hasattr(weaviate, "connect_to_local"):
        port = int(args.get("port", 8080))
        grpc_port = int(args.get("grpc_port", 50051))
        return weaviate.connect_to_local(port=port, grpc_port=grpc_port, skip_init_checks=True)

    auth_creds = (
        weaviate.auth.AuthApiKey(api_key)
        if api_key and hasattr(weaviate, "auth")
        else None
    )
    return weaviate.connect_to_wcs(
        cluster_url=url,
        auth_credentials=auth_creds,
        skip_init_checks=True,
    )


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = getattr(DBNames, "WEAVIATE", "weaviate")

    @classmethod
    def make_parser(cls, subparsers: Any) -> None:
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )
        parser_weaviate.add_argument(
            "--url",
            type=str,
            default="http://localhost:8080",
            help="URL of Weaviate instance (local or WCS cloud)",
        )
        parser_weaviate.add_argument(
            "--api_key", type=str, default=None, help="Weaviate API key (for WCS cloud)"
        )
        parser_weaviate.add_argument(
            "--local",
            action="store_true",
            help="Connect to a local Weaviate instance",
        )
        parser_weaviate.add_argument(
            "--classes",
            type=str,
            default=None,
            help="Comma-separated Weaviate classes/collections to export (default: all)",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Batch size for exporting objects",
        )

    @classmethod
    def export_vdb(cls, args: dict[str, Any], client_override: Any = None) -> "ExportWeaviate":
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: 'http://localhost:8080'): ",
            str,
            "http://localhost:8080",
        )
        set_arg_from_password(
            args,
            "api_key",
            "Enter the Weaviate API key (leave blank for local unauthenticated): ",
            "WEAVIATE_API_KEY",
        )
        exporter = cls(args, client_override=client_override)
        exporter.all_classes = exporter._list_all_collections()
        set_arg_from_input(
            exporter.args,
            "classes",
            "Enter the name of the classes to export (comma-separated, all will be exported by default): ",
            str,
            choices=exporter.all_classes,
        )
        exporter.get_data()
        return exporter

    def __init__(self, args: dict[str, Any], client_override: Any = None) -> None:
        super().__init__(args)
        self.client = connect_weaviate_client(self.args, client_override=client_override)
        self.all_classes = self._list_all_collections()
        self.index_metas: dict[str, list[dict[str, Any]]] = {}

    def _list_all_collections(self) -> list[str]:
        if hasattr(self.client, "collections"):
            cols = self.client.collections.list_all()
            if isinstance(cols, dict):
                return list(cols.keys())
            return [str(c) for c in cols]
        return []

    def get_index_names(self) -> list[str]:
        raw = self.args.get("classes")
        if not raw:
            return list(self.all_classes)
        requested = [c.strip() for c in str(raw).split(",") if c.strip()]
        missing = set(requested) - set(self.all_classes)
        if missing:
            tqdm.write(f"Warning: classes not found in Weaviate instance: {sorted(missing)}")
        return [c for c in self.all_classes if c in requested]

    def export_collection(self, class_name: str) -> dict[str, Any]:
        collection = self.client.collections.get(class_name)
        records: list[dict[str, Any]] = []
        vector_columns: dict[str, int] = {}
        distance_metric = "Cosine"

        if hasattr(collection, "config") and hasattr(collection.config, "get"):
            cfg = collection.config.get()
            raw_dist = getattr(cfg, "distance", None)
            distance_metric = map_weaviate_distance_to_vdf(raw_dist)
            props = getattr(cfg, "properties", []) or []
        else:
            props = []

        iterator = (
            collection.iterator(include_vector=True)
            if hasattr(collection, "iterator")
            else getattr(collection, "objects", [])
        )

        for obj in iterator:
            row: dict[str, Any] = {
                ID_COLUMN: str(getattr(obj, "uuid", getattr(obj, "id", ""))),
            }
            obj_props = getattr(obj, "properties", {}) or {}
            row.update(obj_props)

            vec = getattr(obj, "vector", None)
            if isinstance(vec, dict):
                for vec_name, vec_vals in vec.items():
                    col_name = f"vector_{vec_name}" if vec_name != "default" else "vector"
                    row[col_name] = list(vec_vals)
                    vector_columns[col_name] = len(vec_vals)
            elif isinstance(vec, (list, tuple)):
                row["vector"] = list(vec)
                vector_columns["vector"] = len(vec)

            records.append(row)

        out_dir = os.path.join(getattr(self, "vdf_directory", "./vdf_export"), class_name)
        os.makedirs(out_dir, exist_ok=True)
        parquet_path = os.path.join(out_dir, "1.parquet")
        if pd is not None:
            df = pd.DataFrame(records)
            if not df.empty:
                df.to_parquet(parquet_path, index=False)
            else:
                df.to_csv(os.path.join(out_dir, "empty.csv"), index=False)
        else:
            with open(parquet_path, "w", encoding="utf-8") as f:
                json.dump(records, f)

        schema_py_path = os.path.join(out_dir, f"{class_name.lower()}_model.py")
        with open(schema_py_path, "w", encoding="utf-8") as f:
            f.write(generate_python_class_for_collection(class_name, props))

        meta = {
            "namespace": class_name,
            "index_name": class_name,
            "total_vector_count": len(records),
            "exported_vector_count": len(records),
            "dimensions": next(iter(vector_columns.values()), 0),
            "vector_columns": list(vector_columns.keys()) or ["vector"],
            "metric": distance_metric,
            "data_path": parquet_path,
            "schema_model_path": schema_py_path,
        }
        self.index_metas[class_name] = [meta]
        return meta

    def get_data(self) -> dict[str, Any]:
        exported = {}
        for class_name in self.get_index_names():
            exported[class_name] = self.export_collection(class_name)

        meta_path = os.path.join(getattr(self, "vdf_directory", "./vdf_export"), "VDF_META.json")
        os.makedirs(os.path.dirname(meta_path), exist_ok=True)
        vdf_meta = {
            "version": "1.0.0",
            "file_structure": getattr(self, "file_structure", []),
            "author": "vdf_io",
            "exported_from": self.DB_NAME_SLUG,
            "indexes": self.index_metas,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(vdf_meta, f, indent=2)
        return vdf_meta
