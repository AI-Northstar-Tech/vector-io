from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import weaviate
except ImportError:
    weaviate = None  # type: ignore[assignment]

from vdf_io.constants import DEFAULT_BATCH_SIZE, DISK_SPACE_LIMIT, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password


def connect_weaviate_client(
    url: Optional[str],
    api_key: Optional[str] = None,
    grpc_port: int = 50051,
):
    if weaviate is None:
        raise ImportError("weaviate-client>=4.0.0 is required for Weaviate support")

    raw_url = (url or "http://localhost:8080").strip()
    parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
    host = parsed.hostname or "localhost"
    scheme = (parsed.scheme or "http").lower()
    is_secure = scheme == "https"
    http_port = parsed.port or (443 if is_secure else 8080)
    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

    if host in ("localhost", "127.0.0.1") and not is_secure:
        return weaviate.connect_to_local(
            host=host,
            port=http_port,
            grpc_port=grpc_port,
            auth_credentials=auth,
            skip_init_checks=True,
        )

    if any(
        s in host for s in (".weaviate.network", ".weaviate.cloud", ".semi.network")
    ):
        connect_cloud = getattr(weaviate, "connect_to_weaviate_cloud", None) or getattr(
            weaviate, "connect_to_wcs"
        )
        return connect_cloud(
            cluster_url=raw_url, auth_credentials=auth, skip_init_checks=True
        )

    return weaviate.connect_to_custom(
        http_host=host,
        http_port=http_port,
        http_secure=is_secure,
        grpc_host=host,
        grpc_port=grpc_port,
        grpc_secure=is_secure,
        auth_credentials=auth,
        skip_init_checks=True,
    )


def generate_weaviate_dataclass_code(
    class_name: str, properties_meta: List[Dict[str, str]]
) -> str:
    weaviate_to_py = {
        "text": "str",
        "string": "str",
        "int": "int",
        "number": "float",
        "boolean": "bool",
        "bool": "bool",
        "uuid": "str",
        "date": "str",
    }
    lines = [
        "from dataclasses import dataclass",
        "from typing import Any, Optional",
        "",
        "@dataclass",
        f"class {class_name}:",
        "    id: str",
    ]
    for prop in properties_meta:
        pname = prop.get("name", "field")
        dtype = prop.get("data_type", "Any").lower()
        lines.append(
            f"    {pname}: Optional[{weaviate_to_py.get(dtype, 'Any')}] = None"
        )
    return "\n".join(lines) + "\n"


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        p = subparsers.add_parser(cls.DB_NAME_SLUG, help="Export data from Weaviate")
        p.add_argument("-u", "--url", type=str, help="URL of Weaviate instance")
        p.add_argument("--api_key", type=str, help="Weaviate API key")
        p.add_argument(
            "--grpc_port", type=int, default=50051, help="Weaviate gRPC port"
        )
        p.add_argument("-c", "--classes", type=str, help="Comma-separated classes")
        p.add_argument("-b", "--batch_size", type=int, default=DEFAULT_BATCH_SIZE)

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter Weaviate URL (default: 'http://localhost:8080'): ",
            str,
            default_value="http://localhost:8080",
        )
        set_arg_from_input(
            args,
            "grpc_port",
            "Enter Weaviate gRPC port (default: 50051): ",
            int,
            default_value=50051,
        )
        set_arg_from_password(
            args, "api_key", "Enter Weaviate API key: ", "WEAVIATE_API_KEY"
        )
        exporter = cls(args)
        try:
            exporter.all_classes = exporter.get_all_index_names()
            set_arg_from_input(
                exporter.args,
                "classes",
                "Enter classes to export (comma-separated, default all): ",
                str,
                choices=exporter.all_classes,
                default_value=None,
            )
            set_arg_from_input(
                exporter.args,
                "batch_size",
                "Enter batch size: ",
                int,
                default_value=DEFAULT_BATCH_SIZE,
            )
            exporter.get_data()
        finally:
            exporter.close()
        return exporter

    def __init__(self, args, client=None):
        super().__init__(args)
        self.client = client or connect_weaviate_client(
            url=self.args.get("url"),
            api_key=self.args.get("api_key"),
            grpc_port=int(self.args.get("grpc_port") or 50051),
        )
        self.all_classes: List[str] = []

    def close(self):
        fn = getattr(self.client, "close", None)
        if callable(fn):
            fn()

    def get_all_index_names(self) -> List[str]:
        raw = self.client.collections.list_all()
        return list(raw.keys()) if isinstance(raw, dict) else [str(x) for x in raw]

    def get_index_names(self) -> List[str]:
        if not self.all_classes:
            self.all_classes = self.get_all_index_names()
        raw_classes = self.args.get("classes")
        if not raw_classes:
            return list(self.all_classes)
        req = [c.strip() for c in str(raw_classes).split(",") if c.strip()]
        missing = set(req) - set(self.all_classes)
        if missing:
            tqdm.write(f"Classes not present in Weaviate: {missing}")
        return [c for c in self.all_classes if c in req]

    def _extract_config(self, collection) -> Dict[str, Any]:
        distance = "cosine"
        props_meta: List[Dict[str, str]] = []
        named_metrics: Dict[str, str] = {}
        try:
            cfg = collection.config.get()
            vec_idx = getattr(cfg, "vector_index_config", None)
            dist_obj = getattr(vec_idx, "distance_metric", None)
            if dist_obj is not None:
                distance = str(getattr(dist_obj, "value", dist_obj)).lower()
            vec_cfg = getattr(cfg, "vector_config", None)
            if isinstance(vec_cfg, dict):
                for vname, vdef in vec_cfg.items():
                    sub_dist = getattr(
                        getattr(vdef, "vector_index_config", None),
                        "distance_metric",
                        None,
                    )
                    if sub_dist is not None:
                        named_metrics[vname] = str(
                            getattr(sub_dist, "value", sub_dist)
                        ).lower()
                if named_metrics and distance == "cosine":
                    distance = next(iter(named_metrics.values()))
            for prop in getattr(cfg, "properties", []) or []:
                pname = getattr(prop, "name", None)
                dtype = getattr(prop, "data_type", None)
                if pname:
                    props_meta.append(
                        {
                            "name": str(pname),
                            "data_type": str(getattr(dtype, "value", dtype or "text")),
                        }
                    )
        except Exception:
            pass
        return {
            "distance": distance,
            "named_metrics": named_metrics,
            "properties": props_meta,
        }

    def _flush_rows(self, rows: List[Dict[str, Any]], vectors_directory: str) -> int:
        if not rows:
            return 0
        df = pd.DataFrame.from_records(rows)
        pf = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(pf, index=False)
        if not hasattr(self, "parquet_schema"):
            self.parquet_schema = pq.read_schema(pf)
        else:
            self.parquet_schema = pa.unify_schemas(
                [self.parquet_schema, pq.read_schema(pf)]
            )
        self.file_structure.append(pf)
        self.file_ctr += 1
        return len(df)

    def get_data(self):
        batch_size = int(self.args.get("batch_size") or DEFAULT_BATCH_SIZE)
        index_metas: Dict[str, List[NamespaceMeta]] = {}

        for class_name in tqdm(
            self.get_index_names(), desc="Exporting Weaviate classes"
        ):
            collection = self.client.collections.get(class_name)
            vectors_directory = self.create_vec_dir(class_name)
            cfg_info = self._extract_config(collection)

            schema_file = os.path.join(self.vdf_directory, f"{class_name}_schema.py")
            with open(schema_file, "w", encoding="utf-8") as sf:
                sf.write(
                    generate_weaviate_dataclass_code(class_name, cfg_info["properties"])
                )
            self.file_structure.append(schema_file)

            try:
                total_count = int(
                    getattr(
                        collection.aggregate.over_all(total_count=True),
                        "total_count",
                        0,
                    )
                    or 0
                )
            except Exception:
                total_count = 0

            rows_buf: List[Dict[str, Any]] = []
            exported_count = 0
            vec_cols: List[str] = []
            col_dims: Dict[str, int] = {}

            for item in collection.iterator(include_vector=True):
                row: Dict[str, Any] = {ID_COLUMN: str(getattr(item, "uuid", ""))}
                for pk, pv in dict(getattr(item, "properties", {}) or {}).items():
                    row[f"metadata_{pk}" if pk == ID_COLUMN else pk] = pv

                raw_vec = getattr(item, "vector", None)
                if isinstance(raw_vec, dict) and raw_vec:
                    if set(raw_vec.keys()) == {"default"}:
                        v = list(raw_vec["default"])
                        row["vector"] = v
                        if "vector" not in vec_cols:
                            vec_cols.append("vector")
                        col_dims["vector"] = len(v)
                    else:
                        for vcol, vlist in raw_vec.items():
                            if vlist is None:
                                continue
                            arr = list(vlist)
                            if vcol in row:
                                row[f"metadata_{vcol}"] = row.pop(vcol)
                            row[vcol] = arr
                            if vcol not in vec_cols:
                                vec_cols.append(vcol)
                            col_dims[vcol] = len(arr)
                elif isinstance(raw_vec, (list, tuple)):
                    v = list(raw_vec)
                    row["vector"] = v
                    if "vector" not in vec_cols:
                        vec_cols.append("vector")
                    col_dims["vector"] = len(v)

                rows_buf.append(row)
                if (
                    len(rows_buf) >= batch_size
                    or len(str(rows_buf)) >= DISK_SPACE_LIMIT
                ):
                    exported_count += self._flush_rows(rows_buf, vectors_directory)
                    rows_buf = []

            if rows_buf:
                exported_count += self._flush_rows(rows_buf, vectors_directory)

            if total_count == 0:
                total_count = exported_count
            if not vec_cols:
                vec_cols = ["vector"]

            ns_meta = self.get_namespace_meta(
                class_name,
                vectors_directory,
                total=total_count,
                num_vectors_exported=exported_count,
                dim=col_dims.get(vec_cols[0], -1),
                index_config={
                    "properties": cfg_info["properties"],
                    "named_metrics": cfg_info["named_metrics"],
                    "vector_dimensions": col_dims,
                },
                vector_columns=vec_cols,
                distance=cfg_info["distance"],
            )
            if ns_meta.model_map:
                for vc in vec_cols:
                    if vc in ns_meta.model_map and vc in col_dims:
                        ns_meta.model_map[vc]["dimensions"] = col_dims[vc]

            index_metas[class_name] = [ns_meta]
            self.args["exported_count"] += exported_count

        meta_path = os.path.join(self.vdf_directory, "VDF_META.json")
        self.file_structure.append(meta_path)
        meta_text = json.dumps(
            self.get_basic_vdf_meta(index_metas).model_dump(), indent=4
        )
        with open(meta_path, "w", encoding="utf-8") as jf:
            jf.write(meta_text)
        return True
