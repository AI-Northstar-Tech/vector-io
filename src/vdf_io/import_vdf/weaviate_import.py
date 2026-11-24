from __future__ import annotations

import os

import math
import uuid
from typing import Any, Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import weaviate
    from weaviate.classes.config import Configure, VectorDistances
    from weaviate.classes.data import DataObject
except ImportError:
    weaviate = None  # type: ignore[assignment]
    Configure = None  # type: ignore[assignment]
    VectorDistances = None  # type: ignore[assignment]
    DataObject = None  # type: ignore[assignment]

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.export_vdf.weaviate_export import connect_weaviate_client
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)

load_dotenv()


def to_weaviate_uuid(raw_id: Any) -> str:
    s = str(raw_id)
    try:
        return str(uuid.UUID(s))
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))


def resolve_weaviate_distance_enum(metric_str: Optional[str]):
    key = standardize_metric_reverse(metric_str, DBNames.WEAVIATE)
    if VectorDistances is None:
        return key
    mapping = {
        "cosine": VectorDistances.COSINE,
        "l2-squared": VectorDistances.L2_SQUARED,
        "l2_squared": VectorDistances.L2_SQUARED,
        "dot": VectorDistances.DOT,
        "manhattan": VectorDistances.MANHATTAN,
        "hamming": VectorDistances.HAMMING,
    }
    return mapping.get(str(key).lower(), VectorDistances.COSINE)


def normalize_weaviate_class_name(raw_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name)
    return (cleaned[0].upper() + cleaned[1:]) if cleaned else "VdfCollection"


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        p = subparsers.add_parser(cls.DB_NAME_SLUG, help="Import data into Weaviate")
        p.add_argument("-u", "--url", type=str, help="URL of Weaviate instance")
        p.add_argument("--api_key", type=str, help="Weaviate API key")
        p.add_argument(
            "--grpc_port", type=int, default=50051, help="Weaviate gRPC port"
        )
        p.add_argument("-b", "--batch_size", type=int, default=1000, help="Batch size")

    @classmethod
    def import_vdb(cls, args):
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
        set_arg_from_input(
            args,
            "batch_size",
            "Enter batch size (default: 1000): ",
            int,
            default_value=1000,
        )
        importer = cls(args)
        try:
            importer.upsert_data()
        finally:
            importer.close()
        return importer

    def __init__(self, args, client=None):
        args.setdefault("cwd", os.getcwd())
        super().__init__(args)
        self.client = client or connect_weaviate_client(
            url=self.args.get("url"),
            api_key=self.args.get("api_key"),
            grpc_port=int(self.args.get("grpc_port") or 50051),
        )
        self.total_imported_count = 0

    def close(self):
        fn = getattr(self.client, "close", None)
        if callable(fn):
            fn()

    def _ensure_collection(
        self,
        class_name: str,
        vector_columns: List[str],
        metric: Optional[str],
        existing: List[str],
    ) -> str:
        target = normalize_weaviate_class_name(class_name)
        if self.args.get("create_new") and target in existing:
            target = self.create_new_name(target, existing)
        if target in existing:
            return target
        dist_enum = resolve_weaviate_distance_enum(metric)
        if Configure is not None:
            if len(vector_columns) > 1:
                vec_cfg = [
                    Configure.NamedVectors.none(
                        name=vc,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=dist_enum
                        ),
                    )
                    for vc in vector_columns
                ]
                self.client.collections.create(name=target, vectorizer_config=vec_cfg)
            else:
                self.client.collections.create(
                    name=target,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=dist_enum
                    ),
                )
        else:
            self.client.collections.create(
                name=target, vector_columns=vector_columns, distance_metric=dist_enum
            )
        existing.append(target)
        return target

    @staticmethod
    def _coerce_vec(val: Any) -> Optional[List[float]]:
        if val is None:
            return None
        if isinstance(val, np.ndarray):
            return val.astype(float).tolist()
        if isinstance(val, (list, tuple)):
            return [float(x) for x in val]
        return None

    @staticmethod
    def _clean_prop(val: Any) -> Any:
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None
        if isinstance(val, np.generic):
            return val.item()
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    def upsert_data(self):
        self.total_imported_count = 0
        max_rows = self.args.get("max_num_rows") or INT_MAX
        batch_size = int(self.args.get("batch_size") or min(DEFAULT_BATCH_SIZE, 1000))
        indexes_content: Dict[str, List[Dict[str, Any]]] = self.vdf_meta["indexes"]
        if not indexes_content:
            raise ValueError("No indexes found in VDF_META.json")

        raw_ex = self.client.collections.list_all()
        existing = (
            list(raw_ex.keys())
            if isinstance(raw_ex, dict)
            else [str(x) for x in raw_ex]
        )
        max_hit = False

        for index_name, ns_metas in tqdm(
            indexes_content.items(), desc="Importing Weaviate classes"
        ):
            for ns_meta in ns_metas:
                vec_cols, _ = self.get_vector_column_name(index_name, ns_meta)
                if isinstance(vec_cols, str):
                    vec_cols = [vec_cols]
                raw_cls = index_name + (
                    f"_{ns_meta['namespace']}" if ns_meta.get("namespace") else ""
                )
                target_cls = self._ensure_collection(
                    raw_cls, vec_cols, ns_meta.get("metric"), existing
                )
                collection = self.client.collections.get(target_cls)

                final_path = self.get_final_data_path(ns_meta["data_path"])
                for pfile in self.get_parquet_files(final_path):
                    df = cleanup_df(
                        self.read_parquet_progress(
                            self.get_file_path(final_path, pfile),
                            max_num_rows=max_rows - self.total_imported_count,
                        )
                    )
                    for batch_df in divide_into_batches(df, batch_size):
                        if self.total_imported_count + len(batch_df) >= max_rows:
                            batch_df = batch_df[: max_rows - self.total_imported_count]
                            max_hit = True
                        objs = []
                        for _, row in batch_df.iterrows():
                            rd = row.to_dict()
                            obj_uuid = to_weaviate_uuid(
                                rd.pop(self.id_column, None) or uuid.uuid4()
                            )
                            if len(vec_cols) == 1:
                                vec_payload: Any = self._coerce_vec(
                                    rd.pop(vec_cols[0], None)
                                )
                            else:
                                vec_payload = {}
                                for vc in vec_cols:
                                    cv = self._coerce_vec(rd.pop(vc, None))
                                    if cv is not None:
                                        vec_payload[vc] = cv
                            props = {
                                k: self._clean_prop(v)
                                for k, v in rd.items()
                                if self._clean_prop(v) is not None
                            }
                            objs.append(
                                DataObject(
                                    uuid=obj_uuid, properties=props, vector=vec_payload
                                )
                                if DataObject is not None
                                else {
                                    "uuid": obj_uuid,
                                    "properties": props,
                                    "vector": vec_payload,
                                }
                            )
                        if objs:
                            collection.data.insert_many(objs)
                            self.total_imported_count += len(objs)
                        if max_hit:
                            break
                    if max_hit:
                        break
                if max_hit:
                    break
        return self.total_imported_count
