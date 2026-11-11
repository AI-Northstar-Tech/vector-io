"""
Weaviate Import implementation for Vector-IO (#74).

Supports:
- Importing VDF Parquet/CSV bundles into local or cloud Weaviate instances
- Automatic collection/class creation with distance metric mapping
- Single and multi-vector per row batch import
- Interactive CLI options via make_parser & import_vdb
"""

from __future__ import annotations

import json
import os
from typing import Any

try:
    import pandas as pd
except ImportError:
    pd = None

from vdf_io.export_vdf.weaviate_export import (
    connect_weaviate_client,
    map_vdf_distance_to_weaviate,
)

try:
    from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN
    from vdf_io.import_vdf.vdf_import_cls import ImportVDB
    from vdf_io.names import DBNames
    from vdf_io.util import set_arg_from_input, set_arg_from_password
except ImportError:
    DEFAULT_BATCH_SIZE = 1000
    ID_COLUMN = "id"

    class DBNames:
        WEAVIATE = "weaviate"

    class ImportVDB:
        def __init__(self, args: dict[str, Any]):
            self.args = dict(args)

    def set_arg_from_input(
        args, key, prompt, type_name=str, default_value=None, **kwargs
    ):
        if args.get(key) is None and default_value is not None:
            args[key] = default_value

    def set_arg_from_password(args, key, prompt, env_var):
        if args.get(key) is None and os.getenv(env_var):
            args[key] = os.getenv(env_var)


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = getattr(DBNames, "WEAVIATE", "weaviate")

    @classmethod
    def make_parser(cls, subparsers: Any) -> None:
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data into Weaviate"
        )
        parser_weaviate.add_argument(
            "--url",
            type=str,
            default="http://localhost:8080",
            help="URL of Weaviate instance (local or WCS cloud)",
        )
        parser_weaviate.add_argument(
            "--api_key", type=str, default=None, help="Weaviate API key"
        )
        parser_weaviate.add_argument(
            "--local",
            action="store_true",
            help="Connect to a local Weaviate instance",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Batch size for Weaviate upsert",
        )

    @classmethod
    def import_vdb(
        cls, args: dict[str, Any], client_override: Any = None
    ) -> "ImportWeaviate":
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
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter batch size for import (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )
        importer = cls(args, client_override=client_override)
        importer.upsert_data()
        return importer

    def __init__(self, args: dict[str, Any], client_override: Any = None) -> None:
        super().__init__(args)
        self.client = connect_weaviate_client(
            self.args, client_override=client_override
        )
        self.imported_counts: dict[str, int] = {}

    def upsert_data(self) -> dict[str, int]:
        vdf_dir = (
            self.args.get("dir") or self.args.get("vdf_directory") or "./vdf_export"
        )
        meta_path = os.path.join(vdf_dir, "VDF_META.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"VDF_META.json not found in {vdf_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            vdf_meta = json.load(f)

        existing = set()
        if hasattr(self.client, "collections"):
            cols = self.client.collections.list_all()
            existing = set(
                cols.keys() if isinstance(cols, dict) else [str(c) for c in cols]
            )

        for index_name, ns_list in vdf_meta.get("indexes", {}).items():
            for ns_meta in ns_list:
                metric = map_vdf_distance_to_weaviate(ns_meta.get("metric", "Cosine"))
                vec_cols = ns_meta.get("vector_columns") or ["vector"]
                if index_name not in existing and hasattr(self.client, "collections"):
                    self.client.collections.create(
                        name=index_name,
                        distance_metric=metric,
                        vector_columns=vec_cols,
                    )
                    existing.add(index_name)

                collection = self.client.collections.get(index_name)
                data_path = ns_meta.get("data_path") or os.path.join(
                    vdf_dir, index_name, "1.parquet"
                )
                if not os.path.exists(data_path):
                    continue
                if pd is not None:
                    df = pd.read_parquet(data_path)
                    raw_rows = [r.to_dict() for _, r in df.iterrows()]
                else:
                    with open(data_path, "r", encoding="utf-8") as f:
                        raw_rows = json.load(f)
                count = 0
                for row_dict in raw_rows:
                    row_dict = dict(row_dict)
                    obj_id = str(row_dict.pop(ID_COLUMN, f"row-{count}"))
                    vectors: dict[str, list[float]] = {}
                    for vc in vec_cols:
                        if vc in row_dict and row_dict[vc] is not None:
                            val = row_dict.pop(vc)
                            vec_key = (
                                vc.replace("vector_", "")
                                if vc.startswith("vector_")
                                else "default"
                            )
                            vectors[vec_key] = list(val)

                    vector_payload: Any = (
                        vectors["default"]
                        if len(vectors) == 1 and "default" in vectors
                        else vectors
                    )
                    collection.insert(
                        uuid=obj_id,
                        properties=row_dict,
                        vector=vector_payload,
                    )
                    count += 1
                self.imported_counts[index_name] = count

        return self.imported_counts
