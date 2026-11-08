import argparse
from typing import Any

from tqdm import tqdm

try:
    import weaviate
    from weaviate.classes.config import Configure, VectorDistances
    from weaviate.classes.data import DataObject
except ImportError:
    weaviate = None
    Configure = None
    VectorDistances = None
    DataObject = None

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.export_vdf.weaviate_export import connect_weaviate_client
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    clean_value,
    cleanup_df,
    divide_into_batches,
    get_qdrant_id_from_id,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
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
            help="Batch size for importing objects (default: 1000)",
            default=1000,
        )

    @classmethod
    def import_vdb(cls, args):
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
            "Enter batch size for import (default: 1000): ",
            int,
            1000,
        )
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args, client=None):
        super().__init__(args)
        self.client = (
            client if client is not None else connect_weaviate_client(self.args)
        )

    def get_all_index_names(self) -> list[str]:
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [str(c) for c in collections]

    def _resolve_weaviate_distance(self, metric_str: str):
        rev = standardize_metric_reverse(metric_str, self.DB_NAME_SLUG)
        if VectorDistances is None:
            return rev
        mapping = {
            "cosine": VectorDistances.COSINE,
            "l2-squared": VectorDistances.L2_SQUARED,
            "dot": VectorDistances.DOT,
            "manhattan": VectorDistances.MANHATTAN,
        }
        return mapping.get(rev, VectorDistances.COSINE)

    def _ensure_collection_exists(
        self,
        collection_name: str,
        existing_collections: list[str],
        vector_column_names: list[str],
        distance_metric_str: str,
    ):
        if collection_name in existing_collections:
            return self.client.collections.get(collection_name)

        dist_enum = self._resolve_weaviate_distance(distance_metric_str)
        try:
            if Configure is not None and len(vector_column_names) > 1:
                named_cfg = [
                    Configure.NamedVectors.none(
                        name=vec_col,
                        vector_index_config=Configure.VectorIndex.hnsw(
                            distance_metric=dist_enum
                        ),
                    )
                    for vec_col in vector_column_names
                ]
                return self.client.collections.create(
                    name=collection_name,
                    vectorizer_config=named_cfg,
                )
            elif Configure is not None:
                return self.client.collections.create(
                    name=collection_name,
                    vectorizer_config=Configure.Vectorizer.none(),
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=dist_enum
                    ),
                )
            else:
                return self.client.collections.create(name=collection_name)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"Notice when creating collection '{collection_name}': {e}")
            return self.client.collections.get(collection_name)

    def upsert_data(self):
        self.total_imported_count = 0
        max_hit = False
        indexes_content: dict[str, list[NamespaceMeta]] = self.vdf_meta["indexes"]
        if not indexes_content:
            raise ValueError("No indexes found in VDF_META.json")

        existing_collections = self.get_all_index_names()
        batch_size = min(self.args.get("batch_size") or 1000, DEFAULT_BATCH_SIZE)

        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing Weaviate classes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                raw_target_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta.get("namespace")
                    else ""
                )
                # Ensure PascalCase for Weaviate class naming convention
                if raw_target_name:
                    raw_target_name = raw_target_name[0].upper() + raw_target_name[1:]
                new_collection_name = self.create_new_name(
                    raw_target_name, existing_collections, delimiter="_"
                )

                vector_column_names, _ = self.get_vector_column_name(
                    new_collection_name, namespace_meta, multi_vector_supported=True
                )
                metric = namespace_meta.get("metric") or "Cosine"

                collection = self._ensure_collection_exists(
                    new_collection_name,
                    existing_collections,
                    vector_column_names,
                    metric,
                )
                if new_collection_name not in existing_collections:
                    existing_collections.append(new_collection_name)

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

                    for batch_df in tqdm(
                        divide_into_batches(df, batch_size),
                        desc=f"Uploading batches to {new_collection_name}",
                        total=max(1, len(df) // batch_size),
                    ):
                        objects_to_insert = []
                        for _, row in batch_df.iterrows():
                            raw_id = row.get(self.id_column, None)
                            obj_uuid = (
                                str(get_qdrant_id_from_id(str(raw_id)))
                                if raw_id is not None
                                else None
                            )

                            props: dict[str, Any] = {}
                            for col in batch_df.columns:
                                if col == self.id_column or col in vector_column_names:
                                    continue
                                val = clean_value(row[col])
                                if val is not None:
                                    props[col] = val

                            if len(vector_column_names) > 1:
                                vec_payload: Any = {}
                                for vc in vector_column_names:
                                    if vc in row and row[vc] is not None:
                                        vec_payload[vc] = self.extract_vector(row[vc])
                            else:
                                vc = vector_column_names[0]
                                vec_payload = (
                                    self.extract_vector(row[vc])
                                    if vc in row and row[vc] is not None
                                    else None
                                )

                            if DataObject is not None:
                                objects_to_insert.append(
                                    DataObject(
                                        properties=props,
                                        vector=vec_payload,
                                        uuid=obj_uuid,
                                    )
                                )
                            else:
                                objects_to_insert.append(
                                    {
                                        "properties": props,
                                        "vector": vec_payload,
                                        "uuid": obj_uuid,
                                    }
                                )

                        if objects_to_insert:
                            collection.data.insert_many(objects_to_insert)
                            self.total_imported_count += len(objects_to_insert)

                        if self.total_imported_count >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            max_hit = True
                            break
                    if max_hit:
                        break
                if max_hit:
                    break
            if max_hit:
                break

        tqdm.write(
            f"Weaviate data import completed successfully. {self.total_imported_count} objects imported."
        )
        self.args["imported_count"] = self.total_imported_count
        return True
