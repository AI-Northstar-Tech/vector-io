import argparse
from typing import Dict, List
import uuid

from dotenv import load_dotenv
from tqdm import tqdm
import weaviate.classes.config as wvc

from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN, INT_MAX
from vdf_io.export_vdf.weaviate_export import connect_weaviate_client
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)

load_dotenv()

WEAVIATE_DISTANCE_MAP = {
    "cosine": wvc.VectorDistances.COSINE,
    "l2-squared": wvc.VectorDistances.L2_SQUARED,
    "dot": wvc.VectorDistances.DOT,
    "manhattan": wvc.VectorDistances.MANHATTAN,
}


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
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
            help="Target Weaviate classes/collections (comma-separated)",
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
    def import_vdb(cls, args):
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
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate_client(self.args)

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        if not indexes_content:
            raise ValueError("No indexes found in VDF_META.json")

        existing_collections = list(self.client.collections.list_all().keys())

        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing Weaviate indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                raw_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta.get("namespace")
                    else ""
                )
                class_name = raw_name[:1].upper() + raw_name[1:].replace("-", "_")
                class_name = self.create_new_name(class_name, existing_collections)

                vector_column_names, is_multi_vector = self.get_vector_column_name(
                    class_name, namespace_meta, multi_vector_supported=True
                )

                metric_str = standardize_metric_reverse(
                    namespace_meta.get("metric", self.args.get("metric", "Cosine")),
                    self.DB_NAME_SLUG,
                )
                wv_distance = WEAVIATE_DISTANCE_MAP.get(
                    metric_str, wvc.VectorDistances.COSINE
                )

                if class_name not in existing_collections:
                    if is_multi_vector and len(vector_column_names) > 1:
                        vec_cfg = [
                            wvc.Configure.NamedVectors.none(
                                name=col,
                                vector_index_config=wvc.Configure.VectorIndex.hnsw(
                                    distance_metric=wv_distance
                                ),
                            )
                            for col in vector_column_names
                        ]
                    else:
                        vec_cfg = wvc.Configure.Vectorizer.none()
                    self.client.collections.create(
                        name=class_name,
                        vectorizer_config=vec_cfg,
                        vector_index_config=(
                            None
                            if (is_multi_vector and len(vector_column_names) > 1)
                            else wvc.Configure.VectorIndex.hnsw(
                                distance_metric=wv_distance
                            )
                        ),
                    )
                    existing_collections.append(class_name)
                    tqdm.write(f"Created Weaviate collection '{class_name}'")

                collection = self.client.collections.get(class_name)
                id_col = self.args.get("id_column") or ID_COLUMN
                batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE

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

                    for batch_df in divide_into_batches(df, batch_size):
                        if self.total_imported_count + len(batch_df) >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            batch_df = batch_df[
                                : (self.args.get("max_num_rows") or INT_MAX)
                                - self.total_imported_count
                            ]
                            max_hit = True

                        with collection.batch.fixed_size(
                            batch_size=batch_size
                        ) as batch:
                            for _, row in batch_df.iterrows():
                                row_dict = row.to_dict()
                                raw_id = row_dict.pop(id_col, None)
                                try:
                                    obj_uuid = (
                                        str(uuid.UUID(str(raw_id)))
                                        if raw_id
                                        else str(uuid.uuid4())
                                    )
                                except Exception:
                                    obj_uuid = str(
                                        uuid.uuid5(uuid.NAMESPACE_DNS, str(raw_id))
                                    )

                                if len(vector_column_names) > 1:
                                    vec_payload = {
                                        col: list(row_dict.pop(col))
                                        for col in vector_column_names
                                        if col in row_dict and row_dict[col] is not None
                                    }
                                else:
                                    vcol = vector_column_names[0]
                                    vec_payload = (
                                        list(row_dict.pop(vcol))
                                        if vcol in row_dict
                                        and row_dict[vcol] is not None
                                        else None
                                    )

                                batch.add_object(
                                    properties=row_dict,
                                    uuid=obj_uuid,
                                    vector=vec_payload,
                                )
                        self.total_imported_count += len(batch_df)
                        if max_hit:
                            break
                    if max_hit:
                        break
            if max_hit:
                break

        self.args["imported_count"] = self.total_imported_count
        tqdm.write(
            f"Successfully imported {self.total_imported_count} objects into Weaviate."
        )
