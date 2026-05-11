import argparse
import os
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.weaviate_util import (
    batch_context,
    build_vector_config,
    connect_to_weaviate,
    generate_weaviate_uuid,
    is_weaviate_cloud_url,
    list_collection_names,
    sanitize_properties,
    use_collection,
)


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
        )
        parser_weaviate.add_argument(
            "-u", "--url", type=str, help="URL of Weaviate instance"
        )
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--deployment",
            choices=["local", "cloud", "custom"],
            help="Weaviate deployment type. Inferred from --url by default.",
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help="gRPC port for local/custom Weaviate instances",
        )
        parser_weaviate.add_argument(
            "--grpc_secure",
            help="Use TLS for gRPC custom connections",
            default=None,
            action=argparse.BooleanOptionalAction,
        )

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: 'http://localhost:8080'): ",
            str,
            "http://localhost:8080",
        )
        if args.get("deployment") == "cloud" or is_weaviate_cloud_url(
            args.get("url")
        ):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )
        else:
            if os.getenv("WEAVIATE_API_KEY"):
                args["api_key"] = os.getenv("WEAVIATE_API_KEY")
            set_arg_from_input(
                args,
                "api_key",
                "Enter the Weaviate API key (hit return if auth is disabled): ",
                str,
                "DO_NOT_PROMPT",
                env_var="WEAVIATE_API_KEY",
            )
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_to_weaviate(self.args)

    def get_all_index_names(self):
        return list_collection_names(self.client)

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        if not indexes_content:
            raise ValueError("No indexes found in VDF_META.json")

        collections = self.get_all_index_names()
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                new_collection_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_collection_name = self.create_new_name(
                    new_collection_name, collections
                )
                vector_column_names, _ = self.get_vector_column_name(
                    new_collection_name,
                    namespace_meta,
                    multi_vector_supported=True,
                )
                if new_collection_name not in collections:
                    self.create_collection(
                        new_collection_name, vector_column_names, namespace_meta
                    )
                    collections.append(new_collection_name)

                collection = use_collection(self.client, new_collection_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    remaining = (
                        self.args.get("max_num_rows") or INT_MAX
                    ) - self.total_imported_count
                    if remaining <= 0:
                        max_hit = True
                        break
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(
                        file_path,
                        max_num_rows=remaining,
                    )
                    df = cleanup_df(df)
                    batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                    for batch in tqdm(
                        divide_into_batches(df, batch_size),
                        desc="Importing batches",
                        total=max(1, len(df) // batch_size),
                    ):
                        remaining = (
                            self.args.get("max_num_rows") or INT_MAX
                        ) - self.total_imported_count
                        if remaining <= 0:
                            max_hit = True
                            break
                        if len(batch) > remaining:
                            batch = batch.head(remaining)
                            max_hit = True
                        imported_count = self.upsert_batch(
                            collection,
                            new_collection_name,
                            batch,
                            vector_column_names,
                        )
                        self.total_imported_count += imported_count
                    if max_hit:
                        break
            if max_hit:
                tqdm.write(
                    f"Max rows to be imported {self.args['max_num_rows']} hit. Exiting"
                )
                break

        tqdm.write("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count

    def create_collection(self, collection_name, vector_column_names, namespace_meta):
        vector_config = build_vector_config(
            vector_column_names,
            namespace_meta.get("metric"),
        )
        create = self.client.collections.create
        try:
            create(collection_name, vector_config=vector_config)
        except TypeError:
            try:
                create(name=collection_name, vector_config=vector_config)
            except TypeError:
                create(name=collection_name, vectorizer_config=vector_config)

    def upsert_batch(
        self,
        collection,
        collection_name,
        df,
        vector_column_names,
    ):
        imported_count = 0
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        vector_column_names = [
            vector_column
            for vector_column in vector_column_names
            if vector_column in df.columns
        ]
        if not vector_column_names:
            tqdm.write(f"No vector columns found for collection {collection_name}")
            return 0
        df = df.dropna(subset=vector_column_names, how="all")
        with batch_context(collection, batch_size) as batch:
            for _, row in df.iterrows():
                vector_payload = self.get_vector_payload(row, vector_column_names)
                if vector_payload is None:
                    continue
                properties = sanitize_properties(
                    row.to_dict(),
                    excluded_columns=[self.id_column, *vector_column_names],
                )
                batch.add_object(
                    properties=properties,
                    uuid=generate_weaviate_uuid(collection_name, row[self.id_column]),
                    vector=vector_payload,
                )
                imported_count += 1
                if getattr(batch, "number_errors", 0) > 10:
                    raise RuntimeError("Stopping Weaviate import after 10 batch errors")

        failed_objects = getattr(collection.batch, "failed_objects", None)
        if failed_objects:
            tqdm.write(f"Number of failed Weaviate imports: {len(failed_objects)}")
            tqdm.write(f"First failed Weaviate import: {failed_objects[0]}")
        return imported_count

    def get_vector_payload(self, row, vector_column_names):
        vectors = {}
        for vector_column in vector_column_names:
            value = row.get(vector_column)
            if self.is_missing_vector(value):
                continue
            vector = self.extract_vector(value)
            if vector is not None:
                vectors[vector_column] = vector
        if not vectors:
            return None
        if list(vectors.keys()) == ["vector"]:
            return vectors["vector"]
        return vectors

    def is_missing_vector(self, value):
        if value is None:
            return True
        if isinstance(value, (list, tuple, dict)):
            return False
        try:
            is_missing = pd.isna(value)
        except (TypeError, ValueError):
            return False
        try:
            return bool(is_missing)
        except (TypeError, ValueError):
            return False
