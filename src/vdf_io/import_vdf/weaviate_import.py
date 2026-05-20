from typing import Dict, List

from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN, INT_MAX
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    clean_value,
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.weaviate_util import (
    DEFAULT_WEAVIATE_GRPC_PORT,
    DEFAULT_WEAVIATE_URL,
    build_weaviate_vector_config,
    connect_weaviate,
    get_weaviate_collection,
    get_weaviate_object_uuid,
    is_local_weaviate_url,
    list_weaviate_collection_names,
    weaviate_collection_exists,
)


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
        )
        parser_weaviate.add_argument(
            "--url",
            type=str,
            help="URL of a local or cloud Weaviate instance",
            default=DEFAULT_WEAVIATE_URL,
        )
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help=f"Weaviate gRPC port (default: {DEFAULT_WEAVIATE_GRPC_PORT})",
            default=DEFAULT_WEAVIATE_GRPC_PORT,
        )
        parser_weaviate.add_argument(
            "--skip_init_checks",
            type=bool,
            help="Skip Weaviate client startup checks (default: True)",
            default=True,
        )

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            f"Enter the URL of the Weaviate instance (default: '{DEFAULT_WEAVIATE_URL}'): ",
            str,
            DEFAULT_WEAVIATE_URL,
        )
        set_arg_from_input(
            args,
            "grpc_port",
            f"Enter the Weaviate gRPC port (default: {DEFAULT_WEAVIATE_GRPC_PORT}): ",
            int,
            DEFAULT_WEAVIATE_GRPC_PORT,
        )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the batch size for importing data (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )
        if not is_local_weaviate_url(args["url"]):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )
        weaviate_import = cls(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate(args)

    def get_all_index_names(self):
        return list_weaviate_collection_names(self.client)

    def upsert_data(self):
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        if len(indexes_content) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        existing_collections = self.get_all_index_names()
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
                    new_collection_name, existing_collections
                )
                self.create_collection(new_collection_name, namespace_meta)
                collection = get_weaviate_collection(self.client, new_collection_name)
                existing_collections.append(new_collection_name)
                self.import_namespace(collection, namespace_meta, new_collection_name)

        tqdm.write("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count

    def create_collection(self, collection_name, namespace_meta):
        if weaviate_collection_exists(self.client, collection_name):
            return

        vector_columns, _ = self.get_vector_column_name(
            collection_name, namespace_meta, multi_vector_supported=True
        )
        vector_config = build_weaviate_vector_config(
            vector_columns, namespace_meta.get("metric")
        )
        if vector_config is not None:
            try:
                self.client.collections.create(
                    name=collection_name,
                    vector_config=vector_config,
                )
                return
            except TypeError:
                pass

        self.client.collections.create(
            name=collection_name,
            vectorizer_config=None,
        )

    def import_namespace(self, collection, namespace_meta, collection_name):
        data_path = namespace_meta["data_path"]
        final_data_path = self.get_final_data_path(data_path)
        parquet_files = self.get_parquet_files(final_data_path)
        vector_column_names, _ = self.get_vector_column_name(
            collection_name, namespace_meta, multi_vector_supported=True
        )

        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        max_rows = self.args.get("max_num_rows") or INT_MAX
        for file in tqdm(parquet_files, desc="Iterating parquet files"):
            file_path = self.get_file_path(final_data_path, file)
            df = self.read_parquet_progress(
                file_path,
                max_num_rows=max_rows - self.total_imported_count,
            )
            df = cleanup_df(df)
            for batch_df in tqdm(
                divide_into_batches(df, batch_size),
                desc="Importing batches",
                total=max(1, len(df) // batch_size),
            ):
                self.upsert_batch(collection, batch_df, vector_column_names)
                if self.total_imported_count >= max_rows:
                    tqdm.write(f"Max rows to be imported {max_rows} hit. Exiting")
                    return

    def upsert_batch(self, collection, batch_df, vector_column_names):
        with collection.batch.fixed_size(
            batch_size=self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        ) as batch:
            for _, row in batch_df.iterrows():
                vector_payload = self.build_vector_payload(row, vector_column_names)
                if not vector_payload:
                    continue
                properties = self.build_properties(row, vector_column_names)
                object_uuid = get_weaviate_object_uuid(row[self.id_column])
                if object_uuid != str(row[self.id_column]):
                    properties.setdefault("vdf_original_id", str(row[self.id_column]))
                batch.add_object(
                    properties=properties,
                    uuid=object_uuid,
                    vector=vector_payload,
                )
                self.total_imported_count += 1
                if batch.number_errors > 10:
                    tqdm.write("Batch import stopped due to excessive errors.")
                    break

        failed_objects = collection.batch.failed_objects
        if failed_objects:
            tqdm.write(f"Number of failed imports: {len(failed_objects)}")
            tqdm.write(f"First failed object: {failed_objects[0]}")

    def build_vector_payload(self, row, vector_column_names):
        vectors = {}
        for vector_column_name in vector_column_names:
            if vector_column_name not in row or row[vector_column_name] is None:
                continue
            vector = self.extract_vector(row[vector_column_name])
            if vector is not None:
                vectors[vector_column_name] = vector

        if not vectors:
            return None
        if len(vectors) == 1 and "vector" in vectors:
            return vectors["vector"]
        return vectors

    def build_properties(self, row, vector_column_names):
        properties = {}
        excluded_columns = set(vector_column_names) | {self.id_column, ID_COLUMN}
        for key, value in row.to_dict().items():
            if key in excluded_columns:
                continue
            value = clean_value(value)
            if value is not None:
                properties[key] = value
        return properties
