import os
import weaviate
import json
from tqdm import tqdm
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password

# Set these environment variables
URL = os.getenv("YOUR_WCS_URL")
APIKEY = os.getenv("YOUR_WCS_API_KEY")


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data into Weaviate"
        )

        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate instance")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--index_name", type=str, help="Name of the index in Weaviate"
        )

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance: ",
            str,
        )
        set_arg_from_password(
            args,
            "api_key",
            "Enter the Weaviate API key: ",
            "WEAVIATE_API_KEY",
        )
        set_arg_from_input(
            args,
            "index_name",
            "Enter the name of the index in Weaviate: ",
            str,
        )
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        if self.args["connection_type"] == "local":
            self.client = weaviate.connect_to_local()
        else:
            self.client = weaviate.connect_to_wcs(
                cluster_url=self.args["url"],
                auth_credentials=weaviate.auth.AuthApiKey(self.args["api_key"]),
                headers={'X-OpenAI-Api-key': self.args["openai_api_key"]}
                if self.args["openai_api_key"]
                else None,
                skip_init_checks=True,
            )

    def upsert_data(self):
        max_hit = False
        total_imported_count = 0

        # Iterate over the indexes and import the data
        for index_name, index_meta in tqdm(self.vdf_meta["indexes"].items(), desc="Importing indexes"):
            tqdm.write(f"Importing data for index '{index_name}'")
            for namespace_meta in index_meta:
                self.set_dims(namespace_meta, index_name)

            # Create or get the index
            index_name = self.create_new_name(index_name, self.client.collections.list_all().keys())
            index = self.client.collections.get(index_name)

            # Load data from the Parquet files
            data_path = namespace_meta["data_path"]
            final_data_path = self.get_final_data_path(data_path)
            parquet_files = self.get_parquet_files(final_data_path)

            vectors = {}
            metadata = {}

        #     for file in tqdm(parquet_files, desc="Loading data from parquet files"):
        #         file_path = os.path.join(final_data_path, file)
        #         df = self.read_parquet_progress(file_path)

        #         if len(vectors) > (self.args.get("max_num_rows") or INT_MAX):
        #             max_hit = True
        #             break

        #         self.update_vectors(vectors, vector_column_name, df)
        #         self.update_metadata(metadata, vector_column_names, df)
        #         if max_hit:
        #             break

        #     tqdm.write(f"Loaded {len(vectors)} vectors from {len(parquet_files)} parquet files")

        #     # Upsert the vectors and metadata to the Weaviate index in batches
        #     BATCH_SIZE = self.args.get("batch_size", 1000) or 1000
        #     current_batch_size = BATCH_SIZE
        #     start_idx = 0

        #     while start_idx < len(vectors):
        #         end_idx = min(start_idx + current_batch_size, len(vectors))

        #         batch_vectors = [
        #             (
        #                 str(id),
        #                 vector,
        #                 {
        #                     k: v
        #                     for k, v in metadata.get(id, {}).items()
        #                     if v is not None
        #                 } if len(metadata.get(id, {}).keys()) > 0 else None
        #             )
        #             for id, vector in list(vectors.items())[start_idx:end_idx]
        #         ]

        #         try:
        #             resp = index.batch.create(batch_vectors)
        #             total_imported_count += len(batch_vectors)
        #             start_idx += len(batch_vectors)
        #         except Exception as e:
        #             tqdm.write(f"Error upserting vectors for index '{index_name}', {e}")
        #             if current_batch_size < BATCH_SIZE / 100:
        #                 tqdm.write("Batch size is not the issue. Aborting import")
        #                 raise e
        #             current_batch_size = int(2 * current_batch_size / 3)
        #             tqdm.write(f"Reducing batch size to {current_batch_size}")
        #             continue

        # tqdm.write(f"Data import completed successfully. Imported {total_imported_count} vectors")
        # self.args["imported_count"] = total_imported_count