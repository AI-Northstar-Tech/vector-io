import os
import weaviate
from tqdm import tqdm
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.names import DBNames
from vdf_io.constants import INT_MAX, DEFAULT_BATCH_SIZE
from vdf_io.weaviate_util import prompt_for_creds

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
            "--connection-type",
            type=str,
            choices=["local", "cloud"],
            default="cloud",
            help="Type of connection to Weaviate (local or cloud)",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="batch size for fetching",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def import_vdb(cls, args):
        prompt_for_creds(args)
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
                headers={"X-OpenAI-Api-key": self.args.get("openai_api_key", "")},
                skip_init_checks=True,
            )

    def upsert_data(self):
        max_hit = False
        total_imported_count = 0

        # Iterate over the indexes and import the data
        for index_name, index_meta in tqdm(
            self.vdf_meta["indexes"].items(), desc="Importing indexes"
        ):
            tqdm.write(f"Importing data for index '{index_name}'")
            for namespace_meta in index_meta:
                self.set_dims(namespace_meta, index_name)

            # Create or get the index
            index_name = self.create_new_name(
                index_name, self.client.collections.list_all().keys()
            )

            # Load data from the Parquet files
            data_path = namespace_meta["data_path"]
            final_data_path = self.get_final_data_path(data_path)
            parquet_files = self.get_parquet_files(final_data_path)

            vectors = {}
            metadata = {}
            vector_column_names, vector_column_name = self.get_vector_column_name(
                index_name, namespace_meta
            )

            for file in tqdm(parquet_files, desc="Loading data from parquet files"):
                file_path = os.path.join(final_data_path, file)
                df = self.read_parquet_progress(file_path)

                if len(vectors) > (self.args.get("max_num_rows") or INT_MAX):
                    max_hit = True
                    break
                if len(vectors) + len(df) > (self.args.get("max_num_rows") or INT_MAX):
                    df = df.head(
                        (self.args.get("max_num_rows") or INT_MAX) - len(vectors)
                    )
                    max_hit = True
                self.update_vectors(vectors, vector_column_name, df)
                self.update_metadata(metadata, vector_column_names, df)
                if max_hit:
                    break

            tqdm.write(
                f"Loaded {len(vectors)} vectors from {len(parquet_files)} parquet files"
            )

            # Upsert the vectors and metadata to the Weaviate index in batches
            BATCH_SIZE = self.args.get("batch_size")

            with self.client.batch.fixed_size(batch_size=BATCH_SIZE) as batch:
                for _, vector in vectors.items():
                    batch.add_object(
                        vector=vector,
                        collection=index_name,
                        # TODO: Find way to add Metadata
                    )
                    total_imported_count += 1

        tqdm.write(
            f"Data import completed successfully. Imported {total_imported_count} vectors"
        )
        self.args["imported_count"] = total_imported_count
