from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm

from astrapy.db import AstraDB

from vdf_io.constants import INT_MAX
from vdf_io.names import DBNames
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)

load_dotenv()


class ImportAstraDB(ImportVDB):
    DB_NAME_SLUG = DBNames.ASTRADB

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Datastax Astra DB
        """
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of AstraDB instance (default: value of os.environ['ASTRA_DB_API_ENDPOINT']): ",
            str,
            env_var="ASTRA_DB_API_ENDPOINT",
        )
        set_arg_from_password(
            args,
            "astradb_api_key",
            "Enter the AstraDB API key (default: value of os.environ['ASTRA_DB_APPLICATION_TOKEN']): ",
            "ASTRA_DB_APPLICATION_TOKEN",
        )
        astradb_import = ImportAstraDB(args)
        astradb_import.upsert_data()
        return astradb_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_astradb = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Datastax Astra DB"
        )
        parser_astradb.add_argument(
            "--endpoint", type=str, help="Location of AstraDB instance"
        )
        parser_astradb.add_argument(
            "--astradb_api_key", type=str, help="AstraDB API key"
        )

    def __init__(self, args):
        super().__init__(args)
        self.db = AstraDB(
            token=self.args.get("astradb_api_key"),
            api_endpoint=self.args.get("endpoint"),
        )

    def upsert_data(self):
        self.total_imported_count = 0
        max_hit = False
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        # Load Parquet file
        # print(indexes_content[index_names[0]]):List[NamespaceMeta]
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                index_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_index_name = index_name
                # create collection
                collection = self.db.create_collection(
                    new_index_name,
                    dimension=namespace_meta["dimensions"],
                    metric=standardize_metric_reverse(
                        namespace_meta["distance_metric"], self.DB_NAME_SLUG
                    ),
                )
                parquet_files = self.get_parquet_files(final_data_path)
                vectors = {}
                metadata = {}
                for parquet_file in tqdm(parquet_files, desc="Importing parquet files"):
                    (
                        vector_column_names,
                        vector_column_name,
                    ) = self.get_vector_column_name(index_name, namespace_meta)

                    parquet_file_path = self.get_file_path(
                        final_data_path, parquet_file
                    )

                    df = self.read_parquet_progress(parquet_file_path)
                    if len(vectors) > (self.args.get("max_num_rows") or INT_MAX):
                        max_hit = True
                        break
                    if len(vectors) + len(df) > (
                        self.args.get("max_num_rows") or INT_MAX
                    ):
                        df = df.head(
                            (self.args.get("max_num_rows") or INT_MAX) - len(vectors)
                        )
                        max_hit = True
                    self.update_vectors(vectors, vector_column_name, df)
                    self.update_metadata(metadata, vector_column_names, df)
                    if max_hit:
                        break
                self.total_imported_count += self.flush_to_db(vectors, metadata, collection)

        print("Data imported successfully")
        self.args["imported_count"] = self.total_imported_count

    def flush_to_db(vectors, metadata, collection):
        BATCH_SIZE = 20
        for i in range(0, len(vectors), BATCH_SIZE):
            batch_vectors = vectors[i : i + BATCH_SIZE]
            batch_metadata = metadata[i : i + BATCH_SIZE]
            collection.upsert_many(
                documents=[
                    {"_id": id, "vector": vector, **metadata}
                    for id, vector, metadata in zip(batch_vectors, batch_metadata)
                ],
            )
        return len(vectors)
