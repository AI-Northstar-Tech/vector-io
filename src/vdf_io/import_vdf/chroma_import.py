from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm

import chromadb

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    expand_shorthand_path,
    set_arg_from_input,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB


load_dotenv()


class ImportChroma(ImportVDB):
    DB_NAME_SLUG = DBNames.CHROMA

    @classmethod
    def make_parser(cls, subparsers):
        """
        Import data to Chroma
        """
        parser_chroma = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data from Chroma"
        )
        parser_chroma.add_argument(
            "--host_port", type=str, help="Host and port of Chroma instance"
        )
        parser_chroma.add_argument(
            "--persistent_path",
            type=str,
            help="Path to persistent storage for Chroma",
        )

    @classmethod
    def import_vdb(cls, args):
        if args.get("persistent_path") is None:
            set_arg_from_input(
                args,
                "host_port",
                "Enter the host:port of chroma instance: ",
                str,
            )
        if args.get("host_port") is None or args.get("host_port") == "":
            set_arg_from_input(
                args,
                "persistent_path",
                "Enter the path to persistent storage for Chroma: ",
                str,
            )
        if (args.get("host_port") is None or args.get("host_port") == "") and (
            args.get("persistent_path") is None or args.get("persistent_path") == ""
        ):
            set_arg_from_input(
                args,
                "api_key",
                "Enter the API key for Chroma: ",
                str,
            )
            set_arg_from_input(
                args,
                "tenant",
                "Enter the tenant for Chroma: ",
                str,
            )
            set_arg_from_input(
                args,
                "database",
                "Enter the database for Chroma: ",
                str,
            )
        chroma_import = ImportChroma(args)
        chroma_import.upsert_data()
        return chroma_import

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        if self.args.get("host_port") is not None:
            host_port = self.args.get("host_port")
            self.client = chromadb.HttpClient(
                host=host_port.split(":")[0], port=int(host_port.split(":")[1])
            )
        elif self.args.get("persistent_path") is not None:
            self.client = chromadb.PersistentClient(
                path=expand_shorthand_path(self.args.get("persistent_path"))
            )
        else:
            self.client = chromadb.CloudClient(
                tenant=self.args.get("tenant"),
                database=self.args.get("database"),
                api_key=self.args.get("api_key"),
            )

    def get_all_index_names(self):
        return [coll.name for coll in self.client.list_collections()]

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")
        collections = self.get_all_index_names()
        # Load Parquet file
        # print(indexes_content[index_names[0]]):List[NamespaceMeta]
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(final_data_path)

                new_index_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_index_name = self.create_new_name(new_index_name, collections)
                if new_index_name not in collections:
                    collection = self.client.create_collection(
                        new_index_name,
                    )
                else:
                    collection = self.client.get_collection(new_index_name)
                (
                    vector_column_names,
                    vector_column_name,
                ) = self.get_vector_column_name(index_name, namespace_meta)
                tqdm.write(f"Vector column name: {vector_column_name}")
                if len(vector_column_names) > 1:
                    tqdm.write("Chroma does not support multiple vector columns")
                    tqdm.write(f"Skipping the rest : {vector_column_names[1:]}")
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
                    # if there are additional columns in the parquet file, add them to the table
                    # split in batches
                    BATCH_SIZE = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                    for batch in tqdm(
                        divide_into_batches(df, BATCH_SIZE),
                        desc="Importing batches",
                        total=len(df) // BATCH_SIZE,
                    ):
                        if self.total_imported_count + len(batch) >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            batch = batch[
                                : (self.args.get("max_num_rows") or INT_MAX)
                                - self.total_imported_count
                            ]
                            max_hit = True
                        model_map = namespace_meta.get("model_map", {})

                        # filter out rows with empty or None vector column
                        prev_count = len(batch)
                        batch = batch.dropna(subset=[vector_column_name])
                        non_empty_count = len(batch)
                        if prev_count != non_empty_count:
                            tqdm.write(
                                f"Skipped {prev_count - non_empty_count} rows with empty vector column"
                            )
                        ids = [str(x) for x in batch[self.id_column].tolist()]
                        docs = []

                        if (vector_column_name in model_map) and (
                            model_map[vector_column_name].get(
                                "text_column", "NOT_PROVIDED"
                            )
                            != "NOT_PROVIDED"
                        ):
                            text_column = model_map[vector_column_name]["text_column"]
                            docs = batch[text_column].tolist()
                        else:
                            docs = ids

                        embeddings = (
                            batch[vector_column_name]
                            .apply(
                                # if it is iterable, convert to list
                                # else return the value
                                lambda x: list(x) if hasattr(x, "__iter__") else x
                            )
                            .tolist()
                        )
                        metadatas = []
                        for _, row in batch.iterrows():
                            metadata = row.to_dict()
                            metadata.pop(vector_column_name)
                            metadata.pop(self.id_column)
                            # remove values from metadata which are not str, int, float, bool
                            metadata = {
                                k: v
                                for k, v in metadata.items()
                                if isinstance(v, (str, int, float, bool))
                            }
                            metadatas.append(metadata)

                        collection.upsert(
                            ids=ids,
                            embeddings=embeddings,
                            documents=docs,
                            metadatas=metadatas,
                        )

                        self.total_imported_count += len(batch)
                        if max_hit:
                            break
                tqdm.write(
                    f"Imported {self.total_imported_count} rows into {collection.name}"
                )
                tqdm.write(f"New collection size: {collection.count()}")
                if max_hit:
                    break
        print("Data imported successfully")
