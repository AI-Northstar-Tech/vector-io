import argparse
import pandas as pd
from tqdm import tqdm
import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec, PodSpec, Vector

from vdf_io.constants import INT_MAX
from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB

load_dotenv()


class ImportPinecone(ImportVDB):
    DB_NAME_SLUG = DBNames.PINECONE

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Pinecone
        """
        set_arg_from_password(
            args,
            "pinecone_api_key",
            "Enter your Pinecone API key: ",
            "PINECONE_API_KEY",
        )
        if args["serverless"] is False:
            set_arg_from_input(
                args, "environment", "Enter the environment of Pinecone instance: "
            )
        else:
            set_arg_from_input(
                args,
                "cloud",
                "Enter the cloud of Pinecone Serverless instance (default: 'aws'): ",
                str,
                "aws",
            )
            set_arg_from_input(
                args,
                "region",
                "Enter the region of Pinecone Serverless instance (default: 'us-west-2'): ",
                str,
                "us-west-2",
            )

        if args["subset"] is True:
            if "id_list_file" not in args or args["id_list_file"] is None:
                set_arg_from_input(
                    args,
                    "id_range_start",
                    "Enter the start of id range (hit return to skip): ",
                    int,
                )
                if args["id_range_start"] is not None:
                    set_arg_from_input(
                        args,
                        "id_range_end",
                        "Enter the end of id range (hit return to skip): ",
                        int,
                    )
            if args.get("id_range_start") is None and args.get("id_range_end") is None:
                set_arg_from_input(
                    args,
                    "id_list_file",
                    "Enter the path to id list file (hit return to skip): ",
                )

        pinecone_import = ImportPinecone(args)
        pinecone_import.upsert_data()
        return pinecone_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_pinecone = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Pinecone"
        )
        parser_pinecone.add_argument(
            "-e", "--environment", type=str, help="Pinecone environment"
        )
        parser_pinecone.add_argument(
            "--serverless",
            type=bool,
            help="Import data to Pinecone Serverless (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_pinecone.add_argument(
            "-c", "--cloud", type=str, help="Pinecone Serverless cloud"
        )
        parser_pinecone.add_argument(
            "-r", "--region", type=str, help="Pinecone Serverless region"
        )

    def __init__(self, args):
        super().__init__(args)
        self.pc = Pinecone(api_key=self.args["pinecone_api_key"])

    def compliant_name(self, name: str) -> str:
        new_name = name.lower().replace("_", "-")
        return new_name

    def upsert_data(self):
        max_hit = False
        # Iterate over the indexes and import the data
        for index_name, index_meta in tqdm(
            self.vdf_meta["indexes"].items(), desc="Importing indexes"
        ):
            tqdm.write(f"Importing data for index '{index_name}'")
            for namespace_meta in index_meta:
                self.set_dims(namespace_meta, index_name)
            # list indexes
            indexes = self.pc.list_indexes().names()
            # check if index exists
            index_name = self.create_new_name(index_name, indexes)
            compliant_index_name = self.compliant_name(index_name)
            if compliant_index_name not in indexes:
                # create index
                try:
                    if self.args["serverless"] is True:
                        self.pc.create_index(
                            name=compliant_index_name,
                            dimension=index_meta[0]["dimensions"],
                            metric=standardize_metric_reverse(
                                index_meta[0]["metric"], "pinecone"
                            ),
                            spec=ServerlessSpec(
                                cloud=self.args["cloud"],
                                region=self.args["region"],
                            ),
                        )
                    else:
                        self.pc.create_index(
                            name=compliant_index_name,
                            dimension=index_meta[0]["dimensions"],
                            metric=standardize_metric_reverse(
                                index_meta[0]["metric"], "pinecone"
                            ),
                            spec=PodSpec(
                                environment=self.args["environment"],
                                pod_type=(
                                    self.args["pod_type"]
                                    if (
                                        "pod_type" in self.args
                                        and self.args["pod_type"] is not None
                                    )
                                    else "starter"
                                ),
                            ),
                        )
                except Exception as e:
                    tqdm.write(f"{e}")
                    raise Exception(f"Invalid index name '{compliant_index_name}'", e)
            index = self.pc.Index(compliant_index_name)
            BATCH_SIZE = self.args.get("batch_size", 1000) or 1000
            current_batch_size = BATCH_SIZE
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                tqdm.write(
                    f"Importing data for namespace '{namespace_meta['namespace']}'"
                )
                namespace = namespace_meta["namespace"]
                data_path = namespace_meta["data_path"]

                # Check if the data path exists
                final_data_path = self.get_final_data_path(data_path)

                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(final_data_path)

                vectors = {}
                metadata = {}
                vector_column_names, vector_column_name = self.get_vector_column_name(
                    index_name, namespace_meta
                )

                for file in tqdm(parquet_files, desc="Loading data from parquet files"):
                    file_path = os.path.join(final_data_path, file)
                    df = self.read_parquet_progress(file_path)

                    if self.args["subset"] is True:
                        if (
                            "id_list_file" in self.args
                            and self.args["id_list_file"] is not None
                        ):
                            id_list = pd.read_csv(
                                self.args["id_list_file"], header=None
                            )[0].tolist()
                            df = df[df[self.id_column].isin(id_list)]
                        elif (
                            "id_range_start" in self.args
                            and self.args["id_range_start"] is not None
                            and "id_range_end" in self.args
                            and self.args["id_range_end"] is not None
                        ):
                            # convert id to int before comparison
                            df = df[
                                (
                                    df[self.id_column].astype(int)
                                    >= self.args["id_range_start"]
                                )
                                & (
                                    df[self.id_column].astype(int)
                                    <= self.args["id_range_end"]
                                )
                            ]
                        else:
                            raise Exception(
                                "Invalid arguments for subset export. "
                                "Please provide either id_list_file or id_range_start and id_range_end"
                            )
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
                tqdm.write(
                    f"Loaded {len(vectors)} vectors from {len(parquet_files)} parquet files"
                )
                # Upsert the vectors and metadata to the Pinecone index in batches
                self.total_imported_count = 0
                start_idx = 0
                pbar = tqdm(total=len(vectors), desc="Upserting vectors")
                while start_idx < len(vectors):
                    end_idx = min(start_idx + current_batch_size, len(vectors))

                    batch_vectors = [
                        (
                            Vector(
                                id=str(id),
                                values=vector,
                                metadata={
                                    k: v
                                    for k, v in metadata.get(id, {}).items()
                                    if v is not None
                                },
                            )
                            if len(metadata.get(id, {}).keys()) > 0
                            else Vector(
                                id=str(id),
                                values=vector,
                            )
                        )
                        for id, vector in list(vectors.items())[start_idx:end_idx]
                    ]
                    try:
                        resp = index.upsert(vectors=batch_vectors, namespace=namespace)
                        self.total_imported_count += resp["upserted_count"]
                        pbar.update(resp["upserted_count"])
                        start_idx += resp["upserted_count"]
                    except Exception as e:
                        tqdm.write(
                            f"Error upserting vectors for index '{compliant_index_name}', {e}"
                        )
                        if current_batch_size < BATCH_SIZE / 100:
                            tqdm.write("Batch size is not the issue. Aborting import")
                            raise e
                        current_batch_size = int(2 * current_batch_size / 3)
                        tqdm.write(f"Reducing batch size to {current_batch_size}")
                        continue
        tqdm.write(
            f"Data import completed successfully. Imported {self.total_imported_count} vectors"
        )
        self.args["imported_count"] = self.total_imported_count
