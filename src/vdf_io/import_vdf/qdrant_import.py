import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from grpc import RpcError
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from vdf_io.names import DBNames
from vdf_io.util import (
    extract_numerical_hash,
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta

load_dotenv()


class ImportQdrant(ImportVDB):
    DB_NAME_SLUG = DBNames.QDRANT

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Qdrant
        """
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Qdrant instance (default: 'http://localhost:6334'): ",
            str,
            "http://localhost:6334",
        )
        set_arg_from_input(
            args,
            "prefer_grpc",
            "Whether to use GRPC. Recommended. (default: True): ",
            bool,
            True,
        )
        set_arg_from_input(
            args,
            "parallel",
            "Enter the batch size for upserts (default: 1): ",
            int,
            1,
        )
        set_arg_from_input(
            args,
            "batch_size",
            "Enter the number of parallel processes of upload (default: 64): ",
            int,
            64,
        )
        set_arg_from_input(
            args,
            "max_retries",
            "Enter the maximum number of retries in case of a failure (default: 3): ",
            int,
            3,
        )
        set_arg_from_password(
            args, "qdrant_api_key", "Enter your Qdrant API key: ", "QDRANT_API_KEY"
        )
        qdrant_import = ImportQdrant(args)
        qdrant_import.upsert_data()
        return qdrant_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_qdrant = subparsers.add_parser(
            DBNames.QDRANT, help="Import data to Qdrant"
        )
        parser_qdrant.add_argument("-u", "--url", type=str, help="Qdrant url")
        parser_qdrant.add_argument(
            "--prefer_grpc",
            type=bool,
            help="Whether to use Qdrant's GRPC interface",
            default=True,
        )
        parser_qdrant.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for upserts (default: 64).",
            default=64,
        )
        parser_qdrant.add_argument(
            "--parallel",
            type=int,
            help="Number of parallel processes of upload (default: 1).",
            default=1,
        )
        parser_qdrant.add_argument(
            "--max_retries",
            type=int,
            help="Maximum number of retries in case of a failure (default: 3).",
            default=3,
        )
        parser_qdrant.add_argument(
            "--shard_key_selector",
            type=Any,
            help="Shard to be queried (default: None)",
            default=None,
        )

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        self.client = QdrantClient(
            url=self.args["url"],
            api_key=self.args.get("qdrant_api_key", None),
            prefer_grpc=self.args.get("prefer_grpc", True),
        )

    def upsert_data(self):
        # we know that the self.vdf_meta["indexes"] is a list
        index_meta: Dict[str, List[NamespaceMeta]] = {}
        for index_name, index_meta in tqdm(
            self.vdf_meta["indexes"].items(), desc="Importing indexes"
        ):
            tqdm.write(f"Importing data for index '{index_name}'")
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                # list indexes
                collections = [
                    x.name for x in self.client.get_collections().collections
                ]
                # check if index exists
                new_collection_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta["namespace"]
                    else ""
                )
                if new_collection_name not in collections:
                    # create index
                    try:
                        self.client.create_collection(
                            collection_name=new_collection_name,
                            vectors_config=VectorParams(
                                size=namespace_meta["dimensions"],
                                distance=(
                                    namespace_meta["metric"]
                                    if "metric" in namespace_meta
                                    else Distance.COSINE
                                ),
                            ),
                        )
                    except Exception as e:
                        tqdm.write(f"Failed to create index '{new_collection_name}'", e)
                        return
                prev_vector_count = self.client.get_collection(
                    collection_name=new_collection_name
                ).vectors_count
                if prev_vector_count > 0:
                    tqdm.write(
                        f"Index '{new_collection_name}' has {prev_vector_count} vectors before import"
                    )
                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(final_data_path)

                vectors = {}
                metadata = {}
                vector_column_names, vector_column_name = self.get_vector_column_name(
                    new_collection_name, namespace_meta
                )
                for file in parquet_files:
                    file_path = os.path.join(final_data_path, file)
                    df = pd.read_parquet(file_path)
                    vectors.update(
                        {row["id"]: row[vector_column_name] for _, row in df.iterrows()}
                    )
                    metadata.update(
                        {
                            row["id"]: {
                                key: value
                                for key, value in row.items()
                                if key not in ["id"] + vector_column_names
                            }
                            for _, row in df.iterrows()
                        }
                    )
                vectors = {k: v.tolist() for k, v in vectors.items()}
                points = [
                    PointStruct(
                        id=(
                            int(idx)
                            if (isinstance(idx, int) or idx.isdigit())
                            else extract_numerical_hash(idx)
                        ),
                        vector=vectors[idx],
                        payload=metadata.get(idx, {}),
                    )
                    for idx in vectors.keys()
                ]

                try:
                    self.client.upload_points(
                        collection_name=new_collection_name,
                        points=points,
                        batch_size=self.args.get("batch_size", 64),
                        parallel=self.args.get("parallel", 1),
                        max_retries=self.args.get("max_retries", 3),
                        shard_key_selector=self.args.get("shard_key_selector", None),
                        wait=True,
                    )
                except (UnexpectedResponse, RpcError, ValueError):
                    tqdm.write(
                        f"Failed to upsert data for collection '{new_collection_name}'"
                    )
                    continue
                vector_count = self.client.get_collection(
                    collection_name=new_collection_name
                ).vectors_count
                tqdm.write(
                    f"Index '{new_collection_name}' has {vector_count} vectors after import"
                )
                tqdm.write(f"{vector_count - prev_vector_count} vectors were imported")
        tqdm.write("Data import completed successfully.")
