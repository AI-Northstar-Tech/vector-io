import hashlib
from uuid import UUID
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
from grpc import RpcError
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from vdf_io.constants import ID_COLUMN

from vdf_io.names import DBNames
from vdf_io.util import (
    expand_shorthand_path,
    get_qdrant_id_from_id,
    read_parquet_progress,
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
            "qdrant_local_persist_path",
            "Enter the path to the local persist directory (default: None): ",
            str,
            "DO_NOT_PROMPT",
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
        parser_qdrant.add_argument(
            "-u",
            "--url",
            type=str,
            help="Qdrant instance url",
            default="http://localhost:6334",
        )
        parser_qdrant.add_argument(
            "--prefer_grpc",
            type=bool,
            help="Whether to use Qdrant's GRPC interface",
            default=True,
        )
        parser_qdrant.add_argument(
            "--qdrant_local_persist_path",
            type=str,
            help="Path to the local persist directory (default: None)",
            default=None,
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
            help="Number of parallel processes of upload (default: 5).",
            default=5,
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
        url, api_key, prefer_grpc, path = (
            self.args.get("url", None),
            self.args.get("qdrant_api_key", None),
            self.args.get("prefer_grpc", True),
            expand_shorthand_path(self.args.get("qdrant_local_persist_path", None)),
        )
        if path:
            url = None
        if url:
            path = None
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            path=path,
        )

    def upsert_data(self):
        max_hit = False
        total_imported_count = 0
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
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    df = read_parquet_progress(file_path)
                    vectors.update(
                        {
                            row[self.id_column]: row[vector_column_name]
                            for _, row in df.iterrows()
                        }
                    )
                    metadata.update(
                        {
                            row[self.id_column]: {
                                key: value
                                for key, value in row.items()
                                if key not in [ID_COLUMN] + vector_column_names
                            }
                            for _, row in df.iterrows()
                        }
                    )
                    vectors = {k: v.tolist() for k, v in vectors.items()}
                    points = [
                        PointStruct(
                            id=get_qdrant_id_from_id(idx),
                            vector=vectors[idx],
                            payload=metadata.get(idx, {}),
                        )
                        for idx in vectors.keys()
                    ]

                    if total_imported_count + len(points) >= self.args["max_num_rows"]:
                        max_hit = True
                        points = points[
                            : self.args["max_num_rows"] - total_imported_count
                        ]
                        tqdm.write("Truncating data to limit to max rows")
                    try:
                        tqdm.write(f"Starting bulk upload for file '{file_path}'")
                        self.client.upload_points(
                            collection_name=new_collection_name,
                            points=points,
                            batch_size=self.args.get("batch_size", 64),
                            parallel=self.args.get("parallel", 5),
                            max_retries=self.args.get("max_retries", 3),
                            shard_key_selector=self.args.get(
                                "shard_key_selector", None
                            ),
                            wait=True,
                        )
                        tqdm.write(f"Completed bulk upload for file '{file_path}'")
                        total_imported_count += len(points)
                        if total_imported_count >= self.args["max_num_rows"]:
                            max_hit = True
                    except (UnexpectedResponse, RpcError, ValueError) as e:
                        tqdm.write(
                            f"Failed to upsert data for collection '{new_collection_name}', {e}"
                        )
                        continue
                    vector_count = self.client.get_collection(
                        collection_name=new_collection_name
                    ).vectors_count
                    if max_hit:
                        break
                    # END parquet file loop
                tqdm.write(
                    f"Index '{new_collection_name}' has {vector_count} vectors after import"
                )
                tqdm.write(
                    f"{vector_count - prev_vector_count} vectors were imported"
                )
                if max_hit:
                    break
                # END namespace loop
            if max_hit:
                tqdm.write(
                    f"Max rows to be imported {self.args['max_num_rows']} hit. Exiting"
                )
                break
            # END index loop
        tqdm.write("Data import completed successfully.")
        self.args["imported_count"] = total_imported_count