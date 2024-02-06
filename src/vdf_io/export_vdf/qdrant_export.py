import argparse
import datetime
import json
from typing import Dict, List
from qdrant_client import QdrantClient
import os
from tqdm import tqdm
from dotenv import load_dotenv

from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.meta_types import NamespaceMeta, VDFMeta
from vdf_io.util import set_arg_from_input, set_arg_from_password, standardize_metric

load_dotenv()

MAX_FETCH_SIZE = 1_000


def make_qdrant_parser(subparsers):
    parser_qdrant = subparsers.add_parser("qdrant", help="Export data from Qdrant")
    parser_qdrant.add_argument(
        "-u", "--url", type=str, help="Location of Qdrant instance"
    )
    parser_qdrant.add_argument(
        "-c", "--collections", type=str, help="Names of collections to export"
    )
    parser_qdrant.add_argument(
        "--prefer_grpc",
        type=bool,
        help="Whether to use GRPC. Recommended. (default: True)",
        default=True,
        action=argparse.BooleanOptionalAction,
    )


def export_qdrant(args):
    """
    Export data from Qdrant
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
        "collections",
        "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
        str,
    )
    set_arg_from_password(
        args, "qdrant_api_key", "Enter your Qdrant API key: ", "QDRANT_API_KEY"
    )
    qdrant_export = ExportQdrant(args)
    qdrant_export.get_data()
    return qdrant_export


class ExportQdrant(ExportVDB):
    DB_NAME_SLUG = DBNames.QDRANT

    def __init__(self, args):
        """
        Initialize the class
        """
        super().__init__(args)
        self.client = QdrantClient(
            url=self.args["url"],
            api_key=self.args.get("qdrant_api_key", None),
            prefer_grpc=self.args.get("prefer_grpc", True),
        )

    def get_all_collection_names(self):
        """
        Get all collection names from Qdrant
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        return collection_names

    def get_data(self):
        if "collections" not in self.args or self.args["collections"] is None:
            collection_names = self.get_all_collection_names()
        else:
            collection_names = self.args["collections"].split(",")

        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for collection_name in tqdm(collection_names, desc="Fetching indexes"):
            index_meta = self.get_data_for_collection(collection_name)
            index_metas[collection_name] = index_meta

        # Create and save internal metadata JSON
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = VDFMeta(
            version=self.args["library_version"],
            file_structure=self.file_structure,
            author=os.environ.get("USER"),
            exported_from=self.DB_NAME_SLUG,
            indexes=index_metas,
            exported_at=datetime.datetime.now().astimezone().isoformat(),
        )
        meta_text = json.dumps(internal_metadata.dict(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        # print internal metadata properly
        return True

    def try_scroll(self, fetch_size, collection_name, next_offset):
        try:
            records, next_offset = self.client.scroll(
                collection_name=collection_name,
                offset=next_offset,
                limit=fetch_size,
                with_payload=True,
                with_vectors=True,
                shard_key_selector=self.args.get("shard_key_selector", None),
            )
            return records, next_offset, fetch_size
        except Exception as e:
            # if it is keyboard interrupt, raise it
            if isinstance(e, KeyboardInterrupt):
                raise e
            tqdm.write(
                f"Failed to fetch data, reducing fetch size to{(fetch_size * 2) // 3}"
            )
            return self.try_scroll((fetch_size * 2) // 3, collection_name, next_offset)

    def get_data_for_collection(self, collection_name) -> List[NamespaceMeta]:
        vectors_directory = os.path.join(self.vdf_directory, collection_name)
        os.makedirs(vectors_directory, exist_ok=True)

        total = self.client.get_collection(collection_name).vectors_count

        num_vectors_exported = 0
        dim = self.client.get_collection(collection_name).config.params.vectors.size
        next_offset = 0
        records, next_offset, fetch_size = self.try_scroll(
            MAX_FETCH_SIZE, collection_name, next_offset
        )
        num_vectors_exported += self.save_from_records(
            records,
            vectors_directory,
        )
        pbar = tqdm(total=total, desc=f"Exporting {collection_name}")
        while next_offset is not None:
            records, next_offset, fetch_size = self.try_scroll(
                fetch_size, collection_name, next_offset
            )
            num_vectors_exported += self.save_from_records(
                records,
                vectors_directory,
            )
            pbar.update(len(records))

        namespace_meta = NamespaceMeta(
            index_name=collection_name,
            namespace="",
            total_vector_count=total,
            exported_vector_count=num_vectors_exported,
            metric=standardize_metric(
                self.client.get_collection(
                    collection_name
                ).config.params.vectors.distance,
                self.DB_NAME_SLUG,
            ),
            dimensions=dim,
            model_name=self.args["model_name"],
            vector_columns=["vector"],
            data_path="/".join(vectors_directory.split("/")[1:]),
        )
        return [namespace_meta]

    def save_from_records(self, records, vectors_directory):
        num_vectors_exported = 0
        vectors = {}
        metadata = {}
        for point in records:
            vectors[point.id] = point.vector
            metadata[point.id] = point.payload
        num_vectors_exported += self.save_vectors_to_parquet(
            vectors, metadata, vectors_directory
        )
        return num_vectors_exported
