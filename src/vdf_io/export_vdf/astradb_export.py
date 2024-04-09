import json
import os
import sys
from typing import Dict, List
from astrapy.db import AstraDB
from tqdm import tqdm

from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


DISK_SPACE_LIMIT = 1e8  # 100 MB


class ExportAstraDB(ExportVDB):
    DB_NAME_SLUG = DBNames.ASTRADB

    @classmethod
    def make_parser(cls, subparsers):
        parser_astradb = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from AstraDB"
        )

        parser_astradb.add_argument(
            "--endpoint", type=str, help="Location of AstraDB instance"
        )
        parser_astradb.add_argument(
            "--astradb_api_key", type=str, help="AstraDB API key"
        )
        parser_astradb.add_argument(
            "--collections", type=str, help="AstraDB tables to export (comma-separated)"
        )
        parser_astradb.add_argument(
            "--distance_metric",
            type=str,
            help="Distance metric to use for the vectors",
            default="cosine",
            choices=[
                "cosine",
                "euclidean",
                "dot_product",
            ],
        )

    @classmethod
    def export_vdb(cls, args):
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
        astradb_export = ExportAstraDB(args)
        astradb_export.all_collections = astradb_export.get_all_index_names()
        set_arg_from_input(
            args,
            "collections",
            "Enter the name of collections to export (comma-separated, all will be exported by default): ",
            str,
            None,
            choices=astradb_export.all_collections,
        )
        astradb_export.get_data()
        return astradb_export

    def __init__(self, args):
        super().__init__(args)
        self.db = AstraDB(
            token=self.args.get("astradb_api_key"),
            api_endpoint=self.args.get("endpoint"),
        )

    def get_all_index_names(self):
        return self.db.get_collections()["status"]["collections"]

    def get_index_names(self):
        if self.args.get("collections", None) is not None:
            return self.args["collections"].split(",")
        return self.db.get_collections()["status"]["collections"]

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for index_name in index_names:
            tqdm.write(f"Exporting collection: {index_name}")
            namespace_metas = []
            vectors_directory = os.path.join(self.vdf_directory, index_name)
            os.makedirs(vectors_directory, exist_ok=True)
            collection = self.db.collection(index_name)
            next_page_state = None
            tot_docs = 0
            ids = []
            vectors = {}
            metadatas = {}
            pbar = tqdm(desc="Exporting data", unit="documents")
            exported_count = 0
            while True:
                search_results = collection.find(
                    sort=None, options={"pageState": next_page_state}
                )
                tot_docs += len(search_results["data"]["documents"])
                pbar.update(len(search_results["data"]["documents"]))
                next_page_state = search_results["data"]["nextPageState"]
                # search_results["data"]["documents"] is a list of dictionaries containing _id, vector and other fields, which are metadata
                ids += [doc["_id"] for doc in search_results["data"]["documents"]]
                vectors.update(
                    {
                        doc["_id"]: doc["vector"]
                        for doc in search_results["data"]["documents"]
                        if "vector" in doc
                    }
                )
                # exclude _id and vector from metadata
                metadatas.update(
                    {
                        doc["_id"]: {
                            k: v for k, v in doc.items() if k not in ["_id", "vector"]
                        }
                        for doc in search_results["data"]["documents"]
                    }
                )
                # if getsizeof of vectors and metadata is too large, save to parquet
                if sys.getsizeof(vectors) + sys.getsizeof(metadatas) > DISK_SPACE_LIMIT:
                    tqdm.write("Flushing to parquet files on disk")
                    exported_count += self.save_vectors_to_parquet(
                        vectors, metadatas, vectors_directory
                    )
                if search_results["data"]["nextPageState"] is None:
                    break
            exported_count += self.save_vectors_to_parquet(
                vectors, metadatas, vectors_directory
            )
            sample_doc = collection.find_one()["data"]["document"]
            if "$vector" in sample_doc:
                dims = len(sample_doc["$vector"])
            else:
                dims = -1
            namespace_metas = [
                self.get_namespace_meta(
                    index_name,
                    vectors_directory,
                    total=tot_docs,
                    num_vectors_exported=exported_count,
                    dim=dims,
                    vector_columns=["vector"],
                    distance=self.args.get("distance_metric"),
                )
            ]
            index_metas[index_name] = namespace_metas
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        # print internal metadata properly
        return True
