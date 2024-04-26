import argparse
import json
import os
import sys
from typing import Dict, List
from astrapy.db import AstraDB
from tqdm import tqdm
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement

from vdf_io.constants import DISK_SPACE_LIMIT
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


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
        parser_astradb.add_argument(
            "--via_cql",
            type=bool,
            action=argparse.BooleanOptionalAction,
            help="Whether to use CQL to export data",
            default=False,
        )
        parser_astradb.add_argument(
            "--secure_connect_bundle",
            type=str,
            help="Path to the secure connect bundle",
        )
        parser_astradb.add_argument(
            "--keyspace",
            type=str,
            help="Keyspace to connect to",
            default="default_keyspace",
        )
        parser_astradb.add_argument(
            "--fetch_size",
            type=int,
            help="Fetch size for CQL queries",
            default=10000,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_password(
            args,
            "astradb_api_key",
            "Enter the AstraDB API key (default: value of os.environ['ASTRA_DB_APPLICATION_TOKEN']): ",
            "ASTRA_DB_APPLICATION_TOKEN",
        )
        if args.get("via_cql"):
            set_arg_from_input(
                args,
                "secure_connect_bundle",
                "Enter the path to the secure connect bundle: ",
                str,
            )
            astradb_export = ExportAstraDB(args)
            astradb_export.all_collections = astradb_export.get_all_index_names_cql()
            set_arg_from_input(
                args,
                "collections",
                "Enter the name of collections to export (comma-separated, all will be exported by default): ",
                str,
                None,
                choices=astradb_export.all_collections,
            )
            set_arg_from_input(
                args,
                "fetch_size",
                "Enter the fetch size for CQL queries (default: 10000): ",
                int,
                10000,
            )
            astradb_export.get_data_from_cql()
            return astradb_export
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of AstraDB instance (default: value of os.environ['ASTRA_DB_API_ENDPOINT']): ",
            str,
            env_var="ASTRA_DB_API_ENDPOINT",
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
        if not self.args.get("via_cql"):
            self.db = AstraDB(
                token=self.args.get("astradb_api_key"),
                api_endpoint=self.args.get("endpoint"),
            )
        else:
            self.session = Cluster(
                cloud={"secure_connect_bundle": self.args["secure_connect_bundle"]},
                auth_provider=PlainTextAuthProvider(
                    "token", self.args.get("astradb_api_key")
                ),
            ).connect()
            self.session.execute(
                "USE " + self.args.get("keyspace", "default_keyspace"), timeout=100.0
            )

    def get_all_index_names(self):
        return self.db.get_collections()["status"]["collections"]

    def get_index_names(self):
        if self.args.get("collections", None) is not None:
            return self.args["collections"].split(",")
        return self.db.get_collections()["status"]["collections"]

    def get_all_index_names_cql(self):
        res = self.session.execute(
            f"SELECT * FROM system_schema.tables where keyspace_name='{self.args['keyspace']}'",
            timeout=100.0,
        )
        return [row.table_name for row in res]

    def get_data_from_cql(self):
        index_names = (
            self.get_all_index_names_cql()
            if self.args.get("collections") is None
            else self.args.get("collections").split(",")
        )
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        self.paging_state = None
        for index_name in tqdm(index_names, desc="Fetching indexes"):
            # count rows using execute()
            count_query = f"SELECT COUNT(*) FROM {index_name}"
            count = self.session.execute(count_query, timeout=100.0).one()
            tqdm.write(f"Total rows in {index_name}: {count[0]}")
            tqdm.write(f"Exporting collection: {index_name}")
            namespace_metas = []
            vectors_directory = self.create_vec_dir(index_name)
            pbar = tqdm(desc="Exporting data", unit="documents", total=count[0])
            no_queries_run = True
            exported_count = 0
            vectors = {}
            metadatas = {}
            while no_queries_run or self.paging_state:
                rows = self.execute_select_all_once(index_name)
                rows = [json.loads(r.doc_json) for r in rows]
                no_queries_run = False
                for row in rows:
                    if "vector" in row:
                        vectors[row["_id"]] = row["vector"]
                    elif "$vector" in row:
                        vectors[row["_id"]] = row["$vector"]
                    metadatas[row["_id"]] = {
                        k: v
                        for k, v in row.items()
                        if k not in ["_id", "$vector", "vector"]
                    }
                    pbar.update(1)
                    if (
                        sys.getsizeof(vectors) + sys.getsizeof(metadatas)
                        > DISK_SPACE_LIMIT
                    ):
                        tqdm.write("Flushing to parquet files on disk")
                        exported_count += self.save_vectors_to_parquet(
                            vectors, metadatas, vectors_directory
                        )
            vectors_added = self.save_vectors_to_parquet(
                vectors, metadatas, vectors_directory
            )
            exported_count += vectors_added
            tqdm.write("Flushing to parquet files on disk")
            sample_doc = rows[0]
            if "$vector" in sample_doc:
                dims = len(sample_doc["$vector"])
            elif "vector" in sample_doc:
                dims = len(sample_doc["vector"])
            else:
                dims = -1
            namespace_metas = [
                self.get_namespace_meta(
                    index_name,
                    vectors_directory,
                    total=exported_count,
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
        return True

    def execute_select_all_once(self, index_name):
        query = f"SELECT * FROM {index_name}"
        fetch_size = self.args.get("fetch_size", 10000)
        while True:
            try:
                statement = SimpleStatement(query, fetch_size=fetch_size)
                rows = self.session.execute(
                    statement, paging_state=self.paging_state, timeout=100.0
                )
                self.paging_state = rows.paging_state
                break
            except Exception as e:
                tqdm.write(f"Error occurred: {e}. Reducing fetch size by 10%.")
                fetch_size = int(fetch_size * 0.9)  # reduce fetch size by 10%
        self.paging_state = rows.paging_state
        return list(rows)

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for index_name in index_names:
            tqdm.write(f"Exporting collection: {index_name}")
            namespace_metas = []
            vectors_directory = self.create_vec_dir(index_name)
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
