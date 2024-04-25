import argparse
from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
import concurrent.futures

from astrapy.db import AstraDB
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from qdrant_client.http.models import Distance

from vdf_io.constants import INT_MAX
from vdf_io.names import DBNames
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
import re
from vdf_io.util import (
    clean_documents,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)

load_dotenv()


class ImportAstraDB(ImportVDB):
    DB_NAME_SLUG = DBNames.ASTRADB

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

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Datastax Astra DB
        """
        if args.get("via_cql"):
            set_arg_from_input(
                args,
                "secure_connect_bundle",
                "Enter the path to the secure connect bundle: ",
                str,
            )
            astradb_import = ImportAstraDB(args)
            astradb_import.all_collections = astradb_import.get_all_index_names_cql()
            astradb_import.upsert_data(via_cql=True)
            return astradb_import
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

    def get_all_index_names_cql(self):
        query = f"SELECT * FROM system_schema.tables where keyspace_name='{self.args['keyspace']}'"
        result = self.session.execute_cql(query, timeout=100.0)
        return [f"{row['table_name']}" for row in result]

    def upsert_data(self, via_cql=False):
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
                new_index_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_index_name = self.compliant_name(new_index_name)
                if via_cql:
                    collection_list = self.get_all_index_names_cql()
                else:
                    collection_list = self.db.get_collections()["status"]["collections"]
                new_index_name = self.create_new_name(
                    new_index_name,
                    collection_list,
                    delimiter="_",
                )
                tqdm.write(f"Index name that will be used: {new_index_name}")
                # create collection
                if not via_cql:
                    collection = self.db.create_collection(
                        new_index_name,
                        dimension=namespace_meta["dimensions"],
                        metric=standardize_metric_reverse(
                            namespace_meta.get("distance_metric", Distance.COSINE),
                            self.DB_NAME_SLUG,
                        ),
                    )
                else:
                    """
                    CREATE TABLE default_keyspace.users (
                        firstname text,
                        lastname text,
                        email text,
                        "favorite color" text,
                        PRIMARY KEY (firstname, lastname)
                    )"""

                    self.session.execute(
                        f"CREATE TABLE IF NOT EXISTS {self.args['keyspace']}.{new_index_name}"
                        f" (id text PRIMARY KEY, \"$vector\" vector<float,{namespace_meta['dimensions']}>)"
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
                self.total_imported_count += self.flush_to_db(
                    vectors, metadata, collection, via_cql=via_cql
                )
            tqdm.write(
                f"Collection {index_name} imported successfully as {new_index_name}"
            )
            tqdm.write(f"{self.total_imported_count} points imported in total.")

        print("Data imported successfully.")
        self.args["imported_count"] = self.total_imported_count

    def flush_to_db(self, vectors, metadata, collection, via_cql, parallel=True):
        if via_cql:
            keys = list(set(vectors.keys()).union(set(metadata.keys())))
            for id in keys:
                self.session.execute(
                    f"INSERT INTO {self.args['keyspace']}.{collection.name} (id, \"$vector\", {', '.join(metadata[id].keys())}) "
                    f"VALUES ('{id}', {vectors[id]}, {', '.join([str(v) for v in metadata[id].values()])})"
                )
            return len(vectors)

        keys = list(set(vectors.keys()).union(set(metadata.keys())))
        if not parallel:
            return flush_to_db_sync(
                collection,
                keys,
                [vectors.get(k) for k in keys],
                [metadata.get(k, {}) for k in keys],
            )

        def flush_batch_to_db(collection, keys, vectors, metadata):
            documents = [
                {"_id": id, "$vector": vector, **metadata}
                for id, vector, metadata in zip(keys, vectors, metadata)
            ]
            # replace nan with None
            clean_documents(documents)
            try:
                response = collection.upsert_many(documents=documents)
                return response
            except Exception as e:
                tqdm.write(f"Error upserting batch: {e}")
                return

        BATCH_SIZE = 20
        num_parallel_threads = self.args.get("parallel", 5) or 5
        total_points = len(keys)
        batches = [
            (
                keys[i : i + BATCH_SIZE],
                [vectors.get(k) for k in keys[i : i + BATCH_SIZE]],
                [metadata.get(k, {}) for k in keys[i : i + BATCH_SIZE]],
            )
            for i in range(0, total_points, BATCH_SIZE)
        ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_parallel_threads
        ) as executor, tqdm(
            total=total_points,
            desc=f"Flushing to DB in batches of {BATCH_SIZE} in {num_parallel_threads} threads",
        ) as pbar:
            future_to_batch = {
                executor.submit(flush_batch_to_db, collection, *batch): batch
                for batch in batches
            }

            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    future.result()
                    pbar.update(len(batch[0]))
                except Exception as e:
                    tqdm.write(f"Batch upsert failed with error: {e}")

        return total_points

    def compliant_name(self, name):
        return re.sub(r"[- ./]", "_", name)


def flush_to_db_sync(collection, keys, vectors, metadata):
    # in batches of 20 keys using upsert_many
    BATCH_SIZE = 20
    batches = list(range(0, len(keys), BATCH_SIZE))
    for batch in tqdm(batches, desc="Flushing to DB in batches of 20"):
        key_batch = keys[batch : batch + BATCH_SIZE]
        vector_batch = [vectors[k] for k in key_batch]
        metadata_batch = [metadata[k] for k in key_batch]
        upsert_batch(collection, metadata, key_batch, vector_batch, metadata_batch)


def upsert_batch(collection, metadata, key_batch, vector_batch, metadata_batch):
    documents = [
        {"_id": id, "$vector": vector, **metadata}
        for id, vector in zip(key_batch, vector_batch, metadata_batch)
    ]
    # replace nan with None
    clean_documents(documents)

    response = collection.upsert_many(documents=documents)
    # check if any of the elements are errors
    if any([isinstance(r, Exception) for r in response]):
        for i, r in enumerate(response):
            if isinstance(r, Exception):
                tqdm.write(f"Error upserting: {r} for {documents[i]}")
