from typing import Dict, List
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import pyarrow.parquet as pq

from pgvector.psycopg import register_vector
import psycopg

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.pgvector_util import make_pgv_parser, set_pgv_args_from_prompt
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB


load_dotenv()


class ImportPGVector(ImportVDB):
    DB_NAME_SLUG = DBNames.PGVECTOR

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to PGVector
        """
        set_pgv_args_from_prompt(args)
        pgvector_import = ImportPGVector(args)
        pgvector_import.get_all_table_names()
        pgvector_import.get_all_schemas()
        set_arg_from_input(
            args,
            "schema",
            "Enter the name of the schema of the Postgres instance (default: public): ",
            str,
            choices=pgvector_import.all_schemas,
        )
        pgvector_import.upsert_data()
        return pgvector_import

    @classmethod
    def make_parser(cls, subparsers):
        _parser_pgvector = make_pgv_parser(cls.DB_NAME_SLUG, subparsers)

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        # use connection_string
        if args.get("connection_string"):
            self.conn = psycopg.connect(args["connection_string"])
        else:
            self.conn = psycopg.connect(
                user=args["user"],
                password=args["password"],
                host=args["host"] if args.get("host", "") != "" else "localhost",
                port=args["port"] if args.get("port", "") != "" else "5432",
                dbname=args["dbname"] if args.get("dbname", "") != "" else "postgres",
            )

    def get_all_schemas(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT schema_name FROM information_schema.schemata")
            schemas_response = cur.fetchall()
        self.all_schemas = (
            [schema[0] for schema in schemas_response] if schemas_response else []
        )
        return self.all_schemas

    def get_all_table_names(self):
        self.schema_name = self.args.get("schema") or "public"
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT table_name FROM information_schema.tables WHERE table_schema='{self.schema_name}'"
            )
            tables_response = cur.fetchall()
        tqdm.write(f"Tables in schema {self.schema_name}: {tables_response}")
        self.all_tables = (
            [table[0] for table in tables_response] if tables_response else []
        )
        return self.all_tables

    def upsert_data(self):
        # create pgvector extension if not exists
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            # register vector type
        register_vector(self.conn)

        # use the schema
        with self.conn.cursor() as cur:
            cur.execute(f"SET search_path TO {self.schema_name}")

        max_hit = False
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")
        self.tables = self.get_all_table_names()
        # Load Parquet file
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
                new_index_name = self.create_new_name(new_index_name, self.tables)
                if new_index_name not in self.tables:
                    # assemble schema using parquet file's schema
                    schema = pq.read_schema(parquet_files[0])
                    schema_dict = {}
                    if "model_map" not in namespace_meta:
                        namespace_meta["model_map"] = {}
"""
┌──────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│ column_name  │ column_type │  null   │   key   │ default │  extra  │
│   varchar    │   varchar   │ varchar │ varchar │ varchar │ varchar │
├──────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ id           │ BIGINT      │ YES     │         │         │         │
│ vector       │ DOUBLE[]    │ YES     │         │         │         │
│ claps        │ BIGINT      │ YES     │         │         │         │
│ title        │ VARCHAR     │ YES     │         │         │         │
│ responses    │ BIGINT      │ YES     │         │         │         │
│ reading_time │ BIGINT      │ YES     │         │         │         │
│ publication  │ VARCHAR     │ YES     │         │         │         │
│ link         │ VARCHAR     │ YES     │         │         │         │
└──────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘
id: int64
vector: list<element: double>
  child 0, element: double
claps: int64
title: string
responses: int64
reading_time: int64
publication: string
link: string
"""
                    parquet_to_sql_type_map = {
                        
                        "int64": "BIGINT",
                        "float64": "DOUBLE PRECISION",
                        "bool": "BOOLEAN",
                        "datetime64[ns]": "TIMESTAMP",
                        "timedelta64[ns]": "INTERVAL",
                        "object": "VARCHAR",
                    }
                    for field in schema:
                        col_type = field.type
                        col_name = field.name
                        schema_dict[col_name] = col_type
                        # check if the column is a vector column
                        if col_name in namespace_meta["model_map"]:
                            schema_dict[col_name] = "vector()"
                    # create schema string
                    schema_str = ", ".join(
                        [f"{col_name} {col_type}" for col_name, col_type in schema_dict.items()]
                    )
                    # create postgres table
                    with self.conn.cursor() as cur:
                        cur.execute(
                            f"CREATE TABLE {new_index_name} (id SERIAL PRIMARY KEY)"
                        )
                    tqdm.write(f"Created table {new_index_name}")
                    table_name = new_index_name
                else:
                    # set table name
                    table_name = new_index_name
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
                        # convert df into list of dicts
                        with self.conn.cursor() as cur:
                            cur.execute(
                                f"""INSERT INTO {table_name
                                } ({', '.join(batch.columns)
                                }) VALUES {', '.join([str(tuple(row)) for row in batch.itertuples(index=False)])
                                }"""
                            )
                        self.total_imported_count += len(batch)
                        if max_hit:
                            break
                tqdm.write(f"Imported {self.total_imported_count} rows")
                with self.conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    new_table_size = cur.fetchone()[0]
                tqdm.write(f"New table size: {new_table_size}")
                if max_hit:
                    break
        print("Data imported successfully")


def get_default_value(data_type):
    # Define default values for common data types
    default_values = {
        "object": "",
        "int64": 0,
        "float64": 0.0,
        "bool": False,
        "datetime64[ns]": pd.Timestamp("NaT"),
        "timedelta64[ns]": pd.Timedelta("NaT"),
    }
    # Return the default value for the specified data type, or None if not specified
    return default_values.get(data_type.name, None)
