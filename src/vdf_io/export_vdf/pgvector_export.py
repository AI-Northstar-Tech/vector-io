import json
import os
from typing import Dict, List
from tqdm import tqdm

import psycopg2

from vdf_io.pgvector_util import make_pgv_parser, set_pgv_args_from_prompt
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportPGVector(ExportVDB):
    DB_NAME_SLUG = DBNames.PGVECTOR

    @classmethod
    def make_parser(cls, subparsers):
        parser_pgvector = make_pgv_parser(cls.DB_NAME_SLUG, subparsers)
        parser_pgvector.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting data",
            default=10_000,
        )
        parser_pgvector.add_argument(
            "--tables", type=str, help="Postgres tables to export (comma-separated)"
        )

    @classmethod
    def export_vdb(cls, args):
        set_pgv_args_from_prompt(args)
        pgvector_export = ExportPGVector(args)
        pgvector_export.get_all_table_names()
        pgvector_export.get_all_schemas()
        set_arg_from_input(
            args,
            "schema",
            "Enter the name of the schema of the Postgres instance (default: public): ",
            str,
            choices=pgvector_export.all_schemas,
        )
        set_arg_from_input(
            args,
            "tables",
            "Enter the name of tables to import (comma-separated, all will be imported by default): ",
            str,
            choices=pgvector_export.all_tables,
        )
        pgvector_export.get_data()
        return pgvector_export

    def __init__(self, args):
        super().__init__(args)
        if args.get("connection_string"):
            self.conn = psycopg2.connect(args["connection_string"])
        else:
            self.conn = psycopg2.connect(
                user=args["user"],
                password=args["password"],
                host=args["host"] if args.get("host", "") != "" else "localhost",
                port=args["port"] if args.get("port", "") != "" else "5432",
                dbname=args["dbname"] if args.get("dbname", "") != "" else "postgres",
            )
        self.cur = self.conn.cursor()

    def get_all_schemas(self):
        schemas = self.cur.execute(
            "SELECT schema_name FROM information_schema.schemata"
        )
        self.all_schemas = [schema[0] for schema in schemas]
        return [schema[0] for schema in schemas]

    def get_all_table_names(self):
        self.schema_name = self.args.get("schema") or "public"
        tables = self.cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        )
        self.all_tables = [table[0] for table in tables]
        return [table[0] for table in tables]

    def get_all_index_names(self):
        # get all tables in the schema
        return self.cur.execute(
            f"SELECT table_name FROM information_schema.tables WHERE table_schema='{self.schema_name}'"
        )

    def get_index_names(self):
        if self.args.get("tables", None) is not None:
            return self.args["tables"].split(",")
        return self.get_all_index_names()

    def get_data(self):
        index_names = self.get_index_names()
        BATCH_SIZE = self.args["batch_size"]
        total = 0
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for index_name in index_names:
            namespace_metas = []
            vectors_directory = self.create_vec_dir(index_name)
            table = self.db.open_table(index_name)
            offset = 0
            remainder_df = None
            j = 0
            # for batch in tqdm(table.to_lance().to_batches()):
            #     df = batch.to_pandas()
            #     if remainder_df is not None:
            #         df = pd.concat([remainder_df, df])
            #     while len(df) >= BATCH_SIZE:
            #         # TODO: use save_vectors_to_parquet
            #         df[:BATCH_SIZE].to_parquet(
            #             os.path.join(vectors_directory, f"{j}.parquet")
            #         )
            #         j += 1
            #         total += BATCH_SIZE
            #         df = df[BATCH_SIZE:]
            #     offset += BATCH_SIZE
            #     remainder_df = df
            # if remainder_df is not None and len(remainder_df) > 0:
            #     # TODO: use save_vectors_to_parquet
            #     remainder_df.to_parquet(os.path.join(vectors_directory, f"{j}.parquet"))
            #     total += len(remainder_df)
            # dim = -1
            # for name in table.schema.names:
            #     if pyarrow.types.is_fixed_size_list(table.schema.field(name).type):
            #         dim = table.schema.field(name).type.list_size
            # vector_columns = [
            #     name
            #     for name in table.schema.names
            #     if pyarrow.types.is_fixed_size_list(table.schema.field(name).type)
            # ]
            # distance = "Cosine"
            # try:
            #     for index in table.list_indices():
            #         if index.vector_column_name == vector_columns[0]:
            #             distance = vector_columns[0]
            # except Exception:
            #     pass

            namespace_metas = [
                self.get_namespace_meta(
                    index_name,
                    vectors_directory,
                    total=total,
                    num_vectors_exported=total,
                    # dim=dim,
                    # vector_columns=vector_columns,
                    # distance=distance,
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
