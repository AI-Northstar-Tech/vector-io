import ast
import json
import os
import sys
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import psycopg2
except ImportError:
    psycopg2 = None

from vdf_io.constants import DEFAULT_BATCH_SIZE, DISK_SPACE_LIMIT, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    create_pgvector_index_class,
    set_arg_from_input,
    set_arg_from_password,
)


def connect_pgvector(args: dict[str, Any]):
    """
    Connect to either a local or cloud PostgreSQL (pgvector) instance using a
    connection URL/DSN or host/port/dbname/user/password parameters.
    """
    if psycopg2 is None:
        raise ImportError(
            "The 'psycopg2-binary' package is required to connect to pgvector."
        )
    dsn = args.get("url") or args.get("dsn")
    if dsn:
        return psycopg2.connect(dsn)
    return psycopg2.connect(
        host=args.get("host") or "localhost",
        port=int(args.get("port") or 5432),
        dbname=args.get("dbname") or "postgres",
        user=args.get("user") or "postgres",
        password=args.get("password") or "",
        sslmode=args.get("sslmode") or "prefer",
    )


def _parse_pgvector_value(val: Any) -> list[float] | None:
    if val is None:
        return None
    if isinstance(val, list):
        return [float(x) for x in val]
    if hasattr(val, "tolist"):
        return [float(x) for x in val.tolist()]
    if isinstance(val, str):
        parsed = ast.literal_eval(val)
        return [float(x) for x in parsed]
    return list(val)


class ExportPGVector(ExportVDB):
    DB_NAME_SLUG = DBNames.PGVECTOR

    @classmethod
    def make_parser(cls, subparsers):
        parser_pg = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from PostgreSQL pgvector"
        )
        parser_pg.add_argument(
            "-u",
            "--url",
            type=str,
            help="PostgreSQL connection URI (e.g. postgresql://user:pass@host:5432/db)",
        )
        parser_pg.add_argument(
            "--host",
            type=str,
            help="PostgreSQL host (default: 'localhost')",
            default="localhost",
        )
        parser_pg.add_argument(
            "--port",
            type=int,
            help="PostgreSQL port (default: 5432)",
            default=5432,
        )
        parser_pg.add_argument(
            "--dbname",
            type=str,
            help="Database name (default: 'postgres')",
            default="postgres",
        )
        parser_pg.add_argument(
            "--user",
            type=str,
            help="Database user (default: 'postgres')",
            default="postgres",
        )
        parser_pg.add_argument(
            "--password",
            type=str,
            help="Database password",
        )
        parser_pg.add_argument(
            "-t",
            "--tables",
            type=str,
            help="Comma-separated list of pgvector tables to export (exports all vector tables by default)",
        )
        parser_pg.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting rows (default: 10000)",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter PostgreSQL connection URI (or hit return to enter host/db details): ",
            str,
            default_value=None,
            env_var="DATABASE_URL",
        )
        if not args.get("url"):
            set_arg_from_input(
                args,
                "host",
                "Enter PostgreSQL host (default: 'localhost'): ",
                str,
                "localhost",
            )
            set_arg_from_input(
                args,
                "port",
                "Enter PostgreSQL port (default: 5432): ",
                int,
                5432,
            )
            set_arg_from_input(
                args,
                "dbname",
                "Enter PostgreSQL database name (default: 'postgres'): ",
                str,
                "postgres",
            )
            set_arg_from_input(
                args,
                "user",
                "Enter PostgreSQL username (default: 'postgres'): ",
                str,
                "postgres",
            )
            set_arg_from_password(
                args,
                "password",
                "Enter PostgreSQL password: ",
                "PGPASSWORD",
            )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter batch size for export (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )

        pg_export = ExportPGVector(args)
        pg_export.all_tables = pg_export.get_all_index_names()
        set_arg_from_input(
            pg_export.args,
            "tables",
            "Enter the name of pgvector tables to export (comma-separated, hit return to export all): ",
            str,
            choices=pg_export.all_tables,
        )
        pg_export.get_data()
        return pg_export

    def __init__(self, args, connection=None):
        super().__init__(args)
        self.conn = (
            connection if connection is not None else connect_pgvector(self.args)
        )
        self.generated_classes: dict[str, type] = {}

    def get_all_index_names(self) -> list[str]:
        query = """
            SELECT DISTINCT table_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND udt_name = 'vector'
            ORDER BY table_name;
        """
        with self.conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
        return [r[0] for r in rows]

    def get_index_names(self) -> list[str]:
        if not hasattr(self, "all_tables") or self.all_tables is None:
            self.all_tables = self.get_all_index_names()
        if not self.args.get("tables"):
            return self.all_tables
        requested = [t.strip() for t in self.args["tables"].split(",") if t.strip()]
        return [t for t in self.all_tables if t in requested]

    def _inspect_table(
        self, table_name: str
    ) -> tuple[list[str], list[str], dict[str, str], str]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type, udt_name
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position;
                """,
                (table_name,),
            )
            cols = cur.fetchall()

        all_cols: list[str] = []
        vec_cols: list[str] = []
        schema_map: dict[str, str] = {}
        for col_name, data_type, udt_name in cols:
            all_cols.append(col_name)
            if udt_name == "vector" or "vector" in str(data_type).lower():
                vec_cols.append(col_name)
                schema_map[col_name] = "vector"
            else:
                schema_map[col_name] = str(data_type)

        distance = "cosine"
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' AND tablename = %s;",
                    (table_name,),
                )
                for (idx_def,) in cur.fetchall():
                    lower_def = str(idx_def).lower()
                    for op_name in (
                        "vector_l2_ops",
                        "vector_ip_ops",
                        "vector_cosine_ops",
                        "vector_l1_ops",
                    ):
                        if op_name in lower_def:
                            distance = op_name
                            break
        except Exception:  # noqa: BLE001, S110
            pass

        return all_cols, vec_cols, schema_map, distance

    def save_records_to_parquet(
        self, records: list[dict[str, Any]], vectors_directory: str
    ) -> int:
        if not records:
            return 0
        df = pd.DataFrame.from_records(records)
        parquet_file = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(parquet_file)
        if not hasattr(self, "parquet_schema"):
            self.parquet_schema = pq.read_schema(parquet_file)
        else:
            self.parquet_schema = pa.unify_schemas(
                [self.parquet_schema, pq.read_schema(parquet_file)]
            )
        self.file_structure.append(parquet_file)
        self.file_ctr += 1
        return len(df)

    def get_data(self):
        table_names = self.get_index_names()
        index_metas: dict[str, list[NamespaceMeta]] = {}
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE

        for table_name in tqdm(table_names, desc="Exporting pgvector tables"):
            vectors_directory = self.create_vec_dir(table_name)
            all_cols, vec_cols, schema_map, distance = self._inspect_table(table_name)
            self.generated_classes[table_name] = create_pgvector_index_class(
                table_name, schema_map
            )

            with self.conn.cursor() as cur:
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}";')
                total_count = int(cur.fetchone()[0])

            records_buffer: list[dict[str, Any]] = []
            num_exported = 0
            detected_dim = -1

            with self.conn.cursor() as cur:
                cur.execute(f'SELECT * FROM "{table_name}";')
                rows = cur.fetchall()

            for row_idx, row_tuple in enumerate(rows):
                row_dict = dict(zip(all_cols, row_tuple))
                rec: dict[str, Any] = {}
                rec[ID_COLUMN] = str(row_dict.get(ID_COLUMN, row_idx + 1))

                for k, v in row_dict.items():
                    if k == ID_COLUMN:
                        continue
                    if k in vec_cols:
                        vec_list = _parse_pgvector_value(v)
                        rec[k] = vec_list
                        if detected_dim == -1 and isinstance(vec_list, list):
                            detected_dim = len(vec_list)
                    else:
                        rec[k] = v

                records_buffer.append(rec)
                if (
                    len(records_buffer) >= batch_size
                    or sys.getsizeof(records_buffer) > DISK_SPACE_LIMIT
                ):
                    num_exported += self.save_records_to_parquet(
                        records_buffer, vectors_directory
                    )
                    records_buffer = []

            if records_buffer:
                num_exported += self.save_records_to_parquet(
                    records_buffer, vectors_directory
                )

            namespace_meta = self.get_namespace_meta(
                table_name,
                vectors_directory.replace("\\", "/"),
                total=total_count,
                num_vectors_exported=num_exported,
                dim=detected_dim,
                vector_columns=vec_cols or ["vector"],
                distance=distance,
            )
            self.args["exported_count"] += num_exported
            index_metas[table_name] = [namespace_meta]

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True
