import json
from typing import Any

from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.export_vdf.pgvector_export import connect_pgvector
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    clean_value,
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)


class ImportPGVector(ImportVDB):
    DB_NAME_SLUG = DBNames.PGVECTOR

    @classmethod
    def make_parser(cls, subparsers):
        parser_pg = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to PostgreSQL pgvector"
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
            "--batch_size",
            type=int,
            help="Batch size for importing rows (default: 1000)",
            default=1000,
        )

    @classmethod
    def import_vdb(cls, args):
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
            "Enter batch size for import (default: 1000): ",
            int,
            1000,
        )
        pg_import = ImportPGVector(args)
        pg_import.upsert_data()
        return pg_import

    def __init__(self, args, connection=None):
        super().__init__(args)
        self.conn = (
            connection if connection is not None else connect_pgvector(self.args)
        )

    def get_all_index_names(self) -> list[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"
            )
            return [r[0] for r in cur.fetchall()]

    def _ensure_table_exists(
        self,
        table_name: str,
        vector_column_names: list[str],
        dims: int,
        metric: str,
    ):
        ops_map = {
            "cosine": "vector_cosine_ops",
            "vector_cosine_ops": "vector_cosine_ops",
            "l2": "vector_l2_ops",
            "vector_l2_ops": "vector_l2_ops",
            "ip": "vector_ip_ops",
            "inner_product": "vector_ip_ops",
            "vector_ip_ops": "vector_ip_ops",
            "l1": "vector_l1_ops",
            "vector_l1_ops": "vector_l1_ops",
        }
        rev = standardize_metric_reverse(metric, self.DB_NAME_SLUG)
        index_ops = ops_map.get(rev, "vector_cosine_ops")

        vec_col_defs = ", ".join(f'"{vc}" vector({dims})' for vc in vector_column_names)
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f'CREATE TABLE IF NOT EXISTS "{table_name}" ('
                f'"{self.id_column}" TEXT PRIMARY KEY, '
                f"metadata JSONB, "
                f"{vec_col_defs});"
            )
            for vc in vector_column_names:
                cur.execute(
                    f'CREATE INDEX IF NOT EXISTS "{table_name}_{vc}_idx" '
                    f'ON "{table_name}" USING hnsw ("{vc}" {index_ops});'
                )
        self.conn.commit()

    def upsert_data(self):
        self.total_imported_count = 0
        max_hit = False
        indexes_content: dict[str, list[NamespaceMeta]] = self.vdf_meta["indexes"]
        if not indexes_content:
            raise ValueError("No indexes found in VDF_META.json")

        existing_tables = self.get_all_index_names()
        batch_size = min(self.args.get("batch_size") or 1000, DEFAULT_BATCH_SIZE)

        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing pgvector tables"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                dims = int(namespace_meta["dimensions"])
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                raw_table = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta.get("namespace")
                    else ""
                )
                target_table = self.create_new_name(
                    raw_table, existing_tables, delimiter="_"
                )
                vector_column_names, _ = self.get_vector_column_name(
                    target_table, namespace_meta, multi_vector_supported=True
                )
                metric = namespace_meta.get("metric") or "Cosine"

                self._ensure_table_exists(
                    target_table, vector_column_names, dims, metric
                )
                if target_table not in existing_tables:
                    existing_tables.append(target_table)

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

                    for batch_df in tqdm(
                        divide_into_batches(df, batch_size),
                        desc=f"Uploading batches to {target_table}",
                        total=max(1, len(df) // batch_size),
                    ):
                        rows_to_insert = []
                        for _, row in batch_df.iterrows():
                            row_id = str(row[self.id_column])
                            meta_dict: dict[str, Any] = {}
                            for col in batch_df.columns:
                                if col == self.id_column or col in vector_column_names:
                                    continue
                                val = clean_value(row[col])
                                if val is not None:
                                    meta_dict[col] = val

                            vec_values = [
                                json.dumps(self.extract_vector(row[vc]))
                                if vc in row and row[vc] is not None
                                else None
                                for vc in vector_column_names
                            ]
                            rows_to_insert.append(
                                (row_id, json.dumps(meta_dict), *vec_values)
                            )

                        if rows_to_insert:
                            vec_cols_sql = ", ".join(
                                f'"{vc}"' for vc in vector_column_names
                            )
                            placeholders = ", ".join(
                                ["%s", "%s"] + ["%s"] * len(vector_column_names)
                            )
                            update_set = ", ".join(
                                ["metadata = EXCLUDED.metadata"]
                                + [
                                    f'"{vc}" = EXCLUDED."{vc}"'
                                    for vc in vector_column_names
                                ]
                            )
                            sql = (
                                f'INSERT INTO "{target_table}" ("{self.id_column}", metadata, {vec_cols_sql}) '
                                f"VALUES ({placeholders}) "
                                f'ON CONFLICT ("{self.id_column}") DO UPDATE SET {update_set};'
                            )
                            with self.conn.cursor() as cur:
                                for params in rows_to_insert:
                                    cur.execute(sql, params)
                            self.conn.commit()
                            self.total_imported_count += len(rows_to_insert)

                        if self.total_imported_count >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            max_hit = True
                            break
                    if max_hit:
                        break
                if max_hit:
                    break
            if max_hit:
                break

        tqdm.write(
            f"pgvector data import completed successfully. {self.total_imported_count} rows imported."
        )
        self.args["imported_count"] = self.total_imported_count
        return True
