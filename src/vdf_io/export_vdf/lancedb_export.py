import json
import os
from typing import Dict, List
import lancedb
import pandas as pd
import pyarrow
from tqdm import tqdm
from vdf_io.meta_types import NamespaceMeta

from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportLanceDB(ExportVDB):
    DB_NAME_SLUG = DBNames.LANCEDB

    @classmethod
    def make_parser(cls, subparsers):
        parser_lancedb = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from LanceDB"
        )

        parser_lancedb.add_argument(
            "--endpoint", type=str, help="Location of LanceDB instance"
        )
        parser_lancedb.add_argument(
            "--lancedb_api_key", type=str, help="LanceDB API key"
        )
        parser_lancedb.add_argument(
            "--tables", type=str, help="LanceDB tables to export (comma-separated)"
        )
        parser_lancedb.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting data",
            default=10_000,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of LanceDB instance (default: '~/.lancedb'): ",
            str,
            "~/.lancedb",
        )
        set_arg_from_password(
            args,
            "lancedb_api_key",
            "Enter the LanceDB API key: ",
            "LANCEDB_API_KEY",
        )
        lancedb_export = ExportLanceDB(args)
        lancedb_export.all_collections = lancedb_export.get_all_index_names()
        set_arg_from_input(
            args,
            "tables",
            "Enter the name of tables to export (comma-separated, all will be exported by default): ",
            str,
            choices=lancedb_export.all_collections,
            default_value=None,
        )
        lancedb_export.get_data()
        return lancedb_export

    def __init__(self, args):
        super().__init__(args)
        self.db = lancedb.connect(
            self.args["endpoint"], api_key=self.args.get("lancedb_api_key") or None
        )

    def get_all_index_names(self):
        return self.db.table_names()

    def get_index_names(self):
        if self.args.get("tables", None) is not None:
            return self.args["tables"].split(",")
        return self.db.table_names()

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
            for batch in tqdm(table.to_lance().to_batches()):
                df = batch.to_pandas()
                if remainder_df is not None:
                    df = pd.concat([remainder_df, df])
                while len(df) >= BATCH_SIZE:
                    # TODO: use save_vectors_to_parquet
                    df[:BATCH_SIZE].to_parquet(
                        os.path.join(vectors_directory, f"{j}.parquet")
                    )
                    j += 1
                    total += BATCH_SIZE
                    df = df[BATCH_SIZE:]
                offset += BATCH_SIZE
                remainder_df = df
            if remainder_df is not None and len(remainder_df) > 0:
                # TODO: use save_vectors_to_parquet
                remainder_df.to_parquet(os.path.join(vectors_directory, f"{j}.parquet"))
                total += len(remainder_df)
            dim = -1
            for name in table.schema.names:
                if pyarrow.types.is_fixed_size_list(table.schema.field(name).type):
                    dim = table.schema.field(name).type.list_size
            vector_columns = [
                name
                for name in table.schema.names
                if pyarrow.types.is_fixed_size_list(table.schema.field(name).type)
            ]
            distance = "Cosine"
            try:
                for index in table.list_indices():
                    if index.vector_column_name == vector_columns[0]:
                        distance = vector_columns[0]
            except Exception:
                pass

            namespace_metas = [
                self.get_namespace_meta(
                    index_name,
                    vectors_directory,
                    total=total,
                    num_vectors_exported=total,
                    dim=dim,
                    vector_columns=vector_columns,
                    distance=distance,
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
