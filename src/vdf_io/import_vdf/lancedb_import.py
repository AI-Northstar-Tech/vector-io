from dotenv import load_dotenv

import lancedb

from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB


load_dotenv()


class ImportLanceDB(ImportVDB):
    DB_NAME_SLUG = DBNames.LANCEDB

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to LanceDB
        """
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of LanceDB instance (default: '~/.lancedb'): ",
            str,
            default_value="~/.lancedb",
        )
        set_arg_from_password(
            args,
            "lancedb_api_key",
            "Enter the LanceDB API key (default: value of os.environ['LANCEDB_API_KEY']): ",
            "LANCEDB_API_KEY",
        )
        lancedb_import = ImportLanceDB(args)
        lancedb_import.upsert_data()
        return lancedb_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_lancedb = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to lancedb"
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

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        self.db = lancedb.connect(
            self.args["endpoint"], api_key=self.args.get("lancedb_api_key") or None
        )

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
