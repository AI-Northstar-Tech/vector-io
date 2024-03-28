from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportTxtai(ExportVDB):
    DB_NAME_SLUG = DBNames.TXTAI

    @classmethod
    def make_parser(cls, subparsers):
        parser_txtai = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Txtai"
        )

        parser_txtai.add_argument(
            "--endpoint", type=str, help="Location of Txtai instance"
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of Txtai instance (default: 'http://localhost:8080'): ",
            str,
            "http://localhost:8080",
        )
        txtai_export = ExportTxtai(args)
        txtai_export.get_data()
        return txtai_export

    def __init__(self, args):
        super().__init__(args)

    def get_data(self):
        pass
