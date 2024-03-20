from vespa.application import Vespa
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportVespa(ExportVDB):
    DB_NAME_SLUG = DBNames.VESPA

    @classmethod
    def make_parser(cls, subparsers):
        parser_vespa = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Vespa"
        )

        parser_vespa.add_argument(
            "--endpoint", type=str, help="Location of Vespa instance"
        )

        parser_vespa.add_argument(
            "--vespa_cloud_secret_token", type=str, help="Vespa cloud secret token"
        )

        parser_vespa.add_argument(
            "--cert_file", type=str, help="Path to the certificate file"
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of Vespa instance (default: 'http://localhost:8080'): ",
            str,
            "http://localhost:8080",
        )
        if not args.get("cert_file"):
            set_arg_from_password(
                args,
                "vespa_cloud_secret_token",
                "Enter the Vespa cloud secret token (hit return to enter cert file path instead): ",
                "VESPA_CLOUD_SECRET_TOKEN",
            )
        if not args["vespa_cloud_secret_token"]:
            set_arg_from_input(
                args,
                "cert_file",
                "Enter the path to the certificate file: ",
                str,
            )
        vespa_export = ExportVespa(args)
        vespa_export.get_data()
        return vespa_export

    def __init__(self, args):
        super().__init__(args)
        self.app = Vespa(
            url=self.args["endpoint"],
            vespa_cloud_secret_token=self.args.get("vespa_cloud_secret_token"),
            cert=self.args.get("cert_file"),
        )

    def get_data(self):
        # Add code here to fetch the data from Vespa
        with self.app.syncio() as session:
            print(
                session.query(
                    yql="select * from music where true",
                )
            )

    def get_data_for_index(self, index_name):
        # Add code here to fetch the data for a specific index
        for doc in self.app.query(
            body={
                "yql": f"select * from sources * where indexname() contains '{index_name}'"
            }
        ).hits:
            print(doc)
