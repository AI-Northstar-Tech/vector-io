from typing import List
from vdf_io.marqo_vespa_util import VespaClient
from rich import print as rprint

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
        parser_vespa.add_argument(
            "--pk_file", type=str, help="Path to the private key file"
        )
        parser_vespa.add_argument("--schemas", type=str, help="")

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of Vespa instance (default: 'http://localhost:8080'): ",
            str,
            None,
            env_var="VESPA_HOST",
        )
        if not args.get("cert_file"):
            set_arg_from_password(
                args,
                "vespa_cloud_secret_token",
                "Enter the Vespa cloud secret token (hit return to enter cert file path instead): ",
                "VESPA_CLOUD_SECRET_TOKEN",
            )
        if args.get("cert_file"):
            set_arg_from_input(
                args,
                "pk_file",
                "Enter the path to the private key file: ",
                str,
                default_value=None,
            )
        if not args["vespa_cloud_secret_token"]:
            set_arg_from_input(
                args,
                "cert_file",
                "Enter the path to the certificate file: ",
                str,
                default_value=None,
            )
        vespa_export = ExportVespa(args)
        vespa_export.get_data()
        return vespa_export

    def __init__(self, args):
        super().__init__(args)

    def get_index_names(self) -> List[str]:
        raise NotImplementedError()  # not available in pyvespa

    def get_data(self):
        schemas = self.args.get("schemas", []).split(",")
        for schema in schemas:
            self.get_data_for_index(schema)
        # Add code here to fetch the data from Vespa

    def get_data_for_index(self, index_name):
        self.vespa_client = VespaClient(
            config_url=self.args["endpoint"],
            query_url=self.args["endpoint"],
            document_url=self.args["endpoint"],
            content_cluster_name=index_name,
            cert_file=self.args.get("cert_file", None),
            pk_file=self.args.get("pk_file", None),
        )
        all_docs = self.vespa_client.get_all_documents(index_name, stream=True)
        rprint(all_docs.document_count)
