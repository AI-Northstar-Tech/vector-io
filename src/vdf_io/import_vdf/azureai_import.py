from dotenv import load_dotenv


from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB


load_dotenv()


class ImportAzureAI(ImportVDB):
    DB_NAME_SLUG = DBNames.AZUREAI

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Azure AI
        """
        set_arg_from_input(
            args,
            "endpoint",
            "Enter the URL of Azure AI instance (default: '~/.azureai'): ",
            str,
            default_value="~/.azureai",
        )
        set_arg_from_password(
            args,
            arg_name="azureai_api_key",
            prompt="Enter the Azure AI API key (default: value of os.environ['AZUREAI_API_KEY']): ",
            env_var_name="AZUREAI_API_KEY",
        )
        azureai_import = ImportAzureAI(args)
        azureai_import.upsert_data()
        return azureai_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_azureai = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to azureai"
        )
        parser_azureai.add_argument(
            "--endpoint", type=str, help="Location of Azure AI instance"
        )
        parser_azureai.add_argument(
            "--azureai_api_key", type=str, help="Azure AI API key"
        )
        parser_azureai.add_argument(
            "--tables", type=str, help="Azure AI tables to export (comma-separated)"
        )

    def __init__(self, args):
        super().__init__(args)
        self.endpoint = args.endpoint
        self.azureai_api_key = args.azureai_api_key
        self.tables = args.tables.split(",") if args.tables else []

    def upsert_data(self):
        """
        Upsert data to Azure AI
        """
        print("Upserting data to Azure AI")
