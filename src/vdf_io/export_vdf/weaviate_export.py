import os

from tqdm import tqdm
import weaviate

from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password

# Set these environment variables
URL = os.getenv("YOUR_WCS_URL")
APIKEY = os.getenv("YOUR_WCS_API_KEY")


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate instance")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--classes", type=str, help="Classes to export (comma-separated)"
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance: ",
            str,
        )
        set_arg_from_password(
            args,
            "api_key",
            "Enter the Weaviate API key: ",
            "WEAVIATE_API_KEY",
        )
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = list(
            weaviate_export.client.collections.list_all().keys()
        )
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the name of the classes to export (comma-separated, all will be exported by default): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    # Connect to a WCS instance
    def __init__(self, args):
        super().__init__(args)
        self.client = weaviate.connect_to_wcs(
            cluster_url=self.args["url"],
            auth_credentials=weaviate.auth.AuthApiKey(self.args["api_key"]),
            skip_init_checks=True,
        )

    def get_index_names(self):
        if self.args.get("classes") is None:
            return self.all_classes
        else:
            input_classes = self.args["classes"].split(",")
            if set(input_classes) - set(self.all_classes):
                tqdm.write(
                    f"These classes are not present in the Weaviate instance: {set(input_classes) - set(self.all_classes)}"
                )
            return [c for c in self.all_classes if c in input_classes]

    def get_data(self):
        # Get all objects of a class
        index_names = self.get_index_names()
        for class_name in index_names:
            collection = self.client.collections.get(class_name)
            response = collection.aggregate.over_all(total_count=True)
            print(f"{response.total_count=}")

            # objects = self.client.query.get(
            #     wvq.Objects(wvq.Class(class_name)).with_limit(1000)
            # )
            # print(objects)
