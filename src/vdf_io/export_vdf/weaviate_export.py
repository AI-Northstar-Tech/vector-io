import os

from tqdm import tqdm
import weaviate
import json

from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from typing import Dict, List

# Set these environment variables
URL = os.getenv("YOUR_WCS_URL")
APIKEY = os.getenv("YOUR_WCS_API_KEY")
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate instance")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument("--openai_api_key", type=str, help="Openai API key")
        parser_weaviate.add_arguments(
            "--batch_size", type=int, help="batch size for fetching",
            default=1000
        )
        parser_weaviate.add_argument(
            "--connection-type", type=str, choices=["local", "cloud"], default="cloud",
            help="Type of connection to Weaviate (local or cloud)"
        )
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
        set_arg_from_input(
            args,
            "connection_type",
            "Enter 'local' or 'cloud' for connection types: ",
            choices=['local', 'cloud'],
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

    # Connect to a WCS or local instance
    def __init__(self, args):
        super().__init__(args)
        if self.args["connection_type"] == "local":
            self.client = weaviate.connect_to_local()
        else:
            self.client = weaviate.connect_to_wcs(
                cluster_url=self.args["url"],
                auth_credentials=weaviate.auth.AuthApiKey(self.args["api_key"]),
                headers={'X-OpenAI-Api-key': self.args["openai_api_key"]}
                if self.args["openai_api_key"]
                else None,
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
        # Get the index names to export
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}

        # Iterate over index names and fetch data
        for index_name in index_names:
            collection = self.client.collections.get(index_name)
            response = collection.aggregate.over_all(total_count=True)
            total_vector_count = response.total_count

            # Create vectors directory for this index
            vectors_directory = self.create_vec_dir(index_name)

            # Export data in batches
            batch_size = self.args["batch_size"]
            num_batches = (total_vector_count + batch_size - 1) // batch_size
            num_vectors_exported = 0

            for batch_idx in tqdm(range(num_batches), desc=f"Exporting {index_name}"):
                offset = batch_idx * batch_size
                objects = collection.objects.limit(batch_size).offset(offset).get()

                # Extract vectors and metadata
                vectors = {obj.id: obj.vector for obj in objects}
                metadata = {}
                # Need a better way
                for obj in objects:
                    metadata[obj.id] = {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith("__")}


                # Save vectors and metadata to Parquet file
                num_vectors_exported += self.save_vectors_to_parquet(
                    vectors, metadata, vectors_directory
                )

            # Create NamespaceMeta for this index
            namespace_metas = [
                self.get_namespace_meta(
                    index_name,
                    vectors_directory,
                    total=total_vector_count,
                    num_vectors_exported=num_vectors_exported,
                    dim=300, # Not sure of the dimensions
                    distance="Cosine",
                )
            ]
            index_metas[index_name] = namespace_metas

        # Write VDFMeta to JSON file
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)

        print("Data export complete.")

        return True
