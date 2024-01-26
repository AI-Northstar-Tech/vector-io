import json
import os
from packaging.version import Version
from util import expand_shorthand_path
import abc


class ImportVDF(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, 'DB_NAME_SLUG'):
            raise TypeError(f"Class {cls.__name__} lacks required class variable 'DB_NAME_SLUG'")

    def __init__(self, args):
        self.args = args
        self.args["dir"] = expand_shorthand_path(self.args["dir"])
        if not os.path.isdir(self.args["dir"]):
            raise Exception("Invalid dir path")
        if not os.path.isfile(os.path.join(self.args["dir"], "VDF_META.json")):
            raise Exception("Invalid dir path, VDF_META.json not found")
        # Check if the VDF_META.json file exists
        vdf_meta_path = os.path.join(self.args["dir"], "VDF_META.json")
        if not os.path.isfile(vdf_meta_path):
            raise Exception("VDF_META.json not found in the specified directory")
        with open(vdf_meta_path) as f:
            self.vdf_meta = json.load(f)
        if "indexes" not in self.vdf_meta:
            raise Exception("Invalid VDF_META.json, 'indexes' key not found")
        if "version" not in self.vdf_meta:
            print("Warning: 'version' key not found in VDF_META.json")
        elif "library_version" not in self.args:
            print(
                "Warning: 'library_version' not found in args. Skipping version check."
            )
        elif Version(self.vdf_meta["version"]) > Version(self.args["library_version"]):
            print(
                f"Warning: The version of vector-io library: ({self.args['library_version']}) is behind the version of the vdf directory: ({self.vdf_meta['version']})."
            )
            print(
                "Please upgrade the vector-io library to the latest version to ensure compatibility."
            )

    @abc.abstractmethod
    def upsert_data():
        """
        Get data from vector database
        """
        raise NotImplementedError

    def get_vector_column_name(self, index_name, namespace_meta):
        if "vector_columns" not in namespace_meta:
            print(
                "vector_columns not found in namespace metadata. Using 'vector' as the vector column name."
            )
            vector_column_name = "vector"
            vector_column_names = [vector_column_name]
        else:
            vector_column_names = namespace_meta["vector_columns"]
            vector_column_name = vector_column_names[0]
            if len(vector_column_names) > 1:
                print(
                    f"Warning: More than one vector column found for index {index_name}."
                    f" Only the first vector column {vector_column_name} will be imported."
                )
        return vector_column_names, vector_column_name

    def get_parquet_files(self, data_path):
        # Load the data from the parquet files
        if not os.path.isdir(data_path):
            if data_path.endswith(".parquet"):
                return [data_path]
            else:
                raise Exception(f"Invalid data path '{data_path}'")
        else:
            parquet_files = sorted(
                [file for file in os.listdir(data_path) if file.endswith(".parquet")]
            )
            return parquet_files
        
    def get_final_data_path(self, data_path):
        final_data_path = os.path.join(
            self.args["cwd"], self.args["dir"], data_path
        )
        if not os.path.isdir(final_data_path):
            raise Exception(
                f"Invalid data path\n"
                f"data_path: {data_path}',\n"
                f"Joined path: {final_data_path}'"
                f"Current working directory: {self.args['cwd']}'\n"
                f"Command line arg (dir): {self.args['dir']}'"
            )
        return final_data_path