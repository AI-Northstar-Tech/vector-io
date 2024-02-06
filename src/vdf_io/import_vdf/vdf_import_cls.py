import json
import os
from packaging.version import Version
import abc

from vdf_io.util import expand_shorthand_path, get_final_data_path, get_parquet_files


class ImportVDF(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "DB_NAME_SLUG"):
            raise TypeError(
                f"Class {cls.__name__} lacks required class variable 'DB_NAME_SLUG'"
            )

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
        return get_parquet_files(data_path)

    def get_final_data_path(self, data_path):
        return get_final_data_path(self.args["cwd"], self.args["dir"], data_path)
