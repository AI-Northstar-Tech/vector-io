import json
import os

from export.util import expand_shorthand_path


class ImportVDF:
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

    def upsert_data():
        """
        Get data from vector database
        """
        raise NotImplementedError