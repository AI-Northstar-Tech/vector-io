from import_VDF.vdf_import import ImportVDF
import pinecone
import os
import json
import json
from dotenv import load_dotenv


load_dotenv()


class ImportPinecone(ImportVDF):
    def __init__(self, args):
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])
        self.args = args

    def upsert_data(self):
        # check dir exists
        if not os.path.isdir(self.args["dir"]):
            raise Exception("Invalid dir path")
        
        if not os.path.isfile(os.path.join(self.args["dir"], "VDF_META.json")):
            raise Exception("Invalid dir path, VDF_META.json not found")
        
        # open VDF_META.json
        with open(os.path.join(self.args["dir"], "VDF_META.json")) as f:
            vdf_meta = json.load(f)
        
        # list folders in dir
        folders = os.listdir(self.args["dir"])
        # check if folders are valid, start with vectors_
        vector_folders = [folder for folder in folders if folder.startswith("vectors_")]        
        # read vectors parquet folder and get all the parquet files
        
        
        index = pinecone.Index(index_name=self.index_name)
        