import pandas as pd
from import_vdf.vdf_import import ImportVDF
import pinecone
import os
import json
import json
from dotenv import load_dotenv
from pathlib import Path
import math


load_dotenv()

BATCH_SIZE = 1000  # Set the desired batch size

def expand_shorthand_path(shorthand_path):
    """
    Expand shorthand notations in a file path to a full path-like object.

    :param shorthand_path: A string representing the shorthand path.
    :return: A Path object representing the full path.
    """
    # Expand '~' to the user's home directory
    expanded_path = os.path.expanduser(shorthand_path)

    # Resolve '.' and '..' to get the absolute path
    full_path = Path(expanded_path).resolve()

    return str(full_path)

class ImportPinecone(ImportVDF):
    def __init__(self, args):
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])
        self.args = args

    def upsert_data(self):
        # check dir exists
        # convert path string like ~/Documents to absolute path, also . and ..
        
        self.args["dir"] = expand_shorthand_path(self.args["dir"])
        if not os.path.isdir(self.args["dir"]):
            raise Exception("Invalid dir path")
        
        if not os.path.isfile(os.path.join(self.args["dir"], "VDF_META.json")):
            raise Exception("Invalid dir path, VDF_META.json not found")
        
        self.import_data()
    
    def import_data(self):
        # Check if the VDF_META.json file exists
        vdf_meta_path = os.path.join(self.args["dir"], "VDF_META.json")
        if not os.path.isfile(vdf_meta_path):
            raise Exception("VDF_META.json not found in the specified directory")

        # Load the VDF_META.json file
        with open(vdf_meta_path) as f:
            vdf_meta = json.load(f)

        # Check if the "indexes" key exists in the VDF_META.json
        if "indexes" not in vdf_meta:
            raise Exception("Invalid VDF_META.json, 'indexes' key not found")

        # Iterate over the indexes and import the data
        for index_name, index_meta in vdf_meta["indexes"].items():
            print(f"Importing data for index '{index_name}'")
            # list indexes
            indexes = pinecone.list_indexes()
            # check if index exists
            if index_name not in indexes:
                # create index
                try:
                    pinecone.create_index(name=index_name, dimension=index_meta[0]["dimensions"])
                except Exception as e:
                    print(e)
                    raise Exception(f"Invalid index name '{index_name}'", e)
            index = pinecone.Index(index_name=index_name)
            for namespace_meta in index_meta:
                print(f"Importing data for namespace '{namespace_meta['namespace']}'")
                namespace = namespace_meta["namespace"]
                data_path = namespace_meta["data_path"]

                # Check if the data path exists
                if not os.path.isdir(data_path):
                    raise Exception(f"Invalid data path for index '{index_name}'")

                # Load the data from the parquet files
                parquet_files = sorted(
                    [
                        file
                        for file in os.listdir(data_path)
                        if file.endswith(".parquet")
                    ]
                )

                vectors = {}
                metadata = {}
                for file in parquet_files:
                    file_path = os.path.join(data_path, file)
                    df = pd.read_parquet(file_path)
                    vectors.update({row["id"]: row["vector"] for _, row in df.iterrows()})
                    metadata.update(
                        {
                            row["id"]: {
                                key: value
                                for key, value in row.items()
                                if key != "id" and key != "vector"
                            }
                            for _, row in df.iterrows()
                        }
                    )
                vectors = {k: v.tolist() for k, v in vectors.items()}
                # Upsert the vectors and metadata to the Pinecone index in batches
                num_batches = math.ceil(len(vectors) / BATCH_SIZE)

                for i in range(num_batches):
                    start_idx = i * BATCH_SIZE
                    end_idx = min((i + 1) * BATCH_SIZE, len(vectors))

                    batch_vectors = [
                        pinecone.Vector(
                            id=str(id),
                            values=vector,
                            metadata=metadata.get(id, {}),
                        )
                        for id, vector in list(vectors.items())[start_idx:end_idx]
                    ]

                    index.upsert(vectors=batch_vectors, namespace=namespace)

        print("Data import completed successfully.")