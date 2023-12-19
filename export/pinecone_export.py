import datetime
from export.util import extract_data_hash
from export.vdb_export import ExportVDB
import pinecone
import os
import json
import pandas as pd
import numpy as np
import json

PINECONE_MAX_K = 10_000


class ExportPinecone(ExportVDB):
    def __init__(self, args):
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])
        self.args = args

    def get_all_index_names(self):
        return pinecone.list_indexes()

    def get_ids_from_query(self, index, input_vector):
        print("searching pinecone...")
        results = index.query(
            vector=input_vector, include_values=False, top_k=PINECONE_MAX_K
        )
        ids = set()
        print(type(results))
        for result in results["matches"]:
            ids.add(result["id"])
        return ids

    def get_all_ids_from_index(self, index, num_dimensions, namespace=""):
        print("index.describe_index_stats()", index.describe_index_stats())
        num_vectors = index.describe_index_stats()["namespaces"][namespace][
            "vector_count"
        ]
        all_ids = set()
        while len(all_ids) < num_vectors:
            print("Length of ids list is shorter than the number of total vectors...")
            input_vector = np.random.rand(num_dimensions).tolist()
            print("creating random vector...")
            ids = self.get_ids_from_query(index, input_vector)
            print("getting ids from a vector query...")
            all_ids.update(ids)
            print("updating ids set...")
            print(f"Collected {len(all_ids)} ids out of {num_vectors}.")

        return all_ids

    def get_data(self, index_name):
        self.index = pinecone.Index(index_name=index_name)
        info = self.index.describe_index_stats()
        namespace = info["namespaces"]
        print(
            "info.__dict__['_data_store']",
            info.__dict__["_data_store"],
            type(info.__dict__["_data_store"]),
        )
        print(type(info.__dict__["_data_store"]["namespaces"]))
        # hash_value based on args
        # convert info to dict
        info_dict = info.__dict__["_data_store"]
        hash_value = extract_data_hash(info_dict)
        timestamp = datetime.datetime.now().strftime("%y%m%d%H%M%S")

        # Fetch the actual data from the Pinecone index
        for namespace in info["namespaces"]:
            vdf_directory = f"vdf_{index_name}_{namespace}_{timestamp}_{hash_value}"
            vectors_directory = os.path.join(
                vdf_directory, "vectors_" + self.args["model_name"]
            )
            os.makedirs(vdf_directory, exist_ok=True)
            os.makedirs(vectors_directory, exist_ok=True)

            data = self.index.fetch(
                list(
                    self.get_all_ids_from_index(
                        index=pinecone.Index(index_name=index_name),
                        num_dimensions=info["dimension"],
                        namespace=namespace,
                    )
                )
            )

            parquet_file = os.path.join(
                vectors_directory, f"_n_{namespace}.parquet"
            )
            print('data', data)
            vectors = data["vectors"]
            # vectors is a dict of string to dict with keys id, values, metadata
            # store the vector in values as a column in the parquet file, and store the metadata as columns in the parquet file
            vectors = {k: v["values"] for k, v in vectors.items()}
            metadata = {k: v["metadata"] for k, v in data["vectors"].items()}
            ids = list(vectors.keys())
            df = pd.DataFrame.from_dict(vectors, orient="index")
            df["id"] = ids
            for k, v in metadata.items():
                df[k] = v
            df.to_parquet(parquet_file)
            print("info", info)
            # Create and save internal metadata JSON
            internal_metadata = {
                "file_structure": ["vectors/", "metadata.db", "VDF_META.json"],
                # author is from unix username
                "author": os.getlogin(),
                "dimensions": info["dimension"],
                "total_vector_count": info["total_vector_count"],
                "exported_from": "pinecone",
                "model_name": "PLEASE_FILL_IN",
            }
            with open(os.path.join(vdf_directory, "VDF_META.json"), "w") as json_file:
                json.dump(internal_metadata, json_file, indent=4)
        
        return True