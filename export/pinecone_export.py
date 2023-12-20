import datetime
from export.util import extract_data_hash
from export.vdb_export import ExportVDB
import pinecone
import os
import json
import pandas as pd
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

PINECONE_MAX_K = 10_000
MAX_TRIES_OVERALL = 5  # 100
MAX_FETCH_SIZE = 1_000


class ExportPinecone(ExportVDB):
    def __init__(self, args):
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])
        self.args = args

    def get_all_index_names(self):
        return pinecone.list_indexes()

    def get_ids_from_query(self, index, input_vector):
        results = index.query(
            vector=input_vector, include_values=False, top_k=PINECONE_MAX_K
        )
        ids = set()
        for result in results["matches"]:
            ids.add(result["id"])
        return ids

    def get_all_ids_from_index(self, index, num_dimensions, namespace=""):
        print("index.describe_index_stats()", index.describe_index_stats())
        num_vectors = index.describe_index_stats()["namespaces"][namespace][
            "vector_count"
        ]
        all_ids = set()
        max_tries = min((num_vectors // PINECONE_MAX_K) * 20, MAX_TRIES_OVERALL)
        try_count = 0
        with tqdm(total=num_vectors, desc="Collecting IDs") as pbar:
            while len(all_ids) < num_vectors:
                print(
                    "Length of ids list is shorter than the number of total vectors..."
                )
                input_vector = np.random.rand(num_dimensions).tolist()
                ids = self.get_ids_from_query(index, input_vector)
                prev_size = len(all_ids)
                all_ids.update(ids)
                curr_size = len(all_ids)
                if curr_size > prev_size:
                    print(f"updating ids set with {curr_size - prev_size} new ids...")
                # sorted_add_ids = sorted(list(all_ids))
                # print("first 10 ids:", sorted_add_ids[:10])
                # print("last 10 ids:", sorted_add_ids[-10:])
                try_count += 1
                if try_count > max_tries:
                    print(
                        f"Could not collect all ids after {max_tries} random searches."
                        " Please provide range of ids instead. Exporting the ids collected so far."
                    )
                    break
                pbar.update(curr_size - prev_size)
        print(f"Collected {len(all_ids)} ids out of {num_vectors}.")
        return all_ids

    def get_data(self, index_name):
        self.index = pinecone.Index(index_name=index_name)
        info = self.index.describe_index_stats()
        namespace = info["namespaces"]
        # hash_value based on args
        # convert info to dict
        info_dict = info.__dict__["_data_store"]
        hash_value = extract_data_hash(info_dict)

        # Fetch the actual data from the Pinecone index
        for namespace in info["namespaces"]:
            timestamp_in_format = datetime.datetime.now().strftime("%y%m%d%H%M%S")
            vdf_directory = (
                f"vdf_{index_name}_{namespace}_{timestamp_in_format}_{hash_value}"
            )
            vectors_directory = os.path.join(
                vdf_directory, "vectors_" + self.args["model_name"]
            )
            os.makedirs(vdf_directory, exist_ok=True)
            os.makedirs(vectors_directory, exist_ok=True)

            all_ids = list(
                self.get_all_ids_from_index(
                    index=pinecone.Index(index_name=index_name),
                    num_dimensions=info["dimension"],
                    namespace=namespace,
                )
            )

            parquet_file = os.path.join(vectors_directory, f"_n_{namespace}.parquet")
            # vectors is a dict of string to dict with keys id, values, metadata
            vectors = {}
            metadata = {}
            for i in tqdm(range(0, len(all_ids), MAX_FETCH_SIZE), desc="Fetching data"):
                batch_ids = all_ids[i : i + MAX_FETCH_SIZE]
                data = self.index.fetch(batch_ids)
                batch_vectors = data["vectors"]
                print('data', data.keys(), len(data['vectors'].keys()), len(batch_ids))
                # verify that the ids are the same
                assert set(batch_ids) == set(batch_vectors.keys())
                metadata.update({k: v["metadata"] for k, v in batch_vectors.items()})
                vectors.update({k: v["values"] for k, v in batch_vectors.items()})

            ids = list(vectors.keys())
            
            # store the vector in values as a column in the parquet file, and store the metadata as columns in the parquet file
            # vectors.keys() are the indices of the df
            # vectors.values() are the "vectors" column of the df
            # metadata.keys() are the columns of the df
            # metadata.values() are the values of the df in those columns
            
            # convert vectors to a dataframe
            df = pd.DataFrame.from_dict(vectors, orient="index")
            
            print("df", df.head())
            # convert metadata to a dataframe
            metadata_df = pd.DataFrame.from_dict(metadata, orient="index")
            print("metadata_df", metadata_df.head())
            # concatenate the two dataframes so that each column in metadata_df becomes a column in df
            df.columns = df.columns.astype(str)  # Convert column names to strings
            df = pd.concat([df, metadata_df], axis=1)

            df.to_parquet(parquet_file)
            # Create and save internal metadata JSON
            internal_metadata = {
                "file_structure": ["vectors/", "metadata.db", "VDF_META.json"],
                # author is from unix username
                "author": os.getlogin(),
                "dimensions": info["dimension"],
                "total_vector_count": info["total_vector_count"],
                "exported_vector_count": len(ids),
                "exported_from": "pinecone",
                "model_name": "PLEASE_FILL_IN",
            }
            with open(os.path.join(vdf_directory, "VDF_META.json"), "w") as json_file:
                json.dump(internal_metadata, json_file, indent=4)

        return True
