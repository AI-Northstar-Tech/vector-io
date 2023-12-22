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
MAX_TRIES_OVERALL = 100
MAX_FETCH_SIZE = 1_000
MAX_PARQUET_FILE_SIZE = 1_000_000_000  # 1GB


class ExportPinecone(ExportVDB):
    def __init__(self, args):
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])
        self.args = args
        self.file_structure = []

    def get_all_index_names(self):
        return pinecone.list_indexes()

    def get_ids_from_query(self, index, input_vector):
        results = index.query(
            vector=input_vector, include_values=False, top_k=PINECONE_MAX_K
        )
        ids = set(result["id"] for result in results["matches"])
        return ids

    def get_all_ids_from_index(self, index, num_dimensions, namespace=""):
        print("index.describe_index_stats()", index.describe_index_stats())
        if (
            self.args["id_range_start"] is not None
            and self.args["id_range_end"] is not None
        ):
            print(
                "Using id range {} to {}".format(
                    self.args["id_range_start"], self.args["id_range_end"]
                )
            )
            return [
                str(x)
                for x in range(
                    int(self.args["id_range_start"]),
                    int(self.args["id_range_end"]) + 1,
                )
            ]
        if self.args["id_list_file"]:
            with open(self.args["id_list_file"]) as f:
                return [line.strip() for line in f.readlines()]
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
                try_count += 1
                if try_count > max_tries and len(all_ids) < num_vectors:
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
            timestamp_in_format = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

            # vectors is a dict of string to dict with keys id, values, metadata
            vectors = {}
            metadata = {}
            batch_ctr = 1
            total_size = 0
            for i in tqdm(range(0, len(all_ids), MAX_FETCH_SIZE), desc="Fetching data"):
                batch_ids = all_ids[i : i + MAX_FETCH_SIZE]
                data = self.index.fetch(batch_ids)
                batch_vectors = data["vectors"]
                # verify that the ids are the same
                assert set(batch_ids) == set(batch_vectors.keys())
                metadata.update({k: v["metadata"] for k, v in batch_vectors.items()})
                vectors.update({k: v["values"] for k, v in batch_vectors.items()})
                dimensions = info["dimension"]
                # if size of vectors is greater than 1GB, save the vectors to a parquet file
                if vectors.__sizeof__() > MAX_PARQUET_FILE_SIZE:
                    total_size += self.save_vectors_to_parquet(
                        vectors, metadata, batch_ctr, vectors_directory
                    )
                    batch_ctr += 1
            total_size += self.save_vectors_to_parquet(
                vectors, metadata, batch_ctr, vectors_directory
            )
            # Create and save internal metadata JSON
            self.file_structure.append(os.path.join(vdf_directory, "VDF_META.json"))
            internal_metadata = {
                "file_structure": self.file_structure,
                # author is from unix username
                "author": os.getlogin(),
                "dimensions": info["dimension"],
                "total_vector_count": info["total_vector_count"],
                "exported_vector_count": total_size,
                "exported_from": "pinecone",
                "model_name": self.args["model_name"],
            }
            with open(os.path.join(vdf_directory, "VDF_META.json"), "w") as json_file:
                json.dump(internal_metadata, json_file, indent=4)
            # print internal metadata properly
            print(json.dumps(internal_metadata, indent=4))

        return True
