import pandas as pd
from tqdm import tqdm
from export.util import expand_shorthand_path
from import_vdf.vdf_import import ImportVDF
import pinecone
import os
import json
import json
from dotenv import load_dotenv
import math
from packaging.version import Version

load_dotenv()

BATCH_SIZE = 1000  # Set the desired batch size


class ImportPinecone(ImportVDF):
    def __init__(self, args):
        super().__init__(args)
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])

    def upsert_data(self):
        # if self.vdf_meta["version"] is ahead fo self.args["library_version"], prompt user to upgrade vector-io library
        if Version(self.vdf_meta["version"]) > Version(self.args["library_version"]):
            print(
                f"Warning: The version of vector-io library: ({self.args['library_version']}) is behind the version of the vdf directory: ({self.vdf_meta['version']})."
            )
            print(
                "Please upgrade the vector-io library to the latest version to ensure compatibility."
            )
        # Iterate over the indexes and import the data
        for index_name, index_meta in tqdm(self.vdf_meta["indexes"].items(), desc="Importing indexes"):
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
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
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
                    vectors.update({row["id"]: row["vector"].tolist() for _, row in df.iterrows()})
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
                print(f"Loaded {len(vectors)} vectors from {len(parquet_files)} parquet files")
                # Upsert the vectors and metadata to the Pinecone index in batches
                num_batches = math.ceil(len(vectors) / BATCH_SIZE)

                for i in tqdm(range(num_batches), desc="Importing data in batches"):
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