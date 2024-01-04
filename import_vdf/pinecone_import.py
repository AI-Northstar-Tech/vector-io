import pandas as pd
from export.util import expand_shorthand_path
from import_vdf.vdf_import import ImportVDF
import pinecone
import os
import json
import json
from dotenv import load_dotenv
import math


load_dotenv()

BATCH_SIZE = 1000  # Set the desired batch size


class ImportPinecone(ImportVDF):
    def __init__(self, args):
        super().__init__(args)
        pinecone.init(api_key=self.args["pinecone_api_key"], environment=self.args["environment"])

    def upsert_data(self):
        # Load data from parquet files
        vectors, metadata = self.load_data()

        # Convert vectors to a list
        vectors = self.convert_vectors_to_list(vectors)

        # Upsert vectors and metadata in batches
        self.upsert_in_batches(vectors, metadata)

        print("Data import completed successfully.")

    def load_data(self):
        # Load the data from the parquet files
        parquet_files = self.get_parquet_files()

        vectors = {}
        metadata = {}
        for file in parquet_files:
            file_path = os.path.join(self.data_path, file)
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

        return vectors, metadata

    def get_parquet_files(self):
        # Get the list of parquet files in the data path
        return sorted(
            [
                file
                for file in os.listdir(self.data_path)
                if file.endswith(".parquet")
            ]
        )

    def convert_vectors_to_list(self, vectors):
        # Convert vectors from ndarray to list
        return {k: v.tolist() for k, v in vectors.items()}

    def upsert_in_batches(self, vectors, metadata):
        # Upsert the vectors and metadata in batches
        num_batches = math.ceil(len(vectors) / self.batch_size)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(vectors))

            batch_vectors = [
                pinecone.Vector(
                    id=str(id),
                    values=vector,
                    metadata=metadata.get(id, {}),
                )
                for id, vector in list(vectors.items())[start_idx:end_idx]
            ]

            self.index.upsert(vectors=batch_vectors, namespace=self.namespace)