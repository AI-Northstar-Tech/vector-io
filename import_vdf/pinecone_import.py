import pandas as pd
from tqdm import tqdm
from export_vdf.util import standardize_metric_reverse
from import_vdf.vdf_import_cls import ImportVDF
import pinecone
import os
from dotenv import load_dotenv
import math

load_dotenv()

BATCH_SIZE = 1000  # Set the desired batch size


class ImportPinecone(ImportVDF):
    def __init__(self, args):
        super().__init__(args)
        pinecone.init(api_key=args["pinecone_api_key"], environment=args["environment"])

    def upsert_data(self):
        # Iterate over the indexes and import the data
        for index_name, index_meta in tqdm(
            self.vdf_meta["indexes"].items(), desc="Importing indexes"
        ):
            print(f"Importing data for index '{index_name}'")
            # list indexes
            indexes = pinecone.list_indexes()
            # check if index exists
            if index_name not in indexes:
                # create index
                try:
                    pinecone.create_index(
                        name=index_name,
                        dimension=index_meta[0]["dimensions"],
                        metric=standardize_metric_reverse(
                            index_meta[0]["metric"], "pinecone"
                        ),
                    )
                except Exception as e:
                    print(e)
                    raise Exception(f"Invalid index name '{index_name}'", e)
            index = pinecone.Index(index_name=index_name)
            current_batch_size = BATCH_SIZE
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
                vector_column_name = self.get_vector_column_name(
                    index_name, namespace_meta
                )

                for file in tqdm(parquet_files, desc="Loading data from parquet files"):
                    file_path = os.path.join(data_path, file)
                    df = pd.read_parquet(file_path)
                    vectors.update(
                        {
                            row["id"]: row[vector_column_name].tolist()
                            for _, row in df.iterrows()
                        }
                    )
                    metadata.update(
                        {
                            row["id"]: {
                                key: value
                                for key, value in row.items()
                                if key != "id" and key != vector_column_name
                            }
                            for _, row in df.iterrows()
                        }
                    )
                print(
                    f"Loaded {len(vectors)} vectors from {len(parquet_files)} parquet files"
                )
                # Upsert the vectors and metadata to the Pinecone index in batches
                num_batches = math.ceil(len(vectors) / current_batch_size)
                imported_count = 0
                for i in tqdm(range(num_batches), desc="Importing data in batches"):
                    start_idx = i * current_batch_size
                    end_idx = min((i + 1) * current_batch_size, len(vectors))

                    batch_vectors = [
                        pinecone.Vector(
                            id=str(id),
                            values=vector,
                            metadata=metadata.get(id, {}),
                        )
                        for id, vector in list(vectors.items())[start_idx:end_idx]
                    ]
                    try:
                        index.upsert(vectors=batch_vectors, namespace=namespace)
                        imported_count += len(batch_vectors)
                    except Exception as e:
                        print(f"Error upserting vectors for index '{index_name}'", e)
                        current_batch_size = int(2 * current_batch_size / 3)
                        print(f"Reducing batch size to {current_batch_size}")
                        continue
        print(f"Data import completed successfully. Imported {imported_count} vectors")
