import datetime
import pandas as pd
import os

from export_vdf.util import extract_data_hash


class ExportVDB:
    def __init__(self, args):
        self.args = args
        self.file_structure = []
        self.file_ctr = 1
        self.hash_value = extract_data_hash(self.args)
        self.timestamp_in_format = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.vdf_directory = f"vdf_{self.timestamp_in_format}_{self.hash_value}"
        os.makedirs(self.vdf_directory, exist_ok=True)

    def get_data():
        """
        Get data from vector database
        """
        raise NotImplementedError

    def save_vectors_to_parquet(self, vectors, metadata, batch_ctr, vectors_directory):
        vectors_df = pd.DataFrame(list(vectors.items()), columns=["id", "vector"])
        # Store the vector in values as a column in the parquet file, and store the metadata as columns in the parquet file
        # Convert metadata to a dataframe with each of metadata_keys as a column
        # Convert metadata to a list of dictionaries
        metadata_list = [{**{"id": k}, **v} for k, v in metadata.items()]
        # Convert the list to a DataFrame
        metadata_df = pd.DataFrame.from_records(metadata_list)
        # Now merge this metadata_df with your main DataFrame
        df = vectors_df.merge(metadata_df, on="id", how="left")

        # Save the DataFrame to a parquet file
        parquet_file = os.path.join(vectors_directory, f"{batch_ctr}.parquet")
        df.to_parquet(parquet_file)
        self.file_structure.append(parquet_file)

        # Reset vectors and metadata
        vectors = {}
        metadata = {}
        return len(df)
