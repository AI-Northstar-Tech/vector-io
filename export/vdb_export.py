import pandas as pd
import os


class ExportVDB:
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
