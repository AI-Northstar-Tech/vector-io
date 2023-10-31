from import_VDF.vdf_import import ImportVDF
import pinecone
import os
import json
import pandas as pd
import numpy as np
import sqlite3
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
import json
from dotenv import load_dotenv



class ImportPinecone(ImportVDF):
    def __init__(self, args):
        load_dotenv()
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if pinecone_api_key is None:
            raise ValueError("PINECONE_API_KEY is not set in the .env file")
        pinecone.init(api_key=pinecone_api_key, environment=args.environment)
        self.index_name = args.index
        self.dir = args.dir


    def upsert_data(self):
        index = pinecone.Index(index_name=self.index_name)
        # Read vectors from Parquet file
        vectors_df = pq.read_table(os.path.join(self.dir, "vectors", "part-1.parquet")).to_pandas()

        # Connect to the SQLite database and fetch metadata
        conn = sqlite3.connect(os.path.join(self.dir, "metadata.db"))
        c = conn.cursor()
        c.execute("SELECT * FROM metadata")
        metadata_rows = c.fetchall()

        # Convert metadata rows to a dictionary
        metadata_dict = {row[0]: dict(zip([column[0] for column in c.description][1:], row[1:])) for row in metadata_rows}

        # Prepare data for upsert
        data_to_upsert = [(row['id'], row['vector'].tolist(), metadata_dict[row['id']]) for _, row in vectors_df.iterrows()]
        # Upsert data to Pinecone
        index.upsert(data_to_upsert)

        print(data_to_upsert)

