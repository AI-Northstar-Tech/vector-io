from qdrant_client import QdrantClient
from qdrant_client.http import models
import re
from tqdm import tqdm
import pandas as pd
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq

class ExportQdrant:
    def __init__(self, client):
        """
        Initialize the class
        """
        self.client = client

    def get_data(self, class_name):
        """
        Get data from Qdrant
        """
        total = self.client.get_collection(collection_name=class_name).points_count
        con = sqlite3.connect(f'{class_name}_qdrant.db')
        cur = con.cursor()
        property_names = []
        first = self.client.scroll(collection_name=class_name, limit = 1, with_payload=True)
        for name in first[0][0].payload:
            property_names.append(name)
        cur.execute(f"DROP TABLE IF EXISTS {class_name}_qdrant")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {class_name}_qdrant (id, {','.join(property_names)})")
        insert_query = f"INSERT INTO {class_name}_qdrant (id, {','.join(property_names)}) VALUES ({','.join(['?']*(len(property_names) + 1))})"
        objects = self.client.scroll(collection_name=class_name, limit = 100, with_payload=True, with_vectors=True)
        df = pd.DataFrame(columns=["Vectors"])
        self.insert_data(f'{class_name}_qdrant.parquet', objects[0], property_names, insert_query, cur)
        for i in tqdm(range((total//100)-1)):
            uuid = objects[-1]
            objects = self.client.scroll(collection_name=class_name, limit = 100, offset=uuid, with_payload=True, with_vectors=True)
            self.insert_data(f'{class_name}_qdrant.parquet', objects[0], property_names, insert_query, cur)

    def insert_data(self, file_path, objects, property_names, insert_query, cur):
        """
        Insert data into sqlite database and parquet file
        """
        data_to_insert = []
        vectors = []
        for object in objects:
            vectors.append({"Vectors" : object.vector})
            data_dict = {}
            data_dict['id'] = object.id
            for property_name in property_names:
                if property_name in object.payload:
                    data_dict[property_name] = object.payload[property_name]
                else:
                    data_dict[property_name] = ''
            data_tuple = ()
            for property in data_dict.values():
                data_tuple += (property,)
            data_to_insert.append(data_tuple)
        vectors = pd.DataFrame(vectors)
        schema = pa.Table.from_pandas(vectors).schema
        with pq.ParquetWriter(file_path, schema) as writer:
            table = pa.Table.from_pandas(vectors, schema=schema)
            writer.write_table(table)
        cur.executemany(insert_query, data_to_insert)