from export.vdb_export import ExportVDB
import pinecone
import os
from tqdm import tqdm
import pandas as pd
import sqlite3
from dotenv import load_dotenv

load_dotenv()


class ExportPinecone(ExportVDB):
    def __init__(self, args):
        """
        Initialize the index
        """
        pinecone.init(
            api_key=args.pinecone_api_key, environment=args.environment
        )

    def get_all_index_names(self):
        """
        Get all index names from Pinecone
        """
        return pinecone.list_indexes()

    def get_data(self, index_name):
        """
        Get data from Pinecone
        """
        self.index = pinecone.Index(index_name=index_name)
        info = self.index.describe_index_stats()
        namespaces = info["namespaces"]
        vector_dim = int(pinecone.describe_index(index_name).dimension)
        zero_array = [0] * vector_dim
        data = []
        for key, value in namespaces.items():
            response = self.index.query(
                namespace=key,
                top_k=value["vector_count"],
                include_metadata=True,
                include_values=True,
                vector=zero_array,
            )
            data.append(response)
        con = sqlite3.connect(f"{index_name}_pinecone.db")
        cur = con.cursor()
        df = pd.DataFrame(columns=["Vectors"])
        for response in tqdm(data):
            namespace = response["namespace"]
            property_names = list(response["matches"][0]["metadata"].keys())
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {namespace}_{index_name} (id, {','.join(property_names)})"
            )
            insert_query = f"INSERT INTO {namespace}_{index_name} (id, {','.join(property_names)}) VALUES ({','.join(['?']*(len(property_names) + 1))})"
            df.to_parquet(f"{namespace}_{index_name}.parquet", index=False)
            self.insert_data(
                f"{namespace}_{index_name}.parquet",
                response["matches"],
                property_names,
                insert_query,
                cur,
            )

    def insert_data(self, file_path, objects, property_names, insert_query, cur):
        """
        Insert data into sqlite database and parquet file
        """
        data_to_insert = []
        vectors = []
        for object in objects:
            vectors.append({"Vectors": object.values})
            data_dict = {}
            data_dict["id"] = object.id
            for property_name in property_names:
                if property_name in object.metadata:
                    data_dict[property_name] = object.metadata[property_name]
                else:
                    data_dict[property_name] = ""
            data_tuple = ()
            for property in data_dict.values():
                data_tuple += (property,)
            data_to_insert.append(data_tuple)
        vectors = pd.DataFrame(vectors)
        vectors.to_parquet(file_path, index=False, mode="a", header=False)
        cur.executemany(insert_query, data_to_insert)