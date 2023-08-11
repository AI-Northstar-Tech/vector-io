from vdb_export import ExportVDB
from qdrant_client import QdrantClient
import os
from tqdm import tqdm
import pandas as pd
import sqlite3
from dotenv import load_dotenv

load_dotenv()


class ExportQdrant(ExportVDB):
    def __init__(self, args):
        """
        Initialize the class
        """
        try:
            self.client = QdrantClient(
                url=args.qdrant_url, api_key=os.getenv("QDRANT_API_KEY")
            )
        except:
            self.client = QdrantClient(url=args.qdrant_url)
    
    def get_all_class_names(self):
        """
        Get all class names from Qdrant
        """
        collections = self.client.get_collections().collections
        class_names = [collection.name for collection in collections]
        return class_names

    def get_data(self, class_name):
        """
        Get data from Qdrant
        """
        total = self.client.get_collection(collection_name=class_name).points_count
        con = sqlite3.connect(f"{class_name}_qdrant.db")
        cur = con.cursor()
        property_names = []
        first = self.client.scroll(
            collection_name=class_name, limit=1, with_payload=True
        )
        for name in first[0][0].payload:
            property_names.append(name)
        cur.execute(f"DROP TABLE IF EXISTS {class_name}_qdrant")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {class_name}_qdrant (id, {','.join(property_names)})"
        )
        insert_query = f"INSERT INTO {class_name}_qdrant (id, {','.join(property_names)}) VALUES ({','.join(['?']*(len(property_names) + 1))})"
        objects = self.client.scroll(
            collection_name=class_name, limit=100, with_payload=True, with_vectors=True
        )
        df = pd.DataFrame(columns=["Vectors"])
        df.to_csv(f"{class_name}_qdrant.csv", index=False)
        self.insert_data(
            f"{class_name}_qdrant.csv", objects[0], property_names, insert_query, cur
        )
        for i in tqdm(range((total // 100) - 1)):
            uuid = objects[-1]
            objects = self.client.scroll(
                collection_name=class_name,
                limit=100,
                offset=uuid,
                with_payload=True,
                with_vectors=True,
            )
            self.insert_data(
                f"{class_name}_qdrant.csv",
                objects[0],
                property_names,
                insert_query,
                cur,
            )

    def insert_data(self, file_path, objects, property_names, insert_query, cur):
        """
        Insert data into sqlite database and csv file
        """
        data_to_insert = []
        vectors = []
        for object in objects:
            vectors.append({"Vectors": object.vector})
            data_dict = {}
            data_dict["id"] = object.id
            for property_name in property_names:
                if property_name in object.payload:
                    data_dict[property_name] = object.payload[property_name]
                else:
                    data_dict[property_name] = ""
            data_tuple = ()
            for property in data_dict.values():
                data_tuple += (property,)
            data_to_insert.append(data_tuple)
        vectors = pd.DataFrame(vectors)
        vectors.to_csv(file_path, mode="a", header=False, index=False)
        cur.executemany(insert_query, data_to_insert)

print(ExportQdrant("http://localhost:6333").get_all_class_names())