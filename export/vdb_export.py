import argparse
import os
import re
from dotenv import load_dotenv
from tqdm import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import sqlite3
from qdrant_client import QdrantClient
from qdrant_client.http import models
import pinecone
import weaviate

load_dotenv()


class ExportVDB:
    def get_data():
        """
        Get data from vector database
        """
        raise NotImplementedError

    def insert_data():
        """
        Insert data into sqlite database and csv file
        """
        raise NotImplementedError


class ExportPinecone(ExportVDB):
    def __init__(self, args):
        """
        Initialize the index
        """
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"), environment=args.environment
        )
        self.index = pinecone.Index(index_name=args.index_name)

    def get_data(self, index_name):
        """
        Get data from Pinecone
        """
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
            df.to_csv(f"{namespace}_{index_name}.csv", index=False)
            self.insert_data(
                f"{namespace}_{index_name}.csv",
                response["matches"],
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
        vectors.to_csv(file_path, index=False, mode="a", header=False)
        cur.executemany(insert_query, data_to_insert)


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


class ExportWeaviate(ExportVDB):
    data_types = [
        "text",
        "text[]",
        "int",
        "int[]",
        "number",
        "number[]",
        "boolean",
        "boolean[]",
        "date",
        "date[]",
        "geoCoordinates",
        "phoneNumber",
        "blob",
        "string",
        "string[]",
        "uuid",
        "uuid[]",
    ]

    def __init__(self, args):
        """
        Initialize the class
        """
        try:
            auth_client_secret = weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
            self.weaviate_client = weaviate.Client(
                url=args.weaviate_url, auth_client_secret=auth_client_secret
            )
        except:
            self.weaviate_client = weaviate.Client(url=weaviate_url)

    def get_data(self, class_name, include_crossrefs=False):
        """
        Get data from weaviate
        """
        schema = self.weaviate_client.schema.get(class_name="Patent")
        property_names = [
            property["name"]
            for property in schema["properties"]
            if property["dataType"][0] in self.data_types
        ]
        property_names = sorted(property_names)
        con = sqlite3.connect(f"{class_name}_weaviate.db")
        cur = con.cursor()
        cur.execute(f"DROP TABLE IF EXISTS {class_name}_weaviate")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {class_name}_weaviate (uuid, {','.join(property_names)})"
        )
        insert_query = f"INSERT INTO {class_name}_weaviate (uuid, {','.join(property_names)}) VALUES ({','.join(['?']*(len(property_names) + 1))})"

        if include_crossrefs:
            cross_refs_schemas, cross_refs = self.check_crossref(
                schema, self.data_types
            )
            if cross_refs_schemas is not None:
                insert_queries, property_names_dict = self.create_tables(
                    class_name, cross_refs_schemas, cur
                )

        total = (
            self.weaviate_client.query.aggregate(f"{class_name}")
            .with_meta_count()
            .do()["data"]["Aggregate"][f"{class_name}"][0]["meta"]["count"]
        )
        objects = self.weaviate_client.data_object.get(
            class_name=class_name, limit=100, with_vector=True
        )
        df = pd.DataFrame(columns=["Vectors"])
        df.to_csv(f"{class_name}_weaviate.csv", index=False)

        if include_crossrefs:
            self.insert_data(
                f"{class_name}_weaviate.csv",
                objects,
                property_names,
                insert_query,
                cur,
                cross_refs,
                insert_queries,
                property_names_dict,
            )
            for _ in tqdm(range(total // 100)):
                try:
                    uuid = objects["objects"][-1]["id"]
                    objects = self.weaviate_client.data_object.get(
                        class_name=class_name, limit=100, with_vector=True, after=uuid
                    )
                    self.insert_data(
                        f"{class_name}_weaviate.csv",
                        objects,
                        property_names,
                        insert_query,
                        cur,
                        cross_refs,
                        insert_queries,
                        property_names_dict,
                    )
                except Exception as e:
                    break

        else:
            self.insert_data(
                f"{class_name}_weaviate.csv",
                objects,
                property_names,
                insert_query,
                cur,
                None,
                None,
                None,
            )
            for _ in tqdm(range(total // 100)):
                try:
                    uuid = objects["objects"][-1]["id"]
                    objects = self.weaviate_client.data_object.get(
                        class_name=class_name, limit=100, with_vector=True, after=uuid
                    )
                    self.insert_data(
                        f"{class_name}_weaviate.csv",
                        objects,
                        property_names,
                        insert_query,
                        cur,
                        None,
                        None,
                        None,
                    )
                except Exception as e:
                    break

    def check_crossref(self, schema, data_types):
        """
        Check if there are cross references in the schema
        """
        cross_refs = [
            (property["name"], property["dataType"][0])
            for property in schema["properties"]
            if property["dataType"][0] not in data_types
        ]
        cross_refs_schemas = []
        if len(cross_refs) > 0:
            for _, class_name in cross_refs:
                schema = self.weaviate_client.schema.get(class_name=class_name)
                cross_refs_schemas.append(schema)
            return cross_refs_schemas, cross_refs
        else:
            return None

    def create_tables(self, parent_class, cross_refs_schemas, data_types, cur):
        """
        Create tables for cross references
        """
        insert_queries = {}
        property_names_dict = {}
        for schema in cross_refs_schemas:
            class_name = schema["class"]
            property_names = [
                property["name"]
                for property in schema["properties"]
                if property["dataType"][0] in data_types
            ]
            property_names = sorted(property_names)
            property_names_dict[class_name] = property_names
            cur.execute(f"DROP TABLE IF EXISTS {class_name}_weaviate")
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {class_name}_weaviate (main_uuid REFERENCES {parent_class} (uuid), uuid, {','.join(property_names)})"
            )
            insert_query = f"INSERT INTO {class_name}_weaviate (main_uuid, uuid, {','.join(property_names)}) VALUES ({','.join(['?']*(len(property_names) + 2))})"
            insert_queries[class_name] = insert_query
        if insert_queries == {}:
            return None, None
        else:
            return insert_queries, property_names_dict

    def insert_data(
        self,
        file_path,
        objects,
        property_names,
        insert_query,
        cur,
        cross_refs,
        insert_query_crossrefs,
        property_names_dict,
    ):
        """
        Insert data into sqlite database and csv file
        """
        data_to_insert = []
        vectors = []
        for object in objects["objects"]:
            vectors.append({"Vectors": object["vector"]})
            data_dict = {}
            data_dict["uuid"] = object["id"]
            for property_name in property_names:
                if property_name in object["properties"]:
                    data_dict[property_name] = object["properties"][property_name]
                else:
                    data_dict[property_name] = ""
            data_tuple = ()
            for property in data_dict.values():
                data_tuple += (property,)
            data_to_insert.append(data_tuple)
        vectors = pd.DataFrame(vectors)
        vectors.to_csv(file_path, mode="a", header=False, index=False)
        cur.executemany(insert_query, data_to_insert)
        if cross_refs is not None:
            for cross_ref_name, cross_ref_class_name in cross_refs:
                data_to_insert = []
                for object in objects["objects"]:
                    if cross_ref_name in object["properties"]:
                        for refs in object["properties"][cross_ref_name]:
                            pattern = r"weaviate://localhost/[^/]+/([^/]+)"
                            match = re.search(pattern, refs["beacon"])
                            if match:
                                uuid = match.group(1)
                                obj = self.weaviate_client.data_object.get_by_id(
                                    class_name=cross_ref_class_name, uuid=uuid
                                )
                                if obj is not None:
                                    data_dict = {}
                                    data_dict["main_uuid"] = object["id"]
                                    data_dict["uuid"] = obj["id"]
                                    for property_name in property_names_dict[
                                        cross_ref_class_name
                                    ]:
                                        if property_name in obj["properties"]:
                                            data_dict[property_name] = obj[
                                                "properties"
                                            ][property_name]
                                        else:
                                            data_dict[property_name] = ""
                                    data_tuple = ()
                                    for property in data_dict.values():
                                        data_tuple += (property,)
                                    data_to_insert.append(data_tuple)
                for key, value in insert_query_crossrefs.items():
                    if key == cross_ref_class_name:
                        insert_query_crossref = value
                cur.executemany(insert_query_crossref, data_to_insert)
