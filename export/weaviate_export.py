from export.vdb_export import ExportVDB
import weaviate
import os
import re
from tqdm import tqdm
import pandas as pd
import sqlite3
from dotenv import load_dotenv

load_dotenv()


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
                url=args["weaviate_url"], auth_client_secret=auth_client_secret
            )
        except:
            self.weaviate_client = weaviate.Client(url=args["weaviate_url"])
    
    def get_all_class_names(self):
        """
        Get all class names from weaviate
        """
        class_names = [clss['class'] for clss in self.weaviate_client.schema.get().get("classes")]
        print(class_names)
        return class_names
    
    def get_data(self, class_name, include_crossrefs=False):
        """
        Get data from weaviate
        """
        schema = self.weaviate_client.schema.get(class_name=class_name)
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
        df.to_parquet(f"{class_name}_weaviate.parquet", index=False)

        if include_crossrefs:
            self.insert_data(
                f"{class_name}_weaviate.parquet",
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
                        f"{class_name}_weaviate.parquet",
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
                f"{class_name}_weaviate.parquet",
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
                        f"{class_name}_weaviate.parquet",
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
        Insert data into sqlite database and parquet file
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
        vectors.to_parquet(file_path, mode="a", header=False, index=False)
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
