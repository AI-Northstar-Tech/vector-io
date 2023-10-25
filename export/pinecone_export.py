from export.vdb_export import ExportVDB
import pinecone
import os
import sqlite3
import json
import pandas as pd
import numpy as np


class ExportPinecone(ExportVDB):
    def __init__(self, args):
        pinecone.init(
            api_key='8d48a593-b099-4ba1-a790-6cd416d67cf3',
            environment=args.environment
        )

    def get_all_index_names(self):
        return pinecone.list_indexes()

    def get_ids_from_query(self,index,input_vector):
        print("searching pinecone...")
        results = index.query(vector=input_vector,include_values=False,top_k=100)
        ids = set()
        print(type(results))
        for result in results['matches']:
            ids.add(result['id'])
        return ids

    def get_all_ids_from_index(self,index, num_dimensions, namespace=""):
        num_vectors = index.describe_index_stats()["namespaces"][namespace]['vector_count']
        all_ids = set()
        while len(all_ids) < num_vectors:
            print("Length of ids list is shorter than the number of total vectors...")
            input_vector = np.random.rand(num_dimensions).tolist()
            print("creating random vector...")
            ids = self.get_ids_from_query(index,input_vector)
            print("getting ids from a vector query...")
            all_ids.update(ids)
            print("updating ids set...")
            print(f"Collected {len(all_ids)} ids out of {num_vectors}.")

        return all_ids

    def get_data(self, index_name):
        self.index = pinecone.Index(index_name=index_name)
        info = self.index.describe_index_stats()
        namespace = info["namespaces"]
        zero_array = [0] * info["dimension"]

        vdf_directory = "VDF_dataset"
        vectors_directory = os.path.join(vdf_directory, "vectors")
        os.makedirs(vdf_directory, exist_ok=True)
        os.makedirs(vectors_directory, exist_ok=True)

        con = sqlite3.connect(os.path.join(vdf_directory, "metadata.db"))
        cur = con.cursor()

        # Fetch the actual data from the Pinecone index
        data = self.index.fetch(list(self.get_all_ids_from_index(index=pinecone.Index("pinecone-index"),num_dimensions=8)))

        vectors = data['vectors']
        for id, vector_data in vectors.items():
            namespace = data["namespace"]
            property_names = list(vector_data["metadata"].keys())

            # Modify the table name to replace hyphens with underscores
            table_name = f"{namespace}_{index_name}".replace('-', '_')

            parquet_file = os.path.join(vectors_directory, f"{namespace}.parquet")

            cur.execute(
                f'CREATE TABLE IF NOT EXISTS {table_name} (id, {", ".join(property_names)}, "values")'
            )
            insert_query = f'INSERT INTO {table_name} (id, {", ".join(property_names)}, "values") VALUES ({", ".join(["?"] * (len(property_names) + 1))})'
            self.insert_data(parquet_file, vector_data, property_names, insert_query, cur)

        con.commit()
        con.close()

        # Create and save internal metadata JSON
        internal_metadata = {
            "file_structure": ["vectors/", "metadata.db", "VDF_META.json"]
        }
        with open(os.path.join(vdf_directory, "VDF_META.json"), 'w') as json_file:
            json.dump(internal_metadata, json_file)

    def insert_data(self, file_path, vector_data, property_names, insert_query, cur):
        data_to_insert = []

        data_dict = {"id": vector_data["id"]}
        for property_name in property_names:
            data_dict[property_name] = vector_data["metadata"].get(property_name, "")

        # Ensure 'values' is a list, otherwise skip this data point
        values = vector_data["values"]
        if not isinstance(values, list):
            return

        data_dict["values"] = values  # Store it as a list
        data_tuple = tuple(data_dict.get(property_name, "") for property_name in ["id"] + property_names + ["values"])
        data_to_insert.append(data_tuple)

        new_df = pd.DataFrame(data_to_insert, columns=["id"] + property_names + ["values"])

        # Now, convert the 'values' values to JSON strings in the DataFrame
        new_df['values'] = new_df['values'].apply(json.dumps)

        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            df = pd.concat([df, new_df])
        else:
            df = new_df

        # Ensure all values in 'values' column are lists and fill null values with empty lists
        df['values'] = df['values'].apply(lambda x: x if isinstance(x, list) else [])

        # Get the actual columns in the DataFrame
        actual_columns = df.columns.tolist()
        namespace = ''
        index_name = 'pinecone-index'
        # Modify the table name to replace hyphens with underscores
        table_name = f"{namespace}_{index_name}".replace('-', '_')

        # Create the table with the actual columns
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(actual_columns)})"
        )

        # Update the insert query with the actual columns
        insert_query = f"INSERT INTO {table_name} ({', '.join(actual_columns)}) VALUES ({', '.join(['?'] * len(actual_columns))})"

        df.to_parquet(file_path, index=False)
        cur.executemany(insert_query, data_to_insert)


    