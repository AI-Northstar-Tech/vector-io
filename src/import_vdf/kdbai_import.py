import os
from dotenv import load_dotenv
import kdbai_client as kdbai
from names import DBNames
from import_vdf.vdf_import_cls import ImportVDF
from util import standardize_metric_reverse, standardize_metric
import json
import pyarrow.parquet as pq

load_dotenv()


class ImportKDBAI(ImportVDF):
    DB_NAME_SLUG = DBNames.KDBAI

    def __init__(self, args):
        super().__init__(args)
        api_key = args.get("kdbai_api_key")
        endpoint = args.get("url")
        self.index = args.get("ind")
        self.dir_path = args.get("dir")
        self.session = kdbai.Session(api_key=api_key, endpoint=endpoint)

    def upsert_data(self):
        json_file_path = os.path.join(self.dir_path, "VDF_META.json")
        # return json_file_path
        with open(json_file_path, "r") as json_file:
            data = json.load(json_file)

        indexes_content = data.get("indexes", {})
        index_names = list(data.get("indexes", {}).keys())

        # Load Parquet file
        parquet_file_path = indexes_content[index_names[0]][""][0]["data_path"]
        parquet_table = pq.read_table(parquet_file_path)
        parquet_schema = parquet_table.schema
        parquet_columns = [
            {"name": field.name, "type": str(field.type)} for field in parquet_schema
        ]

        # Extract information from JSON
        namespace = indexes_content[index_names[0]][""][0]["namespace"]
        vector_columns = indexes_content[index_names[0]][""][0]["vector_columns"]

        # Define the schema
        schema = {"columns": []}

        # Add vector column information from JSON and Parquet
        schema["columns"].append(
            {
                "name": vector_columns,
                "vectorIndex": {
                    "dims": indexes_content[index_names[0]][""][0]["dimensions"],
                    "metric": standardize_metric_reverse(indexes_content[index_names[0]][""][0]["metric"],self.DB_NAME_SLUG),
                    "type": self.index.lower(),
                },
            }
        )

        allowed_vector_types = ["flat", "ivf", "ivfpq", "hnsw"]
        if self.index.lower() not in allowed_vector_types:
            raise ValueError(
                f"Invalid vectorIndex type: {self.index}. "
                f"Allowed types are {', '.join(allowed_vector_types)}"
            )

        # Add other columns from Parquet (excluding vector columns)
        for col in parquet_columns:
            if col["name"] != vector_columns:
                schema["columns"].append({"name": col["name"], "pytype": col["type"]})

        for column in schema["columns"]:
            if "pytype" in column and column["pytype"] == "string":
                column["pytype"] = "str"

        # First ensure the table does not already exist
        try:
            self.session.table(index_names).drop()
            time.sleep(5)
        except kdbai.KDBAIException:
            pass

        # create table
        table_name = index_names[0]
        table = self.session.create_table(table_name, schema)
        print("Table created")

        # insert data
        # Set the batch size
        batch_size = 10000
        df = parquet_table.to_pandas()

        for i in range(0, df.shape[0], batch_size):
            chunk = df[i : i + batch_size].reset_index(drop=True)
            # Assuming 'table' has an 'insert' method
            try:
                table.insert(chunk)
                print(
                    f"Inserted {min(i + batch_size, df.shape[0])} out of {df.shape[0]} rows."
                )
            except kdbai.KDBAIException as e:
                print(f"Error inserting chunk: {e}")

        # table.insert(df)
        print("Data fully added")
