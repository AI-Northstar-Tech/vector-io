import json
import kdbai_client as kdbai
import os
from dotenv import load_dotenv
from export_vdf.vdb_export_cls import ExportVDB
from names import DBNames


load_dotenv()


class ExportKDBAI(ExportVDB):
    DB_NAME_SLUG = DBNames.KDBAI

    def __init__(self, args):
        super().__init__(args)
        api_key = args.get("kdbai_api_key")
        endpoint = args.get("url")
        self.session = kdbai.Session(api_key=api_key, endpoint=endpoint)

    def get_all_table_names(self):
        return self.session.list()

    def get_data(self):
        table_name = self.args["tables"]
        model = self.args["model"]
        vectors_directory = os.path.join(self.vdf_directory, table_name)
        os.makedirs(vectors_directory, exist_ok=True)

        table = self.session.table(table_name)
        table_res = table.query()
        save_path = vectors_directory + "/" + table_name + ".parquet"
        table_res.to_parquet(save_path, index=False)

        embedding_name = None
        embedding_dims = None
        embedding_dist = None
        tab_schema = table.schema()

        for i in range(len(tab_schema["columns"])):
            if "vectorIndex" in tab_schema["columns"][i].keys():
                embedding_name = tab_schema["columns"][i]["name"]
                embedding_dims = tab_schema["columns"][i]["vectorIndex"]["dims"]
                embedding_dist = tab_schema["columns"][i]["vectorIndex"]["metric"]

        namespace_meta = {
            "namespace": "",
            "total_vector_count": len(table_res.index),
            "exported_vector_count": len(table_res.index),
            "dimensions": embedding_dims,
            "model_name": model,
            "vector_columns": embedding_name,
            "data_path": save_path,
            "metric": embedding_dist,
        }

        internal_metadata = {
            # "version": self.args["library_version"],
            "version": "0.0.6",
            "file_structure": self.file_structure,
            "author": os.environ.get("USER"),
            "exported_from": self.DB_NAME_SLUG,
            "indexes": {table_name: {"": [namespace_meta]}},
        }

        internal_metadata_path = os.path.join(self.vdf_directory, "VDF_META.json")
        with open(internal_metadata_path, "w") as json_file:
            json.dump(internal_metadata, json_file, indent=4)
