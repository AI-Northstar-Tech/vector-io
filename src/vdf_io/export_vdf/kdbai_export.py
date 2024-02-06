import datetime
import json
from typing import Dict, List

from tqdm import tqdm
import kdbai_client as kdbai
import os
from dotenv import load_dotenv
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from names import DBNames
from meta_types import NamespaceMeta, VDFMeta
from util import standardize_metric


load_dotenv()


class ExportKDBAI(ExportVDB):
    DB_NAME_SLUG = DBNames.KDBAI

    def __init__(self, args):
        super().__init__(args)
        api_key = args.get("kdbai_api_key")
        endpoint = args.get("url")
        self.session = kdbai.Session(api_key=api_key, endpoint=endpoint)
        self.model = args.get("model_name")

    def get_all_table_names(self):
        return self.session.list()

    def get_data(self):
        if "tables" not in self.args or self.args["tables"] is None:
            table_names = self.get_all_table_names()
        else:
            table_names = self.args["tables"].split(",")
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for table_name in tqdm(table_names, desc="Fetching indexes"):
            index_metas[table_name] = self.export_table(table_name)
        internal_metadata = VDFMeta(
            version=self.args["library_version"],
            file_structure=self.file_structure,
            author=os.environ.get("USER"),
            exported_from=self.DB_NAME_SLUG,
            indexes=index_metas,
            exported_at=datetime.datetime.now().astimezone().isoformat(),
        )

        internal_metadata_path = os.path.join(self.vdf_directory, "VDF_META.json")
        meta_json_text = json.dumps(internal_metadata.dict(), indent=4)
        print(meta_json_text)
        with open(internal_metadata_path, "w") as json_file:
            json_file.write(meta_json_text)

    def export_table(self, table_name):
        model = self.model
        vectors_directory = os.path.join(self.vdf_directory, table_name)
        os.makedirs(vectors_directory, exist_ok=True)

        table = self.session.table(table_name)
        table_res = table.query()
        save_path = f"{vectors_directory}/{table_name}.parquet"
        table_res.to_parquet(save_path, index=False)

        embedding_name = None
        embedding_dims = None
        embedding_dist = None
        tab_schema = table.schema()

        for i in range(len(tab_schema["columns"])):
            if "vectorIndex" in tab_schema["columns"][i].keys():
                embedding_name = tab_schema["columns"][i]["name"]
                embedding_dims = tab_schema["columns"][i]["vectorIndex"]["dims"]
                embedding_dist = standardize_metric(
                    tab_schema["columns"][i]["vectorIndex"]["metric"], self.DB_NAME_SLUG
                )

        namespace_meta = NamespaceMeta(
            namespace="",
            index_name=table_name,
            total_vector_count=len(table_res.index),
            exported_vector_count=len(table_res.index),
            dimensions=embedding_dims,
            model_name=model,
            vector_columns=[embedding_name],
            data_path="/".join(vectors_directory.split("/")[1:]),
            metric=embedding_dist,
        )
        return [namespace_meta]
