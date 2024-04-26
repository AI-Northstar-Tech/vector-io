import os
import json
from typing import Dict, List
from dotenv import load_dotenv
import datetime
from tqdm import tqdm

import kdbai_client as kdbai

from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.meta_types import NamespaceMeta, VDFMeta
from vdf_io.util import (
    get_author_name,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric,
)


load_dotenv()


class ExportKDBAI(ExportVDB):
    DB_NAME_SLUG = DBNames.KDBAI

    @classmethod
    def make_parser(cls, subparsers):
        parser_kdbai = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from KDB.AI"
        )
        parser_kdbai.add_argument(
            "-u", "--url", type=str, help="KDB.AI cloud endpoint to connect"
        )
        parser_kdbai.add_argument(
            "-t", "--tables", type=str, help="KDB.AI tables to export (comma-separated)"
        )

    @classmethod
    def export_vdb(cls, args):
        """
        Export data from KDBAI
        """
        set_arg_from_input(
            args,
            "url",
            "Enter the KDB.AI endpoint instance: ",
            str,
        )
        set_arg_from_password(
            args, "kdbai_api_key", "Enter your KDB.AI API key: ", "KDBAI_API_KEY"
        )
        kdbai_export = ExportKDBAI(args)
        set_arg_from_input(
            args,
            "tables",
            f"Enter the name of table to export: {kdbai_export.get_all_table_names()}",
            str,
            None,
        )
        if args.get("tables", None) == "":
            args["tables"] = ",".join(kdbai_export.get_all_table_names())
        kdbai_export.get_data()
        return kdbai_export

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
            author=get_author_name(),
            exported_from=self.DB_NAME_SLUG,
            indexes=index_metas,
            exported_at=datetime.datetime.now().astimezone().isoformat(),
        )

        internal_metadata_path = os.path.join(self.vdf_directory, "VDF_META.json")
        meta_json_text = json.dumps(internal_metadata.model_dump(), indent=4)
        print(meta_json_text)
        with open(internal_metadata_path, "w") as json_file:
            json_file.write(meta_json_text)

    def export_table(self, table_name):
        model = self.model
        vectors_directory = self.create_vec_dir(table_name)

        table = self.session.table(table_name)
        table_res = table.query()
        save_path = f"{vectors_directory}/{table_name}.parquet"
        table_res.to_parquet(save_path, index=False)

        # TODO: use save_vectors_to_parquet
        # vectors = table_res["vector"].apply(pd.Series)
        # metadata = table_res.drop(columns=["vector"]).to_dict(orient="records")
        # self.save_vectors_to_parquet(vectors, metadata, vectors_directory)
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
        self.args["exported_count"] += len(table_res.index)
        return [namespace_meta]
