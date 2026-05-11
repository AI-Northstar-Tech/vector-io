import os
import json
import argparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.weaviate_util import (
    collection_count,
    connect_to_weaviate,
    find_distance_metric,
    get_collection_config,
    is_weaviate_cloud_url,
    list_collection_names,
    normalize_vector_map,
    sanitize_properties,
    to_plain_dict,
    use_collection,
)


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        parser_weaviate.add_argument(
            "-u", "--url", type=str, help="URL of Weaviate instance"
        )
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--deployment",
            choices=["local", "cloud", "custom"],
            help="Weaviate deployment type. Inferred from --url by default.",
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help="gRPC port for local/custom Weaviate instances",
        )
        parser_weaviate.add_argument(
            "--grpc_secure",
            help="Use TLS for gRPC custom connections",
            default=None,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "-c", "--classes", type=str, help="Classes to export (comma-separated)"
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for writing exported objects to parquet",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: 'http://localhost:8080'): ",
            str,
            "http://localhost:8080",
        )
        if args.get("deployment") == "cloud" or is_weaviate_cloud_url(args.get("url")):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )
        else:
            if os.getenv("WEAVIATE_API_KEY"):
                args["api_key"] = os.getenv("WEAVIATE_API_KEY")
            set_arg_from_input(
                args,
                "api_key",
                "Enter the Weaviate API key (hit return if auth is disabled): ",
                str,
                "DO_NOT_PROMPT",
                env_var="WEAVIATE_API_KEY",
            )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the export batch size (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = weaviate_export.get_all_index_names()
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the name of the classes to export (comma-separated, all will be exported by default): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_to_weaviate(self.args)

    def get_all_index_names(self):
        return list_collection_names(self.client)

    def get_index_names(self):
        if self.args.get("classes") is None:
            return self.get_all_index_names()
        else:
            input_classes = self.args["classes"].split(",")
            all_classes = self.get_all_index_names()
            if set(input_classes) - set(all_classes):
                tqdm.write(
                    f"These classes are not present in the Weaviate instance: {set(input_classes) - set(all_classes)}"
                )
            return [c for c in all_classes if c in input_classes]

    def get_data(self):
        index_metas = {}
        index_names = self.get_index_names()
        for class_name in tqdm(index_names, desc="Exporting Weaviate classes"):
            index_metas[class_name] = self.get_data_for_collection(class_name)

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def get_data_for_collection(self, class_name):
        collection = use_collection(self.client, class_name)
        total = collection_count(collection)
        collection_config = get_collection_config(collection)
        vectors_directory = self.create_vec_dir(class_name)
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        vector_columns = []
        rows = []
        num_vectors_exported = 0
        dimensions = -1

        for item in tqdm(
            collection.iterator(include_vector=True),
            total=total,
            desc=f"Exporting {class_name}",
        ):
            row = self._item_to_row(item, vector_columns)
            if row is None:
                continue
            if dimensions == -1:
                dimensions = self._first_vector_dimension(row, vector_columns)
            rows.append(row)
            if len(rows) >= batch_size:
                num_vectors_exported += self._save_rows_to_parquet(
                    rows, vectors_directory
                )
                rows = []

        num_vectors_exported += self._save_rows_to_parquet(rows, vectors_directory)
        self.args["exported_count"] += num_vectors_exported
        return [
            self.get_namespace_meta(
                class_name,
                vectors_directory,
                total=total,
                num_vectors_exported=num_vectors_exported,
                dim=dimensions,
                vector_columns=vector_columns or ["vector"],
                distance=find_distance_metric(collection_config),
                index_config=to_plain_dict(collection_config),
            )
        ]

    def _item_to_row(self, item, vector_columns):
        vector_map = normalize_vector_map(getattr(item, "vector", None))
        if not vector_map:
            tqdm.write(f"Skipping Weaviate object without vector: {item}")
            return None
        for vector_column in vector_map.keys():
            if vector_column not in vector_columns:
                vector_columns.append(vector_column)
        properties = sanitize_properties(
            dict(getattr(item, "properties", {}) or {}),
            excluded_columns=[ID_COLUMN, *vector_map.keys()],
        )
        row = {ID_COLUMN: str(getattr(item, "uuid"))}
        row.update(properties)
        row.update(vector_map)
        return row

    def _first_vector_dimension(self, row, vector_columns):
        for vector_column in vector_columns:
            vector = row.get(vector_column)
            if isinstance(vector, list):
                return len(vector)
        return -1

    def _save_rows_to_parquet(self, rows, vectors_directory):
        if not rows:
            return 0
        df = pd.DataFrame(rows)
        parquet_file = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(parquet_file)
        if not hasattr(self, "parquet_schema"):
            self.parquet_schema = pq.read_schema(parquet_file)
        else:
            self.parquet_schema = pa.unify_schemas(
                [self.parquet_schema, pq.read_schema(parquet_file)]
            )
        self.file_structure.append(parquet_file)
        self.file_ctr += 1
        return len(df)
