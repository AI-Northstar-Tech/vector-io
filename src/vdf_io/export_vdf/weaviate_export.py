import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, DISK_SPACE_LIMIT, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.weaviate_util import (
    collection_names,
    connect_weaviate,
    extract_distance,
    object_to_plain_dict,
)


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate instance")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--grpc_host", type=str, help="Weaviate gRPC host for custom deployments"
        )
        parser_weaviate.add_argument(
            "--grpc_port", type=int, help="Weaviate gRPC port", default=50051
        )
        parser_weaviate.add_argument(
            "--classes", type=str, help="Classes to export (comma-separated)"
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting data",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: http://localhost:8080): ",
            str,
            "http://localhost:8080",
        )
        set_arg_from_password(
            args,
            "api_key",
            "Enter the Weaviate API key (leave blank for local unauthenticated instances): ",
            "WEAVIATE_API_KEY",
        )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the batch size for exporting data (default: {DEFAULT_BATCH_SIZE}): ",
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

    # Connect to a WCS instance
    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate(self.args)

    def get_all_index_names(self):
        return collection_names(self.client)

    def get_index_names(self):
        self.all_classes = getattr(self, "all_classes", self.get_all_index_names())
        if self.args.get("classes") is None:
            return self.all_classes
        else:
            input_classes = self.args["classes"].split(",")
            if set(input_classes) - set(self.all_classes):
                tqdm.write(
                    f"These classes are not present in the Weaviate instance: {set(input_classes) - set(self.all_classes)}"
                )
            return [c for c in self.all_classes if c in input_classes]

    def get_data(self):
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        index_names = self.get_index_names()
        index_metas = {}
        for class_name in tqdm(index_names, desc="Exporting Weaviate collections"):
            collection = self.client.collections.get(class_name)
            vectors_directory = self.create_vec_dir(class_name)
            index_config = object_to_plain_dict(collection.config.get())
            distance = extract_distance(index_config)
            total = self._collection_count(collection)
            rows = []
            exported = 0
            dim = -1
            vector_columns = set()

            for item in tqdm(
                collection.iterator(include_vector=True),
                desc=f"Exporting {class_name}",
                total=total if total >= 0 else None,
            ):
                vectors = self._extract_vectors(getattr(item, "vector", None))
                if not vectors:
                    continue
                vector_columns.update(vectors.keys())
                if dim == -1:
                    dim = len(next(iter(vectors.values())))

                row = {
                    ID_COLUMN: str(getattr(item, "uuid", "")),
                    **(getattr(item, "properties", {}) or {}),
                    **vectors,
                }
                rows.append(row)
                if len(rows) >= batch_size or sys_getsizeof(rows) > DISK_SPACE_LIMIT:
                    exported += self._save_rows_to_parquet(rows, vectors_directory)
                    rows = []

            if rows:
                exported += self._save_rows_to_parquet(rows, vectors_directory)
            self.args["exported_count"] += exported
            index_metas[class_name] = [
                self.get_namespace_meta(
                    class_name,
                    vectors_directory,
                    total=total,
                    num_vectors_exported=exported,
                    dim=dim,
                    index_config=index_config,
                    vector_columns=sorted(vector_columns) or ["vector"],
                    distance=distance,
                )
            ]

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4, default=str)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def _collection_count(self, collection):
        try:
            return len(collection)
        except Exception:
            response = collection.aggregate.over_all(total_count=True)
            return response.total_count

    def _extract_vectors(self, vector_payload):
        if vector_payload is None:
            return {}
        if isinstance(vector_payload, dict):
            return {
                key: list(value)
                for key, value in vector_payload.items()
                if value is not None
            }
        return {"vector": list(vector_payload)}

    def _save_rows_to_parquet(self, rows, vectors_directory):
        parquet_file = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df = pd.DataFrame.from_records(rows)
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


def sys_getsizeof(value):
    import sys

    return sys.getsizeof(value)
