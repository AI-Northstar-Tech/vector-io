import argparse
import json
import os
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm
import weaviate

from vdf_io.constants import ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password

load_dotenv()

MAX_FETCH_SIZE = 1_000


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )
        parser_weaviate.add_argument(
            "-u",
            "--url",
            type=str,
            help="URL of Weaviate instance (e.g. http://localhost:8080 for local or WCS cluster URL for cloud)",
        )
        parser_weaviate.add_argument(
            "--api_key", type=str, help="Weaviate API key (for cloud instances)"
        )
        parser_weaviate.add_argument(
            "--local",
            type=bool,
            help="Connect to a local Weaviate instance (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--classes", type=str, help="Classes to export (comma-separated)"
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for fetching objects (default: 1000)",
            default=MAX_FETCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "local",
            "Connect to a local Weaviate instance? (default: False): ",
            bool,
            False,
        )
        if not args.get("local", False):
            set_arg_from_input(
                args,
                "url",
                "Enter the URL of Weaviate instance (WCS cluster URL for cloud): ",
                str,
            )
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key (hit enter to skip for local/anonymous instances): ",
                "WEAVIATE_API_KEY",
            )
        else:
            set_arg_from_input(
                args,
                "url",
                "Enter the URL of the local Weaviate instance (default: 'http://localhost:8080'): ",
                str,
                "http://localhost:8080",
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
        auth_credentials = None
        if self.args.get("api_key"):
            auth_credentials = weaviate.auth.AuthApiKey(self.args["api_key"])
        if self.args.get("local", False):
            # Connect to a local Weaviate instance
            url = self.args.get("url") or "http://localhost:8080"
            host = url.split("://")[-1].split(":")[0]
            port = int(url.split(":")[-1]) if ":" in url.split("://")[-1] else 8080
            self.client = weaviate.connect_to_local(
                host=host,
                port=port,
                auth_credentials=auth_credentials,
                skip_init_checks=True,
            )
        else:
            # Connect to a Weaviate Cloud (WCS) instance
            self.client = weaviate.connect_to_wcs(
                cluster_url=self.args["url"],
                auth_credentials=auth_credentials,
                skip_init_checks=True,
            )

    def get_all_index_names(self) -> List[str]:
        return list(self.client.collections.list_all().keys())

    def get_index_names(self) -> List[str]:
        if self.args.get("classes") is None:
            return self.all_classes
        input_classes = [c.strip() for c in self.args["classes"].split(",")]
        if set(input_classes) - set(self.all_classes):
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {set(input_classes) - set(self.all_classes)}"
            )
        return [c for c in self.all_classes if c in input_classes]

    def get_distance_metric(self, collection) -> str:
        try:
            config = collection.config.get()
            vector_configs = config.vector_config
            if vector_configs:
                first = next(iter(vector_configs.values()))
                return first.vector_index_config.distance_metric.value
            if config.vector_index_config is not None:
                return config.vector_index_config.distance_metric.value
        except Exception:
            pass
        return "cosine"

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for class_name in tqdm(index_names, desc="Fetching classes"):
            index_meta = self.get_data_for_class(class_name)
            index_metas[class_name] = index_meta

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def get_data_for_class(self, class_name) -> List[NamespaceMeta]:
        vectors_directory = self.create_vec_dir(class_name)
        collection = self.client.collections.get(class_name)

        total = collection.aggregate.over_all(total_count=True).total_count
        batch_size = self.args.get("batch_size") or MAX_FETCH_SIZE

        num_vectors_exported = 0
        dim = -1
        vector_columns = ["vector"]
        offset = 0
        pbar = tqdm(total=total, desc=f"Exporting {class_name}")
        while True:
            response = collection.query.fetch_objects(
                limit=batch_size, offset=offset, include_vector=True
            )
            objects = response.objects
            if not objects:
                break
            num_vectors_exported += self.save_from_objects(
                objects, vectors_directory
            )
            if dim == -1 or len(vector_columns) == 1:
                vector_columns, dim = self.infer_vector_columns(objects, dim)
            offset += len(objects)
            pbar.update(len(objects))
            if len(objects) < batch_size:
                break
        pbar.close()

        namespace_meta = self.get_namespace_meta(
            class_name,
            vectors_directory,
            total,
            num_vectors_exported,
            dim,
            index_config=collection.config.get().model_dump(mode="json"),
            vector_columns=vector_columns,
            distance=self.get_distance_metric(collection),
        )
        self.args["exported_count"] += num_vectors_exported
        return [namespace_meta]

    def infer_vector_columns(self, objects, dim):
        vector_columns = ["vector"]
        for obj in objects:
            vector = obj.vector
            if isinstance(vector, dict):
                vector_columns = list(vector.keys())
                first = next(iter(vector.values()))
                if first is not None:
                    dim = len(first)
                break
            elif vector is not None:
                dim = len(vector)
                break
        return vector_columns, dim

    def save_from_objects(self, objects, vectors_directory):
        rows = []
        for obj in objects:
            row = {ID_COLUMN: str(obj.uuid)}
            vector = obj.vector
            if isinstance(vector, dict):
                for vec_name, vec in vector.items():
                    row[vec_name] = list(vec) if vec is not None else None
            else:
                row["vector"] = list(vector) if vector is not None else None
            if obj.properties:
                for key, value in obj.properties.items():
                    if key not in row:
                        row[key] = value
            rows.append(row)
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
