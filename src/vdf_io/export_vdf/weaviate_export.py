import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input
from vdf_io.weaviate_util import (
    collection_names,
    connect_weaviate,
    first_vector_dimension,
    get_weaviate_distance,
    make_weaviate_parser,
    normalize_weaviate_vectors,
    serialize_weaviate_config,
)


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )

        make_weaviate_parser(parser_weaviate)
        parser_weaviate.add_argument(
            "--classes",
            type=str,
            help="Collections/classes to export (comma-separated)",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting data",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = weaviate_export.get_all_index_names()
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the name of the collections/classes to export (comma-separated, all will be exported by default): ",
            str,
            choices=weaviate_export.all_classes,
        )
        set_arg_from_input(
            weaviate_export.args,
            "batch_size",
            f"Enter the batch size for exporting data (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate(self.args)

    def get_all_index_names(self):
        return collection_names(self.client)

    def get_index_names(self):
        if self.args.get("classes") is None:
            return self.get_all_index_names()
        else:
            input_classes = self.args["classes"].split(",")
            all_classes = self.get_all_index_names()
            if set(input_classes) - set(all_classes):
                tqdm.write(
                    f"These collections/classes are not present in the Weaviate instance: {set(input_classes) - set(all_classes)}"
                )
            return [c for c in all_classes if c in input_classes]

    def get_data(self):
        index_metas = {}
        index_names = self.get_index_names()
        for class_name in tqdm(index_names, desc="Exporting collections"):
            rows = []
            total_exported = 0
            dimensions = -1
            vector_columns = []
            collection = self.client.collections.get(class_name)
            collection_config = collection.config.get()
            response = collection.aggregate.over_all(total_count=True)
            total = response.total_count or 0
            vectors_directory = self.create_vec_dir(class_name)
            batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE

            for item in tqdm(
                collection.iterator(include_vector=True, cache_size=batch_size),
                desc=f"Exporting {class_name}",
                total=total,
            ):
                item_vectors = normalize_weaviate_vectors(item.vector)
                if not item_vectors:
                    continue
                if not vector_columns:
                    vector_columns = list(item_vectors.keys())
                    dimensions = first_vector_dimension(item_vectors[vector_columns[0]])

                row = {ID_COLUMN: str(item.uuid)}
                row.update(item_vectors)
                row.update(dict(item.properties or {}))
                rows.append(row)

                if len(rows) >= batch_size:
                    total_exported += self.save_rows_to_parquet(
                        rows, vectors_directory
                    )
                    rows = []

            if rows:
                total_exported += self.save_rows_to_parquet(rows, vectors_directory)

            if not vector_columns:
                vector_columns = ["vector"]

            namespace_meta = self.get_namespace_meta(
                class_name,
                vectors_directory,
                total=total,
                num_vectors_exported=total_exported,
                dim=dimensions,
                index_config=serialize_weaviate_config(collection_config),
                vector_columns=vector_columns,
                distance=get_weaviate_distance(collection_config, vector_columns[0]),
            )
            index_metas[class_name] = [namespace_meta]
            self.args["exported_count"] += total_exported

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def save_rows_to_parquet(self, rows, vectors_directory):
        if not rows:
            return 0
        df = pd.DataFrame.from_records(rows)
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
