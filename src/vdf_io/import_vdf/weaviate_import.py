from __future__ import annotations

import json
from typing import Dict, List
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.weaviate_util import (
    build_vector_config,
    clean_property_value,
    collection_names,
    connect_weaviate,
)


load_dotenv()


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import VDF data to Weaviate"
        )
        parser_weaviate.add_argument(
            "--url",
            type=str,
            help="Weaviate URL (default: http://localhost:8080)",
            default="http://localhost:8080",
        )
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--grpc_host", type=str, help="Weaviate gRPC host for custom deployments"
        )
        parser_weaviate.add_argument(
            "--grpc_port", type=int, help="Weaviate gRPC port", default=50051
        )

    @classmethod
    def import_vdb(cls, args):
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
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate(self.args)

    def get_all_index_names(self):
        return collection_names(self.client)

    def upsert_data(self):
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        collections = self.get_all_index_names()
        max_num_rows = self.args.get("max_num_rows") or INT_MAX
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                new_collection_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_collection_name = self.create_new_name(
                    new_collection_name, collections
                )
                vector_column_names, _ = self.get_vector_column_name(
                    index_name, namespace_meta, multi_vector_supported=True
                )

                collection = None
                property_types = {}
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    remaining = max_num_rows - self.total_imported_count
                    if remaining <= 0:
                        break
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(
                        file_path,
                        max_num_rows=remaining,
                    )
                    df = cleanup_df(df)

                    if collection is None:
                        collection, property_types = self._get_or_create_collection(
                            new_collection_name,
                            collections,
                            namespace_meta,
                            vector_column_names,
                            df,
                        )
                        if new_collection_name not in collections:
                            collections.append(new_collection_name)

                    batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                    for batch_df in tqdm(
                        divide_into_batches(df, batch_size),
                        desc="Importing batches",
                        total=max(len(df) // batch_size, 1),
                    ):
                        self._insert_batch(
                            collection,
                            new_collection_name,
                            batch_df,
                            vector_column_names,
                            property_types,
                            batch_size,
                        )
                        if self.total_imported_count >= max_num_rows:
                            break

                if collection is not None:
                    tqdm.write(
                        f"Imported {self.total_imported_count} rows into {new_collection_name}"
                    )
        print("Data imported successfully")

    def _get_or_create_collection(
        self,
        collection_name,
        collections,
        namespace_meta,
        vector_column_names,
        df,
    ):
        property_types = self._property_types(df, vector_column_names)
        if collection_name not in collections:
            kwargs = build_vector_config(
                vector_column_names,
                namespace_meta.get("metric"),
            )
            properties = self._properties(property_types)
            try:
                self.client.collections.create(
                    name=collection_name,
                    properties=properties,
                    **kwargs,
                )
            except TypeError:
                self.client.collections.create(
                    collection_name,
                    properties=properties,
                    **kwargs,
                )
        return self.client.collections.get(collection_name), property_types

    def _insert_batch(
        self,
        collection,
        collection_name,
        batch_df,
        vector_column_names,
        property_types,
        batch_size,
    ):
        inserted = 0
        skipped = 0
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            for _, row in batch_df.iterrows():
                vectors = {
                    vector_column_name: self.extract_vector(row[vector_column_name])
                    for vector_column_name in vector_column_names
                    if vector_column_name in row
                    and not self._is_empty(row[vector_column_name])
                }
                vectors = {k: v for k, v in vectors.items() if v}
                if not vectors:
                    skipped += 1
                    continue

                row_id = row.get(self.id_column)
                properties = self._row_properties(
                    row,
                    vector_column_names,
                    property_types,
                    row_id,
                )
                vector = (
                    next(iter(vectors.values()))
                    if len(vector_column_names) == 1
                    else vectors
                )
                batch.add_object(
                    properties=properties,
                    uuid=self._object_uuid(collection_name, row_id),
                    vector=vector,
                )
                inserted += 1

        failed_objects = getattr(collection.batch, "failed_objects", None)
        if failed_objects:
            tqdm.write(f"Warning: {len(failed_objects)} Weaviate objects failed import")
            tqdm.write(f"First failed object: {failed_objects[0]}")
        if skipped:
            tqdm.write(f"Skipped {skipped} rows with no usable vector")
        self.total_imported_count += inserted

    def _property_types(self, df, vector_column_names):
        property_types = {}
        for column in df.columns:
            if column in vector_column_names or column == self.id_column:
                continue
            property_types[column] = self._data_type_for_column(df[column])
        property_types["vdf_original_id"] = self._data_type("text")
        return property_types

    def _properties(self, property_types):
        from weaviate.classes.config import Property

        return [
            Property(name=column, data_type=data_type)
            for column, data_type in property_types.items()
        ]

    def _row_properties(self, row, vector_column_names, property_types, row_id):
        properties = {}
        for column, data_type in property_types.items():
            if column == "vdf_original_id":
                continue
            if column in vector_column_names or column == self.id_column:
                continue
            value = clean_property_value(row.get(column))
            if value is not None:
                properties[column] = self._coerce_property(value, data_type)
        properties["vdf_original_id"] = str(row_id)
        return properties

    def _data_type_for_column(self, series):
        series = series.dropna()
        if len(series) == 0:
            return self._data_type("text")
        sample = clean_property_value(series.iloc[0])
        if isinstance(sample, bool):
            return self._data_type("bool")
        if isinstance(sample, int) and not isinstance(sample, bool):
            return self._data_type("int")
        if isinstance(sample, float):
            return self._data_type("number")
        if isinstance(sample, list) and sample:
            item = sample[0]
            if isinstance(item, bool):
                return self._data_type("bool_array")
            if isinstance(item, int) and not isinstance(item, bool):
                return self._data_type("int_array")
            if isinstance(item, float):
                return self._data_type("number_array")
            if isinstance(item, str):
                return self._data_type("text_array")
        return self._data_type("text")

    def _data_type(self, name):
        from weaviate.classes.config import DataType

        return {
            "text": DataType.TEXT,
            "int": DataType.INT,
            "number": DataType.NUMBER,
            "bool": DataType.BOOL,
            "text_array": DataType.TEXT_ARRAY,
            "int_array": DataType.INT_ARRAY,
            "number_array": DataType.NUMBER_ARRAY,
            "bool_array": DataType.BOOL_ARRAY,
        }[name]

    def _coerce_property(self, value, data_type):
        type_name = getattr(data_type, "value", str(data_type)).lower()
        if "array" in type_name and isinstance(value, list):
            return value
        if "text" in type_name and not isinstance(value, str):
            return json.dumps(value, default=str)
        return value

    def _object_uuid(self, collection_name, row_id):
        try:
            return str(UUID(str(row_id)))
        except (TypeError, ValueError):
            return str(uuid5(NAMESPACE_URL, f"vdf_io:{collection_name}:{row_id}"))

    def _is_empty(self, value):
        if value is None:
            return True
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value) == 0
        return bool(pd.isna(value))
