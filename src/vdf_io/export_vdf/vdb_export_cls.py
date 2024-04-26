from __future__ import annotations
import datetime
from typing import List
import pandas as pd
import os
import abc
import pyarrow.parquet as pq
import pyarrow as pa

from vdf_io.meta_types import NamespaceMeta, VDFMeta
from vdf_io.util import extract_data_hash, get_author_name, standardize_metric
from vdf_io.constants import ID_COLUMN


class ExportVDB(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "DB_NAME_SLUG"):
            raise TypeError(
                f"Class {cls.__name__} lacks required class variable 'DB_NAME_SLUG'"
            )

    def __init__(self, args):
        self.file_structure = []
        self.file_ctr = 1
        self.hash_value = extract_data_hash(args)
        self.args = args
        self.args["hash_value"] = self.hash_value
        self.args["exported_count"] = 0
        self.timestamp_in_format = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.vdf_directory = f"vdf_{self.timestamp_in_format}_{self.hash_value}"
        os.makedirs(self.vdf_directory, exist_ok=True)

    @abc.abstractmethod
    def get_index_names(self) -> List[str]:
        """
        Get index names from vector database
        """
        # raise NotImplementedError()
        pass

    @abc.abstractmethod
    def get_data(self) -> ExportVDB:
        """
        Get data from vector database
        """
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def make_parser(cls, subparsers):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def export_vdb(cls, args):
        raise NotImplementedError()

    def save_vectors_to_parquet(self, vectors, metadata, vectors_directory):
        vectors_df = pd.DataFrame(list(vectors.items()), columns=[ID_COLUMN, "vector"])

        if metadata:
            metadata_list = [{**{ID_COLUMN: k}, **v} for k, v in metadata.items()]
            metadata_df = pd.DataFrame.from_records(metadata_list)

            # Check for duplicate column names and rename as necessary
            common_columns = set(vectors_df.columns) & set(metadata_df.columns) - {
                ID_COLUMN
            }
            metadata_df.rename(
                columns={col: f"metadata_{col}" for col in common_columns}, inplace=True
            )

            df = vectors_df.merge(metadata_df, on=ID_COLUMN, how="outer")
        else:
            df = vectors_df

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

        vectors = {}
        metadata = {}
        return len(df)

    def create_vec_dir(self, index_name):
        vectors_directory = os.path.join(self.vdf_directory, index_name)
        os.makedirs(vectors_directory, exist_ok=True)
        return vectors_directory

    def get_basic_vdf_meta(self, index_metas):
        return VDFMeta(
            version=self.args["library_version"],
            file_structure=self.file_structure,
            author=get_author_name(),
            exported_from=self.DB_NAME_SLUG,
            indexes=index_metas,
            exported_at=datetime.datetime.now().astimezone().isoformat(),
        )

    def get_namespace_meta(
        self,
        index_name,
        vectors_directory,
        total,
        num_vectors_exported,
        dim,
        index_config=None,
        vector_columns=None,
        distance=None,
    ):
        vec_cols = ["vector"] if vector_columns is None else vector_columns
        model_name = self.args.get("model_name", "NOT_PROVIDED")
        namespace_meta = NamespaceMeta(
            index_name=index_name,
            namespace="",
            total_vector_count=total,
            exported_vector_count=num_vectors_exported,
            metric=standardize_metric(
                distance,
                self.DB_NAME_SLUG,
            ),
            dimensions=dim,
            model_name=model_name,
            vector_columns=vec_cols,
            model_map={
                vec_col: {
                    "model_name": model_name,
                    "text_column": "NOT_PROVIDED",
                    "dimensions": dim,
                    "vector_column": vec_col,
                }
                for vec_col in vec_cols
            },
            data_path="/".join(vectors_directory.split("/")[1:]),
            schema_dict_str=(
                self.parquet_schema.to_string()
                if hasattr(self, "parquet_schema")
                else None
            ),
            index_config=index_config,
        )

        return namespace_meta
