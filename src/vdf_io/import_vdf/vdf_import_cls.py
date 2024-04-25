import ast
import datetime
from functools import lru_cache
import json
import os
import numpy as np
from packaging.version import Version
import abc
from tqdm import tqdm
from halo import Halo

from qdrant_client.http.models import Distance

import vdf_io
from vdf_io.constants import ID_COLUMN
from vdf_io.meta_types import NamespaceMeta, VDFMeta
from vdf_io.util import (
    expand_shorthand_path,
    extract_data_hash,
    get_final_data_path,
    get_parquet_files,
    read_parquet_progress,
)


class ImportVDB(abc.ABC):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not hasattr(cls, "DB_NAME_SLUG"):
            raise TypeError(
                f"Class {cls.__name__} lacks required class variable 'DB_NAME_SLUG'"
            )

    def __init__(self, args):
        self.args = args
        self.args["hash_value"] = extract_data_hash(args)
        self.hash_value = extract_data_hash(args)
        self.temp_file_paths = []
        self.abnormal_vector_format = False
        if self.args.get("hf_dataset", None) is None:
            self.args["dir"] = expand_shorthand_path(self.args["dir"])
            if not os.path.isdir(self.args["dir"]):
                raise Exception("Invalid dir path")
            if not os.path.isfile(os.path.join(self.args["dir"], "VDF_META.json")):
                raise Exception("Invalid dir path, VDF_META.json not found")
            # Check if the VDF_META.json file exists
            vdf_meta_path = os.path.join(self.args["dir"], "VDF_META.json")
            if not os.path.isfile(vdf_meta_path):
                raise Exception("VDF_META.json not found in the specified directory")
            with open(vdf_meta_path) as f:
                self.vdf_meta = json.load(f)
        else:
            hf_dataset = self.args.get("hf_dataset", None)
            index_name = hf_dataset.split("/")[-1]
            from huggingface_hub import HfFileSystem

            fs = HfFileSystem()
            with Halo(text="Checking for VDF_META.json", spinner="dots"):
                hf_files = fs.ls(f"datasets/{hf_dataset}", detail=False)
            if f"datasets/{hf_dataset}/VDF_META.json" in hf_files:
                print(f"Found VDF_META.json in {hf_dataset} on HuggingFace Hub")
                self.vdf_meta = json.loads(
                    fs.read_text(f"datasets/{hf_dataset}/VDF_META.json")
                )
            else:
                self.vdf_meta = VDFMeta(
                    version=vdf_io.__version__,
                    file_structure=[],
                    author=hf_dataset.split("/")[0],
                    exported_from="hf",
                    exported_at=datetime.datetime.now().astimezone().isoformat(),
                    id_column=self.args.get("id_column", ID_COLUMN) or ID_COLUMN,
                    indexes={
                        index_name: [
                            NamespaceMeta(
                                namespace="",
                                index_name=index_name,
                                total_vector_count=self.args.get("max_num_rows"),
                                exported_vector_count=self.args.get("max_num_rows"),
                                dimensions=self.args.get("vector_dim", -1),
                                model_name=self.args.get("model_name", ""),
                                vector_columns=self.args.get(
                                    "vector_columns", "vector"
                                ).split(","),
                                data_path=".",
                                metric=self.args.get("metric", Distance.COSINE),
                            )
                        ]
                    },
                ).model_dump()
            print(json.dumps(self.vdf_meta, indent=4))
        self.id_column = self.vdf_meta.get("id_column", ID_COLUMN) or ID_COLUMN
        if "indexes" not in self.vdf_meta:
            raise Exception("Invalid VDF_META.json, 'indexes' key not found")
        if "version" not in self.vdf_meta:
            print("Warning: 'version' key not found in VDF_META.json")
        elif "library_version" not in self.args:
            print(
                "Warning: 'library_version' not found in args. Skipping version check."
            )
        elif Version(self.vdf_meta["version"]) > Version(self.args["library_version"]):
            print(
                f"Warning: The version of vector-io library: ({self.args['library_version']}) is behind the version of the vdf directory: ({self.vdf_meta['version']})."
            )
            print(
                "Please upgrade the vector-io library to the latest version to ensure compatibility."
            )
        print("ImportVDB initialized successfully.")

    @abc.abstractmethod
    def upsert_data():
        """
        Get data from vector database
        """
        raise NotImplementedError

    def get_vector_column_name(
        self, index_name, namespace_meta, multi_vector_supported=False
    ):
        if "vector_columns" not in namespace_meta:
            print(
                "vector_columns not found in namespace metadata. Using 'vector' as the vector column name."
            )
            vector_column_name = "vector"
            vector_column_names = [vector_column_name]
        else:
            vector_column_names = namespace_meta["vector_columns"]
            vector_column_name = vector_column_names[0]
            if len(vector_column_names) > 1 and not multi_vector_supported:
                tqdm.write(
                    f"Warning: More than one vector column found for index {index_name}."
                    f" Only the first vector column {vector_column_name} will be imported."
                )
        return vector_column_names, vector_column_name

    @lru_cache(maxsize=1)
    def get_parquet_files(self, data_path):
        return get_parquet_files(
            data_path, self.args, self.temp_file_paths, self.id_column
        )

    def get_final_data_path(self, data_path):
        return get_final_data_path(
            self.args["cwd"], self.args["dir"], data_path, self.args
        )

    def get_file_path(self, final_data_path, parquet_file):
        if self.args.get("hf_dataset", None):
            return parquet_file
        return os.path.join(final_data_path, parquet_file)

    def set_dims(self, namespace_meta, index_name):
        if namespace_meta["dimensions"] == -1:
            tqdm.write(f"Resolving dimensions for index '{index_name}'")
            dims = self.resolve_dims(namespace_meta, index_name)
            if dims != -1:
                namespace_meta["dimensions"] = dims
                tqdm.write(f"Resolved dimensions: {dims}")
            else:
                tqdm.write(f"Failed to resolve dimensions for index '{index_name}'")
                raise ValueError(
                    f"Failed to resolve dimensions for index '{index_name}'"
                )

    def resolve_dims(self, namespace_meta, index_name):
        final_data_path = self.get_final_data_path(namespace_meta["data_path"])
        parquet_files = self.get_parquet_files(final_data_path)
        _, vector_column_name = self.get_vector_column_name(index_name, namespace_meta)
        dims = -1
        for file in tqdm(parquet_files, desc="Iterating parquet files"):
            file_path = self.get_file_path(final_data_path, file)
            df = self.read_parquet_progress(file_path, columns=[vector_column_name])
            i = 0
            while i < len(df[vector_column_name]):
                first_el = df[vector_column_name].iloc[i]
                if first_el is None:
                    i += 1
                    continue
                dims = len(self.extract_vector(first_el))
                break
        return dims

    def extract_vector(self, v):
        ret_v = None
        if isinstance(v, list) and len(v) > 1:
            ret_v = v
        elif isinstance(v, np.ndarray):
            if v.shape[0] > 1:
                self.abnormal_vector_format = True
                ret_v = v.tolist()
            if v.shape[0] == 1:
                self.abnormal_vector_format = True
                ret_v = v[0].tolist()
        elif isinstance(v, bytes):
            self.abnormal_vector_format = True
            tqdm.write("Warning: Vector is in bytes format. Converting to list")
            ret_v = ast.literal_eval(v.decode("utf-8"))
        elif isinstance(v, str):
            self.abnormal_vector_format = True
            ret_v = ast.literal_eval(v)
        else:
            ret_v = v
        # convert each element to float
        if ret_v is not None:
            ret_v = [float(x) for x in ret_v]
        return ret_v

    def update_metadata(self, metadata, vector_column_names, df):
        metadata.update(
            {
                row[self.id_column]: {
                    key: value
                    for key, value in row.items()
                    if key not in vector_column_names
                }
                for _, row in df.iterrows()
                if self.id_column in row
            }
        )

    def update_vectors(self, vectors, vector_column_name, df):
        for _, row in tqdm(df.iterrows(), desc="Extracting vectors", total=len(df)):
            if self.id_column in row:
                vectors[row[self.id_column]] = self.extract_vector(
                    row[vector_column_name]
                )

    def read_parquet_progress(self, file_path, **kwargs):
        return read_parquet_progress(file_path, self.id_column, **kwargs)

    def create_new_name(self, index_name, indexes, delimiter="-"):
        if not self.args.get("create_new", False):
            return index_name

        # Original name to use as the base for appending suffixes
        og_name = index_name

        # Find all indexes that start with the original name followed by a hyphen
        suffixes = [
            name[len(og_name) + 1 :]
            for name in indexes
            if name.startswith(og_name + delimiter)
        ]

        # Convert suffixes to integers where possible
        suffixes = [int(suffix) for suffix in suffixes if suffix.isdigit()]

        # Determine the next suffix to use
        suffix = max(suffixes) + 1 if suffixes else 2

        # Generate new names until a unique one is found
        while True:
            new_name = og_name + f"{delimiter}{suffix}"
            if new_name not in indexes:
                return new_name
            suffix += 1

    # destructor
    def cleanup(self):
        for temp_file_path in self.temp_file_paths:
            if os.path.isfile(temp_file_path):
                os.remove(temp_file_path)
