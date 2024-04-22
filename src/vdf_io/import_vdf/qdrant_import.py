import json
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm
from grpc import RpcError
from typing import Any, Dict, List
from PIL import Image
from halo import Halo

import concurrent.futures

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import VectorParams, Distance, PointStruct

from vdf_io.constants import INT_MAX
from vdf_io.names import DBNames
from vdf_io.util import (
    expand_shorthand_path,
    get_qdrant_id_from_id,
    set_arg_from_input,
    set_arg_from_password,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta

load_dotenv()


class ImportQdrant(ImportVDB):
    DB_NAME_SLUG = DBNames.QDRANT

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Qdrant
        """
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Qdrant instance (default: 'http://localhost:6334'): ",
            str,
            "http://localhost:6334",
        )
        set_arg_from_input(
            args,
            "prefer_grpc",
            "Whether to use GRPC. Recommended. (default: True): ",
            bool,
            True,
        )
        set_arg_from_input(
            args,
            "qdrant_local_persist_path",
            "Enter the path to the local persist directory (default: None): ",
            str,
            "DO_NOT_PROMPT",
        )
        set_arg_from_input(
            args,
            "parallel",
            "Enter the number of parallel processes of upload (default: 64): ",
            int,
            1,
        )
        set_arg_from_input(
            args,
            "max_retries",
            "Enter the maximum number of retries in case of a failure (default: 3): ",
            int,
            3,
        )
        set_arg_from_password(
            args, "qdrant_api_key", "Enter your Qdrant API key: ", "QDRANT_API_KEY"
        )
        qdrant_import = ImportQdrant(args)
        try:
            qdrant_import.upsert_data()
        # keyboard interrupt
        except KeyboardInterrupt:
            tqdm.write(
                f"Data import interrupted. {qdrant_import.total_imported_count} rows imported."
            )
        return qdrant_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_qdrant = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Qdrant"
        )
        parser_qdrant.add_argument(
            "-u",
            "--url",
            type=str,
            help="Qdrant instance url",
            default="http://localhost:6334",
        )
        parser_qdrant.add_argument(
            "--prefer_grpc",
            type=bool,
            help="Whether to use Qdrant's GRPC interface",
            default=True,
        )
        parser_qdrant.add_argument(
            "--qdrant_local_persist_path",
            type=str,
            help="Path to the local persist directory (default: None)",
            default=None,
        )
        parser_qdrant.add_argument(
            "--parallel",
            type=int,
            help="Number of parallel processes of upload (default: 5).",
            default=5,
        )
        parser_qdrant.add_argument(
            "--max_retries",
            type=int,
            help="Maximum number of retries in case of a failure (default: 3).",
            default=3,
        )
        parser_qdrant.add_argument(
            "--shard_key_selector",
            type=Any,
            help="Shard to be queried (default: None)",
            default=None,
        )

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        url, api_key, prefer_grpc, path = (
            self.args.get("url", None),
            self.args.get("qdrant_api_key", None),
            self.args.get("prefer_grpc", True),
            expand_shorthand_path(self.args.get("qdrant_local_persist_path", None)),
        )
        if path:
            url = None
        if url:
            path = None
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            prefer_grpc=prefer_grpc,
            path=path,
        )

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        # we know that the self.vdf_meta["indexes"] is a list
        index_meta: Dict[str, List[NamespaceMeta]] = {}
        for index_name, index_meta in tqdm(
            self.vdf_meta["indexes"].items(), desc="Importing indexes"
        ):
            tqdm.write(f"Importing data for index '{index_name}'")
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                # list indexes
                collections = [
                    x.name for x in self.client.get_collections().collections
                ]
                # check if index exists
                new_collection_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta["namespace"]
                    else ""
                )
                new_collection_name = self.create_new_name(
                    new_collection_name, collections
                )
                vector_column_names, _ = self.get_vector_column_name(
                    new_collection_name, namespace_meta, multi_vector_supported=True
                )
                if new_collection_name not in collections:
                    # create index
                    try:

                        def get_nested_config(config, keys, default=None):
                            """Helper function to get nested dictionary values."""
                            if not config:
                                return default
                            for key in keys:
                                if not config:
                                    return default
                                config = config.get(key, {}) or {}
                            if not config:
                                return default
                            return config or default

                        index_config = namespace_meta.get("index_config", {})
                        dims = (
                            namespace_meta["dimensions"]
                            if "dimensions" in namespace_meta
                            else get_nested_config(
                                namespace_meta,
                                ["index_config", "params", "vectors"],
                                {},
                            ).get("size")
                        )
                        on_disk = get_nested_config(
                            namespace_meta,
                            ["index_config", "params", "vectors", "on_disk"],
                            None,
                        )
                        configs = [
                            "hnsw_config",
                            "optimizers_config",
                            "wal_config",
                            "quantization_config",
                            "on_disk_payload",
                            "sparse_vectors_config",
                        ]
                        (
                            hnsw_config,
                            optimizers_config,
                            wal_config,
                            quantization_config,
                            on_disk_payload,
                            sparse_vectors_config,
                        ) = [
                            get_nested_config(index_config, [config], None)
                            for config in configs
                        ]
                        distance = (
                            namespace_meta.get("metric", Distance.COSINE)
                            or Distance.COSINE
                        )
                        vectors_config = {
                            vector_column_name: VectorParams(
                                size=dims,
                                distance=distance,
                                on_disk=on_disk,
                            )
                            for vector_column_name in vector_column_names
                        }
                        self.client.create_collection(
                            collection_name=new_collection_name,
                            vectors_config=vectors_config,
                            sparse_vectors_config=sparse_vectors_config,
                            hnsw_config=hnsw_config,
                            optimizers_config=optimizers_config,
                            wal_config=wal_config,
                            quantization_config=quantization_config,
                            on_disk_payload=on_disk_payload,
                        )

                    except Exception as e:
                        tqdm.write(
                            f"Failed to create index '{new_collection_name}' {e}"
                        )
                        return
                prev_vector_count = self.client.get_collection(
                    collection_name=new_collection_name
                ).vectors_count
                if prev_vector_count > 0:
                    tqdm.write(
                        f"Index '{new_collection_name}' has {prev_vector_count} vectors before import"
                    )
                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(final_data_path)

                vectors_all = {}
                for vec_col in namespace_meta.get("vector_columns", []):
                    vectors_all[vec_col] = {}
                metadata = {}
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(
                        file_path,
                        max_num_rows=(
                            (self.args.get("max_num_rows") or INT_MAX)
                            - self.total_imported_count
                        ),
                    )
                    with Halo(text="Processing vectors", spinner="dots"):
                        for vec_col in namespace_meta.get("vector_columns", []):
                            self.update_vectors(vectors_all[vec_col], vec_col, df)
                    with Halo(text="Processing metadata", spinner="dots"):
                        self.update_metadata(metadata, vector_column_names, df)
                    self.make_metadata_qdrant_compliant(metadata)
                    # union of all keys in vectors_all
                    keys = set().union(
                        *[vectors_all[vec_col].keys() for vec_col in vectors_all.keys()]
                    )
                    points = [
                        PointStruct(
                            id=get_qdrant_id_from_id(idx),
                            vector={
                                vec_col: vectors_all[vec_col].get(idx, [])
                                for vec_col in vectors_all.keys()
                            },
                            payload=metadata.get(idx, {}),
                        )
                        for idx in keys
                    ]

                    if self.total_imported_count + len(points) >= (
                        self.args.get("max_num_rows") or INT_MAX
                    ):
                        max_hit = True
                        points = points[
                            : (self.args.get("max_num_rows") or INT_MAX)
                            - self.total_imported_count
                        ]
                        tqdm.write("Truncating data to limit to max rows")
                    try:
                        BATCH_SIZE = self.args.get("batch_size", 64) or 64
                        batches = list(divide_into_batches(points, BATCH_SIZE))
                        total_points = len(points)

                        num_parallel_threads = self.args.get("parallel", 5) or 5
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=num_parallel_threads
                        ) as executor, tqdm(
                            total=total_points,
                            desc=f"Uploading points in batches of {BATCH_SIZE} in {num_parallel_threads} threads",
                        ) as pbar:
                            # Create a future to batch mapping to update progress bar correctly after each batch completion
                            future_to_batch = {
                                executor.submit(
                                    self.upsert_batch, batch, new_collection_name
                                ): batch
                                for batch in batches
                            }

                            for future in concurrent.futures.as_completed(
                                future_to_batch
                            ):
                                batch = future_to_batch[future]
                                try:
                                    # Attempt to get the result, which will re-raise any exceptions
                                    future.result()
                                    # Update the progress bar by the size of the successfully processed batch
                                    pbar.update(len(batch))
                                except Exception as e:
                                    tqdm.write(
                                        f"Batch upsert failed with error: {e} "  # {batch}
                                    )
                                    # Optionally, you might want to handle failed batches differently
                        self.total_imported_count += len(points)
                        if self.total_imported_count >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            max_hit = True
                    except (UnexpectedResponse, RpcError, ValueError) as e:
                        tqdm.write(
                            f"Failed to upsert data for collection '{new_collection_name}', {e}"
                        )
                        continue
                    vector_count = self.client.get_collection(
                        collection_name=new_collection_name
                    ).vectors_count
                    if max_hit:
                        break
                    # END parquet file loop
                tqdm.write(
                    f"Index '{new_collection_name}' has {vector_count} vectors after import"
                )
                tqdm.write(f"{vector_count - prev_vector_count} vectors were imported")
                if max_hit:
                    break
                # END namespace loop
            if max_hit:
                tqdm.write(
                    f"Max rows to be imported {self.args['max_num_rows']} hit. Exiting"
                )
                break
            # END index loop
        tqdm.write("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count

    def make_metadata_qdrant_compliant(self, metadata):
        deleted_images = False
        parsed_json = False
        for k, v in metadata.items():
            deleted_images, parsed_json, zeroed_nan = self.normalize_dict(
                metadata, k, v
            )
        if deleted_images:
            tqdm.write("Images were deleted from metadata")
        if parsed_json:
            tqdm.write("Metadata was parsed to JSON")
        if zeroed_nan:
            tqdm.write("NaN values were replaced with 0 in metadata")

    def replace_nan_with_zero(self, data, zeroed_nan=False):
        if isinstance(data, dict):
            ret_val = {k: self.replace_nan_with_zero(v) for k, v in data.items()}
            for _, v in ret_val.items():
                if v[1]:
                    zeroed_nan = True
            return {k: v[0] for k, v in ret_val.items()}, zeroed_nan
        elif isinstance(data, list):
            ret_val = [self.replace_nan_with_zero(item) for item in data]
            return [x[0] for x in ret_val], any(x[1] for x in ret_val)
        elif isinstance(data, float) and np.isnan(data):
            return 0, True
        else:
            return data, False

    def normalize_dict(self, metadata, k, v):
        deleted_images = False
        parsed_json = False
        zeroed_nan = False
        # Check for np.nan and convert to 0 for scalar values
        if np.isscalar(v) and (
            (isinstance(v, (float, int)) and np.isnan(v))
            or (isinstance(v, str) and v.lower() == "nan")
        ):
            metadata[k] = 0
            zeroed_nan = True
        elif isinstance(v, np.ndarray):
            metadata[k] = v.tolist()
        elif isinstance(v, Image.Image):
            del metadata[k]
            deleted_images = True
        elif isinstance(v, bytes) or isinstance(v, str):
            if isinstance(v, bytes):
                metadata[k] = v.decode("utf-8")
            try:
                parsed_value = json.loads(metadata[k])
                # Replace nan with 0 in the parsed JSON object
                metadata[k], zeroed_nan_rec = self.replace_nan_with_zero(parsed_value)
                if zeroed_nan_rec:
                    zeroed_nan = True
                parsed_json = True
            except json.JSONDecodeError:
                pass
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                (
                    deleted_images_rec,
                    parsed_json_rec,
                    zeroed_nan_rec,
                ) = self.normalize_dict(v, k2, v2)
                if zeroed_nan_rec:
                    zeroed_nan = True
                if deleted_images_rec:
                    deleted_images = True
                if parsed_json_rec:
                    parsed_json = True
        return deleted_images, parsed_json, zeroed_nan

    def upsert_batch(self, batch, new_collection_name):
        RETRIES = self.args.get("max_retries", 3)
        for attempt in range(RETRIES):
            try:
                self.client.upsert(
                    collection_name=new_collection_name,
                    points=batch,
                    shard_key_selector=self.args.get("shard_key_selector", None),
                    wait=True,
                )
                break  # Break the loop on success
            except Exception:
                if attempt == RETRIES - 1:
                    raise  # Re-raise the last exception if all retries fail
                else:
                    continue
        return len(batch)


# Function to divide your points into batches
def divide_into_batches(points, batch_size):
    for i in range(0, len(points), batch_size):
        yield points[i : i + batch_size]
