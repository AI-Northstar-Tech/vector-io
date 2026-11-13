import argparse
import re
import uuid as uuid_lib
from typing import Any, List

from dotenv import load_dotenv
from tqdm import tqdm
import weaviate
from weaviate.classes.config import Configure, DataType
from weaviate.classes.data import DataObject

from vdf_io.constants import INT_MAX
from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB

load_dotenv()

DEFAULT_BATCH_SIZE = 500


def weaviate_distance_metric(metric_name: str):
    distance_map = {
        "cosine": Configure.VectorIndex.Hnsw.Distance.COSINE,
        "dot": Configure.VectorIndex.Hnsw.Distance.DOT,
        "l2-squared": Configure.VectorIndex.Hnsw.Distance.L2_SQUARED,
        "manhattan": Configure.VectorIndex.Hnsw.Distance.MANHATTAN,
        "hamming": Configure.VectorIndex.Hnsw.Distance.HAMMING,
    }
    return distance_map.get(metric_name, Configure.VectorIndex.Hnsw.Distance.COSINE)


def sanitize_class_name(name: str) -> str:
    """
    Weaviate class names must start with an uppercase letter and contain
    only alphanumeric characters and underscores.
    """
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not name or not name[0].isalpha():
        name = "C" + name
    return name[0].upper() + name[1:]


def infer_data_type(value: Any) -> DataType:
    if isinstance(value, bool):
        return DataType.BOOL
    if isinstance(value, int):
        return DataType.INT
    if isinstance(value, float):
        return DataType.NUMBER
    if isinstance(value, list):
        return DataType.TEXT_ARRAY
    return DataType.TEXT


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Weaviate
        """
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
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the batch size for import (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )
        weaviate_import = ImportWeaviate(args)
        try:
            weaviate_import.upsert_data()
        except KeyboardInterrupt:
            tqdm.write(
                f"Data import interrupted. {weaviate_import.total_imported_count} rows imported."
            )
        return weaviate_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
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
            "--batch_size",
            type=int,
            help=f"Batch size for import (default: {DEFAULT_BATCH_SIZE})",
            default=DEFAULT_BATCH_SIZE,
        )

    def __init__(self, args):
        super().__init__(args)
        auth_credentials = None
        if self.args.get("api_key"):
            auth_credentials = weaviate.auth.AuthApiKey(self.args["api_key"])
        if self.args.get("local", False):
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
            self.client = weaviate.connect_to_wcs(
                cluster_url=self.args["url"],
                auth_credentials=auth_credentials,
                skip_init_checks=True,
            )
        self.total_imported_count = 0

    def get_or_create_collection(self, class_name, namespace_meta, vector_column_names):
        class_name = sanitize_class_name(class_name)
        metric = standardize_metric_reverse(
            namespace_meta.get("metric"), self.DB_NAME_SLUG
        )
        distance = weaviate_distance_metric(metric)
        if self.client.collections.exists(class_name):
            return self.client.collections.get(class_name)

        # Automatically create the Weaviate class (collection) for the index
        vector_index_config = Configure.VectorIndex.hnsw(distance_metric=distance)
        if len(vector_column_names) > 1:
            vector_config = Configure.NamedVectors.from_dict(
                {name: vector_index_config for name in vector_column_names}
            )
            self.client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_config=vector_config,
            )
        else:
            self.client.collections.create(
                name=class_name,
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=vector_index_config,
            )
        return self.client.collections.get(class_name)

    def build_data_object(self, row, vector_column_names):
        properties = {}
        vector = {}
        for key, value in row.items():
            if key == self.id_column:
                continue
            if key in vector_column_names:
                vector[key] = self.extract_vector(value)
            else:
                properties[key] = value
        if len(vector_column_names) == 1:
            vector = vector.get(vector_column_names[0])
        try:
            obj_uuid = uuid_lib.UUID(str(row[self.id_column]))
        except (ValueError, AttributeError, TypeError):
            obj_uuid = uuid_lib.uuid5(uuid_lib.NAMESPACE_DNS, str(row[self.id_column]))
        return DataObject(properties=properties, vector=vector, uuid=obj_uuid)

    def upsert_batch(self, collection, batch):
        collection.data.insert_many(batch)
        return len(batch)

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        existing_classes = list(self.client.collections.list_all().keys())
        for index_name, index_meta in self.vdf_meta["indexes"].items():
            print(f'Importing data for class "{index_name}"')
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                class_name = self.create_new_name(index_name, existing_classes)
                vector_column_names, _ = self.get_vector_column_name(
                    index_name, namespace_meta, multi_vector_supported=True
                )
                collection = self.get_or_create_collection(
                    class_name, namespace_meta, vector_column_names
                )

                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                BATCH_SIZE = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                current_batch_size = BATCH_SIZE
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(file_path)
                    data_objects: List[DataObject] = []
                    for _, row in df.iterrows():
                        if self.total_imported_count + len(data_objects) >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            max_hit = True
                            break
                        data_objects.append(
                            self.build_data_object(row, vector_column_names)
                        )
                    tqdm.write(
                        f"Inserting {len(data_objects)} objects in batches of {BATCH_SIZE}"
                    )
                    i = 0
                    pbar = tqdm(
                        total=len(data_objects), desc="Upserting data in batches"
                    )
                    while i < len(data_objects):
                        batch = data_objects[i : i + current_batch_size]
                        try:
                            inserted = self.upsert_batch(collection, batch)
                            self.total_imported_count += inserted
                            i += current_batch_size
                            pbar.update(len(batch))
                        except Exception as e:
                            tqdm.write(f"Error inserting data: {e}")
                            current_batch_size = max(1, current_batch_size * 2 // 3)
                            tqdm.write(f"Reducing batch size to {current_batch_size}")
                            continue
                    pbar.close()
                    if max_hit:
                        break
                count = collection.aggregate.over_all(total_count=True).total_count
                print(f'Class "{class_name}" has {count} objects after import')
        print("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count
