from typing import Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import cleanup_df, divide_into_batches, set_arg_from_input
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.weaviate_util import (
    collection_names,
    compliant_collection_name,
    connect_weaviate,
    infer_weaviate_properties,
    make_weaviate_parser,
    row_to_properties,
    uuid_for_id,
    vector_config_for_columns,
)


load_dotenv()


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
        )
        make_weaviate_parser(parser_weaviate)

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the batch size for importing data (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
        )
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate(self.args)

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        if len(indexes_content) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        collections = collection_names(self.client)
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                collection_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta["namespace"]
                    else ""
                )
                collection_name = compliant_collection_name(collection_name)
                collection_name = self.create_new_name(
                    collection_name, collections, delimiter="_"
                )
                collection_name = compliant_collection_name(collection_name)

                vector_column_names, _ = self.get_vector_column_name(
                    collection_name, namespace_meta, multi_vector_supported=True
                )

                if collection_name not in collections:
                    sample_df = self.load_sample_df(final_data_path, parquet_files)
                    properties = infer_weaviate_properties(
                        sample_df, vector_column_names, self.id_column
                    )
                    self.client.collections.create(
                        collection_name,
                        vector_config=vector_config_for_columns(
                            vector_column_names, namespace_meta.get("metric")
                        ),
                        properties=properties,
                    )
                    collections.append(collection_name)

                collection = self.client.collections.get(collection_name)
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(
                        file_path,
                        max_num_rows=(
                            (self.args.get("max_num_rows") or INT_MAX)
                            - self.total_imported_count
                        ),
                    )
                    df = cleanup_df(df)
                    batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                    for batch_df in tqdm(
                        divide_into_batches(df, batch_size),
                        desc="Importing batches",
                        total=max(len(df) // batch_size, 1),
                    ):
                        with collection.batch.fixed_size(
                            batch_size=batch_size
                        ) as batch:
                            for _, row in batch_df.iterrows():
                                vector = self.row_to_vector(row, vector_column_names)
                                if not vector:
                                    continue
                                batch.add_object(
                                    properties=row_to_properties(
                                        row, vector_column_names, self.id_column
                                    ),
                                    uuid=uuid_for_id(row[self.id_column]),
                                    vector=vector,
                                )
                                self.total_imported_count += 1
                                if self.total_imported_count >= (
                                    self.args.get("max_num_rows") or INT_MAX
                                ):
                                    max_hit = True
                                    break
                        failed_objects = collection.batch.failed_objects
                        if failed_objects:
                            raise RuntimeError(
                                f"Weaviate import failed for {len(failed_objects)} objects. "
                                f"First failure: {failed_objects[0]}"
                            )
                        if max_hit:
                            break
                    if max_hit:
                        break

                tqdm.write(
                    f"Imported {self.total_imported_count} rows into {collection_name}"
                )
                if max_hit:
                    break
            if max_hit:
                tqdm.write(
                    f"Max rows to be imported {self.args['max_num_rows']} hit. Exiting"
                )
                break

        tqdm.write("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count

    def load_sample_df(self, final_data_path, parquet_files):
        if not parquet_files:
            raise ValueError("No parquet files found for Weaviate import")
        first_file = self.get_file_path(final_data_path, parquet_files[0])
        return self.read_parquet_progress(first_file, max_num_rows=100)

    def row_to_vector(self, row, vector_column_names):
        vectors = {}
        for vector_column_name in vector_column_names:
            vector_value = row.get(vector_column_name)
            if vector_value is None:
                continue
            vectors[vector_column_name] = self.extract_vector(vector_value)

        if len(vector_column_names) == 1 and vector_column_names[0] == "vector":
            return vectors.get("vector")
        return vectors
