import argparse
from typing import Dict, List
from tqdm import tqdm
from rich import print as rprint

import turbopuffer as tpuf

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    clean_value,
    cleanup_df,
    divide_into_batches,
    set_arg_from_password,
)


class ImportTurbopuffer(ImportVDB):
    DB_NAME_SLUG = DBNames.TURBOPUFFER

    @classmethod
    def make_parser(cls, subparsers):
        parser_tpuf = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Turbopuffer"
        )
        parser_tpuf.add_argument(
            "--namespaces",
            help="The Turbopuffer namespaces to export (comma-separated). If not provided, all namespaces will be exported.",
        )
        parser_tpuf.add_argument(
            "--api_key",
            help="The API key for the Turbopuffer instance.",
        )

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_password(
            args,
            "api_key",
            "Enter the API key for Turbopuffer (default: from TURBOPUFFER_API_KEY env var): ",
            env_var_name="TURBOPUFFER_API_KEY",
        )
        turbopuffer_import = cls(args)
        turbopuffer_import.upsert_data()
        return turbopuffer_import

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        tpuf.api_key = args.get("api_key")

    def get_all_index_names(self):
        nses = tpuf.namespaces()
        return [ns.name for ns in nses]

    def upsert_data(self):
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")
        collections = self.get_all_index_names()
        # Load Parquet file
        # print(indexes_content[index_names[0]]):List[NamespaceMeta]
        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(final_data_path)

                new_index_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_index_name = self.create_new_name(new_index_name, collections)
                ns = tpuf.Namespace(new_index_name)
                (
                    vector_column_names,
                    vector_column_name,
                ) = self.get_vector_column_name(index_name, namespace_meta)
                tqdm.write(f"Vector column name: {vector_column_name}")
                if len(vector_column_names) > 1:
                    tqdm.write("Turbopuffer does not support multiple vector columns")
                    tqdm.write(f"Skipping the rest : {vector_column_names[1:]}")
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
                    BATCH_SIZE = min(
                        self.args.get("batch_size") or DEFAULT_BATCH_SIZE, 10_000
                    )

                    # keeping track of updated keys
                    updated_keys = set()
                    for batch in tqdm(
                        divide_into_batches(df, BATCH_SIZE),
                        desc="Importing batches",
                        total=len(df) // BATCH_SIZE,
                    ):
                        # filter out rows with empty or None vector column
                        prev_count = len(batch)
                        batch = batch.dropna(subset=[vector_column_name])
                        non_empty_count = len(batch)
                        if prev_count != non_empty_count:
                            tqdm.write(
                                f"Skipped {prev_count - non_empty_count} rows with empty vector column"
                            )
                        # Attributes are key/value mappings. Keys are strings, and values can be strings, unsigned integers, or arrays of either.
                        metadata = batch.drop(
                            columns=[self.id_column] + vector_column_names
                        ).to_dict(orient="records")
                        for i in range(len(metadata)):
                            for key, val in metadata[i].items():
                                if isinstance(val, list):
                                    # check for str or unsigned int
                                    if not all(
                                        isinstance(x, str) for x in val
                                    ) and not (
                                        all(isinstance(x, int) for x in val)
                                        and all(x >= 0 for x in val)
                                    ):
                                        # convert all elements to string
                                        metadata[i][key] = [str(x) for x in val]
                                        updated_keys.add(key)
                                # value can be converted to string
                                else:
                                    metadata[i][key] = str(val)
                                    updated_keys.add(key)
                            for key, val in metadata[i].items():
                                metadata[i][key] = clean_value(val)

                        # rprint(metadata)
                        tqdm.write(
                            f"Updated values of keys: {updated_keys} to string type"
                        )
                        try:
                            ns.upsert(
                                data=[
                                    {
                                        "id": row[self.id_column],
                                        "vector": row[vector_column_name],
                                        "attributes": metadata[idx],
                                    }
                                    for idx, row in batch.iterrows()
                                ],
                            )
                        except Exception as e:
                            tqdm.write(f"Error: {e} {batch}")
                            continue
                        self.total_imported_count += len(batch)
                tqdm.write(
                    f"Finished importing {self.total_imported_count} vectors into {new_index_name}"
                )
