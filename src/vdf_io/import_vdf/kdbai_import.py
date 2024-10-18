from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
import pyarrow.parquet as pq

import kdbai_client as kdbai

from vdf_io.names import DBNames
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)

load_dotenv()


_parquettype_to_pytype = {
    "BOOLEAN": "bool",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "FLOAT": "float32",
    "list<element: float>": "float32s",
    "DOUBLE": "float64",
    "BYTE_ARRAY": "bytes",
    "FIXED_LEN_BYTE_ARRAY": "bytes",
    "string": "str",
    "BINARY": "bytes",
    "timestamp[ns]": "datetime64[ns]",
    "TIMESTAMP_MILLIS": "datetime64[ms]",
    "TIMESTAMP_MICROS": "datetime64[us]",
    "DATE": "datetime64[D]",
    "TIME_MILLIS": "timedelta64[ms]",
    "TIME_MICROS": "timedelta64[us]",
    "DECIMAL": "float64",
    "UINT8": "uint8",
    "UINT16": "uint16",
    "UINT32": "uint32",
    "UINT64": "uint64",
    "INTERVAL": "timedelta64",
}


class ImportKDBAI(ImportVDB):
    DB_NAME_SLUG = DBNames.KDBAI

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to KDB.AI
        """
        if args.get("kdbai_endpoint") is None:
            set_arg_from_input(
                args,
                "kdbai_endpoint",
                "Enter the KDB.AI endpoint instance: ",
                str,
                env_var="KDBAI_ENDPOINT",
            )

        if args.get("kdbai_api_key") is None:
            set_arg_from_password(
                args, "kdbai_api_key", "Enter your KDB.AI API key: ", "KDBAI_API_KEY"
            )

        if args.get("index") is None:
            set_arg_from_input(
                args,
                "index",
                "Enter the index type used (Flat, IVF, IVFPQ, HNSW, QFLAT, QHNSW): ",
                str,
            )

        kdbai_import = ImportKDBAI(args)
        kdbai_import.upsert_data()
        return kdbai_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_kdbai = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to KDB.AI"
        )
        parser_kdbai.add_argument(
            "-u", "--url", type=str, help="KDB.AI Cloud instance Endpoint url"
        )
        parser_kdbai.add_argument(
            "-i", "--index", type=str, help="Index used", default="hnsw"
        )

    def __init__(self, args):
        super().__init__(args)
        api_key = args.get("kdbai_api_key")
        endpoint = args.get("kdbai_endpoint")
        self.index = args.get("index")
        allowed_vector_types = ["flat", "ivf", "ivfpq", "hnsw", "qflat", "qhnsw"]
        if self.index.lower() not in allowed_vector_types:
            raise ValueError(
                f"Invalid vectorIndex type: {self.index}. "
                f"Allowed types are {', '.join(allowed_vector_types)}"
            )

        session = kdbai.Session(api_key=api_key, endpoint=endpoint)
        self.db = session.database("default")

    def compliant_name(self, name: str) -> str:
        new_name = name.replace("-", "_")
        if new_name.startswith("_"):
            new_name = "col" + new_name
        return new_name

    def upsert_data(self):
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        index_names: List[str] = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                index_name = index_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                new_index_name = self.compliant_name(index_name)
                parquet_files = self.get_parquet_files(final_data_path)
                for parquet_file in tqdm(parquet_files, desc="Importing parquet files"):
                    (
                        vector_column_names,
                        vector_column_name,
                    ) = self.get_vector_column_name(index_name, namespace_meta)

                    parquet_file_path = self.get_file_path(
                        final_data_path, parquet_file
                    )

                    if self.abnormal_vector_format:
                        pandas_table = self.read_parquet_progress(parquet_file_path)
                        pandas_table[vector_column_name] = pandas_table[
                            vector_column_name
                        ].apply(lambda x: self.extract_vector(x))
                        # TODO: use save_vectors_to_parquet
                        pandas_table.to_parquet(parquet_file_path)

                    parquet_table = pq.read_table(parquet_file_path)

                    old_column_name_to_new = {
                        col: self.compliant_name(col)
                        for col in parquet_table.column_names
                    }
                    parquet_table = parquet_table.rename_columns(
                        [
                            old_column_name_to_new[col]
                            for col in parquet_table.column_names
                        ]
                    )
                    parquet_schema = parquet_table.schema
                    parquet_columns = [
                        {"name": field.name, "type": str(field.type)}
                        for field in parquet_schema
                    ]

                    # Extract information from JSON
                    vector_column_names = [
                        self.compliant_name(col) for col in vector_column_names
                    ]
                    vector_column_name = self.compliant_name(vector_column_name)

                    # Define the schema
                    schema = []
                    for c in parquet_columns:
                        column_name = c["name"]
                        column_type = c["type"]

                        try:
                            schema.append(
                                {
                                    "name": column_name,
                                    "type": _parquettype_to_pytype[column_type],
                                }
                            )
                        except KeyError:
                            raise ValueError(
                                f"Cannot create the table. The column '{column_name}' with type '{column_type}' is not mapped. Please update the schema."
                            )

                    index = {
                        "name": "flat",
                        "column": vector_column_name,
                        "type": namespace_meta["model_name"],
                        "params": {
                            "dims": namespace_meta["dimensions"],
                            "metric": standardize_metric_reverse(
                                namespace_meta.get("metric"),
                                self.DB_NAME_SLUG,
                            ),
                        },
                    }

                    try:
                        if new_index_name in [name.name for name in self.db.tables]:
                            table = self.db.table(new_index_name)
                            tqdm.write(
                                f"Table '{new_index_name}' already exists. Upserting data into it."
                            )
                        else:
                            table = self.db.create_table(
                                new_index_name, schema=schema, indexes=[index]
                            )
                            tqdm.write("Table created")

                    except kdbai.KDBAIException as e:
                        tqdm.write(f"Error creating table: {e}")
                        raise RuntimeError(f"Error creating table: {e}")

                    df = parquet_table.to_pandas()

                    batch_size = self.args.get("batch_size", 10_000) or 10_000
                    pbar = tqdm(total=df.shape[0], desc="Inserting data")

                    i = 0
                    try:
                        while i < df.shape[0]:
                            chunk = df.iloc[
                                i : min(i + batch_size, df.shape[0])
                            ].reset_index(drop=True)

                            try:
                                table.insert(chunk)
                                pbar.update(chunk.shape[0])
                                i += batch_size
                            except kdbai.KDBAIException as e:
                                raise RuntimeError(f"Error inserting chunk: {e}")
                    finally:
                        pbar.close()

        print("Data imported successfully")
