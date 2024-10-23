from dotenv import load_dotenv
from tqdm import tqdm
import pymongo
import logging
import re
import ast
import numpy as np
from bson import ObjectId, Binary, Regex, Timestamp, Decimal128, Code
import json
from datetime import datetime
from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.names import DBNames
from vdf_io.util import (
    cleanup_df,
    divide_into_batches,
    set_arg_from_input,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB

load_dotenv()
logger = logging.getLogger(__name__)


class ImportMongoDB(ImportVDB):
    DB_NAME_SLUG = DBNames.MONGODB

    @classmethod
    def make_parser(cls, subparsers):
        parser_mongodb = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to MongoDB"
        )
        parser_mongodb.add_argument(
            "--connection_string", type=str, help="MongoDB Atlas Connection string"
        )
        parser_mongodb.add_argument(
            "--database", type=str, help="MongoDB Atlas Database name"
        )
        parser_mongodb.add_argument(
            "--collection", type=str, help="MongoDB Atlas collection to export"
        )
        parser_mongodb.add_argument(
            "--vector_dim", type=int, help="Expected dimension of vector columns"
        )

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to MongoDB
        """
        set_arg_from_input(
            args,
            "connection_string",
            "Enter the MongoDB connection string: ",
            str,
        )
        set_arg_from_input(
            args,
            "database",
            "Enter the MongoDB database name: ",
            str,
        )
        set_arg_from_input(
            args,
            "collection",
            "Enter the name of collection: ",
            str,
        )
        set_arg_from_input(
            args, "vector_dim", "Enter the expected dimension of vector columns: ", int
        )
        mongodb_import = ImportMongoDB(args)
        mongodb_import.upsert_data()
        return mongodb_import

    def __init__(self, args):
        super().__init__(args)

        try:
            self.client = pymongo.MongoClient(
                args["connection_string"], serverSelectionTimeoutMS=5000
            )
            self.client.server_info()
            logger.info("Successfully connected to MongoDB")
        except pymongo.errors.ServerSelectionTimeoutError as err:
            logger.error(f"Failed to connect to MongoDB: {err}")
            raise

        try:
            self.db = self.client[args["database"]]
        except Exception as err:
            logger.error(f"Failed to select MongoDB database: {err}")
            raise

        try:
            self.collection = self.db[args["collection"]]
        except Exception as err:
            logger.error(f"Failed to select MongoDB collection: {err}")
            raise

    def convert_types(self, documents):
        return [self.convert_document(doc) for doc in documents]

    def convert_document(self, doc):
        converted_doc = {}
        for key, value in doc.items():
            parts = key.split("#SEP#")
            value = self.convert_value(value)
            self.nested_set(converted_doc, parts, value)
        return converted_doc

    def nested_set(self, dic, keys, value):
        for key in keys[:-1]:
            # If the key already exists and is not a dictionary, we need to handle it
            if key in dic and not isinstance(dic[key], dict):
                dic[key] = {}  # Overwrite with an empty dictionary

            dic = dic.setdefault(key, {})

        dic[keys[-1]] = value  # Set the final key to the value

    def convert_value(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()  # Convert numpy array to list : MongoDB can't handle the numpy array directly

        # Check if the value is a string
        if isinstance(value, str):
            # Check if the string is a date in "YYYY-MM-DD" format or extended ISO format
            date_pattern = r"^\d{4}-\d{2}-\d{2}$"  # Regex for "YYYY-MM-DD"
            iso_pattern = (
                r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$"  # Extended ISO
            )

            if re.match(date_pattern, value):  # If it matches "YYYY-MM-DD"
                try:
                    return datetime.strptime(value, "%Y-%m-%d")  # Convert to datetime
                except ValueError:
                    pass

            if re.match(iso_pattern, value):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    pass

            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    pass

            try:
                # Try to evaluate if the string is a list (e.g., for arrays like genres)
                value = ast.literal_eval(value)
                if isinstance(value, list):
                    return value  # Return as a list if it is a list
            except (ValueError, SyntaxError):
                # If it's not an array or number, leave it as a string
                pass

            # Handle special BSON formats, as before
            if value.startswith("BSON_ObjectId_"):
                return ObjectId(value[14:])
            elif value.startswith("BSON_Binary_"):
                return Binary(value[12:].encode("utf-8"))
            elif value.startswith("BSON_Regex_"):
                regex_dict = json.loads(value[11:])
                return Regex(regex_dict["pattern"], regex_dict["options"])
            elif value.startswith("BSON_Timestamp_"):
                return Timestamp(datetime.fromisoformat(value[16:]))
            elif value.startswith("BSON_Decimal128_"):
                return Decimal128(value[16:])
            elif value.startswith("BSON_Code_"):
                return Code(value[10:])

        elif isinstance(value, list):
            return [self.convert_value(item) for item in value]

        return value

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        indexes_content = self.vdf_meta["indexes"]
        index_names = list(indexes_content.keys())
        if len(index_names) == 0:
            raise ValueError("No indexes found in VDF_META.json")

        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing indexes"
        ):
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    try:
                        df = self.read_parquet_progress(
                            file_path,
                            max_num_rows=(self.args.get("max_num_rows") or INT_MAX),
                        )
                    except Exception as e:
                        logger.error(
                            f"Error reading Parquet file {file_path}: {str(e)}"
                        )
                        continue
                    df = cleanup_df(df)

                    BATCH_SIZE = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
                    for batch in tqdm(
                        divide_into_batches(df, BATCH_SIZE),
                        desc="Importing batches",
                        total=len(df) // BATCH_SIZE,
                    ):
                        if self.total_imported_count + len(batch) >= (
                            self.args.get("max_num_rows") or INT_MAX
                        ):
                            batch = batch[
                                : (self.args.get("max_num_rows") or INT_MAX)
                                - self.total_imported_count
                            ]
                            max_hit = True

                        documents = batch.to_dict("records")

                        try:
                            documents = self.convert_types(documents)
                            self.collection.insert_many(documents)
                            self.total_imported_count += len(batch)
                        except pymongo.errors.BulkWriteError as e:
                            logger.error(f"Error during bulk insert: {str(e.details)}")

                        if max_hit:
                            break

                tqdm.write(f"Imported {self.total_imported_count} rows")
                tqdm.write(
                    f"New collection size: {self.collection.count_documents({})}"
                )
                if max_hit:
                    break

        logger.info(
            f"Data import completed. Total rows imported: {self.total_imported_count}"
        )
