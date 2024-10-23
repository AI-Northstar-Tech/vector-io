import json
import os
from typing import Dict, List
import pymongo
import pandas as pd
from tqdm import tqdm
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from bson import ObjectId, Binary, Regex, Timestamp, Decimal128, Code
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExportMongoDB(ExportVDB):
    DB_NAME_SLUG = DBNames.MONGODB

    @classmethod
    def make_parser(cls, subparsers):
        parser_mongodb = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from MongoDB"
        )
        parser_mongodb.add_argument(
            "--connection_string", type=str, help="MongoDB Atlas Connection string"
        )
        parser_mongodb.add_argument(
            "--vector_dim", type=int, help="Expected dimension of vector columns"
        )
        parser_mongodb.add_argument(
            "--database", type=str, help="MongoDB Atlas Database name"
        )
        parser_mongodb.add_argument(
            "--collection", type=str, help="MongoDB Atlas collection to export"
        )
        parser_mongodb.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting data",
            default=10_000,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "connection_string",
            "Enter the MongoDB Atlas connection string: ",
            str,
        )
        set_arg_from_input(
            args,
            "database",
            "Enter the MongoDB Atlas database name: ",
            str,
        )
        set_arg_from_input(
            args,
            "collection",
            "Enter the name of collection to export: ",
            str,
        )
        set_arg_from_input(
            args,
            "vector_dim",
            "Enter the expected dimension of vector columns: ",
            int,
        )
        mongodb_atlas_export = ExportMongoDB(args)
        mongodb_atlas_export.all_collections = mongodb_atlas_export.get_index_names()
        mongodb_atlas_export.get_data()
        return mongodb_atlas_export

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

    def get_index_names(self):
        collection_name = self.args.get("collection", None)
        if collection_name is not None:
            if collection_name not in self.db.list_collection_names():
                logger.error(
                    f"Collection '{collection_name}' does not exist in the database."
                )
                raise ValueError(
                    f"Collection '{collection_name}' does not exist in the database."
                )
            return [collection_name]
        else:
            return self.get_all_index_names()

    def get_all_index_names(self):
        return self.db.list_collection_names()

    def flatten_dict(self, d, parent_key="", sep="#SEP#"):
        items = []
        type_conversions = {
            ObjectId: lambda v: f"BSON_ObjectId_{str(v)}",
            Binary: lambda v: f"BSON_Binary_{v.decode('utf-8', errors='ignore')}",
            Regex: lambda v: f"BSON_Regex_{json.dumps({'pattern': v.pattern, 'options': v.options})}",
            Timestamp: lambda v: f"BSON_Timestamp_{v.as_datetime().isoformat()}",
            Decimal128: lambda v: f"BSON_Decimal128_{float(v.to_decimal())}",
            Code: lambda v: f"BSON_Code_{str(v.code)}",
        }

        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            conversion = type_conversions.get(type(value))

            if conversion:
                items.append((new_key, conversion(value)))
            elif isinstance(value, dict):
                items.extend(self.flatten_dict(value, new_key, sep=sep).items())
            elif isinstance(value, list):
                if all(isinstance(v, dict) and "$numberDouble" in v for v in value):
                    float_list = [float(v["$numberDouble"]) for v in value]
                    items.append((new_key, float_list))
                else:
                    items.append((new_key, value))
            else:
                items.append((new_key, value))

        return dict(items)

    def get_data(self):
        object_columns_list = []
        vector_columns = []
        expected_dim = self.args.get("vector_dim")
        collection_name = self.args["collection"]
        batch_size = self.args["batch_size"]

        vectors_directory = self.create_vec_dir(collection_name)

        total_documents = self.collection.count_documents({})
        total_batches = (total_documents + batch_size - 1) // batch_size
        total = 0
        index_metas: Dict[str, List[NamespaceMeta]] = {}

        if expected_dim is None:
            logger.info("Vector dimension not provided. Detecting from data...")
            sample_doc = self.collection.find_one()
            if sample_doc:
                flat_doc = self.flatten_dict(sample_doc)
                for key, value in flat_doc.items():
                    if isinstance(value, list) and all(
                        isinstance(x, (int, float)) for x in value
                    ):
                        expected_dim = len(value)
                        logger.info(
                            f"Detected vector dimension: {expected_dim} from column: {key}"
                        )
                        break

            if expected_dim is None:
                expected_dim = 0
                logger.warning("No vector columns detected in the data")

        for i in tqdm(range(total_batches), desc=f"Exporting {collection_name}"):
            cursor = self.collection.find().skip(i * batch_size).limit(batch_size)
            batch_data = list(cursor)
            if not batch_data:
                break

            flattened_data = []
            for document in batch_data:
                flat_doc = self.flatten_dict(document)

                for key in flat_doc:
                    if isinstance(flat_doc[key], dict):
                        flat_doc[key] = json.dumps(flat_doc[key])
                    elif flat_doc[key] == "":
                        flat_doc[key] = None

                flattened_data.append(flat_doc)

            df = pd.DataFrame(flattened_data)
            df = df.dropna(axis=1, how="all")

            for column in df.columns:
                if (
                    isinstance(df[column].iloc[0], list)
                    and len(df[column].iloc[0]) == expected_dim
                ):
                    vector_columns.append(column)
                else:
                    object_columns_list.append(column)
                    df[column] = df[column].astype(str)

            parquet_file = os.path.join(vectors_directory, f"{i}.parquet")
            df.to_parquet(parquet_file)
            total += len(df)

        namespace_metas = [
            self.get_namespace_meta(
                collection_name,
                vectors_directory,
                total=total,
                num_vectors_exported=total,
                dim=expected_dim,
                vector_columns=vector_columns,
                distance="cosine",
            )
        ]
        index_metas[collection_name] = namespace_metas

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)

        logger.info(f"Export completed. Total documents exported: {total}")
        return True
