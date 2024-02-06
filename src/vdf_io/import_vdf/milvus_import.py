import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import json

from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

from vdf_io.names import DBNames
from vdf_io.util import standardize_metric_reverse
from vdf_io.import_vdf.vdf_import_cls import ImportVDF


load_dotenv()


class ImportMilvus(ImportVDF):
    DB_NAME_SLUG = DBNames.MILVUS

    def __init__(self, args):
        # call super class constructor
        super().__init__(args)
        if args is None:
            args = {}
        assert isinstance(args, dict), "Invalid args."
        super().__init__(args)

        uri = self.args.get("uri", "http://localhost:19530")
        token = self.args.get("token", "")
        connections.connect(uri=uri, token=token)

    def upsert_data(self):
        # we know that the self.vdf_meta["indexes"] is a list
        for collection_name, index_meta in self.vdf_meta["indexes"].items():
            # load data
            print(f'Importing data for collection "{collection_name}"')
            for namespace_meta in index_meta:
                data_path = namespace_meta["data_path"]
                index_name = collection_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
                _, vector_column_name = self.get_vector_column_name(
                    collection_name, namespace_meta
                )
                # replace - with _
                old_vector_column_name = vector_column_name
                vector_column_name = vector_column_name.replace("-", "_")
                index_name = index_name.replace("-", "_")
                print(f"Index name: {index_name}")

                # check if collection exists
                if utility.has_collection(index_name):
                    collection = Collection(index_name)
                    f_vector = None
                    f_pk = None
                    for f in collection.schema.fields:
                        if f.dtype.value in [100, 101]:
                            f_vector = f
                        if hasattr(f, "is_primary") and f.is_primary:
                            f_pk = f
                else:
                    # create collection
                    try:
                        f_pk = FieldSchema(
                            name="id",
                            dtype=DataType.VARCHAR,
                            max_length=65_535,
                            is_primary=True,
                            auto_id=False,
                        )
                        f_vector = FieldSchema(
                            name=vector_column_name,
                            dtype=DataType.FLOAT_VECTOR,
                            dim=namespace_meta["dimensions"],
                        )
                        schema = CollectionSchema(
                            fields=[f_pk, f_vector], enable_dynamic_field=True
                        )
                        collection = Collection(name=index_name, schema=schema)
                    except Exception as e:
                        print(f'Failed to create collection "{index_name}"', e)
                        raise RuntimeError("Failed to create collection.") from e

                # check if index exists
                if index_name in [index.index_name for index in collection.indexes]:
                    print(f"Using existed index: {index_name}")
                else:
                    # create index
                    try:
                        index_params = {
                            "metric_type": standardize_metric_reverse(
                                namespace_meta["metric"], self.DB_NAME_SLUG
                            ),
                            "index_type": "AUTOINDEX",
                        }
                        collection.create_index(
                            field_name=f_vector.name, index_params=index_params
                        )
                    except Exception as e:
                        print(
                            f"Faild to create index {index_name} for collection {index_name}."
                        )
                        raise RuntimeError("Failed to create index.") from e

                prev_vector_count = collection.num_entities
                if prev_vector_count > 0:
                    print(
                        f'Collection "{index_name}" has {prev_vector_count} vectors before import'
                    )

                # Load the data from the parquet files
                final_data_path = self.get_final_data_path(data_path)
                parquet_files = self.get_parquet_files(final_data_path)

                num_inserted = 0
                for file in tqdm(parquet_files, desc="Inserting data"):
                    file_path = os.path.join(final_data_path, file)
                    df = pd.read_parquet(file_path)
                    df["id"] = df["id"].apply(lambda x: str(x))
                    data_rows = []
                    for _, row in df.iterrows():
                        row = json.loads(row.to_json())
                        # replace old_vector_column_name with vector_column_name
                        row[vector_column_name] = row[old_vector_column_name]
                        del row[old_vector_column_name]
                        assert isinstance(row[f_pk.name], str), row[f_pk.name]
                        assert isinstance(row[f_vector.name][0], float), type(
                            row[f_vector.name][0]
                        )
                        data_rows.append(row)
                    BATCH_SIZE = 100
                    for i in tqdm(
                        range(0, len(data_rows), BATCH_SIZE),
                        desc="Upserting in Batches",
                    ):
                        mr = collection.insert(data_rows[i : i + BATCH_SIZE])
                        num_inserted += mr.succ_count

                collection.flush()
                vector_count = collection.num_entities
                print(f"Index '{index_name}' has {vector_count} vectors after import")
                print(f"{num_inserted} vectors were imported")
        print("Data import completed successfully.")
