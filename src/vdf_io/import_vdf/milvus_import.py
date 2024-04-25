from dotenv import load_dotenv
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

from vdf_io.constants import INT_MAX
from vdf_io.names import DBNames
from vdf_io.util import (
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB


load_dotenv()


class ImportMilvus(ImportVDB):
    DB_NAME_SLUG = DBNames.MILVUS

    @classmethod
    def import_vdb(cls, args):
        """
        Import data to Milvus
        """
        set_arg_from_input(
            args,
            "uri",
            "Enter the Milvus URI (default: 'http://localhost:19530'): ",
            str,
            "http://localhost:19530",
        )
        set_arg_from_password(
            args,
            "token",
            "Enter your Milvus/Zilliz token (hit enter to skip): ",
            "ZILLIZ_CLOUD_TOKEN",
        )
        milvus_import = ImportMilvus(args)
        milvus_import.upsert_data()
        return milvus_import

    @classmethod
    def make_parser(cls, subparsers):
        parser_milvus = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Milvus"
        )
        parser_milvus.add_argument(
            "-u", "--uri", type=str, help="URI of Milvus instance"
        )
        parser_milvus.add_argument("-t", "--token", type=str, help="Milvus token")

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
        max_hit = False
        self.total_imported_count = 0
        # we know that the self.vdf_meta["indexes"] is a list
        for collection_name, index_meta in self.vdf_meta["indexes"].items():
            # load data
            print(f'Importing data for collection "{collection_name}"')
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, collection_name)
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
                            name=self.id_column,
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
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(file_path)
                    df[self.id_column] = df[self.id_column].apply(lambda x: str(x))
                    data_rows = []

                    for _, row in df.iterrows():
                        row[old_vector_column_name] = self.extract_vector(
                            row[old_vector_column_name]
                        )
                        row = json.loads(row.to_json())
                        # replace old_vector_column_name with vector_column_name
                        if old_vector_column_name != vector_column_name:
                            row[vector_column_name] = row[old_vector_column_name]
                            del row[old_vector_column_name]
                        assert isinstance(row[f_pk.name], str), row[f_pk.name]
                        assert isinstance(row[f_vector.name][0], float), type(
                            row[f_vector.name][0]
                        )
                        data_rows.append(row)
                    BATCH_SIZE = self.args.get("batch_size", 1000) or 1000
                    current_batch_size = BATCH_SIZE
                    tqdm.write(
                        f"Inserting {len(data_rows)} rows in batches of {BATCH_SIZE}"
                    )
                    if self.total_imported_count + BATCH_SIZE >= (
                        self.args.get("max_num_rows") or INT_MAX
                    ):
                        data_rows = data_rows[
                            : (self.args.get("max_num_rows") or INT_MAX)
                            - self.total_imported_count
                        ]
                        max_hit = True
                    i = 0
                    pbar = tqdm(total=len(data_rows), desc="Upserting data in batches")
                    while i < len(data_rows):
                        try:
                            mr = collection.upsert(
                                data_rows[i : i + current_batch_size]
                            )
                            num_inserted += mr.upsert_count
                            self.total_imported_count += mr.upsert_count
                            i += current_batch_size
                            pbar.update(current_batch_size)
                        except Exception as e:
                            tqdm.write(f"Error inserting data: {e}")
                            # reduce batch size
                            current_batch_size = current_batch_size * 2 // 3
                            tqdm.write(f"Reducing batch size to {current_batch_size}")
                            continue
                    if max_hit:
                        break
                collection.flush()
                vector_count = collection.num_entities
                print(f"Index '{index_name}' has {vector_count} vectors after import")
                print(f"{num_inserted} vectors were imported")
        print("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count
