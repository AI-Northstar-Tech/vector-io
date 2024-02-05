#!/usr/bin/env python3

import argparse
import os
import time
from dotenv import load_dotenv
from names import DBNames

from util import set_arg_from_input, set_arg_from_password
from import_vdf.pinecone_import import ImportPinecone
from import_vdf.qdrant_import import ImportQdrant
from import_vdf.milvus_import import ImportMilvus
from import_vdf.vertexai_vector_search_import import ImportVertexAIVectorSearch
from import_vdf.vdf_import_cls import ImportVDF

load_dotenv()


def import_milvus(args):
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
        args, "token", "Enter your Milvus token (hit enter to skip): ", "Milvus Token"
    )
    milvus_import = ImportMilvus(args)
    milvus_import.upsert_data()


def import_qdrant(args):
    """
    Import data to Qdrant
    """
    set_arg_from_input(
        args,
        "url",
        "Enter the url of Qdrant instance (default: 'http://localhost:6333'): ",
        str,
        "http://localhost:6333",
    )
    set_arg_from_password(
        args, "qdrant_api_key", "Enter your Qdrant API key: ", "QDRANT_API_KEY"
    )
    qdrant_import = ImportQdrant(args)
    qdrant_import.upsert_data()


def import_pinecone(args):
    """
    Import data to Pinecone
    """
    set_arg_from_password(
        args, "pinecone_api_key", "Enter your Pinecone API key: ", "PINECONE_API_KEY"
    )
    if args["serverless"] is False:
        set_arg_from_input(
            args, "environment", "Enter the environment of Pinecone instance: "
        )
    else:
        set_arg_from_input(
            args,
            "cloud",
            "Enter the cloud of Pinecone Serverless instance (default: 'aws'): ",
            str,
            "aws",
        )
        set_arg_from_input(
            args,
            "region",
            "Enter the region of Pinecone Serverless instance (default: 'us-west-2'): ",
            str,
            "us-west-2",
        )

    if args["subset"] is True:
        if "id_list_file" not in args or args["id_list_file"] is None:
            set_arg_from_input(
                args,
                "id_range_start",
                "Enter the start of id range (hit return to skip): ",
                int,
            )
            if args["id_range_start"] is not None:
                set_arg_from_input(
                    args,
                    "id_range_end",
                    "Enter the end of id range (hit return to skip): ",
                    int,
                )
        if args["id_range_start"] is None and args["id_range_end"] is None:
            set_arg_from_input(
                args,
                "id_list_file",
                "Enter the path to id list file (hit return to skip): ",
            )

    pinecone_import = ImportPinecone(args)
    pinecone_import.upsert_data()


def import_vertexai_vectorsearch(args):
    """
    Import data to Vertex AI Vector Search
    """
    set_arg_from_input(args, "project_id", "Enter the Google Cloud Project ID:")
    set_arg_from_input(args, "location", "Enter the region hosting your index: ")
    set_arg_from_input(
        args,
        "batch_size",
        "Enter size of upsert batches (default: 100):",
        default_value=100,
        type_name=int,
    )
    set_arg_from_input(
        args,
        "filter_restricts",
        "Optional. Enter list of dicts describing string filters for each data point: ",
    )
    set_arg_from_input(
        args, "numeric_restricts", "Optional. Enter list of dicts for each datapoint: "
    )
    set_arg_from_input(
        args, "crowding_tag", "Optional. CrowdingTag of the datapoint: ", type_name=str
    )
    if args["create_new"] is True:
        set_arg_from_input(
            args,
            "gcs_bucket",
            "Optional. Enter valid gcs bucket name (or one will be created per index_name): ",
            type_name=str,
        )
        set_arg_from_input(
            args,
            "approx_nn_count",
            "Optional. The default number of neighbors to find via approximate search (default: 150): ",
            type_name=int,
            default_value=150,
        )
        set_arg_from_input(
            args,
            "leaf_node_emb_count",
            "Optional. Number of embeddings on each leaf node (default: 1000): ",
            type_name=int,
            default_value=1000,
        )
        set_arg_from_input(
            args,
            "leaf_nodes_percent",
            "Optional. The default percentage of leaf nodes that any query may be searched (default: 10 [10%]): ",
            type_name=int,
            default_value=10,
            # choices=range(1, 101)
        )
        set_arg_from_input(
            args,
            "distance_measure",
            "Optional. The distance measure used in nearest neighbor search (default: `DOT_PRODUCT_DISTANCE`): ",
            type_name=str,
            default_value="DOT_PRODUCT_DISTANCE",
            # choices=[
            #     "DOT_PRODUCT_DISTANCE",
            #     "COSINE_DISTANCE",
            #     "L1_DISTANCE",
            #     "SQUARED_L2_DISTANCE"
            # ],
        )
        set_arg_from_input(
            args,
            "shard_size",
            "Optional. Size of the shards (default: `SHARD_SIZE_MEDIUM`): ",
            type_name=str,
            default_value="SHARD_SIZE_MEDIUM",
            # choices=[
            #     "SHARD_SIZE_SMALL",
            #     "SHARD_SIZE_MEDIUM",
            #     "SHARD_SIZE_LARGE",
            # ],
        )
    if args["deploy_new_index"] is True:
        set_arg_from_input(
            args,
            "machine_type",
            "Optional. The type of machine (default: `e2-standard-16`): ",
            type_name=str,
            default_value="e2-standard-16",
            # choices=[
            #     "n1-standard-16",
            #     "n1-standard-32",
            #     "e2-standard-2",
            #     "e2-standard-16",
            #     "e2-highmem-16",
            #     "n2d-standard-32",
            # ],
        )
        set_arg_from_input(
            args,
            "min_replicas",
            "Optional. The minimum number of machine replicas for deployed index (default: 1): ",
            type_name=int,
            default_value=1,
        )
        set_arg_from_input(
            args,
            "max_replicas",
            "Optional. The maximum number of machine replicas for deployed index (default: 1): ",
            type_name=int,
            default_value=1,
        )

    vertexai_vectorsearch_import = ImportVertexAIVectorSearch(args)
    vertexai_vectorsearch_import.upsert_data()


def main():
    """
    Import data to Pinecone using a vector dataset directory in the VDF format.
    """
    parser = argparse.ArgumentParser(
        description="Import data from VDF to a vector database"
    )
    # list of all subclasses of ImportVDF
    db_choices = [c.DB_NAME_SLUG for c in ImportVDF.__subclasses__()]
    subparsers = parser.add_subparsers(
        title="Vector Databases",
        description="Choose the vectors database to export data from",
        dest="vector_database",
    )

    parser.add_argument("-d", "--dir", type=str, help="Directory to import")
    parser.add_argument(
        "-s",
        "--subset",
        type=bool,
        help="Import a subset of data (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--create_new",
        type=bool,
        help="Create a new index (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    # Milvus
    parser_milvus = subparsers.add_parser(DBNames.MILVUS, help="Import data to Milvus")
    parser_milvus.add_argument("-u", "--uri", type=str, help="URI of Milvus instance")
    parser_milvus.add_argument("-t", "--token", type=str, help="Milvus token")

    # Pinecone
    parser_pinecone = subparsers.add_parser(
        DBNames.PINECONE, help="Import data to Pinecone"
    )
    parser_pinecone.add_argument(
        "-e", "--environment", type=str, help="Pinecone environment"
    )
    parser_pinecone.add_argument(
        "--serverless",
        type=bool,
        help="Import data to Pinecone Serverless (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser_pinecone.add_argument(
        "-c", "--cloud", type=str, help="Pinecone Serverless cloud"
    )
    parser_pinecone.add_argument(
        "-r", "--region", type=str, help="Pinecone Serverless region"
    )

    # Qdrant
    parser_qdrant = subparsers.add_parser(DBNames.QDRANT, help="Import data to Qdrant")
    parser_qdrant.add_argument("-u", "--url", type=str, help="Qdrant url")

    # Vertex AI VectorSearch
    parser_vertexai_vectorsearch = subparsers.add_parser(
        DBNames.VERTEXAI, help="Import data to Vertex AI Vector Search"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-p", "--project-id", type=str, help="Google Cloud Project ID"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-l", "--location", type=str, help="Google Cloud region hosting your index"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-b",
        "--batch-size",
        type=str,
        help="Enter size of upsert batches:",
        default=100,
    )
    parser_vertexai_vectorsearch.add_argument(
        "-f", "--filter-restricts", type=str, help="string filters"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-n", "--numeric-restricts", type=str, help="numeric filters"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-c",
        "--crowding-tag",
        type=str,
        help="string value to enforce diversity in retrieval",
    )
    parser_vertexai_vectorsearch.add_argument(
        "--deploy_new_index",
        type=bool,
        help="deploy new index (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()
    args = vars(args)
    # open VERSION.txt which is in the parent directory of this script
    args["library_version"] = open(
        os.path.join(os.path.dirname(__file__), "../VERSION.txt")
    ).read()
    set_arg_from_input(
        args, "dir", "Enter the directory of vector dataset to be imported: ", str
    )

    args["cwd"] = os.getcwd()

    start_time = time.time()

    if (
        ("vector_database" not in args)
        or (args["vector_database"] is None)
        or (args["vector_database"] not in db_choices)
    ):
        print("Please choose a vector database to export data from:", db_choices)
        return
    if args["vector_database"] == DBNames.PINECONE:
        import_pinecone(args)
    elif args["vector_database"] == DBNames.QDRANT:
        import_qdrant(args)  # Add the function to import data to Qdrant
    elif args["vector_database"] == DBNames.MILVUS:
        import_milvus(args)
    elif args["vector_database"] == DBNames.VERTEXAI:
        import_vertexai_vectorsearch(args)
    else:
        print(
            "Unrecognized DB. Please choose a vector database to export data from:",
            db_choices,
        )

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
