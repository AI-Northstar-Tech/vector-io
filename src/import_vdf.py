#!/usr/bin/env python3

import argparse
import os
import time
from dotenv import load_dotenv
from names import DBNames

from util import set_arg_from_input, set_arg_from_password
from import_vdf.pinecone_import import ImportPinecone
from import_vdf.qdrant_import import ImportQdrant
from import_vdf.kdbai_import import ImportKDBAI
from import_vdf.milvus_import import ImportMilvus
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


def import_kdbai(args):
    """
    Import data to KDB.AI
    """
    set_arg_from_input(
        args,
        "url",
        "Enter the endpoint for KDB.AI Cloud instance: ",
        str,
    )
    set_arg_from_password(
        args, "kdbai_api_key", "Enter your KDB.AI API key: ", "KDBAI_API_KEY"
    )
    set_arg_from_input(
        args,
        "ind",
        "Enter the index type used (Flat, IVF, IVFPQ, HNSW): ",
        str,
    )
    kdbai_import = ImportKDBAI(args)
    kdbai_import.upsert_data()


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

    # KDB.AI
    parser_kdbai = subparsers.add_parser(DBNames.KDBAI, help="Import data to KDB.AI")
    parser_kdbai.add_argument(
        "-u", "--endpoint", type=str, help="KDB.AI Cloud instance Endpoint url"
    )
    parser_kdbai.add_argument("-i", "--index", type=str, help="Index used")

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
    elif args["vector_database"] == DBNames.KDBAI:
        import_kdbai(args)
    elif args["vector_database"] == DBNames.MILVUS:
        import_milvus(args)
    else:
        print(
            "Unrecognized DB. Please choose a vector database to export data from:",
            db_choices,
        )

    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
