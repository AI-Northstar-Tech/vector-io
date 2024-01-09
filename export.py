#!/usr/bin/env python3

import argparse
import os
import sys
import time
from dotenv import load_dotenv
from export.pinecone_export import ExportPinecone
from export.util import set_arg_from_input, set_arg_from_password
from export.weaviate_export import ExportWeaviate
from export.qdrant_export import ExportQdrant
from getpass import getpass
import warnings

# Suppress specific warnings
warnings.simplefilter("ignore", ResourceWarning)

load_dotenv()

DEFAULT_MAX_FILE_SIZE = 1024  # in MB

def export_pinecone(args):
    """
    Export data from Pinecone
    """
    set_arg_from_input(
        args, "environment", "Enter the environment of Pinecone instance: "
    )
    set_arg_from_input(
        args,
        "index",
        "Enter the name of index to export (hit return to export all): ",
    )
    if "id_list_file" not in args or args["id_list_file"] is None:
        set_arg_from_input(
            args,
            "id_range_start",
            "Enter the start of id range (hit return to skip): ",
            int,
        )
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
    set_arg_from_password(
        args, "pinecone_api_key", "Enter your Pinecone API key: ", "PINECONE_API_KEY"
    )
    set_arg_from_input(
        args,
        "modify_to_search",
        "Allow modifying data to search, enter Y or N: ",
        bool,
    )
    pinecone_export = ExportPinecone(args)
    pinecone_export.get_data()


def export_weaviate(args):
    """
    Export data from Weaviate
    """
    set_arg_from_input(args, "url", "Enter the location of Weaviate instance: ")
    set_arg_from_input(
        args,
        "class_name",
        "Enter the name of class to export, or type all to export all classes: ",
    )
    set_arg_from_input(
        args, "include_crossrefs", "Include cross references, enter Y or N: "
    )
    if args["include_crossrefs"] == "Y":
        args["include_crossrefs"] = True
    else:
        args["include_crossrefs"] = False
    weaviate = ExportWeaviate(args)
    weaviate.get_data()


def export_qdrant(args):
    """
    Export data from Qdrant
    """
    set_arg_from_input(args, "url", "Enter the url of Qdrant instance (hit return for 'http://localhost:6333'): ", str, "http://localhost:6333")
    set_arg_from_input(
        args,
        "collections",
        "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
        lambda x: x.split(","),
    )
    set_arg_from_password(
        args, "qdrant_api_key", "Enter your Qdrant API key: ", "QDRANT_API_KEY"
    )
    qdrant = ExportQdrant(args)
    qdrant.get_data()


def main():
    """
    Export data from Pinecone, Weaviate and Qdrant to sqlite database and parquet file.

    Usage:
        python export.py <vector_database> [options]

    Arguments:
        vector_database (str): Choose the vectors database to export data from.
            Possible values: "pinecone", "weaviate", "qdrant".

    Options:
        Pinecone:
            -e, --environment (str): Environment of Pinecone instance.
            -i, --index (str): Name of indexes to export (comma-separated).

        Weaviate:
            -u, --url (str): Location of Weaviate instance.
            -c, --class_name (str): Name of class to export.
            -i, --include_crossrefs (bool): Include cross references, set Y or N.

        Qdrant:
            -u, --url (str): Location of Qdrant instance.
            -c, --collections (str): Names of collections to export (comma-separated).

    Examples:
        Export data from Pinecone:
        python export.py pinecone -e my_env -i my_index

        Export data from Weaviate:
        python export.py weaviate -u http://localhost:8080 -c my_class -i Y

        Export data from Qdrant:
        python export.py qdrant -u http://localhost:6333 -c my_collection
    """
    parser = argparse.ArgumentParser(
        description="Export data from Pinecone, Weaviate and Qdrant to sqlite database and parquet file"
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        help="Name of model used",
        default="text-embedding-ada-002",
    )
    parser.add_argument(
        "--max_file_size",
        type=int,
        help="Maximum file size in MB (default: 1024)",
        default=DEFAULT_MAX_FILE_SIZE,
    )
    subparsers = parser.add_subparsers(
        title="Vector Databases",
        description="Choose the vectors database to export data from",
        dest="vector_database",
    )

    # Pinecone
    parser_pinecone = subparsers.add_parser(
        "pinecone", help="Export data from Pinecone"
    )
    parser_pinecone.add_argument(
        "-e", "--environment", type=str, help="Environment of Pinecone instance"
    )
    parser_pinecone.add_argument(
        "-i", "--index", type=str, help="Name of index to export"
    )
    parser_pinecone.add_argument(
        "-s", "--id_range_start", type=int, help="Start of id range", default=None
    )
    parser_pinecone.add_argument(
        "--id_range_end", type=int, help="End of id range", default=None
    )
    parser_pinecone.add_argument(
        "-f", "--id_list_file", type=str, help="Path to id list file", default=None
    )
    parser_pinecone.add_argument(
        "--modify_to_search", type=bool, help="Allow modifying data to search", default=True
    )

    # Weaviate
    parser_weaviate = subparsers.add_parser(
        "weaviate", help="Export data from Weaviate"
    )
    parser_weaviate.add_argument(
        "-u", "--url", type=str, help="Location of Weaviate instance"
    )
    parser_weaviate.add_argument(
        "-c", "--class_name", type=str, help="Name of class to export"
    )
    parser_weaviate.add_argument(
        "-i",
        "--include_crossrefs",
        type=bool,
        help="Include cross references, set Y or N",
    )

    # Qdrant
    parser_qdrant = subparsers.add_parser("qdrant", help="Export data from Qdrant")
    parser_qdrant.add_argument(
        "-u", "--url", type=str, help="Location of Qdrant instance"
    )
    parser_qdrant.add_argument(
        "-c", "--collections", type=str, help="Names of collections to export"
    )

    args = parser.parse_args()
    # convert args to dict
    args = vars(args)
    args["library_version"] = open("VERSION.txt").read()
    t_start = time.time()
    if args["vector_database"] == "pinecone":
        export_pinecone(args)
    elif args["vector_database"] == "weaviate":
        export_weaviate(args)
    elif args["vector_database"] == "qdrant":
        export_qdrant(args)
    else:
        print("Invalid vector database")
        args["vector_database"] = input("Enter the name of vector database to export: ")
        sys.argv.extend(["--vector_database", args["vector_database"]])
        main()
    t_end = time.time()
    # formatted time
    print("Time taken to export data: ", time.strftime("%H:%M:%S", time.gmtime(t_end - t_start)))
    print("Export completed.")


if __name__ == "__main__":
    main()
