#!/usr/bin/env python3

import argparse
import os
import sys
import time
from dotenv import load_dotenv
from export_vdf.pinecone_export import ExportPinecone
from export_vdf.qdrant_export import ExportQdrant
from export_vdf.milvus_export import ExportMilvus
from export_vdf.vertexai_vectorsearch_export import ExportVertexAIVectorSearch
from export_vdf.vdb_export_cls import ExportVDB
from names import DBNames
from push_to_hub import push_to_hub
from util import set_arg_from_input, set_arg_from_password
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
    set_arg_from_password(
        args, "pinecone_api_key", "Enter your Pinecone API key: ", "PINECONE_API_KEY"
    )
    set_arg_from_input(
        args,
        "modify_to_search",
        "Allow modifying data to search, enter Y or N: ",
        bool,
    )
    if args["subset"] is True:
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
    pinecone_export = ExportPinecone(args)
    pinecone_export.get_data()
    return pinecone_export


def export_qdrant(args):
    """
    Export data from Qdrant
    """
    set_arg_from_input(
        args,
        "url",
        "Enter the url of Qdrant instance (hit return for 'http://localhost:6333'): ",
        str,
        "http://localhost:6333",
    )
    set_arg_from_input(
        args,
        "collections",
        "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
        str,
    )
    set_arg_from_password(
        args, "qdrant_api_key", "Enter your Qdrant API key: ", "QDRANT_API_KEY"
    )
    qdrant_export = ExportQdrant(args)
    qdrant_export.get_data()
    return qdrant_export


def export_milvus(args):
    """
    Export data from Milvus
    """
    set_arg_from_input(
        args,
        "uri",
        "Enter the uri of Milvus (hit return for 'http://localhost:19530'): ",
        str,
        "http://localhost:19530",
    )
    set_arg_from_input(
        args,
        "collections",
        "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
        str,
    )
    set_arg_from_password(
        args, "token", "Enter your Milvus Token (hit return to skip): ", "Milvus Token"
    )
    milvus_export = ExportMilvus(args)
    milvus_export.get_data()
    return milvus_export


def export_vertexai_vectorsearch(args):
    """
    Export data from Vertex AI Vector Search
    """
    set_arg_from_input(
        args, 
        "project_id", 
        "Enter the Google Cloud Project ID: "
    )
    set_arg_from_input(
        args,
        "index",
        "Enter name of index to export (hit return to export all. Comma separated for multiple indexes): ",
    )
    set_arg_from_input(
        args, 
        "gcloud_credentials_file", 
        "Enter path to service account credentials file (hit return to use application default credentials): ", 
    )
    vertexai_vectorsearch_export = ExportVertexAIVectorSearch(args)
    vertexai_vectorsearch_export.get_data()
    return vertexai_vectorsearch_export


def main():
    """
    Export data from various vector databases to the VDF format for vector datasets.

    Usage:
        python export.py <vector_database> [options]

    Arguments:
        vector_database (str): Choose the vectors database to export data from.
            Possible values: "pinecone", "qdrant", "vertexai_vectorsearch".

    Options:
        Pinecone:
            -e, --environment (str): Environment of Pinecone instance.
            -i, --index (str): Name of indexes to export (comma-separated).

        Qdrant:
            -u, --url (str): Location of Qdrant instance.
            -c, --collections (str): Names of collections to export (comma-separated).

        Vertex AI Vector Search:
            -p, --project-id (str): Google Cloud Project ID.
            -i, --index (str): Name of indexes to export (comma-separated).
            -c, --gcloud-credentials-file: Path to Goofle Cloud Service Account credentials
            
    Examples:
        Export data from Pinecone:
        python export_vdf.py pinecone -e my_env -i my_index

        Export data from Qdrant:
        python export_vdf.py qdrant -u http://localhost:6333 -c my_collection

        Export data from Vertex AI Vector Search:
        python export_vdf.py vertexai_vectorsearch -p your_project_id -i your_index
    """
    parser = argparse.ArgumentParser(
        description="Export data from various vector databases to the VDF format for vector datasets"
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

    parser.add_argument(
        "--push_to_hub",
        type=bool,
        help="Push to hub",
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--public",
        type=bool,
        help="Make dataset public (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
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
        "--modify_to_search",
        type=bool,
        help="Allow modifying data to search",
        default=False,
        action=argparse.BooleanOptionalAction
    )
    parser_pinecone.add_argument(
        "--subset",
        type=bool,
        help="Export a subset of data (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction
    )
    db_choices = [c.DB_NAME_SLUG for c in ExportVDB.__subclasses__()]
    # Qdrant
    parser_qdrant = subparsers.add_parser("qdrant", help="Export data from Qdrant")
    parser_qdrant.add_argument(
        "-u", "--url", type=str, help="Location of Qdrant instance"
    )
    parser_qdrant.add_argument(
        "-c", "--collections", type=str, help="Names of collections to export"
    )
    # Milvus
    parser_milvus = subparsers.add_parser("milvus", help="Export data from Milvus")
    parser_milvus.add_argument(
        "-u", "--uri", type=str, help="Milvus connection URI"
    )
    parser_milvus.add_argument(
        "-t", "--token", type=str, required=False, help="Milvus connection token"
    )
    parser_milvus.add_argument(
        "-c", "--collections", type=str, help="Names of collections to export"
    )

    # Vertex AI VectorSearch
    parser_vertexai_vectorsearch = subparsers.add_parser(
        "vertexai_vectorsearch", help="Export data from Vertex AI Vector Search"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-p", "--project-id", type=str, help="Google Cloud Project ID"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-i", "--index", type=str, help="Name of the index or indexes to export"
    )
    parser_vertexai_vectorsearch.add_argument(
        "-c", "--gcloud-credentials-file", type=str, help="Path to Google Cloud service account credentials file", default=None
    )

    args = parser.parse_args()
    # convert args to dict
    args = vars(args)
    # open VERSION.txt which is in the parent directory of this script
    args["library_version"] = open(
        os.path.join(os.path.dirname(__file__), "../VERSION.txt")
    ).read()
    t_start = time.time()
    if (
        ("vector_database" not in args)
        or (args["vector_database"] is None)
        or (args["vector_database"] not in db_choices)
    ):
        print("Please choose a vector database to export data from:", db_choices)
        return
    if args["vector_database"] == DBNames.PINECONE:
        export_obj = export_pinecone(args)
    elif args["vector_database"] == DBNames.QDRANT:
        export_obj = export_qdrant(args)
    elif args["vector_database"] == DBNames.MILVUS:
        export_obj = export_milvus(args)
    elif args["vector_database"] == DBNames.VERTEXAI:
        export_obj = export_vertexai_vectorsearch(args)
    else:
        print("Invalid vector database")
        args["vector_database"] = input("Enter the name of vector database to export: ")
        sys.argv.extend(["--vector_database", args["vector_database"]])
        main()
    t_end = time.time()
    # formatted time
    print(f"Export to disk completed. Exported to: {export_obj.vdf_directory}/")
    print(
        "Time taken to export data: ",
        time.strftime("%H:%M:%S", time.gmtime(t_end - t_start)),
    )

    if args["push_to_hub"]:
        push_to_hub(export_obj, args)


if __name__ == "__main__":
    main()
