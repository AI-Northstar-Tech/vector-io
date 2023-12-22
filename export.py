import argparse
import os
import sys
from dotenv import load_dotenv
from export.pinecone_export import ExportPinecone
from export.weaviate_export import ExportWeaviate
from export.qdrant_export import ExportQdrant
from getpass import getpass
import warnings

# Suppress specific warnings
warnings.simplefilter("ignore", ResourceWarning)

load_dotenv()


def set_arg_from_input(args, arg_name, prompt, type_name=str):
    """
    Set the value of an argument from user input if it is not already present
    """
    if arg_name not in args or args[arg_name] is None:
        inp = input(prompt)
        if inp == "":
            args[arg_name] = None
        else:
            args[arg_name] = type_name(inp)
    return


def set_arg_from_password(args, arg_name, prompt, env_var_name):
    """
    Set the value of an argument from user input if it is not already present
    """
    if os.getenv(env_var_name) is not None:
        args[arg_name] = os.getenv(env_var_name)
    elif arg_name not in args or args[arg_name] is None:
        args[arg_name] = getpass(prompt)
    return


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
    set_arg_from_input(args, "url", "Enter the location of Qdrant instance: ")
    set_arg_from_input(
        args,
        "collections",
        "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
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
    print("Export completed.")
    import ssl
    import socket

    # Create an SSL socket
    ssl_socket = ssl.wrap_socket(socket.socket(socket.AF_INET, socket.SOCK_STREAM))

    # Use the SSL socket

    # Close the SSL socket
    ssl_socket.close()


if __name__ == "__main__":
    main()
