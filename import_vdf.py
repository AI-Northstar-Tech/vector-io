import argparse
import os
from dotenv import load_dotenv
# from export.vdb_export import ExportPinecone, ExportWeaviate, ExportQdrant
from getpass import getpass
from export.util import set_arg_from_input, set_arg_from_password
from import_VDF.pinecone_import import ImportPinecone


load_dotenv()


def import_pinecone(args):
    """
    Export data from Pinecone
    """
    set_arg_from_input(
        args, "environment", "Enter the environment of Pinecone instance: "
    )
    set_arg_from_input(
        args,
        "index",
        "Enter the name of indexes to import (hit return to export all): ",
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
    pinecone_import = ImportPinecone(args)
    pinecone_import.upsert_data()



def main():
    """
    Import data to Pinecone using a vector dataset directory in the VDF format.
    """
    parser = argparse.ArgumentParser(
        description="Import data from VDF to a vector database"
    )
    subparsers = parser.add_subparsers(
        title="Vector Databases",
        description="Choose the vectors database to export data from",
        dest="vector_database",
    )

    # Pinecone
    parser_pinecone = subparsers.add_parser(
        "pinecone", help="Import data to Pinecone"
    )
    parser_pinecone.add_argument(
        "-d", "--dir", type=str, help="Directory to import"
    )

    args = parser.parse_args()

    if args.vector_database == "pinecone":
        ImportPinecone(args)
    else:
        print("Please choose a vector database to export data from")


if __name__ == "__main__":
    main()
