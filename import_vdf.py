import argparse
import os
from dotenv import load_dotenv

# from export.vdb_export import ExportPinecone, ExportWeaviate, ExportQdrant
from getpass import getpass
from export.util import set_arg_from_input, set_arg_from_password
from import_vdf.pinecone_import import ImportPinecone
from import_vdf.qdrant_import import ImportQdrant


load_dotenv()

def import_qdrant(args):
    """
    Import data to Qdrant
    """
    set_arg_from_input(
        args, "url", "Enter the url of Qdrant instance (default: 'http://localhost:6333'): ", str, "http://localhost:6333"
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
    set_arg_from_input(
        args, "environment", "Enter the environment of Pinecone instance: "
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
    db_choices = ["pinecone", "qdrant"]  # Add "qdrant" to the list of database choices
    subparsers = parser.add_subparsers(
        title="Vector Databases",
        description="Choose the vectors database to export data from",
        dest="vector_database",
    )

    parser.add_argument("-d", "--dir", type=str, help="Directory to import")
    # Pinecone
    parser_pinecone = subparsers.add_parser("pinecone", help="Import data to Pinecone")
    parser_pinecone.add_argument("-e", "--environment", type=str, help="Pinecone environment")

    # Qdrant
    parser_qdrant = subparsers.add_parser("qdrant", help="Import data to Qdrant")
    parser_qdrant.add_argument("-u", "--url", type=str, help="Qdrant url")

    args = parser.parse_args()
    args = vars(args)
    args["library_version"] = open("VERSION.txt").read()
    set_arg_from_input(
        args, "dir", "Enter the directory of vector dataset to be imported: ", str
    )
    
    if (
        ("vector_database" not in args)
        or (args["vector_database"] is None)
        or (args["vector_database"] not in db_choices)
    ):
        print("Please choose a vector database to export data from:", db_choices)
        return
    if args["vector_database"] == "pinecone":
        import_pinecone(args)
    elif args["vector_database"] == "qdrant":
        import_qdrant(args)  # Add the function to import data to Qdrant
    else:
        print(
            "Unrecognized DB. Please choose a vector database to export data from:",
            db_choices,
        )


if __name__ == "__main__":
    main()
