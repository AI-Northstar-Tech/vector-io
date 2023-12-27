import argparse
import os
from dotenv import load_dotenv
# from export.vdb_export import ExportPinecone, ExportWeaviate, ExportQdrant
from getpass import getpass
from import_VDF.pinecone_import import ImportPinecone


load_dotenv()


def set_arg_from_input(arg_name, prompt):
    """
    Set the value of an argument from user input if it is not already present
    """
    if arg_name is None:
        arg_name = input(prompt)
    return arg_name


def import_pinecone(args):
    """
    Export data from Pinecone
    """
    args.environment = set_arg_from_input(
        args.environment, "Enter the environment of Pinecone instance: "
    )
    args.index = set_arg_from_input(args.index, "Enter the name of index to export, or type all to get all indexes: ")
    args.pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_import = ImportPinecone(args)
    # if args.index == "all":
    #     index_names = pinecone.get_all_index_names()
    #     for index_name in index_names:
    #         pinecone.insert_data(index_name)
    pinecone_import.upsert_data()



def main():
    """
    Import data to Pinecone using a dir in a VDF format of an sqlite database and parquet file.

    Usage:
        python import.py <vector_database> [options]

    Arguments:
        vector_database (str): Choose the vectors database to export data from.
            Possible values: "pinecone", "weaviate", "qdrant".

    Options:
        Pinecone:
            -e, --environment (str): Environment of Pinecone instance.
            -i, --index (str): Name of index to export.

        Weaviate:
            -u, --url (str): Location of Weaviate instance.
            -c, --class_name (str): Name of class to export.
            -i, --include_crossrefs (bool): Include cross references, set Y or N.

        Qdrant:
            -u, --url (str): Location of Qdrant instance.
            -c, --collection (str): Name of collection to export.

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
        "-e", "--environment", type=str, help="Environment of Pinecone instance"
    )
    parser_pinecone.add_argument(
        "-i", "--index", type=str, help="Name of index to import to"
    )
    parser_pinecone.add_argument(
        "-d", "--dir", type=str, help="Directory to import"
    )

    args = parser.parse_args()

    if args.vector_database == "pinecone":
        import_pinecone(args)
    else:
        print("Please choose a vector database to export data from")


if __name__ == "__main__":
    main()
