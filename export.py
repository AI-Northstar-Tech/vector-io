import argparse
import os
from dotenv import load_dotenv
from export.vdb_export import ExportPinecone, ExportWeaviate, ExportQdrant
from getpass import getpass

load_dotenv()


def set_arg_from_input(arg_name, prompt):
    """
    Set the value of an argument from user input if it is not already present
    """
    if arg_name is None:
        arg_name = input(prompt)
    return arg_name


def export_pinecone(args):
    """
    Export data from Pinecone
    """
    args.environment = set_arg_from_input(
        args.environment, "Enter the environment of Pinecone instance: "
    )
    args.index = set_arg_from_input(args.index, "Enter the name of index to export, or type all to get all indexes: ")
    args.pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone = ExportPinecone(args)
    if args.index == "all":
        index_names = pinecone.get_all_index_names()
        for index_name in index_names:
            pinecone.get_data(index_name)
    pinecone.get_data(args.index)


def export_weaviate(args):
    """
    Export data from Weaviate
    """
    set_arg_from_input(args.url, "Enter the location of Weaviate instance: ")
    set_arg_from_input(args.class_name, "Enter the name of class to export, or type all to export all classes: ")
    if args.include_crossrefs is None:
        args.include_crossrefs = input("Include cross references, enter Y or N: ")
        if args.include_crossrefs == "Y":
            args.include_crossrefs = True
        else:
            args.include_crossrefs = False
    if args.class_name == "all":
        weaviate = ExportWeaviate(args)
        class_names = weaviate.get_all_class_names()
        for class_name in class_names:
            weaviate.get_data(class_name, args.include_crossrefs)
    else:
        weaviate = ExportWeaviate(args)
        weaviate.get_data(args.class_name, args.include_crossrefs)


def export_qdrant(args):
    """
    Export data from Qdrant
    """
    set_arg_from_input(args.url, "Enter the location of Qdrant instance: ")
    set_arg_from_input(args.collection, "Enter the name of collection to export, or type all to export all collections: ")
    qdrant = ExportQdrant(args)
    if args.collection == "all":
        collection_names = qdrant.get_all_collection_names()
        for collection_name in collection_names:
            qdrant.get_data(collection_name)
    else:
        qdrant.get_data(args.collection)


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
        "pinecone", help="Export data from Pinecone"
    )
    parser_pinecone.add_argument(
        "-e", "--environment", type=str, help="Environment of Pinecone instance"
    )
    parser_pinecone.add_argument(
        "-i", "--index", type=str, help="Name of index to export"
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
        "-c", "--collection", type=str, help="Name of collection to export"
    )

    args = parser.parse_args()

    if args.vector_database == "pinecone":
        export_pinecone(args)
    elif args.vector_database == "weaviate":
        export_weaviate(args)
    elif args.vector_database == "qdrant":
        export_qdrant(args)
    else:
        print("Please choose a vector database to export data from")


if __name__ == "__main__":
    main()
