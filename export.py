from scripts.ExportPinecone import ExportPinecone
from scripts.ExportWeaviate import ExportWeaviate
from scripts.ExportQdrant import ExportQdrant

import argparse
import os
from dotenv import load_dotenv
load_dotenv()

def export_pinecone(args):
    """
    Export data from Pinecone
    """
    if args.environment is None:    
        args.environment = input("Enter the environment of Pinecone instance: ")
    if args.index is None:
        args.index = input("Enter the name of index to export: ")
    pinecone = ExportPinecone(args.environment, args.index)
    pinecone.get_data(args.index)

def export_weaviate(args):
    """
    Export data from Weaviate
    """
    if args.url is None:
        args.url = input("Enter the location of Weaviate instance: ")
    if args.class_name is None:
        args.class_name = input("Enter the name of class to export: ")
    if args.include_crossrefs is None:
        args.include_crossrefs = input("Include cross references, enter Y or N: ")
        if args.include_crossrefs == 'Y':
            args.include_crossrefs = True
        else:
            args.include_crossrefs = False
    weaviate = ExportWeaviate(args.url)
    weaviate.get_data(args.class_name, args.include_crossrefs)

def export_qdrant(args):
    """
    Export data from Qdrant
    """
    if args.url is None:
        args.url = input("Enter the location of Qdrant instance: ")
    if args.collection is None:
        args.collection = input("Enter the name of collection to export: ")
    qdrant = ExportQdrant(args.url)
    qdrant.get_data(args.collection)

def main():
    parser = argparse.ArgumentParser(description='Export data from Pinecone, Weaviate and Qdrant to sqlite database and csv file')
    subparsers = parser.add_subparsers(title='Vectors Databases', description='Choose the vectors database to export data from', dest='vectors_database')

    # Pinecone
    parser_pinecone = subparsers.add_parser('pinecone', help='Export data from Pinecone')
    parser_pinecone.add_argument('-e', '--environment', type=str, help='Environment of Pinecone instance')
    parser_pinecone.add_argument('-i','--index', type=str, help='Name of index to export')

    # Weaviate
    parser_weaviate = subparsers.add_parser('weaviate', help='Export data from Weaviate')
    parser_weaviate.add_argument('-u', '--url', type=str, help='Location of Weaviate instance')
    parser_weaviate.add_argument('-c','--class_name', type=str, help='Name of class to export')
    parser_weaviate.add_argument('-i','--include_crossrefs', type=bool, help='Include cross references, set Y or N')

    # Qdrant
    parser_qdrant = subparsers.add_parser('qdrant', help='Export data from Qdrant')
    parser_qdrant.add_argument('-u', '--url', type=str, help='Location of Qdrant instance')
    parser_qdrant.add_argument('-c','--collection', type=str, help='Name of collection to export')

    args = parser.parse_args()

    if args.vectors_database == 'pinecone':
        export_pinecone(args)
    elif args.vectors_database == 'weaviate':
        export_weaviate(args)
    elif args.vectors_database == 'qdrant':
        export_qdrant(args)
    else:
        print('Please choose a vectors database to export data from')

if __name__ == "__main__":
    main()