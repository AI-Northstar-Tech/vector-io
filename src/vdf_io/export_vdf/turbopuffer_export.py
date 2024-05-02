import argparse
import turbopuffer as tpuf
from vdf_io.util import clean_documents
from vdf_io.export_vdf.export_vdb import ExportVDB


class ExportTurbopuffer(ExportVDB):
    def make_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            "--turbopuffer-namespace",
            help="The Turbopuffer namespace to connect to",
        )
        parser.add_argument(
            "--turbopuffer-index-names",
            nargs="+",
            help="The names of the indexes to export. If not provided, all indexes will be exported.",
        )
        return parser

    def get_data(self, args):
        namespace = args.get("turbopuffer_namespace")
        if namespace is None:
            namespace = input("Enter the Turbopuffer namespace to connect to: ")

        ns = tpuf.Namespace(namespace)

        index_names = args.get("turbopuffer_index_names")
        if index_names is None:
            index_names = ns.list_indexes()
            print(f"Available indexes in namespace '{namespace}': {index_names}")
            index_names = input(
                "Enter the index names to export (comma-separated), or leave empty to export all: "
            ).split(",")
            index_names = [name.strip() for name in index_names if name.strip()]
            if not index_names:
                index_names = ns.list_indexes()

        for index_name in index_names:
            for row in ns.vectors(index_name):
                attributes = row.attributes
                clean_documents([attributes])

                if isinstance(row.vector, list):
                    for vector in row.vector:
                        yield {"id": row.id, "vector": vector, **attributes}
                else:
                    yield {"id": row.id, "vector": row.vector, **attributes}
