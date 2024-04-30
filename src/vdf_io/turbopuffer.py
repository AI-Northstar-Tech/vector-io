import argparse
import turbopuffer as tpuf
from vdf_io.names import DBNames
from vdf_io.util import standardize_metric, clean_documents


def make_parser(parser=None):
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


def export_vdb(args):
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


def import_vdb(args, input_data):
    namespace = args.get("turbopuffer_namespace")
    if namespace is None:
        namespace = input("Enter the Turbopuffer namespace to connect to: ")

    ns = tpuf.Namespace(namespace)

    distance_metric = args.get("distance_metric")
    if distance_metric is None:
        distance_metric = input(
            "Enter the distance metric to use (cosine_distance, euclidean_distance, dot_product): "
        )
    distance_metric = standardize_metric(distance_metric, DBNames.TURBOPUFFER)

    batch_size = 1000
    batch = []

    for doc in input_data:
        if len(batch) >= batch_size:
            ns.upsert(
                ids=[d["id"] for d in batch],
                vectors=[d["vector"] for d in batch],
                attributes={
                    k: [d.get(k) for d in batch]
                    for k in batch[0].keys()
                    if k not in ["id", "vector"]
                },
                distance_metric=distance_metric,
            )
            batch = []

        batch.append(doc)

    if batch:
        ns.upsert(
            ids=[d["id"] for d in batch],
            vectors=[d["vector"] for d in batch],
            attributes={
                k: [d.get(k) for d in batch]
                for k in batch[0].keys()
                if k not in ["id", "vector"]
            },
            distance_metric=distance_metric,
        )
