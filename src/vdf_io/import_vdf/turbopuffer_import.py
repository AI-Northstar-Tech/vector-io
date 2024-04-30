import argparse
import turbopuffer as tpuf
from vdf_io.names import DBNames
from vdf_io.util import standardize_metric


class ImportTurbopuffer:
    def make_parser(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        parser.add_argument(
            "--turbopuffer-namespace",
            help="The Turbopuffer namespace to connect to",
        )
        return parser

    def import_vdb(self, args, input_data):
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