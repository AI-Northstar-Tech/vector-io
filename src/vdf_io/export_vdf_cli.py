#!/usr/bin/env python3

import argparse
import os
import sys
import time
import traceback
from dotenv import load_dotenv
import warnings


# Suppress specific warnings
warnings.simplefilter("ignore", ResourceWarning)
warnings.filterwarnings("ignore", module="numpy")
warnings.simplefilter("ignore", DeprecationWarning)

import sentry_sdk  # noqa: E402
from opentelemetry import trace  # noqa: E402
from opentelemetry.propagate import set_global_textmap  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from sentry_sdk.integrations.opentelemetry import (  # noqa: E402
    SentrySpanProcessor,  # noqa: E402
    SentryPropagator,  # noqa: E402
)  # noqa: E402

import vdf_io  # noqa: E402
from vdf_io.export_vdf.milvus_export import ExportMilvus  # noqa: E402
from vdf_io.export_vdf.pinecone_export import ExportPinecone  # noqa: E402
from vdf_io.export_vdf.qdrant_export import ExportQdrant  # noqa: E402
from vdf_io.export_vdf.kdbai_export import ExportKDBAI  # noqa: E402
from vdf_io.export_vdf.vertexai_vector_search_export import (  # noqa: E402
    ExportVertexAIVectorSearch,  # noqa: E402
)  # noqa: E402
from vdf_io.export_vdf.vespa_export import ExportVespa  # noqa: E402
from vdf_io.export_vdf.lancedb_export import ExportLanceDB  # noqa: E402
from vdf_io.export_vdf.txtai_export import ExportTxtai  # noqa: E402
from vdf_io.export_vdf.chroma_export import ExportChroma  # noqa: E402
from vdf_io.export_vdf.astradb_export import ExportAstraDB
from vdf_io.export_vdf.weaviate_export import ExportWeaviate
from vdf_io.names import DBNames  # noqa: E402
from vdf_io.scripts.check_for_updates import check_for_updates  # noqa: E402
from vdf_io.scripts.push_to_hub_vdf import push_to_hub  # noqa: E402


load_dotenv()

DEFAULT_MAX_FILE_SIZE = 1024  # in MB


if os.environ.get("DISABLE_TELEMETRY_VECTORIO", False) != "1":
    sentry_sdk.init(
        dsn="https://4826b78415eeaf0135c12416e222596d@o1284436.ingest.sentry.io/4506716331573248",
        enable_tracing=True,
        # set the instrumenter to use OpenTelemetry instead of Sentry
        instrumenter="otel",
        default_integrations=False,
    )


provider = TracerProvider()
provider.add_span_processor(SentrySpanProcessor())
trace.set_tracer_provider(provider)
set_global_textmap(SentryPropagator())

tracer = trace.get_tracer(__name__)


def main():
    with tracer.start_as_current_span("export_vdf_cli_main") as span:
        try:
            run_export(span)
            sentry_sdk.flush()
        except Exception as e:
            sentry_sdk.flush()
            print(f"Error: {e}")
            traceback.print_exc()
            sys.exit(1)
        finally:
            sentry_sdk.flush()
    sentry_sdk.flush()
    return


ARGS_ALLOWLIST = [
    "vector_database",
    "library_version",
    "hash_value",
    "exported_count",
]

slug_to_export_func = {
    DBNames.PINECONE: ExportPinecone.export_vdb,
    DBNames.QDRANT: ExportQdrant.export_vdb,
    DBNames.KDBAI: ExportKDBAI.export_vdb,
    DBNames.MILVUS: ExportMilvus.export_vdb,
    DBNames.VERTEXAI: ExportVertexAIVectorSearch.export_vdb,
    DBNames.VESPA: ExportVespa.export_vdb,
    DBNames.LANCEDB: ExportLanceDB.export_vdb,
    DBNames.TXTAI: ExportTxtai.export_vdb,
    DBNames.CHROMA: ExportChroma.export_vdb,
    DBNames.WEAVIATE: ExportWeaviate.export_vdb,
    DBNames.ASTRADB: ExportAstraDB.export_vdb,
}

slug_to_parser_func = {
    DBNames.PINECONE: ExportPinecone.make_parser,
    DBNames.QDRANT: ExportQdrant.make_parser,
    DBNames.KDBAI: ExportKDBAI.make_parser,
    DBNames.MILVUS: ExportMilvus.make_parser,
    DBNames.VERTEXAI: ExportVertexAIVectorSearch.make_parser,
    DBNames.VESPA: ExportVespa.make_parser,
    DBNames.LANCEDB: ExportLanceDB.make_parser,
    DBNames.TXTAI: ExportTxtai.make_parser,
    DBNames.CHROMA: ExportChroma.make_parser,
    DBNames.WEAVIATE: ExportWeaviate.make_parser,
    DBNames.ASTRADB: ExportAstraDB.make_parser,
}


def run_export(span):
    parser = argparse.ArgumentParser(
        description="Export data from various vector databases to the VDF format for vector datasets"
    )
    make_common_options(parser)
    subparsers = parser.add_subparsers(
        title="Vector Databases",
        description="Choose the vectors database to export data from",
        dest="vector_database",
    )

    db_choices = slug_to_export_func.keys()
    for db in db_choices:
        slug_to_parser_func[db](subparsers)

    args = parser.parse_args()
    # convert args to dict
    args = vars(args)
    args["library_version"] = vdf_io.__version__

    t_start = time.time()
    if (
        ("vector_database" not in args)
        or (args["vector_database"] is None)
        or (args["vector_database"] not in db_choices)
    ):
        print("Please choose a vector database to export data from:", db_choices)
        return

    if args["vector_database"] in db_choices:
        export_obj = slug_to_export_func[args["vector_database"]](args)
    else:
        print("Invalid vector database")
        args["vector_database"] = input("Enter the name of vector database to export: ")
        sys.argv.extend(["--vector_database", args["vector_database"]])
        main()
    t_end = time.time()

    for key in list(export_obj.args.keys()):
        if key in ARGS_ALLOWLIST:
            span.set_attribute(key, export_obj.args[key])

    # formatted time
    print(f"Export to disk completed. Exported to: {export_obj.vdf_directory}/")
    print(
        "Time taken to export data: ",
        time.strftime("%H:%M:%S", time.gmtime(t_end - t_start)),
    )
    span.set_attribute("export_time", t_end - t_start)
    if args["push_to_hub"]:
        push_to_hub(export_obj, args)

    check_for_updates()


def make_common_options(parser):
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
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--public",
        type=bool,
        help="Make dataset public (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )


if __name__ == "__main__":
    main()
