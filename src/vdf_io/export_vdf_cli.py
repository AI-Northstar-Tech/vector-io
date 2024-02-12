#!/usr/bin/env python3

import argparse
import os
import sys
import time
from typing import Dict
from dotenv import load_dotenv
import warnings

import sentry_sdk
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor, SentryPropagator

import vdf_io
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.export_vdf.kdbai_export import ExportKDBAI
from vdf_io.scripts.push_to_hub_vdf import push_to_hub

# Suppress specific warnings
warnings.simplefilter("ignore", ResourceWarning)

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
            sys.exit(1)
        finally:
            sentry_sdk.flush()
    sentry_sdk.flush()
    return


ARGS_ALLOWLIST = [
    "vector_database",
    "library_version",
    "hash_value",
]


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

    db_choices = [c.DB_NAME_SLUG for c in ExportVDB.__subclasses__()]
    db_class_map = {c.DB_NAME_SLUG: c for c in ExportVDB.__subclasses__()}
    for db in db_choices:
        print(db_class_map[db], type(db_class_map[db]))
        print(ExportKDBAI)
        print(dir(ExportKDBAI))
        print(ExportKDBAI.make_parser)
        # call static method make_parser of the class
        db_class_map[db].make_parser(subparsers)

        print(dir(db_class_map[db]))
        # print(db_class_map[db])

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
        export_obj = db_class_map[args["vector_database"]].export_vdb(args)
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
