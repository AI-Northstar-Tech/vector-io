#!/usr/bin/env python3

import argparse
import importlib
import os
import pkgutil
import time
import warnings
from dotenv import find_dotenv, load_dotenv
import traceback

import sentry_sdk
from opentelemetry import trace
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.trace import TracerProvider
from sentry_sdk.integrations.opentelemetry import SentrySpanProcessor, SentryPropagator

import vdf_io
from vdf_io.constants import ID_COLUMN
from vdf_io.scripts.check_for_updates import check_for_updates
from vdf_io.util import set_arg_from_input
from vdf_io.import_vdf.vdf_import_cls import ImportVDB

# Path to the directory containing all export modules
package_dir = "vdf_io.import_vdf"


def load_subclasses(package_dir):
    slug_to_import_func = {}
    slug_to_parser_func = {}
    # Dynamically import all modules in the specified package
    package = importlib.import_module(package_dir)
    for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
        if not is_pkg:
            # Import the module
            module = importlib.import_module(f"{package_dir}.{module_name}")
            # Check each class in the module
            for name, cls in module.__dict__.items():
                if (
                    isinstance(cls, type)
                    and issubclass(cls, ImportVDB)
                    and cls is not ImportVDB
                ):
                    # Assign functions based on the class
                    slug_to_import_func[cls.DB_NAME_SLUG] = cls.import_vdb
                    slug_to_parser_func[cls.DB_NAME_SLUG] = cls.make_parser
    return slug_to_import_func, slug_to_parser_func


slug_to_import_func, slug_to_parser_func = load_subclasses(package_dir)

warnings.filterwarnings("ignore", module="numpy")

load_dotenv(find_dotenv(), override=True)

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
    """
    Import data to Pinecone using a vector dataset directory in the VDF format.
    """
    with tracer.start_as_current_span("import_vdf_cli_main") as span:
        try:
            run_import(span)
            sentry_sdk.flush()
        except Exception as e:
            sentry_sdk.flush()
            print(f"Error: {e}")
            traceback.print_exc()
            return
        finally:
            sentry_sdk.flush()
    sentry_sdk.flush()
    return


def add_subparsers_for_dbs(subparsers, slugs):
    for slug in slugs:
        parser_func = slug_to_parser_func[slug]
        parser_func(subparsers)


def run_import(span):
    parser = argparse.ArgumentParser(
        description="Import data from VDF to a vector database"
    )
    # list of all subclasses of ImportVDB
    db_choices = [c.DB_NAME_SLUG for c in ImportVDB.__subclasses__()]
    subparsers = parser.add_subparsers(
        title="Vector Databases",
        description="Choose the vectors database to import data from",
        dest="vector_database",
    )

    make_common_options(parser)
    add_subparsers_for_dbs(subparsers, db_choices)

    args = parser.parse_args()
    args = vars(args)
    args["library_version"] = vdf_io.__version__
    if args.get("hf_dataset") is None:
        set_arg_from_input(
            args, "dir", "Enter the directory of vector dataset to be imported: ", str
        )
    if args["subset"]:
        set_arg_from_input(
            args,
            "max_num_rows",
            "Maximum number of vectors you'd like to load",
            int,
        )

    args["cwd"] = os.getcwd()

    start_time = time.time()

    if (
        ("vector_database" not in args)
        or (args["vector_database"] is None)
        or (args["vector_database"] not in db_choices)
    ):
        print("Please choose a vector database to import data from:", db_choices)
        return
    import_obj = slug_to_import_func[args["vector_database"]](args)

    end_time = time.time()
    ARGS_ALLOWLIST = [
        "vector_database",
        "library_version",
        "hash_value",
        "imported_count",
    ]

    for key in list(import_obj.args.keys()):
        if key in ARGS_ALLOWLIST:
            span.set_attribute(key, import_obj.args[key])

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    span.set_attribute("import_time", end_time - start_time)
    import_obj.cleanup()
    check_for_updates()


def make_common_options(parser):
    parser.add_argument("-d", "--dir", type=str, help="Directory to import")
    parser.add_argument(
        "--hf_dataset",
        type=str,
        help="Hugging Face dataset name eg: 'aintech/vdf_20240125_130746_ac5a6_medium_articles'",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=bool,
        help="Import a subset of data (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--max_num_rows",
        type=int,
        help="Maximum number of rows you'd like to load",
    )
    parser.add_argument(
        "--create_new",
        type=bool,
        help="Create a new index (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--vector_columns",
        type=str,
        help="Vector column names (comma separated) eg: 'vector1,vector2'",
        default="vector",
    )
    parser.add_argument(
        "--metric",
        type=str,
        help="Distance metric to use (default: 'Cosine')",
        default="Cosine",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        help="ID column name (default: 'id')",
        default=ID_COLUMN,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for import (default: based on DB)",
    )


if __name__ == "__main__":
    main()
