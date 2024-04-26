from pathlib import Path
from collections import OrderedDict
from getpass import getpass
import hashlib
import json
import os
import time
from uuid import UUID
import numpy as np
import pandas as pd
from io import StringIO
import sys
from tqdm import tqdm
from PIL import Image
from halo import Halo

from qdrant_client.http.models import Distance

from vdf_io.constants import ID_COLUMN, INT_MAX
from vdf_io.names import DBNames


def sort_recursive(d):
    """
    Recursively sort the nested dictionary by its keys.
    """
    # if isinstance(d, list):
    #     return [sort_recursive(v) for v in d]
    # if isinstance(d, tuple):
    #     return tuple(sort_recursive(v) for v in d)
    # if isinstance(d, set):
    #     return list({sort_recursive(v) for v in d}).sort()
    if (
        isinstance(d, str)
        or isinstance(d, int)
        or isinstance(d, float)
        or isinstance(d, bool)
        or d is None
        or isinstance(d, OrderedDict)
    ):
        return d
    if hasattr(d, "attribute_map"):
        return sort_recursive(d.attribute_map)
    if not isinstance(d, dict):
        try:
            d = dict(d)
        except Exception:
            d = {"": str(d)}

    sorted_dict = OrderedDict()
    for key, value in sorted(d.items()):
        sorted_dict[key] = sort_recursive(value)

    return sorted_dict


def convert_to_consistent_value(d):
    """
    Convert a nested dictionary to a consistent string regardless of key order.
    """
    sorted_dict = sort_recursive(d)
    return json.dumps(sorted_dict, sort_keys=True)


def extract_data_hash(arg_dict_combined):
    arg_dict_combined_copy = arg_dict_combined.copy()
    data_hash = hashlib.md5(
        convert_to_consistent_value(arg_dict_combined_copy).encode("utf-8")
    )
    # make it 5 characters long
    data_hash = data_hash.hexdigest()[:5]
    return data_hash


def extract_numerical_hash(string_value):
    """
    Extract a numerical hash from a string
    """
    return int(hashlib.md5(string_value.encode("utf-8")).hexdigest(), 16)


def set_arg_from_input(
    args,
    arg_name,
    prompt,
    type_name=str,
    default_value=None,
    choices=None,
    env_var=None,
):
    """
    Set the value of an argument from user input if it is not already present
    """
    if (
        (default_value is None)
        and (env_var is not None)
        and (os.getenv(env_var) is not None)
    ):
        default_value = os.getenv(env_var)
    if arg_name not in args or (
        args[arg_name] is None and default_value != "DO_NOT_PROMPT"
    ):
        while True:
            inp = input(
                prompt
                + (" " + str(list(choices)) + ": " if choices is not None else "")
            )
            if len(inp) >= 2:
                if inp[0] == '"' and inp[-1] == '"':
                    inp = inp[1:-1]
                elif inp[0] == "'" and inp[-1] == "'":
                    inp = inp[1:-1]
            if inp == "":
                args[arg_name] = (
                    None if default_value is None else type_name(default_value)
                )
                break
            elif choices is not None and not all(
                choice in choices for choice in inp.split(",")
            ):
                print(f"Invalid input. Please choose from {choices}")
                continue
            else:
                args[arg_name] = type_name(inp)
                break
    return


def set_arg_from_password(args, arg_name, prompt, env_var_name):
    """
    Set the value of an argument from user input if it is not already present
    """
    if os.getenv(env_var_name) is not None:
        args[arg_name] = os.getenv(env_var_name)
    elif arg_name not in args or args[arg_name] is None:
        args[arg_name] = getpass(prompt)
    return


def expand_shorthand_path(shorthand_path):
    """
    Expand shorthand notations in a file path to a full path-like object.

    :param shorthand_path: A string representing the shorthand path.
    :return: A Path object representing the full path.
    """
    if shorthand_path is None:
        return None
    # Expand '~' to the user's home directory
    expanded_path = os.path.expanduser(shorthand_path)

    # Resolve '.' and '..' to get the absolute path
    full_path = Path(expanded_path).resolve()

    return str(full_path)


db_metric_to_standard_metric = {
    DBNames.PINECONE: {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dotproduct": Distance.DOT,
    },
    DBNames.QDRANT: {
        Distance.COSINE: Distance.COSINE,
        Distance.EUCLID: Distance.EUCLID,
        Distance.DOT: Distance.DOT,
        Distance.MANHATTAN: Distance.MANHATTAN,
    },
    DBNames.MILVUS: {
        "COSINE": Distance.COSINE,
        "IP": Distance.DOT,
        "L2": Distance.EUCLID,
    },
    DBNames.KDBAI: {
        "L2": Distance.EUCLID,
        "CS": Distance.COSINE,
        "IP": Distance.DOT,
    },
    DBNames.VERTEXAI: {
        "DOT_PRODUCT_DISTANCE": Distance.DOT,
        "SQUARED_L2_DISTANCE": Distance.EUCLID,
        "COSINE_DISTANCE": Distance.COSINE,
        "L1_DISTANCE": Distance.MANHATTAN,
    },
    DBNames.LANCEDB: {
        "L2": Distance.EUCLID,
        "Cosine": Distance.COSINE,
        "Dot": Distance.DOT,
    },
    DBNames.CHROMA: {
        "l2": Distance.EUCLID,
        "cosine": Distance.COSINE,
        "ip": Distance.DOT,
    },
    DBNames.ASTRADB: {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dot_product": Distance.DOT,
    },
    DBNames.WEAVIATE: {
        "cosine": Distance.COSINE,
        "l2-squared": Distance.EUCLID,
        "dot": Distance.DOT,
        "manhattan": Distance.MANHATTAN,
    },
    DBNames.VESPA: {
        "angular": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dotproduct": Distance.DOT,
    },
}


def standardize_metric(metric, db):
    """
    Standardize the metric name to the one used in the standard library.
    """
    if (
        db in db_metric_to_standard_metric
        and metric in db_metric_to_standard_metric[db]
    ):
        return db_metric_to_standard_metric[db][metric]
    else:
        raise Exception(f"Invalid metric '{metric}' for database '{db}'")


def standardize_metric_reverse(metric, db):
    """
    Standardize the metric name to the one used in the standard library.
    """
    if (
        db in db_metric_to_standard_metric
        and metric in db_metric_to_standard_metric[db].values()
    ):
        for key, value in db_metric_to_standard_metric[db].items():
            if value == metric:
                return key
    else:
        raise Exception(f"Invalid metric '{metric}' for database '{db}'")


def get_final_data_path(cwd, dir, data_path, args):
    if args.get("hf_dataset", None):
        return data_path
    final_data_path = os.path.join(cwd, dir, data_path)
    if not os.path.isdir(final_data_path):
        raise Exception(
            f"Invalid data path\n"
            f"data_path: {data_path},\n"
            f"Joined path: {final_data_path}\n"
            f"Current working directory: {cwd}\n"
            f"Command line arg (dir): {dir}"
        )
    return final_data_path


def list_configs_and_splits(name):
    if "HUGGING_FACE_TOKEN" not in os.environ:
        yield "train", None
    import requests

    headers = {"Authorization": f"Bearer {os.environ['HUGGING_FACE_TOKEN']}"}
    API_URL = f"https://datasets-server.huggingface.co/splits?dataset={name}"

    def query():
        response = requests.get(API_URL, headers=headers)
        return response.json()

    data = query()
    if "splits" in data:
        for split in data["splits"]:
            if "config" in split:
                yield split["split"], split["config"]
            else:
                yield split["split"], None
    else:
        yield "train", None


def get_parquet_files(data_path, args, temp_file_paths=[], id_column=ID_COLUMN):
    # Load the data from the parquet files
    if args.get("hf_dataset", None):
        if args.get("max_num_rows", None):
            from datasets import load_dataset

            total_rows_loaded = 0
            for i, (split, config) in enumerate(
                list_configs_and_splits(args.get("hf_dataset"))
            ):
                tqdm.write(f"Split: {split}, Config: {config}")
                ds = load_dataset(
                    args.get("hf_dataset"), name=config, split=split, streaming=True
                )
                with Halo(text="Taking a subset of the dataset", spinner="dots"):
                    it_ds = ds.take(args.get("max_num_rows") - total_rows_loaded)
                start_time = time.time()
                with Halo(
                    text="Converting to pandas dataframe (this may take a while)",
                    spinner="dots",
                ):
                    df = pd.DataFrame(it_ds)
                end_time = time.time()
                tqdm.write(
                    f"Time taken to convert to pandas dataframe: {end_time - start_time:.2f} seconds"
                )
                df = cleanup_df(df)
                if id_column not in df.columns:
                    # remove all rows
                    tqdm.write(
                        (
                            f"ID column '{id_column}' not found in parquet file '{data_path}'."
                            f" Skipping split '{split}', config '{config}'."
                        )
                    )
                    continue
                total_rows_loaded += len(df)
                temp_file_path = f"{os.getcwd()}/temp_{args['hash_value']}_{i}.parquet"
                with Halo(text="Saving to parquet", spinner="dots"):
                    df.to_parquet(temp_file_path)
                temp_file_paths.append(temp_file_path)
                if total_rows_loaded >= args.get("max_num_rows"):
                    break
            return temp_file_paths
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        return [
            "hf://" + x
            for x in fs.glob(
                f"datasets/{args.get('hf_dataset')}/{data_path if data_path!='.' else ''}/**.parquet"
            )
        ]
    if not os.path.isdir(data_path):
        if data_path.endswith(".parquet"):
            return [data_path]
        else:
            raise Exception(f"Invalid data path '{data_path}'")
    else:
        # recursively find all parquet files (it should be a file acc to OS)
        parquet_files = []
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".parquet"):
                    parquet_files.append(os.path.join(root, file))
        return parquet_files


def cleanup_df(df):
    for col in df.columns:
        if df[col].dtype == "object":
            first_el = df[col].iloc[0]
            # if isinstance(first_el, bytes):
            #     df[col] = df[col].apply(lambda x: x.decode("utf-8"))
            if isinstance(first_el, Image.Image):
                # delete the image column
                df = df.drop(columns=[col])
                tqdm.write(
                    f"Warning: Image column '{col}' detected. Image columns are not supported in parquet files. The column has been removed."
                )
        # replace NaT with start of epoch
        if df[col].dtype == "datetime64[ns]":
            df[col] = df[col].fillna(pd.Timestamp(0))

    # for float columns, replace inf with nan
    numeric_cols = df.select_dtypes(include=[np.number])
    df[numeric_cols.columns] = numeric_cols.map(lambda x: np.nan if np.isinf(x) else x)

    return df


# Function to recursively print help messages
def print_help_recursively(parser, level=0):
    # Temporarily redirect stdout to capture the help message
    old_stdout = sys.stdout
    sys.stdout = StringIO()

    # Print the current parser's help message
    parser.print_help()

    # Retrieve and print the help message from the StringIO object
    help_message = sys.stdout.getvalue()
    sys.stdout = old_stdout  # Restore stdout

    # Print the captured help message with indentation for readability
    print("\n" + "\t" * level + "Help message for level " + str(level) + ":")
    for line in help_message.split("\n"):
        print("\t" * level + line)

    # Check if the current parser has subparsers
    if hasattr(parser, "_subparsers"):
        for _, subparser in parser._subparsers._group_actions[0].choices.items():
            # Recursively print help for each subparser
            print_help_recursively(subparser, level + 1)


def is_str_uuid(id_str):
    try:
        uuid_obj = UUID(id_str)
        return str(uuid_obj)
    except ValueError:
        return False


def get_qdrant_id_from_id(idx):
    if isinstance(idx, int) or idx.isdigit():
        return int(idx)
    elif not is_str_uuid(idx):
        hex_string = hashlib.md5(idx.encode("UTF-8")).hexdigest()
        return str(UUID(hex=hex_string))
    else:
        return str(UUID(idx))


def read_parquet_progress(file_path, id_column, **kwargs):
    if file_path.startswith("hf://"):
        from huggingface_hub import HfFileSystem
        from huggingface_hub import hf_hub_download

        fs = HfFileSystem()
        resolved_path = fs.resolve_path(file_path)
        cache_path = hf_hub_download(
            repo_id=resolved_path.repo_id,
            filename=resolved_path.path_in_repo,
            repo_type=resolved_path.repo_type,
        )
        file_path_to_be_read = cache_path
    else:
        file_path = os.path.abspath(file_path)
        file_path_to_be_read = file_path
    # read schema of the parquet file to check if columns are present
    from pyarrow import parquet as pq

    schema = pq.read_schema(file_path_to_be_read)
    # list columns
    columns = schema.names
    # if kwargs has columns, check if all columns are present
    cols = set()
    cols.add(id_column)
    return_empty = False
    if "columns" in kwargs:
        for col in kwargs["columns"]:
            cols.add(col)
            if col not in columns:
                tqdm.write(
                    f"Column '{col}' not found in parquet file '{file_path_to_be_read}'. Returning empty DataFrame."
                )
                return_empty = True
    if return_empty:
        return pd.DataFrame(columns=list(cols))
    with Halo(text=f"Reading parquet file {file_path_to_be_read}", spinner="dots"):
        if (
            "max_num_rows" in kwargs
            and (kwargs.get("max_num_rows", INT_MAX) or INT_MAX) < INT_MAX
        ):
            from pyarrow.parquet import ParquetFile
            import pyarrow as pa

            pf = ParquetFile(file_path_to_be_read)
            first_ten_rows = next(pf.iter_batches(batch_size=kwargs["max_num_rows"]))
            df = pa.Table.from_batches([first_ten_rows]).to_pandas()
        else:
            df = pd.read_parquet(file_path_to_be_read)
    tqdm.write(f"{file_path_to_be_read} read successfully. {len(df)=} rows")
    return df


def get_author_name():
    return (os.environ.get("USER", os.environ.get("USERNAME"))) or "unknown"


def clean_value(v):
    if hasattr(v, "__iter__") and not isinstance(v, str):
        if any(pd.isna(x) for x in v):
            return [None if pd.isna(x) else x for x in v]
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.datetime64) and np.isnat(v):
        return None
    if not hasattr(v, "__iter__") and pd.isna(v):
        return None
    return v


def clean_documents(documents):
    for doc in documents:
        to_be_replaced = []
        for k, v in doc.items():
            doc[k] = clean_value(v)
            # if k doesn't conform to CQL standards, replace it
            # like spaces
            if " " in k:
                to_be_replaced.append(k)
        for k in to_be_replaced:
            doc[k.replace(" ", "_")] = doc.pop(k)


def divide_into_batches(df, batch_size):
    """
    Divide the dataframe into batches of size batch_size
    """
    for i in range(0, len(df), batch_size):
        yield df[i : i + batch_size]
