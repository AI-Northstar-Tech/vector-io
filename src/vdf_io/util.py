from pathlib import Path
from collections import OrderedDict
from getpass import getpass
import hashlib
import json
import os
from uuid import UUID
import pandas as pd
from io import StringIO
import sys
from tqdm import tqdm
from PIL import Image

from qdrant_client.http.models import Distance

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


def set_arg_from_input(args, arg_name, prompt, type_name=str, default_value=None):
    """
    Set the value of an argument from user input if it is not already present
    """
    if arg_name not in args or (
        args[arg_name] is None and default_value != "DO_NOT_PROMPT"
    ):
        inp = input(prompt)
        if inp == "":
            args[arg_name] = None if default_value is None else type_name(default_value)
        else:
            args[arg_name] = type_name(inp)
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


def get_parquet_files(data_path, args, temp_file_paths=[]):
    # Load the data from the parquet files
    if args.get("hf_dataset", None):
        if args.get("max_num_rows", None):
            from datasets import load_dataset

            tqdm.write("Loading a subset of the dataset")
            ds = load_dataset(args.get("hf_dataset"), split="train", streaming=True)
            tqdm.write("Taking a subset of the dataset")
            it_ds = ds.take(args.get("max_num_rows"))
            tqdm.write("Converting to pandas dataframe")
            df = pd.DataFrame(it_ds)
            tqdm.write("Writing to parquet")
            df = cleanup_df(df)
            temp_file_path = f"{os.getcwd()}/temp.parquet"
            df.to_parquet(temp_file_path)
            temp_file_paths.append(temp_file_path)
            tqdm.write("Writing complete")
            return [temp_file_path]
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
        parquet_files = sorted(
            [file for file in os.listdir(data_path) if file.endswith(".parquet")]
        )
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


def read_parquet_progress(file_path, **kwargs):
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
        df = pd.read_parquet(cache_path, **kwargs)
    else:
        df = pd.read_parquet(file_path, **kwargs)
    return df
