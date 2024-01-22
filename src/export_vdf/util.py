from pathlib import Path
from collections import OrderedDict
from getpass import getpass
import hashlib
import json
import os
from qdrant_client.http.models import Distance


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
        except Exception as e:
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
    if (
        arg_name not in args
        or args[arg_name] is None
        or args[arg_name] == default_value
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
    # Expand '~' to the user's home directory
    expanded_path = os.path.expanduser(shorthand_path)

    # Resolve '.' and '..' to get the absolute path
    full_path = Path(expanded_path).resolve()

    return str(full_path)


db_metric_to_standard_metric = {
    "pinecone": {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dotproduct": Distance.DOT,
    },
    "qdrant": {
        Distance.COSINE: Distance.COSINE,
        Distance.EUCLID: Distance.EUCLID,
        Distance.DOT: Distance.DOT,
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
