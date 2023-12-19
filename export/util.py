from collections import OrderedDict
import hashlib
import json


def sort_recursive(d):
    """
    Recursively sort the nested dictionary by its keys.
    """
    print("t", type(d), d)
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
        d = dict(d)

    sorted_dict = OrderedDict()
    for key, value in sorted(d.items()):
        sorted_dict[key] = sort_recursive(value)

    return sorted_dict


def convert_to_consistent_value(d):
    """
    Convert a nested dictionary to a consistent string regardless of key order.
    """
    print(type(d), d)
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
