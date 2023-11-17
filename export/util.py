import hashlib
import json


def sort_recursive(d):
    """
    Recursively sort the nested dictionary by its keys.
    """
    if not isinstance(d, dict):
        return d
    
    sorted_dict = {}
    for key, value in sorted(d.items()):
        sorted_dict[key] = sort_recursive(value)
    
    return sorted_dict

def convert_to_consistent_value(d):
    """
    Convert a nested dictionary to a consistent string regardless of key order.
    """
    sorted_dict = sort_recursive(d)
    return json.dumps(sorted_dict, sort_keys=True)

file_path_keys = [
    "file1_path",
    "file2_path",
    "sttm_file",
    "ground_truth_file",
    "arg_file",
]

def extract_data_hash(arg_dict_combined):
    return 'ABCD'
    # TODO: fix this
    arg_dict_combined_copy = arg_dict_combined.copy()
    data_hash = hashlib.md5(convert_to_consistent_value(arg_dict_combined_copy).encode("utf-8"))
    # make it 5 characters long
    data_hash = data_hash.hexdigest()[:5]
    return data_hash

