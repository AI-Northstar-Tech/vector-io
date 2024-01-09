#!/usr/bin/env python

from pathlib import Path
import os
import sys
import json
import yaml

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



print(yaml.dump(json.load(open(expand_shorthand_path(sys.argv[1]))), default_flow_style=False))