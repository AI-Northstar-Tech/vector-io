#!/usr/bin/env python3

import os
import pandas as pd

from vdf_io.util import expand_shorthand_path

# script to get list of ids from directory of parquet files


def get_ids_from_parquet(directory):
    # for os.walk
    ids = set()
    for root, dirs, files in os.walk(expand_shorthand_path(dir)):
        # for each file in the directory
        for file in files:
            # if the file ends with .parquet
            if file.endswith(".parquet"):
                # get the path to the file
                path = os.path.join(root, file)
                # open the file
                # read only the id column
                try:
                    # Read only the 'id' column from the parquet file
                    df = pd.read_parquet(path, columns=["id"])
                    ids.update(df["id"].tolist())
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    return ids


dir = input("Enter directory of parquet files: ")
ids = get_ids_from_parquet(dir)
print(f"Found {len(ids)} ids")
print("Writing ids to ids.csv")
pd.DataFrame(list(ids)).sort_values(0).to_csv("ids.csv", index=False, header=False)

# if they are numeric ids, find the missing ids
if all([isinstance(i, int) for i in ids]):
    print("Finding missing ids")
    ids = set(ids)
    missing_ids = [i for i in range(1, max(ids)) if i not in ids]
    print(f"Found {len(missing_ids)} missing ids")
    print("Writing missing ids to missing_ids.csv")
    pd.DataFrame(missing_ids).to_csv("missing_ids.csv", index=False, header=False)
print("Done")
