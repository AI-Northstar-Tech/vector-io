#!/usr/bin/env python3

import json
import pandas as pd
import pyarrow.parquet as pq
import os
import argparse
from tqdm import tqdm
from pyarrow import Table
from collections import defaultdict


def get_file_size_in_gb(file_path):
    """
    Calculate the size of a file in gigabytes.

    Args:
        file_path (str): The path to the file.

    Returns:
        float: The size of the file in gigabytes.
    """
    return os.path.getsize(file_path) / (1024 * 1024 * 1024)  # Convert bytes to GB


def main():
    """
    Consolidate multiple Parquet files into combined files based on a maximum size.

    Example command:
    python consolidate_parquet.py /path/to/parquet_files /path/to/output_directory 2
    """
    # Directory containing your parquet files
    parser = argparse.ArgumentParser(description="Consolidate Parquet files")
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing your parquet files",
        default=None,
    )
    parser.add_argument(
        "--max_size_gb",
        type=int,
        help="Maximum size in GB for each combined file",
        default=0.5,
    )
    args = parser.parse_args()

    directory = args.directory
    output_directory = args.directory
    max_size_gb = args.max_size_gb

    # open VDF_META.json from parent of directory
    vdf_meta_path = os.path.join(directory, "..", "VDF_META.json")
    with open(vdf_meta_path, "r") as f:
        vdf_meta = json.load(f)

    # Initialize variables
    pqwriter = None
    file_count = 1

    # update file_structure by replacing
    old_file_structure = vdf_meta["file_structure"]
    new_file_set = set()
    # add metadata to file_structure
    old_files = os.listdir(directory)
    all_columns = set()
    for filename in tqdm(old_files):
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            # remove filepath from old_file_structure which has last segment matching filename
            old_file_structure = [
                x for x in old_file_structure if not x.endswith("/" + filename)
            ]
            # Read the parquet file as a PyArrow table
            table = pq.read_table(filepath)
            all_columns.update(table.column_names)

    for filename in tqdm(old_files):
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            # Read the parquet file as a PyArrow table
            table = pq.read_table(filepath)
            # Rearrange the table columns according to the union of all columns
            data = defaultdict(list)
            for column in all_columns:
                if column in table.column_names:
                    data[column] = table[column]
                else:
                    data[column] = [None] * len(table)
            table = Table.from_pandas(pd.DataFrame(data))

            if pqwriter is None:
                output_file = os.path.join(
                    output_directory, f"combined_file_{file_count}.parquet"
                )
                pqwriter = pq.ParquetWriter(output_file, table.schema)
            else:
                # If the schema of the new table is different, close the current writer and start a new file
                if table.schema != pqwriter.schema:
                    new_file_set.add(output_file)
                    pqwriter.close()
                    file_count += 1
                    output_file = os.path.join(
                        output_directory, f"combined_file_{file_count}.parquet"
                    )
                    pqwriter = pq.ParquetWriter(output_file, table.schema)

            # Write the table to the output file
            pqwriter.write_table(table)

            # Check if the current output file size is approaching the maximum size
            if get_file_size_in_gb(output_file) >= max_size_gb:
                # Close the current writer and start a new file
                new_file_set.add(output_file)
                pqwriter.close()
                pqwriter = None
                file_count += 1
    # delete old files
    # Close the last parquet writer if it's open
    if pqwriter:
        new_file_set.add(output_file)
        pqwriter.close()
    print("written files", new_file_set)
    print("deleting old files", old_files)
    for filename in old_files:
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            os.remove(filepath)
    with open(vdf_meta_path, "w") as f:
        vdf_meta["file_structure"] = old_file_structure
        f.seek(0)
        json.dump(vdf_meta, f, indent=4)


if __name__ == "__main__":
    main()
