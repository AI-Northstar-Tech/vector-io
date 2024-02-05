#!/usr/bin/env python3

import pyarrow.parquet as pq
import os
import argparse


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
    python count_rows.py /path/to/parquet_files
    """
    # Directory containing your parquet files
    parser = argparse.ArgumentParser(description="Consolidate Parquet files")
    parser.add_argument(
        "--directory",
        type=str,
        help="Directory containing your parquet files",
        default=None,
    )

    args = parser.parse_args()
    directory = args.directory
    cnt = 0
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)
            table = pq.read_table(filepath)
            cnt += table.num_rows
    print("row count", cnt)


if __name__ == "__main__":
    main()
