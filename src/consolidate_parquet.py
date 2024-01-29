#!/usr/bin/env python3

import json
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
    python consolidate_parquet.py /path/to/parquet_files /path/to/output_directory 2
    """
    # Directory containing your parquet files
    parser = argparse.ArgumentParser(description="Consolidate Parquet files")
    parser.add_argument(
        "directory", type=str, help="Directory containing your parquet files"
    )
    parser.add_argument(
        "output_directory", type=str, help="Directory to save the combined files"
    )
    parser.add_argument(
        "max_size_gb", type=int, help="Maximum size in GB for each combined file"
    )
    args = parser.parse_args()

    directory = args.directory
    output_directory = args.output_directory
    max_size_gb = args.max_size_gb

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    
    # open VDF_META.json from parent of directory
    vdf_meta_path = os.path.join(directory, "..", "VDF_META.json")
    with open(vdf_meta_path, "r") as f:
        vdf_meta = json.load(f)
    
    # Initialize variables
    pqwriter = None
    file_count = 1

    # update file_structure by replacing 
    old_file_structure = vdf_meta["file_structure"]
    # add metadata to file_structure
    for filename in os.listdir(directory):
        if filename.endswith(".parquet"):
            filepath = os.path.join(directory, filename)

            # Read the parquet file as a PyArrow table
            table = pq.read_table(filepath)

            if pqwriter is None:
                output_file = os.path.join(
                    output_directory, f"combined_file_{file_count}.parquet"
                )
                pqwriter = pq.ParquetWriter(output_file, table.schema)

            # Write the table to the output file
            pqwriter.write_table(table)

            # Check if the current output file size is approaching the maximum size
            if get_file_size_in_gb(output_file) >= max_size_gb:
                # Close the current writer and start a new file
                pqwriter.close()
                pqwriter = None
                file_count += 1

    # Close the last parquet writer if it's open
    if pqwriter:
        pqwriter.close()


if __name__ == "__main__":
    main()
