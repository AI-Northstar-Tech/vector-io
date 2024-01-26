#!/usr/bin/env python3

import argparse
import json
import os
import litellm
from tqdm import tqdm
import pandas as pd
from dotenv import load_dotenv
import sys
from IPython.core import ultratb
import warnings

from util import (
    get_final_data_path,
    get_parquet_files,
    set_arg_from_input,
    set_arg_from_password,
)


warnings.filterwarnings("ignore", module="litellm")
sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Reembed a vector dataset")

    parser.add_argument(
        "-d",
        "--dir",
        type=str,
        help="Directory of vector dataset in the VDF format",
        required=True,
    )

    parser.add_argument(
        "-m",
        "--new_model_name",
        type=str,
        help="Name of new model to be used",
    )

    parser.add_argument(
        "-t",
        "--text_column",
        type=str,
        help="Name of the column containing text to be embedded",
        default="text",
    )

    args = parser.parse_args()
    args = vars(args)

    set_arg_from_input(
        args,
        "new_model_name",
        "Enter the name of the new model to be used (default: 'text-embedding-3-small'): ",
        str,
        "text-embedding-3-small",
    )

    set_arg_from_password(
        args,
        "OPENAI_API_KEY",
        "Enter the OpenAI API key (hit return to skip): ",
        "OPENAI_API_KEY",
    )

    set_arg_from_input(
        args,
        "text_column",
        "Enter the name of the column containing text to be embedded (default: 'text'): ",
        str,
        "text",
    )

    set_arg_from_input(
        args,
        "dir",
        "Enter the directory of vector dataset in the VDF format: ",
        str,
    )

    reembed_count = 0
    # open VDF_META.json
    with open(os.path.join(args["dir"], "VDF_META.json"), "r+") as f:
        vdf_meta = json.load(f)
        for _, index_meta in tqdm(vdf_meta["indexes"].items(), desc=f"Iterating over indexes"):
            for namespace_meta in tqdm(index_meta, desc=f"Iterating over namespaces"):
                data_path = namespace_meta["data_path"]
                final_data_path = get_final_data_path(
                    os.getcwd(), args["dir"], data_path
                )
                parquet_files = get_parquet_files(final_data_path)
                for file in tqdm(parquet_files, desc=f"Iterating over parquet files"):
                    file_path = os.path.join(final_data_path, file)
                    new_vector_column = (
                        f"vector_{args['new_model_name'].replace('/', '_')}"
                    )
                    tqdm.write(f"Reembedding {file_path}")
                    # read parquet file
                    df = pd.read_parquet(file_path)
                    # get text column
                    text_column = args["text_column"]
                    if text_column not in df.columns:
                        raise Exception(
                            f"Text column {text_column} not found in {file_path}"
                        )
                    # get embeddings
                    BATCH_SIZE = 100
                    all_embeddings = []
                    for i in tqdm(
                        range(0, len(df), BATCH_SIZE), desc="Iterating over batches"
                    ):
                        batch_text = df[text_column][i : i + BATCH_SIZE].tolist()
                        embeddings = litellm.embedding(
                            model=args["new_model_name"],
                            input=batch_text,
                        )
                        # add embeddings to df
                        # embeddings.data is a list of dicts. Each dict has keys "embedding" and "index"
                        # first sort by "index" and then extract "embedding"
                        vectors = [x["embedding"] for x in sorted(embeddings.data, key=lambda x: x["index"])]
                        dim = len(vectors[0])
                        all_embeddings.extend(vectors)
                    # add only the new_vector_column in parquet file
                    df[new_vector_column] = all_embeddings
                    df.to_parquet(file_path)
                    tqdm.write(f"Computed {len(all_embeddings)} vectors for {len(df)} rows in {file_path}")
                    reembed_count += len(all_embeddings)
                # prepend new_vector_column to vector_columns in namespace_meta
                namespace_meta["vector_columns"].insert(0, new_vector_column)
                if "model_map" not in namespace_meta:
                    namespace_meta["model_map"] = {}
                    for vector_column in namespace_meta["vector_columns"]:
                        namespace_meta["model_map"][vector_column] = {
                            "model_name": namespace_meta["model_name"],
                            "text_column": args["text_column"],
                            "dimensions": namespace_meta["dimensions"],
                        }
                namespace_meta["model_map"][new_vector_column] = {
                    "model_name": args["new_model_name"],
                    "text_column": args["text_column"],
                    "dimensions": dim,
                }
                namespace_meta["model_name"] = args["new_model_name"]
                namespace_meta["dimensions"] = dim
        # write vdf_meta to VDF_META.json
        f.seek(0)
        json.dump(vdf_meta, f, indent=4)
        tqdm.write(f"Reembedding complete. Computed {reembed_count} vectors. Updated VDF_META.json")


if __name__ == "__main__":
    main()
