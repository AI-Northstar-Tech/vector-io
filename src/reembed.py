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

from util import (
    get_final_data_path,
    get_parquet_files,
    set_arg_from_input,
    set_arg_from_password,
)

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

    # open VDF_META.json
    with open(os.path.join(args["dir"], "VDF_META.json"), "r+") as f:
        vdf_meta = json.load(f)
        for collection_name, index_meta in vdf_meta["indexes"].items():
            for namespace_meta in index_meta:
                data_path = namespace_meta["data_path"]
                index_name = collection_name + (
                    f'_{namespace_meta["namespace"]}'
                    if namespace_meta["namespace"]
                    else ""
                )
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
                        tqdm.write(f"{len(batch_text)},{len(embeddings.data)}")
                        # add embeddings to df
                        all_embeddings.extend(embeddings.data)
                    # add only the new_vector_column in parquet file
                    tqdm.write(f"total embeddings: {len(all_embeddings)}, {len(df)}")
                    df[new_vector_column] = all_embeddings
                    df.to_parquet(file_path)
                # prepend new_vector_column to vector_columns in namespace_meta
                namespace_meta["vector_columns"].insert(0, new_vector_column)
                if "model_map" not in namespace_meta:
                    namespace_meta["model_map"] = {}
                    for vector_column in namespace_meta["vector_columns"]:
                        namespace_meta["model_map"][vector_column] = (
                            namespace_meta["model_name"],
                            args["text_column"],
                        )
                namespace_meta["model_map"][new_vector_column] = (
                    args["new_model_name"],
                    args["text_column"],
                )
                namespace_meta["model_name"] = args["new_model_name"]
        # write vdf_meta to VDF_META.json
        f.seek(0)
        json.dump(vdf_meta, f, indent=4)


if __name__ == "__main__":
    main()
