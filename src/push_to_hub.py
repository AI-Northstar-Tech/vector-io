#!/usr/bin/env python3

from getpass import getpass
import os
from huggingface_hub import HfApi
import argparse


def push_to_hub(export_obj, args):
    args = vars(args)
    print("Pushing to HuggingFace Hub...")

    # Log in to Hugging Face
    if (
        "HUGGING_FACE_TOKEN" not in os.environ
        or os.environ["HUGGING_FACE_TOKEN"] is None
    ):
        # set HUGGINGFACEHUB_API_TOKEN env var
        os.environ["HUGGING_FACE_TOKEN"] = getpass(
            prompt="Enter your HuggingFace API token (with write access): "
        )
    if (
        "HF_USERNAME" not in args
        or args["HF_USERNAME"] is None
        or args["HF_USERNAME"] == ""
    ):
        if "HF_USERNAME" not in os.environ or os.environ["HF_USERNAME"] is None:
            # set HF_USERNAME env var
            os.environ["HF_USERNAME"] = input("Enter your HuggingFace username: ")
    else:
        os.environ["HF_USERNAME"] = args["HF_USERNAME"]
    hf_api = HfApi(token=os.environ["HUGGING_FACE_TOKEN"])
    # put current working directory + vdf_directory in new variable
    data_path = os.path.join(os.getcwd(), export_obj.vdf_directory)
    export_obj.vdf_directory = os.path.basename(export_obj.vdf_directory)
    repo_id = f"{os.environ['HF_USERNAME']}/{export_obj.vdf_directory}"
    dataset_url = hf_api.create_repo(
        token=os.environ["HUGGING_FACE_TOKEN"],
        repo_id=repo_id,
        private=(not args["public"]),
        repo_type="dataset",
        exist_ok=True,
    )
    # for each file/folder in export_obj.vdf_directory, upload to hub
    hf_api.upload_folder(
        repo_id=repo_id,
        folder_path=data_path,
        repo_type="dataset",
    )
    # create hf dataset card in temp README.md
    readme_path = os.path.join(data_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(
            """
---
tags:
- vdf
- vector-io
- vector-dataset
- vector-embeddings
---
This is a dataset created using [vector-io](https://github.com/ai-northstar-tech/vector-io)
"""
        )
    hf_api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_type="dataset",
    )
    print(f"Created a private HuggingFace dataset repo at {dataset_url}")


def main():
    parser = argparse.ArgumentParser(
        description="Push a vector dataset to HuggingFace Hub"
    )
    parser.add_argument(
        "-d",
        "--vdf_directory",
        type=str,
        help="Path to the directory containing the vector dataset",
    )
    parser.add_argument(
        "--public",
        type=bool,
        help="Make the dataset public (default: False)",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--HF_USERNAME",
        type=str,
        help="HuggingFace username (default: existing env var HF_USERNAME)",
        default=None,
    )
    args = parser.parse_args()

    if args.vdf_directory is None:
        print("Please provide a path to the vector dataset directory")
        args.vdf_directory = input("Enter the path to the vector dataset directory: ")

    push_to_hub(args, args)


if __name__ == "__main__":
    main()
