#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import time
import litellm
from litellm import EmbeddingResponse
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm
from dotenv import load_dotenv
import sys
from IPython.core import ultratb
import warnings
import vdf_io
from vdf_io.constants import ID_COLUMN
from vdf_io.meta_types import NamespaceMeta, VDFMeta

from vdf_io.util import (
    get_final_data_path,
    get_parquet_files,
    read_parquet_progress,
    set_arg_from_input,
    set_arg_from_password,
)

litellm.suppress_debug_info = True
warnings.filterwarnings("ignore", module="litellm")
sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

load_dotenv()


def main():
    start_time = time.time()
    reembed()
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")


def reembed():
    parser = argparse.ArgumentParser(description="Reembed a vector dataset")

    add_arguments_to_parser(parser)

    args = parser.parse_args()
    args = vars(args)

    if args.get("env_file_path"):
        load_dotenv(os.path.join(os.getcwd(), args.get("env_file_path")))

    take_input_from_cli_prompt(args)

    reembed_count = 0
    # open VDF_META.json
    handle_new_dataset(args)
    reembed_impl(args, reembed_count)


def reembed_impl(args, reembed_count):
    with open(os.path.join(args["dir"], "VDF_META.json"), "r+") as f:
        vdf_meta = json.load(f)
        for _, index_meta in tqdm(
            vdf_meta["indexes"].items(), desc="Iterating over indexes"
        ):
            new_vector_column = (
                f"vec_{args['text_column']}_{args['new_model_name'].replace('/', '_')}"
            )
            overwrite_bool = args["overwrite"]
            for namespace_meta in tqdm(index_meta, desc="Iterating over namespaces"):
                data_path = namespace_meta["data_path"]
                final_data_path = get_final_data_path(
                    os.getcwd(), args["dir"], data_path, args={}
                )
                parquet_files = get_parquet_files(
                    final_data_path,
                    args={},
                    id_column=vdf_meta.get("id_column", ID_COLUMN),
                )
                for file in tqdm(parquet_files, desc="Iterating over parquet files"):
                    file_path = os.path.join(final_data_path, file)
                    if "vector_columns" not in namespace_meta:
                        namespace_meta["vector_columns"] = []
                    if (
                        new_vector_column in namespace_meta["vector_columns"]
                        and not overwrite_bool
                    ):
                        # ask user if they want to overwrite (y/n)
                        overwrite = args["overwrite"] or input(
                            f"{new_vector_column} already exists in vector_columns. Overwrite? (y/n): "
                        )
                        overwrite_bool = overwrite.lower() == "y"
                        if not overwrite_bool:
                            tqdm.write(
                                f"Skipping {file_path} because {new_vector_column} already exists in vector_columns.\n"
                                "Aborting reembedding."
                            )
                            exit()
                    if "dimensions" in args and args["dimensions"] is not None:
                        new_vector_column += f"_{args['dimensions']}"
                    tqdm.write(f"Reembedding {file_path}")
                    # read parquet file
                    df = read_parquet_progress(file_path, ID_COLUMN)
                    # get text column
                    text_column = args["text_column"]
                    if text_column not in df.columns:
                        text_columns = text_column.split("|")
                        if all([col in df.columns for col in text_columns]):
                            df[text_column] = df[text_columns].apply(
                                lambda x: " ".join(x.dropna().astype(str)), axis=1
                            )
                        else:
                            raise Exception(
                                f"Text column {text_column} not found in {file_path}"
                            )
                    # get embeddings
                    BATCH_SIZE = args.get("batch_size")
                    all_embeddings = []
                    for i in tqdm(
                        range(0, len(df), BATCH_SIZE), desc="Iterating over batches"
                    ):
                        batch_text = df[text_column][i : i + BATCH_SIZE].tolist()

                        embeddings = call_litellm(args, batch_text)
                        # add embeddings to df
                        # embeddings.data is a list of dicts. Each dict has keys "embedding" and "index"
                        # first sort by "index" and then extract "embedding"
                        vectors = [
                            x["embedding"]
                            for x in sorted(embeddings.data, key=lambda x: x["index"])
                        ]
                        dim = len(vectors[0])
                        all_embeddings.extend(vectors)
                    # add only the new_vector_column in parquet file
                    df[new_vector_column] = all_embeddings
                    if ID_COLUMN not in df.columns:
                        df[ID_COLUMN] = df.index
                    df.to_parquet(file_path)
                    tqdm.write(
                        f"Computed {len(all_embeddings)} vectors for {len(df)} rows in column:{new_vector_column} of {file_path}"
                    )
                    reembed_count += len(all_embeddings)
                # prepend new_vector_column to vector_columns in namespace_meta
                if new_vector_column not in namespace_meta["vector_columns"]:
                    namespace_meta["vector_columns"].insert(0, new_vector_column)
                else:
                    tqdm.write(
                        f"Warning: {new_vector_column} already exists in vector_columns. Overwriting."
                    )
                if "model_map" not in namespace_meta:
                    namespace_meta["model_map"] = {}
                    for vector_column in namespace_meta["vector_columns"]:
                        namespace_meta["model_map"][vector_column] = {
                            "model_name": namespace_meta.get("model_name"),
                            "text_column": args["text_column"],
                            "dimensions": namespace_meta.get("dimensions"),
                            "vector_column": vector_column,
                        }
                namespace_meta["model_map"][new_vector_column] = {
                    "model_name": args["new_model_name"],
                    "text_column": args["text_column"],
                    "dimensions": dim,
                    "vector_column": new_vector_column,
                }
                namespace_meta["model_name"] = args["new_model_name"]
                namespace_meta["dimensions"] = dim
        # write vdf_meta to VDF_META.json
        f.seek(0)
        json.dump(vdf_meta, f, indent=4)
        tqdm.write(
            f"Reembedding complete. Computed {reembed_count} vectors. Updated VDF_META.json"
        )


def handle_new_dataset(args):
    if not os.path.exists(os.path.join(args["dir"], "VDF_META.json")):
        # create VDF_META.json
        with open(os.path.join(args["dir"], "VDF_META.json"), "w") as f:
            # index_name is folder name of dir
            index_name = os.path.basename(args["dir"])
            # find parquet files in dir recursively
            import glob

            parquet_files = glob.glob(
                os.path.join(args["dir"], "**", "*.parquet"), recursive=True
            )
            if len(parquet_files) == 0:
                raise Exception("No parquet files found in the specified directory")
            total_vector_count = 0
            for file in parquet_files:
                # read schema of parquet file using pyarrow and get count of rows
                import pyarrow.parquet as pq

                table = pq.read_table(file)
                row_count = table.num_rows
                total_vector_count += row_count
            vdf_meta = VDFMeta(
                version=vdf_io.__version__,
                file_structure=[],
                author=os.environ.get("USER"),
                exported_from="reembed",
                exported_at=datetime.datetime.now().astimezone().isoformat(),
                id_column=ID_COLUMN,
                indexes={
                    index_name: [
                        NamespaceMeta(
                            namespace="",
                            index_name=index_name,
                            total_vector_count=total_vector_count,
                            exported_vector_count=total_vector_count,
                            dimensions=-1,
                            model_name=None,
                            vector_columns=[],
                            data_path=".",
                            metric=None,
                        )
                    ]
                },
            ).model_dump()
            json.dump(vdf_meta, f, indent=4)


def take_input_from_cli_prompt(args):
    set_arg_from_input(
        args,
        "new_model_name",
        "Enter the name of the new model to be used (default: 'text-embedding-3-small'): ",
        str,
        "text-embedding-3-small",
    )

    if args["new_model_name"].startswith("text-embedding-3"):
        set_arg_from_input(
            args,
            "dimensions",
            "Enter the dimensions of the new model to be used (default: 1536 for small, 3072 for large): ",
            int,
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


def add_arguments_to_parser(parser):
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

    parser.add_argument(
        "--dimensions",
        type=int,
        help="Dimensions of the new model to be used",
    )

    parser.add_argument(
        "--env_file_path",
        type=str,
        help="Path to the .env file",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for reembedding",
        default=100,
    )

    parser.add_argument(
        "--overwrite",
        type=bool,
        help="Overwrite existing vector columns",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument(
        "--input_type",
        type=str,
        help="Input type for the model",
        default=None,
    )


use_sentence_transformers = False


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception),
)
def call_litellm(args, batch_text):
    if args["new_model_name"].startswith("huggingface/"):
        return call_sentence_transformers(args, batch_text)
    global use_sentence_transformers
    if use_sentence_transformers:
        return call_sentence_transformers(args, batch_text)
    try:
        embeddings = litellm.embedding(
            model=args["new_model_name"],
            input=batch_text,
            dimensions=args.get("dimensions"),
            input_type=args.get("input_type"),
        )
    except Exception as e:
        if e.message.startswith("Huggingface"):
            use_sentence_transformers = True
            embeddings = call_sentence_transformers(args, batch_text)
        else:
            raise e

    return embeddings


model = None


def call_sentence_transformers(args, batch_text):
    from sentence_transformers import SentenceTransformer

    # check global model
    global model
    if model is None:
        model = SentenceTransformer(
            args["new_model_name"].replace("huggingface/", ""), trust_remote_code=True
        )
    embeddings = model.encode(batch_text, show_progress_bar=True)
    return EmbeddingResponse(
        data=[
            {"index": i, "embedding": emb.tolist()} for i, emb in enumerate(embeddings)
        ]
    )


if __name__ == "__main__":
    main()
