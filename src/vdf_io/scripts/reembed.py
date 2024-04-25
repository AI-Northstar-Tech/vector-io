#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import time
import litellm
from litellm import EmbeddingResponse
import numpy as np
import sentence_transformers
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)
import torch
from tqdm import tqdm
from dotenv import load_dotenv
import sys
from IPython.core import ultratb
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
from mlx_embedding_models.embedding import EmbeddingModel
from sentence_transformers import SentenceTransformer

import vdf_io
from vdf_io.constants import ID_COLUMN
from vdf_io.meta_types import NamespaceMeta, VDFMeta

from vdf_io.util import (
    get_author_name,
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
        load_dotenv(os.path.join(os.getcwd(), args.get("env_file_path")), override=True)

    take_input_from_cli_prompt(args)

    reembed_count = 0
    # open VDF_META.json
    vdf_meta_path = os.path.join(args["dir"], "VDF_META.json")
    if not os.path.exists(vdf_meta_path) or not valid_json(vdf_meta_path):
        handle_new_dataset(args)
    reembed_impl(args, reembed_count)


def valid_json(file_path):
    try:
        with open(file_path) as f:
            json.load(f)
            return True
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def reembed_impl(args, reembed_count):
    with open(os.path.join(args["dir"], "VDF_META.json"), "r+") as f:
        vdf_meta = json.load(f)
        for _, index_meta in tqdm(
            vdf_meta["indexes"].items(), desc="Iterating over indexes"
        ):
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
                    df = read_parquet_progress(file_path, ID_COLUMN)
                    if args["text_column"] not in df.columns:
                        ask_for_text_column(args, file_path, df)
                    new_vector_column = (
                        f"vec_{args['text_column']}_{args['new_model_name'].replace('/', '_')}"
                        + (
                            "_" + args["quantize"]
                            if (args.get("quantize") != "float32")
                            else ""
                        )
                    )
                    if "dimensions" in args and args["dimensions"] is not None:
                        new_vector_column += f"_dim{args['dimensions']}"
                    else:
                        litellm_embed_response = call_litellm(args, ["dummy"])
                        dims = len(litellm_embed_response.data[0]["embedding"])
                        new_vector_column += f"_dim{dims}"
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
                    tqdm.write(f"Reembedding {file_path}")
                    # read parquet file
                    # get text column
                    # get embeddings
                    BATCH_SIZE = args.get("batch_size")
                    all_embeddings = []
                    pbar = tqdm(total=len(df), desc="Iterating over rows")
                    for i in tqdm(
                        range(0, len(df), BATCH_SIZE), desc="Iterating over batches"
                    ):
                        batch_text = df[args["text_column"]][
                            i : i + BATCH_SIZE
                        ].tolist()
                        pbar.update(len(batch_text))

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
                    # TODO: use save_vectors_to_parquet
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
                if not namespace_meta.get("model_map"):
                    namespace_meta["model_map"] = {}
                for vector_column in namespace_meta["vector_columns"]:
                    if vector_column not in namespace_meta["model_map"]:
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
                namespace_meta["schema_dict_str"] = pq.read_schema(
                    file_path
                ).to_string()
        # write vdf_meta to VDF_META.json
        f.seek(0)
        json.dump(vdf_meta, f, indent=4)
        tqdm.write(
            f"Reembedding complete. Computed {reembed_count} vectors. Updated VDF_META.json"
        )


def ask_for_text_column(args, file_path, df):
    text_columns = args["text_column"].split("|")
    if all([col in df.columns for col in text_columns]):
        df[args["text_column"]] = df[text_columns].apply(
            lambda x: " ".join(x.dropna().astype(str)), axis=1
        )
    else:
        # ask user to enter text columns by index1|index2|index3
        tqdm.write(f"Text column {args['text_column']} not found in {file_path}")
        tqdm.write("Select the text column(s) from the following list:")
        # list of string columns
        text_column_options = {}
        for i, col in enumerate(df.columns):
            # check if column is of type string
            # pick first non-null value
            non_null_value = df[col].dropna().iloc[0]
            if isinstance(non_null_value, str):
                tqdm.write(f"{i+1}: {col}")
                text_column_options[i + 1] = col
        choice_correctly_entered = False
        while not choice_correctly_entered:
            text_column_choice = input(
                "Enter the text column index(es) separated by '|' (e.g. 1|2|3): "
            )
            text_columns_choice = [
                int(col_no) for col_no in text_column_choice.split("|")
            ]
            if all([tc in text_column_options for tc in text_columns_choice]):
                choice_correctly_entered = True
                text_column_choice_names = [
                    text_column_options[col_no] for col_no in text_columns_choice
                ]
                text_column_name = "|".join(text_column_choice_names)
                tqdm.write(f"Selected text column(s): {text_column_name}.")
                args["text_column"] = text_column_name
                if len(text_column_choice_names) > 1:
                    tqdm.write("Combining multiple text columns into a single column.")
                    df[text_column_name] = df[text_column_choice_names].apply(
                        lambda x: " ".join(x.dropna().astype(str)),
                        axis=1,
                    )
            else:
                tqdm.write(
                    "Invalid choice. Please enter the text column index(es) separated by '|': "
                )


def handle_new_dataset(args):
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
            # print each column name and type
            # for col in schema_dict["metadata"]["arrow:extension:column_types"]:
            #     tqdm.write(f"Column: {col['name']} of type {col['type']}")
            # # are there any vector columns in the schema?
            # for col in schema_dict["metadata"]["arrow:extension:vector_columns"]:
            #     tqdm.write(f"Vector column: {col['name']} of type {col['type']}")
            # create VDF_META.json
            # if there exists a vector column, add it to vector_columns
            vector_columns = []
            model_map = {}
            model_name = ""
            dimensions = -1
            for field in table.schema:
                if pa.types.is_list(field.type):
                    tqdm.write(
                        f"Vector column: {field.name} of type {field.type.to_pandas_dtype()}"
                    )
                    vector_columns.append(field.name)
                    # check length of vector column
                    dimensions = len(table[field.name].to_pandas().iloc[0])
                    # ask which model to use for embedding
                    tqdm.write(
                        f"Vector column {field.name} has length {dimensions}. Please specify the model to use for embedding."
                    )
                    model_name = input(
                        "Enter the name of the model to be used for embedding: "
                    )
                    old_text_column = input(
                        "Enter the name of the column containing text to be embedded: "
                    )
                    model_map[field.name] = {
                        "model_name": model_name,
                        "text_column": old_text_column,
                        "dimensions": dimensions,
                        "vector_column": field.name,
                    }
                    # add model_name to VDF_META.json
                    # add vector column to vector_columns
                    # add vector column to model_map
                    # add model_name to model_map
                    # add dimensions to model_map

        vdf_meta = VDFMeta(
            version=vdf_io.__version__,
            file_structure=[],
            author=get_author_name(),
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
                        dimensions=dimensions,
                        model_name=model_name,
                        vector_columns=vector_columns,
                        data_path=".",
                        metric=None,
                        model_map=model_map,
                        schema_dict_str=table.schema.to_string(),
                    )
                ]
            },
        )
        vdf_meta = vdf_meta.model_dump()
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
        default=96,
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
    parser.add_argument(
        "--disable_mlx",
        type=bool,
        help="Disable MLX for embedding",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    quantize_options = [
        "int8",
        "binary",
        "float32",
        "int8",
        "uint8",
        "binary",
        "ubinary",
    ]
    parser.add_argument(
        "--quantize",
        type=str,
        choices=quantize_options,
        help=f"Quantization method ({', '.join(quantize_options)})",
        default="float32",
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
            **{
                "model": args["new_model_name"],
                "input": batch_text,
                "dimensions": args.get("dimensions"),
                **(
                    {"input_type": args["input_type"]}
                    if args.get("input_type") is not None
                    else {}
                ),
            }
        )
        if args.get("quantize") != "float32":
            # convert to ndarray
            embeddings = np.array([np.array(x["embedding"]) for x in embeddings.data])
            embeddings = sentence_transformers.quantize_embeddings(
                embeddings, precision=args.get("quantize")
            )
            # convert to EmbeddingResponse
            embeddings = EmbeddingResponse(
                data=[
                    {"index": i, "embedding": emb.tolist()}
                    for i, emb in enumerate(embeddings)
                ]
            )
    except Exception as e:
        # catch BadRequestError: LLM Provider NOT provided. Pass in the LLM provider you are trying to call. You passed model=TaylorAI/bge-micro-v2
        # check type of exception
        if e.message.startswith("Huggingface") or (
            type(e).__name__ == "BadRequestError" and "LLM Provider" in e.message
        ):
            tqdm.write(
                f"Using Sentence Transformers for embedding for model {args['new_model_name']} {e}"
            )
            use_sentence_transformers = True
            embeddings = call_sentence_transformers(args, batch_text)
        else:
            raise e

    return embeddings


model = None

using_mlx = False


def is_apple_silicon():
    # if `uname -m` returns arm64, then it is an apple silicon device
    return os.uname().machine == "arm64"
    # return torch.backends.mps.is_available()


def call_sentence_transformers(args, batch_text):
    global model
    if model is None:
        # figure out device map
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and is_apple_silicon() and not args.get("disable_mlx"):
            global using_mlx
            using_mlx = True
            tqdm.write("Using MLX for embedding")
            model = EmbeddingModel.from_registry(strip_hf_prefix(args))
        else:
            model = SentenceTransformer(
                strip_hf_prefix(args),
                trust_remote_code=True,
                device=device,
            )
    if using_mlx:
        embeddings = model.encode(batch_text)
    else:
        embeddings = model.encode(batch_text, precision=args.get("quantize"))
    return EmbeddingResponse(
        data=[
            {"index": i, "embedding": emb.tolist()} for i, emb in enumerate(embeddings)
        ]
    )


def strip_hf_prefix(args):
    # count /s
    if args["new_model_name"].count("/") == 2:
        # replace first occurrence of "huggingface/" to ""
        return args["new_model_name"].replace("huggingface/", "", 1)
    return args["new_model_name"].replace("huggingface/", "")


if __name__ == "__main__":
    main()
