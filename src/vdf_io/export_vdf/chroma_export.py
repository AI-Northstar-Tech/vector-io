import json
import os
import sys
from tqdm import tqdm

import chromadb

from vdf_io.constants import DEFAULT_BATCH_SIZE, DISK_SPACE_LIMIT
from vdf_io.names import DBNames
from vdf_io.util import expand_shorthand_path, set_arg_from_input
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportChroma(ExportVDB):
    DB_NAME_SLUG = DBNames.CHROMA

    @classmethod
    def make_parser(cls, subparsers):
        parser_chroma = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Chroma"
        )

        parser_chroma.add_argument(
            "--host_port", type=str, help="Host and port of Chroma instance"
        )
        parser_chroma.add_argument(
            "--persistent_path",
            type=str,
            help="Path to persistent storage for Chroma",
        )
        parser_chroma.add_argument(
            "--collections", type=str, help="Names of collections to export"
        )
        parser_chroma.add_argument(
            "--batch_size",
            type=int,
            help="Batch size for exporting data",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        if args.get("persistent_path") is None:
            set_arg_from_input(
                args,
                "host_port",
                "Enter the host:port of chroma instance: ",
                str,
            )
        if args.get("host_port") is None or args.get("host_port") == "":
            set_arg_from_input(
                args,
                "persistent_path",
                "Enter the path to persistent storage for Chroma: ",
                str,
            )
        if (args.get("host_port") is None or args.get("host_port") == "") and (
            args.get("persistent_path") is None or args.get("persistent_path") == ""
        ):
            set_arg_from_input(
                args,
                "api_key",
                "Enter the API key for Chroma: ",
                str,
            )
            set_arg_from_input(
                args,
                "tenant",
                "Enter the tenant for Chroma: ",
                str,
            )
            set_arg_from_input(
                args,
                "database",
                "Enter the database for Chroma: ",
                str,
            )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the batch size for exporting data (default: {DEFAULT_BATCH_SIZE}):",
            int,
            DEFAULT_BATCH_SIZE,
        )
        chroma_export = ExportChroma(args)
        chroma_export.all_collections = chroma_export.get_all_index_names()
        set_arg_from_input(
            args,
            "collections",
            "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
            str,
            choices=chroma_export.all_collections,
        )
        chroma_export.get_data()
        return chroma_export

    def __init__(self, args):
        super().__init__(args)
        if self.args.get("host_port") is not None:
            host_port = self.args.get("host_port")
            self.client = chromadb.HttpClient(
                host=host_port.split(":")[0], port=int(host_port.split(":")[1])
            )
        elif self.args.get("persistent_path") is not None:
            self.client = chromadb.PersistentClient(
                path=expand_shorthand_path(self.args.get("persistent_path"))
            )
        else:
            self.client = chromadb.CloudClient(
                tenant=self.args.get("tenant"),
                database=self.args.get("database"),
                api_key=self.args.get("api_key"),
            )

    def get_all_index_names(self):
        return [coll.name for coll in self.client.list_collections()]

    def get_index_names(self):
        if self.args.get("collections", None) is not None:
            return self.args["collections"].split(",")
        return self.get_all_index_names()

    def get_data(self):
        batch_size = self.args.get("batch_size") or DEFAULT_BATCH_SIZE
        index_metas = {}
        for i, collection_name in tqdm(
            enumerate(self.get_index_names()), desc="Exporting collections"
        ):
            dims = -1
            col = self.client.get_collection(collection_name)
            vectors_directory = self.create_vec_dir(collection_name)
            existing_count = col.count()
            total = 0
            for j in tqdm(
                range(0, existing_count, batch_size),
                desc=f"Exporting {collection_name} collection",
            ):
                batch = col.get(
                    include=["metadatas", "documents", "embeddings"],
                    limit=batch_size,
                    offset=j,
                )
                vectors = {}
                metadata = {}
                chroma_ids = batch["ids"]
                embeddings = batch["embeddings"]
                chroma_metadatas = batch["metadatas"]
                uris = batch["uris"]
                data = batch["data"]
                documents = batch["documents"]
                dims = len(embeddings[0])
                for idx, chroma_id in enumerate(chroma_ids):
                    vectors[chroma_id] = embeddings[idx]
                    metadata[chroma_id] = chroma_metadatas[idx]
                    metadata[chroma_id]["document"] = documents[idx]
                    if uris is not None and len(uris) > idx:
                        metadata[chroma_id]["uri"] = uris[idx]
                    if data is not None and len(data) > idx:
                        metadata[chroma_id]["data"] = data[idx]
                if sys.getsizeof(vectors) + sys.getsizeof(metadata) > DISK_SPACE_LIMIT:
                    # save_vectors_to_parquet
                    total += self.save_vectors_to_parquet(
                        vectors, metadata, vectors_directory
                    )
            total += self.save_vectors_to_parquet(vectors, metadata, vectors_directory)
            namespace_metas = [
                self.get_namespace_meta(
                    collection_name,
                    vectors_directory,
                    total=existing_count,
                    num_vectors_exported=total,
                    dim=dims,
                    vector_columns=["vector"],
                    distance=(
                        col.metadata.get("hnsw:space", "cosine")
                        if (hasattr(col, "metadata") and col.metadata is not None)
                        else "cosine"
                    ),
                )
            ]
            index_metas[collection_name] = namespace_metas
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True
