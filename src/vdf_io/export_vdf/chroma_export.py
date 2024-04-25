import json
import os
import chromadb
import pandas as pd
from tqdm import tqdm
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input
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
            default=10_000,
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
        set_arg_from_input(
            args,
            "collections",
            "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
            str,
        )
        set_arg_from_input(
            args,
            "batch_size",
            "Enter the batch size for exporting data (default: 10,000):",
            int,
            10_000,
        )
        chroma_export = ExportChroma(args)
        chroma_export.get_data()
        return chroma_export

    def __init__(self, args):
        super().__init__(args)
        if self.args.get("host_port") is not None:
            host_port = self.args.get("host_port")
            self.client = chromadb.Client(
                host=host_port.split(":")[0], port=host_port.split(":")[1]
            )
        elif self.args.get("persistent_path") is not None:
            self.client = chromadb.PersistentClient(
                path=self.args.get("persistent_path")
            )

    def get_index_names(self):
        if self.args.get("collections", None) is not None:
            return self.args["collections"].split(",")
        return self.client.list_collections()

    def get_data(self):
        batch_size = self.args.get("batch_size")
        index_metas = {}
        for i, collection in tqdm(
            enumerate(self.get_index_names()), desc="Exporting collections"
        ):
            col = self.client.get_collection(collection)
            vectors_directory = f"{self.vdf_directory}/{collection}"
            existing_count = col.count()
            total = 0
            for j in tqdm(
                range(0, existing_count, batch_size),
                desc=f"Exporting {collection} collection",
            ):
                batch = collection.get(
                    include=["metadatas", "documents", "embeddings"],
                    limit=batch_size,
                    offset=i,
                )
                # put batch["ids"], batch["metadatas"], batch["documents"], batch["embeddings"] into parquet

                vectors = {}
                metadata = {}
                for row in batch:
                    vectors[row["id"]] = row["embedding"]
                    metadata[row["id"]] = row["metadata"]

                # save_vectors_to_parquet
                total += self.save_vectors_to_parquet(
                    vectors, metadata, vectors_directory
                )

            namespace_metas = [
                self.get_namespace_meta(
                    collection,
                    vectors_directory,
                    total=existing_count,
                    num_vectors_exported=total,
                    dim=col["dimension"],
                    vector_columns=["embedding"],
                    distance=col.metadata.get("hnsw:space", "cosine"),
                )
            ]
            index_metas[collection] = namespace_metas
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
