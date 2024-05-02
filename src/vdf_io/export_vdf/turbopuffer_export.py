import json
import os
import sys
from typing import Dict, List
from tqdm import tqdm
import turbopuffer as tpuf
from vdf_io.constants import DISK_SPACE_LIMIT
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportTurbopuffer(ExportVDB):
    DB_NAME_SLUG = DBNames.TURBOPUFFER

    @classmethod
    def make_parser(cls, subparsers):
        parser_tpuf = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Turbopuffer"
        )
        parser_tpuf.add_argument(
            "--namespaces",
            help="The Turbopuffer namespaces to export (comma-separated). If not provided, all namespaces will be exported.",
        )
        parser_tpuf.add_argument(
            "--api_key",
            help="The API key for the Turbopuffer instance.",
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_password(
            args,
            "api_key",
            "Enter the API key for Turbopuffer (default: from TURBOPUFFER_API_KEY env var): ",
            env_var_name="TURBOPUFFER_API_KEY",
        )
        turbopuffer_export = ExportTurbopuffer(args)
        turbopuffer_export.all_namespaces = turbopuffer_export.get_all_index_names()
        set_arg_from_input(
            args,
            "namespaces",
            "Enter the Turbopuffer namespaces to export (hit return to export all): ",
            str,
            choices=turbopuffer_export.all_namespaces,
        )
        turbopuffer_export.get_data()
        return turbopuffer_export

    def __init__(self, args):
        super().__init__(args)
        tpuf.api_key = args.get("api_key")

    def get_all_index_names(self):
        nses = tpuf.namespaces()
        return [ns.name for ns in nses]

    def get_index_names(self):
        if self.args.get("namespaces"):
            return self.args.get("namespaces").split(",")
        return self.get_all_index_names()

    def get_data(self):
        namespace_names = self.get_index_names()
        ids = []
        vectors = {}
        metadata = {}
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        self.total_imported_count = 0
        for ns_name in tqdm(
            namespace_names, desc="Exporting turbopuffer namespaces (indexes)"
        ):
            exported_count = 0
            ns_idx = tpuf.Namespace(ns_name)
            tqdm.write(f"Exporting namespace {ns_name}")
            vectors_directory = self.create_vec_dir(ns_name)
            pbar = tqdm(desc="Exporting vectors", total=ns_idx.approx_count())
            for row in ns_idx.vectors():
                # VectorRow(id=1, vector=[0.1, 0.2], attributes={'name': 'foo', 'public': 1}, dist=None)
                # collect id, vector, attributes in ids, vectors, metadata
                ids.append(row.id)
                vectors[row.id] = row.vector
                metadata[row.id] = row.attributes
                pbar.update(1)
                if sys.getsizeof(vectors) + sys.getsizeof(metadata) > DISK_SPACE_LIMIT:
                    tqdm.write("Flushing to parquet files on disk")
                    exported_count += self.save_vectors_to_parquet(
                        vectors, metadata, vectors_directory
                    )

            tqdm.write("Flushing to parquet files on disk")
            exported_count += self.save_vectors_to_parquet(
                vectors, metadata, vectors_directory
            )
            namespace_metas = [
                self.get_namespace_meta(
                    ns_name,
                    vectors_directory,
                    total=exported_count,
                    num_vectors_exported=exported_count,
                    dim=ns_idx.dimensions(),
                    vector_columns=["vector"],
                    distance=ns_idx.metadata["distance"]
                    if "distance" in ns_idx.metadata
                    else None,
                )
            ]
            self.total_imported_count += exported_count
            index_metas[ns_name] = namespace_metas
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        # print internal metadata properly
        return True
