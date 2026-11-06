import json
import os
from typing import Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
import weaviate
from weaviate.classes.config import VectorDistances

from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.metadata_types import DistanceMetric
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password

load_dotenv()

WEAVIATE_DISTANCE_MAP = {
    VectorDistances.COSINE: DistanceMetric.COSINE,
    VectorDistances.DOT: DistanceMetric.DOT_PRODUCT,
    VectorDistances.L2_SQUARED: DistanceMetric.EUCLIDEAN,
    "cosine": DistanceMetric.COSINE,
    "dot": DistanceMetric.DOT_PRODUCT,
    "l2-squared": DistanceMetric.EUCLIDEAN,
}


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )
        parser_weaviate.add_argument(
            "-u", "--url", type=str, help="URL of Weaviate instance (WCS or local)"
        )
        parser_weaviate.add_argument(
            "--api_key", type=str, help="Weaviate API key (leave empty for local)"
        )
        parser_weaviate.add_argument(
            "-c",
            "--classes",
            type=str,
            help="Classes/collections to export (comma-separated)",
        )
        parser_weaviate.add_argument(
            "--local",
            action="store_true",
            default=False,
            help="Connect to a local Weaviate instance",
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "url",
            "Enter the URL of Weaviate instance (default: 'http://localhost:8080'): ",
            str,
            default_value="http://localhost:8080",
        )
        if not args.get("local") and "localhost" not in (args.get("url") or ""):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = weaviate_export.get_all_index_names()
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the name of the classes to export (comma-separated, hit return to export all): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args):
        super().__init__(args)
        url = self.args.get("url") or "http://localhost:8080"
        api_key = self.args.get("api_key")
        if self.args.get("local") or "localhost" in url or "127.0.0.1" in url:
            self.client = weaviate.connect_to_local(skip_init_checks=True)
        else:
            auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
            self.client = weaviate.connect_to_wcs(
                cluster_url=url,
                auth_credentials=auth,
                skip_init_checks=True,
            )

    def get_all_index_names(self) -> List[str]:
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [str(c) for c in collections]

    def get_index_names(self) -> List[str]:
        if not self.args.get("classes"):
            return self.all_classes
        input_classes = [
            c.strip() for c in self.args["classes"].split(",") if c.strip()
        ]
        missing = set(input_classes) - set(self.all_classes)
        if missing:
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {missing}"
            )
        return [c for c in self.all_classes if c in input_classes]

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for class_name in tqdm(index_names, desc="Fetching Weaviate collections"):
            index_metas[class_name] = self.get_data_for_collection(class_name)

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def get_data_for_collection(self, class_name: str) -> List[NamespaceMeta]:
        vectors_directory = self.create_vec_dir(class_name)
        collection = self.client.collections.get(class_name)
        agg = collection.aggregate.over_all(total_count=True)
        total = getattr(agg, "total_count", 0) or 0

        vectors = {}
        metadata = {}
        num_vectors_exported = 0
        dim = 0

        for item in tqdm(
            collection.iterator(include_vector=True),
            total=total,
            desc=f"Exporting {class_name}",
        ):
            obj_id = str(item.uuid)
            vec = item.vector
            if isinstance(vec, dict):
                vec = next(iter(vec.values()), [])
            if vec and not dim:
                dim = len(vec)
            vectors[obj_id] = vec
            metadata[obj_id] = dict(item.properties or {})
            if len(vectors) >= 1000:
                num_vectors_exported += self.save_vectors_to_parquet(
                    vectors, metadata, vectors_directory
                )

        if vectors:
            num_vectors_exported += self.save_vectors_to_parquet(
                vectors, metadata, vectors_directory
            )

        cfg = collection.config.get()
        raw_dist = getattr(
            getattr(cfg, "vector_index_config", None), "distance_metric", "cosine"
        )
        distance = WEAVIATE_DISTANCE_MAP.get(raw_dist, DistanceMetric.COSINE)

        namespace_meta = self.get_namespace_meta(
            class_name,
            vectors_directory,
            total or num_vectors_exported,
            num_vectors_exported,
            dim,
            distance=distance,
        )
        self.args["exported_count"] += num_vectors_exported
        return [namespace_meta]
