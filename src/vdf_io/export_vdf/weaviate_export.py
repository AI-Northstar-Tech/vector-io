import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import weaviate

from vdf_io.constants import DEFAULT_BATCH_SIZE, DISK_SPACE_LIMIT, ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password

DEFAULT_LOCAL_URL = "http://localhost:8080"
DEFAULT_GRPC_PORT = 50051


@dataclass
class CollectionExportState:
    """Tracks in-memory vector buffers and schema metadata while streaming a Weaviate collection."""

    class_name: str
    vectors_directory: str
    vectors: Dict[str, Any]
    metadata: Dict[str, Dict[str, Any]]
    vector_columns: List[str]
    dimensions: int = -1
    exported_count: int = 0


def _parse_endpoint(raw_url: str):
    """Parses a raw URL string, adding an explicit HTTP scheme when omitted by the user."""
    normalized = raw_url if "://" in raw_url else f"http://{raw_url}"
    return urlparse(normalized)


def _is_local_url(url: str) -> bool:
    """Determines whether the target Weaviate endpoint points to a loopback interface."""
    hostname = (_parse_endpoint(url).hostname or "").lower()
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0", ""}


def _is_wcs_cloud_url(url: str, api_key: Optional[str]) -> bool:
    """Identifies managed Weaviate Cloud Services endpoints that use cluster URL routing."""
    lowered = url.lower()
    is_wcs_domain = ".weaviate.network" in lowered or ".weaviate.cloud" in lowered
    if is_wcs_domain:
        return True
    return bool(api_key) and not _is_local_url(url)


def _connect_local_endpoint(args: Dict[str, Any], auth):
    """Opens a local Weaviate connection over the configured HTTP and gRPC loopback ports."""
    parsed_url = _parse_endpoint((args.get("url") or DEFAULT_LOCAL_URL).strip())
    return weaviate.connect_to_local(
        host=parsed_url.hostname or "localhost",
        port=parsed_url.port or 8080,
        grpc_port=int(args.get("grpc_port") or DEFAULT_GRPC_PORT),
        auth_credentials=auth,
        skip_init_checks=True,
    )


def _connect_custom_endpoint(args: Dict[str, Any], auth):
    """Opens a custom self-hosted Weaviate connection with explicit HTTP and gRPC transport parameters."""
    parsed_url = _parse_endpoint((args.get("url") or DEFAULT_LOCAL_URL).strip())
    is_secure = parsed_url.scheme == "https"
    host = parsed_url.hostname or "localhost"
    default_port = 443 if is_secure else 8080
    return weaviate.connect_to_custom(
        http_host=host,
        http_port=parsed_url.port or default_port,
        http_secure=is_secure,
        grpc_host=host,
        grpc_port=int(args.get("grpc_port") or DEFAULT_GRPC_PORT),
        grpc_secure=is_secure,
        auth_credentials=auth,
        skip_init_checks=True,
    )


def connect_weaviate_client(args: Dict[str, Any]):
    """Connects to a local, custom self-hosted, or managed WCS Weaviate instance based on CLI args."""
    raw_url = (args.get("url") or DEFAULT_LOCAL_URL).strip()
    api_key = (args.get("api_key") or "").strip() or None
    auth = weaviate.auth.AuthApiKey(api_key) if api_key else None

    if _is_local_url(raw_url):
        return _connect_local_endpoint(args, auth)
    if _is_wcs_cloud_url(raw_url, api_key):
        return weaviate.connect_to_wcs(
            cluster_url=raw_url,
            auth_credentials=auth,
            skip_init_checks=True,
        )
    return _connect_custom_endpoint(args, auth)


def _read_distance_value(idx_cfg) -> Optional[str]:
    """Extracts the lowercase distance metric identifier from a Weaviate vector index config object."""
    distance = getattr(idx_cfg, "distance_metric", None)
    if distance is None:
        return None
    return getattr(distance, "value", str(distance)).lower()


def _extract_collection_distance(collection) -> str:
    """Extracts the configured vector distance metric name from a Weaviate collection config."""
    try:
        config = collection.config.get()
    except Exception:
        return "cosine"

    direct = _read_distance_value(getattr(config, "vector_index_config", None))
    if direct:
        return direct

    vector_config = getattr(config, "vector_config", None)
    if isinstance(vector_config, dict) and vector_config:
        first_named = next(iter(vector_config.values()))
        named = _read_distance_value(getattr(first_named, "vector_index_config", None))
        if named:
            return named
    return "cosine"


def _to_float_list(values: Any) -> List[float]:
    """Converts a sequence of numeric coordinates into standard Python floats."""
    return [float(x) for x in values]


def _unpack_object_vectors(raw_vector: Any) -> Dict[str, List[float]]:
    """Normalizes single unnamed vectors and multi-vector dictionaries into column mappings."""
    if isinstance(raw_vector, list):
        return {"vector": _to_float_list(raw_vector)}
    if not isinstance(raw_vector, dict):
        return {}
    if list(raw_vector.keys()) == ["default"]:
        return {"vector": _to_float_list(raw_vector["default"])}
    return {
        name: _to_float_list(vec) for name, vec in raw_vector.items() if vec is not None
    }


class ExportWeaviate(ExportVDB):
    """Exports collections, metadata properties, and single or named vectors from Weaviate."""

    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        """Registers CLI flags for exporting Weaviate collections to VDF."""
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )
        parser_weaviate.add_argument(
            "-u",
            "--url",
            type=str,
            help="URL of Weaviate instance (default: http://localhost:8080)",
        )
        parser_weaviate.add_argument(
            "--api_key",
            type=str,
            help="Weaviate API key (optional for local instances)",
        )
        parser_weaviate.add_argument(
            "-c",
            "--classes",
            type=str,
            help="Classes/collections to export (comma-separated)",
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            default=DEFAULT_GRPC_PORT,
            help="gRPC port for local or custom Weaviate instances (default: 50051)",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            default=DEFAULT_BATCH_SIZE,
            help="Batch size for flushing exported vectors",
        )

    @classmethod
    def export_vdb(cls, args):
        """Prompts for missing Weaviate connection arguments and runs the collection export."""
        set_arg_from_input(
            args,
            "url",
            f"Enter the URL of Weaviate instance (default: '{DEFAULT_LOCAL_URL}'): ",
            str,
            DEFAULT_LOCAL_URL,
            env_var="WEAVIATE_URL",
        )
        if not _is_local_url(args.get("url") or DEFAULT_LOCAL_URL):
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key (hit return to skip for unauthenticated instances): ",
                "WEAVIATE_API_KEY",
            )
        weaviate_export = ExportWeaviate(args)
        weaviate_export.all_classes = weaviate_export.get_all_index_names()
        set_arg_from_input(
            weaviate_export.args,
            "classes",
            "Enter the name of the classes to export (comma-separated, all will be exported by default): ",
            str,
            choices=weaviate_export.all_classes,
        )
        weaviate_export.get_data()
        return weaviate_export

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate_client(self.args)
        self.all_classes: List[str] = []

    def get_all_index_names(self) -> List[str]:
        """Lists all collection names present in the connected Weaviate instance."""
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [getattr(c, "name", str(c)) for c in collections]

    def get_index_names(self) -> List[str]:
        """Resolves the target collections to export from CLI arguments or full discovery."""
        if not getattr(self, "all_classes", None):
            self.all_classes = self.get_all_index_names()
        raw_classes = self.args.get("classes")
        if not raw_classes:
            return self.all_classes
        requested = [c.strip() for c in raw_classes.split(",") if c.strip()]
        missing = set(requested) - set(self.all_classes)
        if missing:
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {missing}"
            )
        return [c for c in self.all_classes if c in requested]

    def _write_multi_vector_parquet(self, state: CollectionExportState) -> int:
        """Writes multi-vector records directly to a Parquet shard and updates the unified schema."""
        all_ids = list(state.vectors.keys() | state.metadata.keys())
        rows = [
            {
                ID_COLUMN: obj_id,
                **state.metadata.get(obj_id, {}),
                **state.vectors.get(obj_id, {}),
            }
            for obj_id in all_ids
        ]
        df = pd.DataFrame.from_records(rows)
        parquet_file = os.path.join(state.vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(parquet_file)
        schema = pq.read_schema(parquet_file)
        self.parquet_schema = (
            pa.unify_schemas([self.parquet_schema, schema])
            if hasattr(self, "parquet_schema")
            else schema
        )
        self.file_structure.append(parquet_file)
        self.file_ctr += 1
        return len(df)

    def _flush_state_to_parquet(self, state: CollectionExportState) -> int:
        """Persists buffered single-vector or multi-vector records to a numbered Parquet shard."""
        if not state.vectors and not state.metadata:
            return 0
        if state.vector_columns == ["vector"]:
            single_vectors = {
                obj_id: vec_map.get("vector")
                for obj_id, vec_map in state.vectors.items()
            }
            written = self.save_vectors_to_parquet(
                single_vectors, state.metadata, state.vectors_directory
            )
        else:
            written = self._write_multi_vector_parquet(state)
        state.vectors.clear()
        state.metadata.clear()
        return written

    def _accumulate_object(self, obj, state: CollectionExportState) -> None:
        """Extracts UUID, properties, and vectors from a Weaviate item into the export buffer."""
        obj_id = str(getattr(obj, "uuid", ""))
        vec_map = _unpack_object_vectors(getattr(obj, "vector", None))
        if vec_map:
            state.vectors[obj_id] = vec_map
            for col_name, vec_values in vec_map.items():
                if col_name not in state.vector_columns:
                    state.vector_columns.append(col_name)
                if state.dimensions == -1 and vec_values:
                    state.dimensions = len(vec_values)
        props = getattr(obj, "properties", None) or {}
        state.metadata[obj_id] = dict(props)

    def export_collection(self, class_name: str) -> List[NamespaceMeta]:
        """Streams all objects from a single Weaviate collection into Parquet files and builds metadata."""
        collection = self.client.collections.get(class_name)
        agg = collection.aggregate.over_all(total_count=True)
        total_count = getattr(agg, "total_count", 0) or 0
        vectors_directory = self.create_vec_dir(class_name)
        state = CollectionExportState(
            class_name=class_name,
            vectors_directory=vectors_directory,
            vectors={},
            metadata={},
            vector_columns=[],
        )

        pbar = tqdm(total=total_count, desc=f"Exporting {class_name}")
        for obj in collection.iterator(include_vector=True):
            self._accumulate_object(obj, state)
            pbar.update(1)
            buffer_bytes = sys.getsizeof(state.vectors) + sys.getsizeof(state.metadata)
            if buffer_bytes > DISK_SPACE_LIMIT:
                state.exported_count += self._flush_state_to_parquet(state)

        state.exported_count += self._flush_state_to_parquet(state)
        pbar.close()

        namespace_meta = self.get_namespace_meta(
            class_name,
            vectors_directory,
            total=total_count or state.exported_count,
            num_vectors_exported=state.exported_count,
            dim=state.dimensions,
            vector_columns=state.vector_columns or ["vector"],
            distance=_extract_collection_distance(collection),
        )
        self.args["exported_count"] += state.exported_count
        return [namespace_meta]

    def get_data(self):
        """Exports all selected Weaviate collections and writes VDF_META.json."""
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for class_name in tqdm(index_names, desc="Fetching Weaviate collections"):
            index_metas[class_name] = self.export_collection(class_name)

        meta_path = os.path.join(self.vdf_directory, "VDF_META.json")
        self.file_structure.append(meta_path)
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(meta_path, "w") as json_file:
            json_file.write(meta_text)
        return True
