from dataclasses import dataclass
from typing import Any, Dict, List
from dotenv import load_dotenv
from tqdm import tqdm
import weaviate.classes.config as wvcc
from weaviate.classes.data import DataObject

from vdf_io.constants import DEFAULT_BATCH_SIZE, INT_MAX
from vdf_io.export_vdf.weaviate_export import (
    DEFAULT_GRPC_PORT,
    DEFAULT_LOCAL_URL,
    _is_local_url,
    connect_weaviate_client,
)
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import (
    clean_value,
    cleanup_df,
    divide_into_batches,
    get_qdrant_id_from_id,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric_reverse,
)

load_dotenv()

WEAVIATE_DISTANCE_ENUM_MAP = {
    "cosine": wvcc.VectorDistances.COSINE,
    "l2-squared": wvcc.VectorDistances.L2_SQUARED,
    "dot": wvcc.VectorDistances.DOT,
    "manhattan": wvcc.VectorDistances.MANHATTAN,
    "hamming": wvcc.VectorDistances.HAMMING,
}


@dataclass
class NamespaceImportContext:
    """Encapsulates target collection metadata and vector column configuration during Parquet import."""

    collection: Any
    collection_name: str
    namespace_meta: Dict[str, Any]
    vector_column_names: List[str]


def _resolve_weaviate_distance(namespace_meta: Dict[str, Any]):
    """Maps a standardized VDF distance metric to the corresponding Weaviate VectorDistances enum."""
    raw_metric = namespace_meta.get("metric")
    metric_key = standardize_metric_reverse(raw_metric, DBNames.WEAVIATE)
    return WEAVIATE_DISTANCE_ENUM_MAP.get(metric_key, wvcc.VectorDistances.COSINE)


def _normalize_weaviate_name(raw_name: str) -> str:
    """Formats collection names to start with an uppercase letter per Weaviate GraphQL naming rules."""
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in raw_name)
    if not cleaned:
        return "VdfCollection"
    return cleaned[0].upper() + cleaned[1:]


def _to_weaviate_uuid(raw_id: Any) -> str:
    """Derives a deterministic UUID string for arbitrary integer or text VDF row identifiers."""
    candidate = get_qdrant_id_from_id(str(raw_id))
    if isinstance(candidate, int):
        return get_qdrant_id_from_id(f"vdf-id-{candidate}")
    return str(candidate)


def _sanitize_properties(row_dict: Dict[str, Any], excluded_cols: List[str]) -> Dict[str, Any]:
    """Extracts non-vector columns and strips NaN or unsupported null coordinates for Weaviate properties."""
    props: Dict[str, Any] = {}
    for key, val in row_dict.items():
        if key in excluded_cols:
            continue
        cleaned = clean_value(val)
        if cleaned is not None:
            props[key] = cleaned
    return props


class ImportWeaviate(ImportVDB):
    """Imports VDF Parquet datasets into local, custom, or cloud Weaviate collections."""

    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        """Registers CLI flags for importing VDF datasets into Weaviate."""
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
        )
        parser_weaviate.add_argument(
            "-u",
            "--url",
            type=str,
            default=DEFAULT_LOCAL_URL,
            help="URL of Weaviate instance (default: http://localhost:8080)",
        )
        parser_weaviate.add_argument(
            "--api_key",
            type=str,
            help="Weaviate API key (optional for local instances)",
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
            help="Batch size for inserting DataObjects into Weaviate",
        )

    @classmethod
    def import_vdb(cls, args):
        """Prompts for missing Weaviate target parameters and executes the VDF import."""
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
        weaviate_import = ImportWeaviate(args)
        weaviate_import.upsert_data()
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = connect_weaviate_client(self.args)
        self.total_imported_count = 0

    def get_all_index_names(self) -> List[str]:
        """Lists existing collection names from the target Weaviate instance."""
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [getattr(c, "name", str(c)) for c in collections]

    def _ensure_collection(
        self, collection_name: str, ctx_meta: Dict[str, Any]
    ):
        """Retrieves an existing Weaviate collection or creates one with matching vector and distance settings."""
        existing = self.get_all_index_names()
        if collection_name in existing:
            return self.client.collections.get(collection_name)

        distance_enum = _resolve_weaviate_distance(ctx_meta)
        vec_cols = ctx_meta.get("vector_columns") or ["vector"]
        if len(vec_cols) <= 1 and vec_cols[0] == "vector":
            return self.client.collections.create(
                name=collection_name,
                vectorizer_config=wvcc.Configure.Vectorizer.none(),
                vector_index_config=wvcc.Configure.VectorIndex.hnsw(
                    distance_metric=distance_enum
                ),
            )

        named_configs = [
            wvcc.Configure.NamedVectors.none(
                name=col_name,
                vector_index_config=wvcc.Configure.VectorIndex.hnsw(
                    distance_metric=distance_enum
                ),
            )
            for col_name in vec_cols
        ]
        return self.client.collections.create(
            name=collection_name,
            vectorizer_config=named_configs,
        )

    def _build_row_vector(
        self, row: Any, vector_column_names: List[str]
    ) -> Any:
        """Extracts either a single float list or a named-vector dictionary for a DataObject."""
        if len(vector_column_names) == 1 and vector_column_names[0] == "vector":
            raw_val = row.get("vector")
            return self.extract_vector(raw_val) if raw_val is not None else None

        named_vectors: Dict[str, List[float]] = {}
        for col in vector_column_names:
            raw_val = row.get(col)
            if raw_val is not None:
                extracted = self.extract_vector(raw_val)
                if extracted is not None:
                    named_vectors[col] = extracted
        return named_vectors or None

    def _build_data_objects(
        self, batch_df, ctx: NamespaceImportContext
    ) -> List[DataObject]:
        """Converts a pandas DataFrame batch into Weaviate DataObject instances ready for bulk insertion."""
        excluded = [self.id_column] + ctx.vector_column_names
        objects: List[DataObject] = []
        for _, row in batch_df.iterrows():
            row_dict = row.to_dict()
            vector_payload = self._build_row_vector(row_dict, ctx.vector_column_names)
            if vector_payload is None:
                continue
            raw_id = row_dict.get(self.id_column)
            properties = _sanitize_properties(row_dict, excluded)
            objects.append(
                DataObject(
                    properties=properties,
                    vector=vector_payload,
                    uuid=_to_weaviate_uuid(raw_id),
                )
            )
        return objects

    def _remaining_row_budget(self) -> int:
        """Calculates the remaining row quota when a user-defined max_num_rows cap is active."""
        user_limit = self.args.get("max_num_rows")
        if user_limit is None:
            return INT_MAX
        return user_limit - self.total_imported_count

    def _import_parquet_file(
        self, file_path: str, ctx: NamespaceImportContext
    ) -> bool:
        """Reads a single Parquet file in batches and inserts DataObjects into the target Weaviate collection."""
        max_rows = self._remaining_row_budget()
        if max_rows <= 0:
            return True
        df = cleanup_df(self.read_parquet_progress(file_path, max_num_rows=max_rows))
        batch_size = int(self.args.get("batch_size") or DEFAULT_BATCH_SIZE)

        for batch_df in divide_into_batches(df, batch_size):
            objects = self._build_data_objects(batch_df, ctx)
            if not objects:
                continue
            ctx.collection.data.insert_many(objects)
            self.total_imported_count += len(objects)
            if self.total_imported_count >= (self.args.get("max_num_rows") or INT_MAX):
                return True
        return False

    def _import_namespace(
        self, index_name: str, namespace_meta: Dict[str, Any]
    ) -> bool:
        """Imports all Parquet shards belonging to a single VDF namespace into Weaviate."""
        self.set_dims(namespace_meta, index_name)
        final_data_path = self.get_final_data_path(namespace_meta["data_path"])
        parquet_files = self.get_parquet_files(final_data_path)

        suffix = f"_{namespace_meta['namespace']}" if namespace_meta.get("namespace") else ""
        base_name = _normalize_weaviate_name(f"{index_name}{suffix}")
        target_name = self.create_new_name(base_name, self.get_all_index_names(), delimiter="_")
        vector_column_names, _ = self.get_vector_column_name(
            target_name, namespace_meta, multi_vector_supported=True
        )
        meta_with_cols = {**namespace_meta, "vector_columns": vector_column_names}
        collection = self._ensure_collection(target_name, meta_with_cols)
        ctx = NamespaceImportContext(
            collection=collection,
            collection_name=target_name,
            namespace_meta=meta_with_cols,
            vector_column_names=vector_column_names,
        )

        for file in tqdm(parquet_files, desc=f"Importing files into {target_name}"):
            file_path = self.get_file_path(final_data_path, file)
            if self._import_parquet_file(file_path, ctx):
                return True
        return False

    def upsert_data(self):
        """Iterates over all VDF indexes and namespaces to populate the Weaviate instance."""
        self.total_imported_count = 0
        indexes_content: Dict[str, List[NamespaceMeta]] = self.vdf_meta["indexes"]
        if not indexes_content:
            raise ValueError("No indexes found in VDF_META.json")

        for index_name, index_meta in tqdm(
            indexes_content.items(), desc="Importing Weaviate indexes"
        ):
            for namespace_meta in index_meta:
                if self._import_namespace(index_name, namespace_meta):
                    break
        self.args["imported_count"] = self.total_imported_count
        tqdm.write(
            f"Weaviate import completed successfully ({self.total_imported_count} objects)."
        )
