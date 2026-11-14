from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import weaviate
from tqdm import tqdm

from vdf_io.constants import ID_COLUMN
from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password


DEFAULT_BATCH_SIZE = 1000


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _normalize_headers(headers: Any) -> Optional[Dict[str, str]]:
    if headers is None:
        return None
    if isinstance(headers, dict):
        return {str(k): str(v) for k, v in headers.items()}
    if isinstance(headers, str):
        headers = headers.strip()
        if not headers:
            return None
        try:
            parsed = json.loads(headers)
            if isinstance(parsed, dict):
                return {str(k): str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            pass
        result: Dict[str, str] = {}
        for item in headers.split(","):
            if ":" in item:
                key, value = item.split(":", 1)
                result[key.strip()] = value.strip()
            elif "=" in item:
                key, value = item.split("=", 1)
                result[key.strip()] = value.strip()
        return result or None
    return None


def _standardize_weaviate_metric(distance: Any) -> Optional[str]:
    if distance is None:
        return None
    value = str(distance).strip().lower()
    value = value.replace("vector_distances.", "")
    value = value.replace("vector_distance.", "")
    value = value.replace("_", "")
    value = value.replace("-", "")
    value = value.replace(" ", "")

    if value in {"cosine", "cos"}:
        return "cosine"
    if value in {"dot", "dotproduct"}:
        return "dot"
    if value in {"l2", "l2squared", "euclidean", "euclideansquared", "l2distance"}:
        return "euclidean"
    if value in {"manhattan", "l1", "cityblock"}:
        return "manhattan"
    if value in {"hamming"}:
        return "hamming"
    return str(distance)


def _vector_name_to_vdf(name: str) -> str:
    if name in {"default", "vector"}:
        return "vector"
    return name


def _vdf_vector_name_to_weaviate(name: str) -> str:
    if name in {"vector", "default"}:
        return "default"
    return name


class ExportWeaviate(ExportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Weaviate"
        )
        parser_weaviate.add_argument(
            "--url",
            type=str,
            help="URL of Weaviate cloud instance, e.g. https://xxx.weaviate.cloud",
        )
        parser_weaviate.add_argument(
            "--api_key",
            type=str,
            help="Weaviate API key for cloud instances",
        )
        parser_weaviate.add_argument(
            "--classes",
            type=str,
            help="Classes to export (comma-separated). Defaults to all classes.",
        )
        parser_weaviate.add_argument(
            "--local",
            type=bool,
            help="Use a local Weaviate instance (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--host",
            type=str,
            help="Weaviate host for local/custom connections (default: localhost)",
            default="localhost",
        )
        parser_weaviate.add_argument(
            "--http_port",
            type=int,
            help="Weaviate HTTP port (default: 8080)",
            default=8080,
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            help="Weaviate gRPC port (default: 50051)",
            default=50051,
        )
        parser_weaviate.add_argument(
            "--http_secure",
            type=bool,
            help="Use HTTPS for local/custom connections (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--grpc_secure",
            type=bool,
            help="Use secure gRPC for local/custom connections (default: False)",
            default=False,
            action=argparse.BooleanOptionalAction,
        )
        parser_weaviate.add_argument(
            "--headers",
            type=str,
            help="Additional headers as JSON or comma-separated key:value pairs",
        )
        parser_weaviate.add_argument(
            "--batch_size",
            type=int,
            help=f"Number of objects to fetch per batch (default: {DEFAULT_BATCH_SIZE})",
            default=DEFAULT_BATCH_SIZE,
        )

    @classmethod
    def export_vdb(cls, args):
        set_arg_from_input(
            args,
            "local",
            "Use a local Weaviate instance? (y/N): ",
            bool,
            False,
        )
        if args.get("local"):
            set_arg_from_input(
                args,
                "host",
                "Enter the Weaviate host (default: 'localhost'): ",
                str,
                "localhost",
            )
            set_arg_from_input(
                args,
                "http_port",
                "Enter the Weaviate HTTP port (default: 8080): ",
                int,
                8080,
            )
            set_arg_from_input(
                args,
                "grpc_port",
                "Enter the Weaviate gRPC port (default: 50051): ",
                int,
                50051,
            )
            set_arg_from_input(
                args,
                "http_secure",
                "Use HTTPS for the local/custom connection? (y/N): ",
                bool,
                False,
            )
            set_arg_from_input(
                args,
                "grpc_secure",
                "Use secure gRPC for the local/custom connection? (y/N): ",
                bool,
                False,
            )
        else:
            set_arg_from_input(
                args,
                "url",
                "Enter the URL of Weaviate cloud instance: ",
                str,
            )
            set_arg_from_password(
                args,
                "api_key",
                "Enter the Weaviate API key: ",
                "WEAVIATE_API_KEY",
            )

        set_arg_from_input(
            args,
            "headers",
            "Enter additional Weaviate headers as JSON or comma-separated key:value pairs (optional): ",
            str,
        )
        set_arg_from_input(
            args,
            "batch_size",
            f"Enter the fetch batch size (default: {DEFAULT_BATCH_SIZE}): ",
            int,
            DEFAULT_BATCH_SIZE,
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
        self.client = self._connect()
        self.all_classes: List[str] = []

    def _connect(self):
        headers = _normalize_headers(self.args.get("headers"))
        api_key = self.args.get("api_key")
        auth_credentials = (
            weaviate.auth.AuthApiKey(api_key) if api_key else None
        )

        local = _as_bool(self.args.get("local"), False)
        url = self.args.get("url")

        if local or not url:
            return weaviate.connect_to_local(
                host=self.args.get("host", "localhost"),
                port=int(self.args.get("http_port", 8080)),
                grpc_port=int(self.args.get("grpc_port", 50051)),
                headers=headers,
                skip_init_checks=True,
            )

        if url.startswith("http://") or url.startswith("https://"):
            return weaviate.connect_to_weaviate_cloud(
                cluster_url=url,
                auth_credentials=auth_credentials,
                headers=headers,
                skip_init_checks=True,
            )

        return weaviate.connect_to_custom(
            http_host=url,
            http_port=int(self.args.get("http_port", 8080)),
            http_secure=_as_bool(self.args.get("http_secure"), False),
            grpc_host=self.args.get("host", url),
            grpc_port=int(self.args.get("grpc_port", 50051)),
            grpc_secure=_as_bool(self.args.get("grpc_secure"), False),
            auth_credentials=auth_credentials,
            headers=headers,
            skip_init_checks=True,
        )

    def get_all_index_names(self) -> List[str]:
        collections = self.client.collections.list_all()
        if isinstance(collections, dict):
            return list(collections.keys())
        return [getattr(collection, "name", str(collection)) for collection in collections]

    def get_index_names(self) -> List[str]:
        if not self.args.get("classes"):
            return self.get_all_index_names()

        input_classes = [
            class_name.strip()
            for class_name in str(self.args["classes"]).split(",")
            if class_name.strip()
        ]
        missing = set(input_classes) - set(self.all_classes)
        if missing:
            tqdm.write(
                f"These classes are not present in the Weaviate instance: {missing}"
            )
        return [class_name for class_name in self.all_classes if class_name in input_classes]

    def get_data(self):
        index_names = self.get_index_names()
        index_metas: Dict[str, List[NamespaceMeta]] = {}

        for class_name in tqdm(index_names, desc="Fetching indexes"):
            index_metas[class_name] = self.get_data_for_class(class_name)

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = self.get_basic_vdf_meta(index_metas)
        meta_text = json.dumps(internal_metadata.model_dump(), indent=4)
        tqdm.write(meta_text)
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json_file.write(meta_text)
        return True

    def _get_collection_config(self, collection):
        try:
            return collection.config.get()
        except Exception:
            return None

    def _get_distance(self, collection) -> Optional[str]:
        config = self._get_collection_config(collection)
        if config is None:
            return None

        candidates = []
        for attr in ("vector_index_config", "vector_config"):
            value = getattr(config, attr, None)
            if value is not None:
                candidates.append(value)

        for candidate in candidates:
            distance = getattr(candidate, "distance", None)
            if distance is not None:
                return str(distance)

            if isinstance(candidate, dict):
                distance = candidate.get("distance")
                if distance is not None:
                    return str(distance)
                for nested in candidate.values():
                    nested_distance = getattr(nested, "distance", None)
                    if nested_distance is not None:
                        return str(nested_distance)
                    if isinstance(nested, dict) and nested.get("distance") is not None:
                        return str(nested["distance"])

        return None

    def _get_vector_names_from_config(self, collection) -> List[str]:
        config = self._get_collection_config(collection)
        if config is None:
            return []

        names: List[str] = []
        vector_config = getattr(config, "vector_config", None)
        if isinstance(vector_config, dict):
            names.extend([str(name) for name in vector_config.keys()])
        elif vector_config is not None:
            for attr in ("named_vectors", "vectors"):
                value = getattr(vector_config, attr, None)
                if isinstance(value, dict):
                    names.extend([str(name) for name in value.keys()])

        vector_index_config = getattr(config, "vector_index_config", None)
        if vector_index_config is not None and not names:
            names.append("default")

        return list(dict.fromkeys(names))

    def _extract_vectors(self, obj) -> Dict[str, List[float]]:
        vectors: Dict[str, List[float]] = {}

        raw_vectors = getattr(obj, "vectors", None)
        if isinstance(raw_vectors, dict):
            for name, vector in raw_vectors.items():
                if vector is not None:
                    vectors[_vector_name_to_vdf(str(name))] = list(vector)

        raw_vector = getattr(obj, "vector", None)
        if isinstance(raw_vector, dict):
            for name, vector in raw_vector.items():
                if vector is not None:
                    vectors[_vector_name_to_vdf(str(name))] = list(vector)
        elif isinstance(raw_vector, list):
            vectors["vector"] = list(raw_vector)

        return vectors

    def _get_dimensions(self, collection, vector_names: List[str]) -> Dict[str, int]:
        dimensions: Dict[str, int] = {}
        if not vector_names:
            return dimensions

        try:
            response = collection.query.fetch_objects(
                limit=1,
                include_vector=True,
                return_properties=False,
            )
            if response.objects:
                vectors = self._extract_vectors(response.objects[0])
                for name, vector in vectors.items():
                    dimensions[name] = len(vector)
        except Exception:
            pass

        return dimensions

    def _get_namespace_meta(
        self,
        class_name: str,
        vectors_directory: str,
        total: int,
        num_vectors_exported: int,
        vector_columns: List[str],
        dimensions: Dict[str, int],
        distance: Optional[str],
        index_config: Any,
    ) -> NamespaceMeta:
        model_name = self.args.get("model_name", "NOT_PROVIDED")
        first_dim = -1
        for vector_column in vector_columns:
            if dimensions.get(vector_column, -1) != -1:
                first_dim = dimensions[vector_column]
                break

        model_map = {}
        for vector_column in vector_columns:
            model_map[vector_column] = {
                "model_name": model_name,
                "text_column": "NOT_PROVIDED",
                "dimensions": dimensions.get(vector_column, first_dim),
                "vector_column": vector_column,
            }

        return NamespaceMeta(
            index_name=class_name,
            namespace="",
            total_vector_count=total,
            exported_vector_count=num_vectors_exported,
            metric=_standardize_weaviate_metric(distance),
            dimensions=first_dim,
            model_name=model_name,
            vector_columns=vector_columns,
            model_map=model_map,
            data_path="/".join(vectors_directory.split("/")[1:]),
            schema_dict_str=(
                self.parquet_schema.to_string()
                if hasattr(self, "parquet_schema")
                else None
            ),
            index_config=index_config,
        )

    def get_data_for_class(self, class_name: str) -> List[NamespaceMeta]:
        collection = self.client.collections.get(class_name)
        vectors_directory = self.create_vec_dir(class_name)

        try:
            total = collection.aggregate.over_all(total_count=True).total_count
        except Exception:
            total = 0

        distance = self._get_distance(collection)
        config = self._get_collection_config(collection)
        index_config = config.model_dump() if hasattr(config, "model_dump") else config

        vector_names_from_config = self._get_vector_names_from_config(collection)
        dimensions = self._get_dimensions(collection, vector_names_from_config)

        num_vectors_exported = 0
        batch_size = int(self.args.get("batch_size", DEFAULT_BATCH_SIZE) or DEFAULT_BATCH_SIZE)
        batch_vectors: Dict[Any, Dict[str, List[float]]] = {}
        batch_metadata: Dict[Any, Dict[str, Any]] = {}
        discovered_vector_names: List[str] = list(vector_names_from_config)

        pbar = tqdm(total=total, desc=f"Exporting {class_name}")
        try:
            iterator = collection.iterator(
                include_vector=True,
                return_properties=True,
            )
            for obj in iterator:
                obj_id = getattr(obj, "uuid", None)
                if obj_id is None:
                    continue

                vectors = self._extract_vectors(obj)
                for vector_name, vector in vectors.items():
                    if vector_name not in discovered_vector_names:
                        discovered_vector_names.append(vector_name)
                    if vector_name not in dimensions:
                        dimensions[vector_name] = len(vector)

                metadata = getattr(obj, "properties", None) or {}
                batch_vectors[obj_id] = vectors
                batch_metadata[obj_id] = metadata

                if len(batch_vectors) >= batch_size:
                    num_vectors_exported += self.save_vectors_to_parquet_multi(
                        batch_vectors,
                        batch_metadata,
                        vectors_directory,
                        discovered_vector_names,
                    )
                    pbar.update(len(batch_vectors))
                    batch_vectors = {}
                    batch_metadata = {}

            if batch_vectors:
                num_vectors_exported += self.save_vectors_to_parquet_multi(
                    batch_vectors,
                    batch_metadata,
                    vectors_directory,
                    discovered_vector_names,
                )
                pbar.update(len(batch_vectors))
        finally:
            pbar.close()

        if not discovered_vector_names:
            discovered_vector_names = ["vector"]

        namespace_meta = self._get_namespace_meta(
            class_name,
            vectors_directory,
            total,
            num_vectors_exported,
            discovered_vector_names,
            dimensions,
            distance,
            index_config,
        )
        self.args["exported_count"] += num_vectors_exported
        return [namespace_meta]

    def save_vectors_to_parquet_multi(
        self,
        vectors: Dict[Any, Dict[str, List[float]]],
        metadata: Dict[Any, Dict[str, Any]],
        vectors_directory: str,
        vector_columns: List[str],
    ) -> int:
        records: List[Dict[str, Any]] = []
        all_vector_columns = list(dict.fromkeys(vector_columns))

        for obj_id, obj_vectors in vectors.items():
            record: Dict[str, Any] = {ID_COLUMN: obj_id}
            obj_metadata = metadata.get(obj_id, {}) or {}

            for key, value in obj_metadata.items():
                if key == ID_COLUMN:
                    continue
                if key in all_vector_columns:
                    record[f"metadata_{key}"] = value
                else:
                    record[key] = value

            for vector_column in all_vector_columns:
                record[vector_column] = obj_vectors.get(vector_column)

            records.append(record)

        if not records:
            return 0

        df = pd.DataFrame.from_records(records)
        parquet_file = os.path.join(vectors_directory, f"{self.file_ctr}.parquet")
        df.to_parquet(parquet_file)

        if not hasattr(self, "parquet_schema"):
            self.parquet_schema = pq.read_schema(parquet_file)
        else:
            self.parquet_schema = pa.unify_schemas(
                [self.parquet_schema, pq.read_schema(parquet_file)]
            )

        self.file_structure.append(parquet_file)
        self.file_ctr += 1
        return len(df)
