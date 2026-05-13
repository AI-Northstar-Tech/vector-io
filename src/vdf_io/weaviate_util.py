import argparse
import math
import re
from urllib.parse import urlparse
from uuid import UUID

import numpy as np
import pandas as pd
import weaviate
from weaviate.classes.config import Configure, DataType, Property, VectorDistances
from weaviate.classes.init import Auth
from weaviate.util import generate_uuid5

from vdf_io.constants import ID_COLUMN
from vdf_io.names import DBNames
from vdf_io.util import standardize_metric_reverse


WEAVIATE_DISTANCE_METRICS = {
    "cosine": VectorDistances.COSINE,
    "l2-squared": VectorDistances.L2_SQUARED,
    "dot": VectorDistances.DOT,
    "manhattan": VectorDistances.MANHATTAN,
}

VALID_PROPERTY_NAME_RE = re.compile(r"^[a-z][A-Za-z0-9_]*$")


def make_weaviate_parser(parser):
    parser.add_argument(
        "--url",
        type=str,
        help="Weaviate Cloud URL or custom HTTP(S) endpoint",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="Weaviate API key. Defaults to WEAVIATE_API_KEY when set.",
    )
    parser.add_argument(
        "--local",
        help="Connect to a local Weaviate instance",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Local/custom Weaviate host. Default: localhost",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Local/custom Weaviate HTTP port. Default: 8080",
    )
    parser.add_argument(
        "--grpc_host",
        type=str,
        help="Custom Weaviate gRPC host. Defaults to --host or --url host",
    )
    parser.add_argument(
        "--grpc_port",
        type=int,
        default=50051,
        help="Local/custom Weaviate gRPC port. Default: 50051",
    )
    parser.add_argument(
        "--secure",
        help="Use HTTPS/gRPC TLS for a custom endpoint",
        default=None,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--skip_init_checks",
        help="Skip Weaviate client startup checks",
        default=True,
        action=argparse.BooleanOptionalAction,
    )


def connect_weaviate(args):
    api_key = args.get("api_key")
    if not api_key:
        import os

        api_key = os.getenv("WEAVIATE_API_KEY")
    auth_credentials = auth_api_key(api_key) if api_key else None
    skip_init_checks = args.get("skip_init_checks", True)
    url = args.get("url")

    if url:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        cluster_url = parsed.geturl()
        host = parsed.hostname or url
        http_secure = (
            parsed.scheme == "https"
            if args.get("secure") is None
            else bool(args.get("secure"))
        )
        http_port = parsed.port or (443 if http_secure else 80)
        grpc_host = args.get("grpc_host") or host
        grpc_port = args.get("grpc_port") or (443 if http_secure else 50051)

        if "weaviate.cloud" in host or "weaviate.network" in host:
            if auth_credentials is None:
                raise ValueError("Weaviate Cloud connections require --api_key")
            connect_to_cloud = getattr(
                weaviate, "connect_to_weaviate_cloud", None
            ) or getattr(weaviate, "connect_to_wcs")
            return connect_to_cloud(
                cluster_url=cluster_url,
                auth_credentials=auth_credentials,
                skip_init_checks=skip_init_checks,
            )

        return weaviate.connect_to_custom(
            http_host=host,
            http_port=http_port,
            http_secure=http_secure,
            grpc_host=grpc_host,
            grpc_port=grpc_port,
            grpc_secure=http_secure,
            auth_credentials=auth_credentials,
            skip_init_checks=skip_init_checks,
        )

    return weaviate.connect_to_local(
        host=args.get("host") or "localhost",
        port=args.get("port") or 8080,
        grpc_port=args.get("grpc_port") or 50051,
        auth_credentials=auth_credentials,
        skip_init_checks=skip_init_checks,
    )


def auth_api_key(api_key):
    if hasattr(Auth, "api_key"):
        return Auth.api_key(api_key)
    return weaviate.auth.AuthApiKey(api_key)


def collection_names(client):
    return list(client.collections.list_all().keys())


def compliant_collection_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", str(name))
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "ImportedCollection"
    if not cleaned[0].isalpha():
        cleaned = f"Collection_{cleaned}"
    return cleaned[0].upper() + cleaned[1:]


def uuid_for_id(value):
    value = str(value)
    try:
        return str(UUID(value))
    except ValueError:
        return generate_uuid5(value)


def normalize_weaviate_vectors(raw_vector):
    if raw_vector is None:
        return {}
    if isinstance(raw_vector, dict):
        vectors = {}
        for name, vector in raw_vector.items():
            if vector is None:
                continue
            column_name = "vector" if name in ("default", None) else str(name)
            vectors[column_name] = normalize_vector_value(vector)
        return vectors
    return {"vector": normalize_vector_value(raw_vector)}


def normalize_vector_value(vector):
    if isinstance(vector, np.ndarray):
        return vector.tolist()
    if hasattr(vector, "tolist"):
        return vector.tolist()
    return list(vector) if hasattr(vector, "__iter__") else vector


def first_vector_dimension(vector):
    if vector is None:
        return -1
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    if isinstance(vector, list) and vector and isinstance(vector[0], list):
        return len(vector[0])
    try:
        return len(vector)
    except TypeError:
        return -1


def get_weaviate_distance(collection_config, vector_column=None):
    vector_index_config = getattr(collection_config, "vector_index_config", None)
    vector_config = getattr(collection_config, "vector_config", None) or {}
    if vector_column and vector_config:
        config_key = "default" if vector_column == "vector" else vector_column
        named_config = vector_config.get(config_key)
        if named_config is not None:
            vector_index_config = getattr(named_config, "vector_index_config", None)

    distance = getattr(vector_index_config, "distance_metric", None)
    if hasattr(distance, "value"):
        return distance.value
    return distance


def serialize_weaviate_config(value):
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): serialize_weaviate_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_weaviate_config(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    if hasattr(value, "__dict__"):
        return {
            k: serialize_weaviate_config(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def vector_index_config_for_metric(metric):
    metric_name = standardize_metric_reverse(metric, DBNames.WEAVIATE)
    distance = WEAVIATE_DISTANCE_METRICS.get(metric_name, VectorDistances.COSINE)
    return Configure.VectorIndex.hnsw(distance_metric=distance)


def vector_config_for_columns(vector_columns, metric):
    vector_index_config = vector_index_config_for_metric(metric)
    if len(vector_columns) == 1 and vector_columns[0] == "vector":
        return Configure.Vectors.self_provided(vector_index_config=vector_index_config)
    return [
        Configure.Vectors.self_provided(
            name=column, vector_index_config=vector_index_config
        )
        for column in vector_columns
    ]


def infer_weaviate_properties(df, vector_columns, id_column=ID_COLUMN):
    properties = []
    for column in df.columns:
        if column == id_column or column in vector_columns:
            continue
        if not VALID_PROPERTY_NAME_RE.match(str(column)):
            continue
        data_type = infer_weaviate_data_type(df[column])
        if data_type is not None:
            properties.append(Property(name=str(column), data_type=data_type))
    return properties


def infer_weaviate_data_type(series):
    first = None
    for value in series:
        value = normalize_metadata_value(value)
        if value is not None:
            first = value
            break
    if first is None:
        return None
    if isinstance(first, bool):
        return DataType.BOOL
    if isinstance(first, int) and not isinstance(first, bool):
        return DataType.INT
    if isinstance(first, float):
        return DataType.NUMBER
    if isinstance(first, str):
        return DataType.TEXT
    if isinstance(first, list):
        if all(isinstance(v, bool) for v in first):
            return DataType.BOOL_ARRAY
        if all(isinstance(v, int) and not isinstance(v, bool) for v in first):
            return DataType.INT_ARRAY
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in first):
            return DataType.NUMBER_ARRAY
        if all(isinstance(v, str) for v in first):
            return DataType.TEXT_ARRAY
    return None


def normalize_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return [normalize_metadata_value(v) for v in value]
    if isinstance(value, dict):
        normalized = {}
        for k, v in value.items():
            normalized_value = normalize_metadata_value(v)
            if normalized_value is not None:
                normalized[str(k)] = normalized_value
        return normalized
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def is_supported_property_value(value):
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(isinstance(v, (str, int, float, bool)) for v in value)
    return False


def row_to_properties(row, vector_columns, id_column=ID_COLUMN):
    properties = {"vdf_id": str(row[id_column])}
    for column, value in row.items():
        if column == id_column or column in vector_columns:
            continue
        if not VALID_PROPERTY_NAME_RE.match(str(column)):
            continue
        normalized = normalize_metadata_value(value)
        if normalized is not None and is_supported_property_value(normalized):
            properties[str(column)] = normalized
    return properties
