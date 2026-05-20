from __future__ import annotations

from enum import Enum
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import weaviate

from qdrant_client.http.models import Distance

from vdf_io.names import DBNames
from vdf_io.util import standardize_metric_reverse


def connect_weaviate(args):
    url = args.get("url") or "http://localhost:8080"
    api_key = args.get("api_key")
    grpc_host = args.get("grpc_host")
    grpc_port = args.get("grpc_port")
    auth_credentials = _api_key_auth(api_key) if api_key else None

    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname or "localhost"
    secure = parsed.scheme == "https"
    http_port = parsed.port or (443 if secure else 8080)

    if _is_local_host(host) and not api_key:
        return _call_connect(
            weaviate.connect_to_local,
            host=host,
            port=http_port,
            grpc_port=grpc_port or 50051,
        )

    if hasattr(weaviate, "connect_to_custom"):
        kwargs = {
            "http_host": host,
            "http_port": http_port,
            "http_secure": secure,
            "grpc_host": grpc_host or host,
            "grpc_port": grpc_port or (443 if secure else 50051),
            "grpc_secure": secure,
        }
        if auth_credentials is not None:
            kwargs["auth_credentials"] = auth_credentials
        return _call_connect(weaviate.connect_to_custom, **kwargs)

    return _call_connect(
        weaviate.connect_to_wcs,
        cluster_url=url,
        auth_credentials=auth_credentials,
    )


def collection_names(client):
    return list(client.collections.list_all().keys())


def object_to_plain_dict(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: object_to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [object_to_plain_dict(v) for v in value]
    for method_name in ("to_dict", "model_dump", "dict"):
        method = getattr(value, method_name, None)
        if callable(method):
            try:
                return object_to_plain_dict(method())
            except TypeError:
                continue
    if hasattr(value, "__dict__"):
        return object_to_plain_dict(
            {k: v for k, v in value.__dict__.items() if not k.startswith("_")}
        )
    return str(value)


def extract_distance(index_config):
    config = object_to_plain_dict(index_config)
    if not isinstance(config, dict):
        return None
    return _find_key(config, "distance_metric") or _find_key(config, "distance")


def to_weaviate_distance(metric):
    distance = standardize_metric_reverse(metric or Distance.COSINE, DBNames.WEAVIATE)
    try:
        from weaviate.classes.config import VectorDistances

        return {
            "cosine": VectorDistances.COSINE,
            "dot": VectorDistances.DOT,
            "l2-squared": VectorDistances.L2_SQUARED,
            "manhattan": VectorDistances.MANHATTAN,
        }.get(distance, VectorDistances.COSINE)
    except Exception:
        return distance


def build_vector_config(vector_column_names, distance):
    from weaviate.classes.config import Configure

    vector_distance = to_weaviate_distance(distance)

    if hasattr(Configure, "Vectors"):
        vector_index_config = Configure.VectorIndex.hnsw(
            distance_metric=vector_distance
        )
        if len(vector_column_names) == 1:
            try:
                return {
                    "vector_config": Configure.Vectors.self_provided(
                        vector_index_config=vector_index_config
                    )
                }
            except TypeError:
                return {
                    "vector_config": Configure.Vectors.self_provided(
                        name=vector_column_names[0],
                        vector_index_config=vector_index_config,
                    )
                }
        return {
            "vector_config": [
                Configure.Vectors.self_provided(
                    name=vector_column_name,
                    vector_index_config=vector_index_config,
                )
                for vector_column_name in vector_column_names
            ]
        }

    return {"vectorizer_config": Configure.Vectorizer.none()}


def clean_property_value(value):
    if value is None:
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        cleaned = [clean_property_value(v) for v in value]
        return [v for v in cleaned if v is not None]
    if isinstance(value, dict):
        return {k: clean_property_value(v) for k, v in value.items()}
    if pd.isna(value):
        return None
    return str(value)


def _api_key_auth(api_key):
    if hasattr(weaviate, "auth") and hasattr(weaviate.auth, "AuthApiKey"):
        return weaviate.auth.AuthApiKey(api_key)

    from weaviate.classes.init import Auth

    return Auth.api_key(api_key)


def _call_connect(connect_fn, **kwargs):
    try:
        return connect_fn(**kwargs, skip_init_checks=True)
    except TypeError:
        return connect_fn(**kwargs)


def _is_local_host(host):
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _find_key(value, target_key):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == target_key:
                return item.value if isinstance(item, Enum) else item
            found = _find_key(item, target_key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_key(item, target_key)
            if found is not None:
                return found
    return None
