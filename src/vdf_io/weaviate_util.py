from __future__ import annotations

import inspect
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5

from vdf_io.names import DBNames
from vdf_io.util import standardize_metric_reverse


DEFAULT_WEAVIATE_URL = "http://localhost:8080"
DEFAULT_WEAVIATE_GRPC_PORT = 50051


def normalized_weaviate_url(url):
    if not url:
        url = DEFAULT_WEAVIATE_URL
    if "://" not in url:
        url = f"http://{url}"
    return url


def parsed_weaviate_url(url):
    return urlparse(normalized_weaviate_url(url))


def is_local_weaviate_url(url):
    parsed = parsed_weaviate_url(url)
    return parsed.hostname in {"localhost", "127.0.0.1", "::1"}


def call_with_supported_kwargs(func, **kwargs):
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(
            **{key: value for key, value in kwargs.items() if value is not None}
        )
    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return func(**supported_kwargs)


def get_weaviate_auth(weaviate_module, api_key):
    if not api_key:
        return None
    return weaviate_module.auth.AuthApiKey(api_key)


def connect_weaviate(args):
    import weaviate

    url = normalized_weaviate_url(args.get("url"))
    api_key = args.get("api_key")
    grpc_port = int(args.get("grpc_port") or DEFAULT_WEAVIATE_GRPC_PORT)
    skip_init_checks = args.get("skip_init_checks", True)
    parsed = parsed_weaviate_url(url)
    host = parsed.hostname or "localhost"
    http_port = parsed.port or (443 if parsed.scheme == "https" else 8080)
    auth_credentials = get_weaviate_auth(weaviate, api_key)

    if (
        is_local_weaviate_url(url)
        and not api_key
        and hasattr(weaviate, "connect_to_local")
    ):
        return call_with_supported_kwargs(
            weaviate.connect_to_local,
            host=host,
            port=http_port,
            grpc_port=grpc_port,
            skip_init_checks=skip_init_checks,
        )

    if api_key and hasattr(weaviate, "connect_to_weaviate_cloud"):
        return call_with_supported_kwargs(
            weaviate.connect_to_weaviate_cloud,
            cluster_url=url,
            auth_credentials=auth_credentials,
            skip_init_checks=skip_init_checks,
        )

    if api_key and hasattr(weaviate, "connect_to_wcs"):
        return call_with_supported_kwargs(
            weaviate.connect_to_wcs,
            cluster_url=url,
            auth_credentials=auth_credentials,
            skip_init_checks=skip_init_checks,
        )

    if hasattr(weaviate, "connect_to_custom"):
        return call_with_supported_kwargs(
            weaviate.connect_to_custom,
            http_host=host,
            http_port=http_port,
            http_secure=parsed.scheme == "https",
            grpc_host=args.get("grpc_host") or host,
            grpc_port=grpc_port,
            grpc_secure=args.get("grpc_secure", parsed.scheme == "https"),
            auth_credentials=auth_credentials,
            skip_init_checks=skip_init_checks,
        )

    return call_with_supported_kwargs(
        weaviate.connect_to_local,
        host=host,
        port=http_port,
        grpc_port=grpc_port,
        skip_init_checks=skip_init_checks,
    )


def list_weaviate_collection_names(client):
    collections = client.collections.list_all()
    if isinstance(collections, dict):
        return list(collections.keys())
    names = []
    for collection in collections:
        names.append(getattr(collection, "name", collection))
    return names


def get_weaviate_collection(client, collection_name):
    collections = client.collections
    if hasattr(collections, "use"):
        return collections.use(collection_name)
    if hasattr(collections, "get"):
        try:
            return collections.get(collection_name)
        except TypeError:
            return collections.get(name=collection_name)
    raise AttributeError("Weaviate client collections object has no use/get method")


def weaviate_collection_exists(client, collection_name):
    if hasattr(client.collections, "exists"):
        return client.collections.exists(collection_name)
    return collection_name in list_weaviate_collection_names(client)


def standard_metric_to_weaviate(metric):
    if metric is None:
        return "cosine"
    metric_text = str(metric).lower()
    if "cos" in metric_text:
        return "cosine"
    if "euclid" in metric_text or "l2" in metric_text:
        return "l2-squared"
    if "dot" in metric_text:
        return "dot"
    if "manhattan" in metric_text:
        return "manhattan"
    return standardize_metric_reverse(metric, DBNames.WEAVIATE)


def get_weaviate_distance_config(metric):
    try:
        import weaviate.classes.config as wvcc
    except ImportError:
        return None

    if not hasattr(wvcc, "VectorDistances"):
        return None

    weaviate_metric = standard_metric_to_weaviate(metric)
    distance_map = {
        "cosine": "COSINE",
        "l2-squared": "L2_SQUARED",
        "dot": "DOT",
        "manhattan": "MANHATTAN",
    }
    distance_name = distance_map.get(weaviate_metric)
    if not distance_name or not hasattr(wvcc.VectorDistances, distance_name):
        return None
    return getattr(wvcc.VectorDistances, distance_name)


def build_weaviate_vector_config(vector_columns, metric=None):
    try:
        import weaviate.classes.config as wvcc
    except ImportError:
        return None

    if not hasattr(wvcc, "Configure") or not hasattr(wvcc.Configure, "Vectors"):
        return None

    distance = get_weaviate_distance_config(metric)
    vector_index_config = None
    if distance is not None and hasattr(wvcc.Configure, "VectorIndex"):
        vector_index_config = wvcc.Configure.VectorIndex.hnsw(distance_metric=distance)

    def self_provided_config(vector_name=None):
        kwargs = {}
        if vector_name is not None:
            kwargs["name"] = vector_name
        if vector_index_config is not None:
            kwargs["vector_index_config"] = vector_index_config
        return wvcc.Configure.Vectors.self_provided(**kwargs)

    if len(vector_columns) == 1 and vector_columns[0] == "vector":
        return self_provided_config()
    return [self_provided_config(vector_column) for vector_column in vector_columns]


def serializable_weaviate_config(config):
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    elif hasattr(config, "dict"):
        config = config.dict()
    elif hasattr(config, "__dict__"):
        config = vars(config)

    if isinstance(config, dict):
        return {
            str(key): serializable_weaviate_config(value)
            for key, value in config.items()
            if not str(key).startswith("_")
        }
    if isinstance(config, (list, tuple, set)):
        return [serializable_weaviate_config(value) for value in config]
    if isinstance(config, (str, int, float, bool)) or config is None:
        return config
    return str(config)


def get_weaviate_object_uuid(raw_id):
    try:
        return str(UUID(str(raw_id)))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, str(raw_id)))
