import inspect
import math
import re
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, UUID, uuid5


LOCAL_WEAVIATE_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def is_weaviate_cloud_url(url):
    if not url:
        return False
    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = parsed.hostname or ""
    return host.endswith(".weaviate.cloud") or host.endswith(".weaviate.network")


def _call_with_supported_kwargs(func, **kwargs):
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(**kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return func(**kwargs)
    supported_kwargs = {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return func(**supported_kwargs)


def _get_weaviate_auth(api_key):
    if not api_key:
        return None
    try:
        from weaviate.classes.init import Auth

        return Auth.api_key(api_key)
    except (ImportError, AttributeError):
        import weaviate

        return weaviate.auth.AuthApiKey(api_key)


def _parse_weaviate_url(url, default_port=8080):
    parsed = urlparse(url if "://" in url else f"http://{url}")
    secure = parsed.scheme == "https"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if secure else default_port)
    return host, port, secure


def connect_to_weaviate(args):
    import weaviate

    url = args.get("url") or "http://localhost:8080"
    deployment = args.get("deployment")
    api_key = args.get("api_key")
    host, port, secure = _parse_weaviate_url(url)
    grpc_port = args.get("grpc_port") or (443 if secure else 50051)
    grpc_secure = args.get("grpc_secure")
    if grpc_secure is None:
        grpc_secure = secure

    auth_credentials = _get_weaviate_auth(api_key)
    if deployment is None:
        if is_weaviate_cloud_url(url):
            deployment = "cloud"
        elif host in LOCAL_WEAVIATE_HOSTS:
            deployment = "local"
        else:
            deployment = "custom"

    if deployment == "cloud":
        cloud_connector = getattr(weaviate, "connect_to_weaviate_cloud", None)
        if cloud_connector is None:
            cloud_connector = getattr(weaviate, "connect_to_wcs", None)
        if cloud_connector is None:
            raise RuntimeError(
                "Installed weaviate-client does not expose a cloud connection helper"
            )
        return _call_with_supported_kwargs(
            cloud_connector,
            cluster_url=url,
            auth_credentials=auth_credentials,
            skip_init_checks=True,
        )

    if deployment == "local":
        return _call_with_supported_kwargs(
            weaviate.connect_to_local,
            host=host,
            port=port,
            grpc_port=grpc_port,
            auth_credentials=auth_credentials,
            skip_init_checks=True,
        )

    custom_connector = getattr(weaviate, "connect_to_custom", None)
    if custom_connector is None:
        raise RuntimeError(
            "Installed weaviate-client does not expose connect_to_custom"
        )
    grpc_host = args.get("grpc_host") or host
    return _call_with_supported_kwargs(
        custom_connector,
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=grpc_secure,
        auth_credentials=auth_credentials,
        skip_init_checks=True,
    )


def list_collection_names(client):
    collections = client.collections.list_all()
    if hasattr(collections, "keys"):
        return list(collections.keys())
    return [getattr(collection, "name", str(collection)) for collection in collections]


def use_collection(client, collection_name):
    if hasattr(client.collections, "use"):
        return client.collections.use(collection_name)
    return client.collections.get(collection_name)


def collection_count(collection):
    try:
        return len(collection)
    except TypeError:
        pass
    response = collection.aggregate.over_all(total_count=True)
    return response.total_count


def get_collection_config(collection):
    if not hasattr(collection, "config"):
        return None
    try:
        return collection.config.get()
    except Exception:
        return None


def to_plain_dict(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except TypeError:
            return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain_dict(item) for item in value]
    if hasattr(value, "__dict__"):
        return {
            key: to_plain_dict(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return str(value)


def find_distance_metric(config):
    config = to_plain_dict(config)

    def find_distance(value):
        if isinstance(value, dict):
            for key in ("distance", "distance_metric", "distanceMetric"):
                if value.get(key):
                    return value[key]
            for nested_value in value.values():
                distance = find_distance(nested_value)
                if distance:
                    return distance
        elif isinstance(value, list):
            for nested_value in value:
                distance = find_distance(nested_value)
                if distance:
                    return distance
        return None

    return find_distance(config)


def _to_python_value(value):
    if hasattr(value, "as_py"):
        value = value.as_py()
    elif hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            value = value.item()
        except ValueError:
            pass

    if not isinstance(value, (list, tuple, dict)):
        try:
            is_missing = value != value
        except (TypeError, ValueError):
            is_missing = False
        if isinstance(is_missing, bool) and is_missing:
            return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            item = _to_python_value(item)
            if item is not None:
                cleaned[str(key)] = item
        return cleaned or None
    if isinstance(value, (list, tuple)):
        return [_to_python_value(item) for item in value]
    return value


def _sanitize_property_name(name, existing_names):
    safe_name = re.sub(r"\W+", "_", str(name)).strip("_")
    if not safe_name:
        safe_name = "property"
    if safe_name[0].isdigit():
        safe_name = f"property_{safe_name}"

    candidate = safe_name
    suffix = 2
    while candidate in existing_names:
        candidate = f"{safe_name}_{suffix}"
        suffix += 1
    existing_names.add(candidate)
    return candidate


def sanitize_properties(row, excluded_columns):
    excluded_columns = set(excluded_columns)
    properties = {}
    used_names = set()
    for key, value in row.items():
        if key in excluded_columns:
            continue
        value = _to_python_value(value)
        if value is None:
            continue
        properties[_sanitize_property_name(key, used_names)] = value
    return properties


def normalize_vector_value(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [normalize_vector_value(item) for item in value]
    return float(value)


def normalize_vector_map(vector):
    if vector is None:
        return {}
    if hasattr(vector, "model_dump"):
        vector = vector.model_dump()
    elif hasattr(vector, "to_dict"):
        vector = vector.to_dict()
    elif hasattr(vector, "__dict__") and not isinstance(vector, (list, tuple, dict)):
        vector = {
            key: value for key, value in vars(vector).items() if not key.startswith("_")
        }

    if isinstance(vector, dict):
        normalized = {}
        for name, value in vector.items():
            vector_value = normalize_vector_value(value)
            if vector_value is None:
                continue
            column_name = "vector" if name in ("", "default", None) else str(name)
            normalized[column_name] = vector_value
        return normalized

    return {"vector": normalize_vector_value(vector)}


def generate_weaviate_uuid(collection_name, object_id):
    try:
        return str(UUID(str(object_id)))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, f"vector-io:{collection_name}:{object_id}"))


def weaviate_metric_from_vdf(metric):
    metric = getattr(metric, "value", metric)
    metric_text = str(metric or "").lower()
    if "euclid" in metric_text or "l2" in metric_text:
        return "l2-squared"
    if "dot" in metric_text or metric_text in {"ip", "inner_product"}:
        return "dot"
    if "manhattan" in metric_text or "l1" in metric_text:
        return "manhattan"
    return "cosine"


def _weaviate_distance_enum(metric):
    metric = weaviate_metric_from_vdf(metric)
    try:
        from weaviate.classes.config import VectorDistances
    except ImportError:
        return metric
    distance_names = {
        "cosine": "COSINE",
        "dot": "DOT",
        "l2-squared": "L2_SQUARED",
        "manhattan": "MANHATTAN",
    }
    return getattr(VectorDistances, distance_names[metric], metric)


def _vector_index_config(metric):
    try:
        from weaviate.classes.config import Configure
    except ImportError:
        return None
    try:
        return Configure.VectorIndex.hnsw(
            distance_metric=_weaviate_distance_enum(metric)
        )
    except (AttributeError, TypeError):
        return None


def _self_provided_vector_config(name=None, vector_index_config=None):
    from weaviate.classes.config import Configure

    kwargs = {}
    if name is not None:
        kwargs["name"] = name
    if vector_index_config is not None:
        kwargs["vector_index_config"] = vector_index_config

    try:
        return Configure.Vectors.self_provided(**kwargs)
    except (AttributeError, TypeError):
        kwargs.pop("vector_index_config", None)
        try:
            return Configure.Vectors.self_provided(**kwargs)
        except (AttributeError, TypeError):
            pass

    try:
        return Configure.NamedVectors.none(**kwargs)
    except (AttributeError, TypeError):
        if name is not None:
            return Configure.NamedVectors.none(name=name)
        return Configure.Vectorizer.none()


def build_vector_config(vector_columns, metric):
    vector_columns = vector_columns or ["vector"]
    vector_index_config = _vector_index_config(metric)
    if vector_columns == ["vector"]:
        return _self_provided_vector_config(vector_index_config=vector_index_config)
    return [
        _self_provided_vector_config(
            name=vector_column, vector_index_config=vector_index_config
        )
        for vector_column in vector_columns
    ]


def batch_context(collection, batch_size):
    if hasattr(collection.batch, "fixed_size"):
        return collection.batch.fixed_size(batch_size=batch_size)
    if hasattr(collection.batch, "dynamic"):
        return collection.batch.dynamic()
    return collection.batch
