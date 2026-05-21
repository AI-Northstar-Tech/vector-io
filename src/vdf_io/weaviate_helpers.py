from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Tuple
from uuid import NAMESPACE_URL, UUID, uuid5

from vdf_io.constants import ID_COLUMN


DEFAULT_VECTOR_COLUMN = "vector"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _to_plain_value(value: Any) -> Any:
    if hasattr(value, "item") and not isinstance(value, (list, tuple, dict)):
        try:
            return value.item()
        except ValueError:
            return value
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            key: _to_plain_value(nested_value)
            for key, nested_value in value.items()
            if not _is_missing(nested_value)
        }
    if isinstance(value, tuple):
        return [_to_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_to_plain_value(item) for item in value]
    return value


def _vector_to_list(vector: Any) -> List[Any] | None:
    if vector is None:
        return None
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    if isinstance(vector, tuple):
        vector = list(vector)
    if not isinstance(vector, list) or len(vector) == 0:
        return None
    return [_to_plain_value(value) for value in vector]


def normalize_weaviate_vectors(raw_vector: Any) -> Dict[str, List[Any]]:
    if isinstance(raw_vector, Mapping):
        vectors = {}
        for name, vector in raw_vector.items():
            normalized = _vector_to_list(vector)
            if normalized is not None:
                vector_name = (
                    DEFAULT_VECTOR_COLUMN
                    if len(raw_vector) == 1 and name == "default"
                    else str(name)
                )
                vectors[vector_name] = normalized
        return vectors

    normalized = _vector_to_list(raw_vector)
    if normalized is None:
        return {}
    return {DEFAULT_VECTOR_COLUMN: normalized}


def split_weaviate_object(
    obj: Any, id_column: str = ID_COLUMN
) -> Tuple[str, Dict[str, List[Any]], Dict[str, Any]]:
    object_id = str(getattr(obj, "uuid", getattr(obj, "id", "")))
    properties = getattr(obj, "properties", {}) or {}
    if not isinstance(properties, Mapping):
        properties = dict(properties)

    metadata = {
        key: _to_plain_value(value)
        for key, value in properties.items()
        if not _is_missing(value)
    }
    metadata[id_column] = object_id
    return object_id, normalize_weaviate_vectors(getattr(obj, "vector", None)), metadata


def build_weaviate_properties(
    row: Mapping[str, Any], id_column: str, vector_columns: Iterable[str]
) -> Dict[str, Any]:
    excluded_columns = set(vector_columns)
    excluded_columns.add(id_column)

    properties = {}
    for key, value in row.items():
        if key in excluded_columns or _is_missing(value):
            continue
        properties[key] = _to_plain_value(value)
    return properties


def coerce_weaviate_uuid(value: Any) -> str:
    value = str(value)
    try:
        return str(UUID(value))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, value))


def sanitize_weaviate_collection_name(name: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", str(name)).strip("_")
    if not sanitized:
        sanitized = "VectorIOCollection"
    if not sanitized[0].isalpha():
        sanitized = f"VectorIO_{sanitized}"
    return sanitized[0].upper() + sanitized[1:]


def get_vector_payload(
    row: Mapping[str, Any], vector_columns: Iterable[str]
) -> Dict[str, List[Any]]:
    vectors = {}
    for vector_column in vector_columns:
        vector = _vector_to_list(row.get(vector_column))
        if vector is not None:
            vectors[vector_column] = vector
    return vectors


def first_vector_dimension(vectors: Mapping[str, List[Any]]) -> int:
    for vector in vectors.values():
        if vector:
            return len(vector)
    return -1


def find_distance_metric(config: Any) -> str | None:
    if config is None:
        return None
    if hasattr(config, "model_dump"):
        config = config.model_dump(mode="json")
    elif hasattr(config, "dict"):
        config = config.dict()

    if isinstance(config, Mapping):
        for key, value in config.items():
            if key in {"distance", "distance_metric", "distanceMetric"} and value:
                return str(getattr(value, "value", value))
            nested = find_distance_metric(value)
            if nested:
                return nested
    elif isinstance(config, (list, tuple)):
        for value in config:
            nested = find_distance_metric(value)
            if nested:
                return nested
    else:
        value = getattr(config, "value", None)
        if value is not None:
            return str(value)
    return None


def make_jsonable_config(config: Any) -> Dict[str, Any] | None:
    if config is None:
        return None
    if hasattr(config, "model_dump"):
        return config.model_dump(mode="json")
    if hasattr(config, "dict"):
        return config.dict()
    if isinstance(config, Mapping):
        return {key: _to_plain_value(value) for key, value in config.items()}
    return {"raw_config": str(config)}
