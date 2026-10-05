from typing import Any

from pydantic import BaseModel, ConfigDict


class NamespaceMeta(BaseModel):
    namespace: str
    index_name: str
    total_vector_count: int
    exported_vector_count: int
    dimensions: int
    model_name: str | None = None
    model_map: dict[str, Any] | None = None
    vector_columns: list[str] = ["vector"]
    data_path: str
    metric: str | None = None
    index_config: dict[Any, Any] | None = None
    # schema_dict is a byte string
    schema_dict_str: str | None = None
    model_config = ConfigDict(protected_namespaces=())


class VDFMeta(BaseModel):
    version: str
    file_structure: list[str]
    author: str
    exported_from: str
    indexes: dict[str, list[NamespaceMeta]]
    exported_at: str
    id_column: str | None = None
