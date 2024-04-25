from pydantic import BaseModel, ConfigDict
from typing import Any, Dict, List, Optional


class NamespaceMeta(BaseModel):
    namespace: str
    index_name: str
    total_vector_count: int
    exported_vector_count: int
    dimensions: int
    model_name: str | None = None
    model_map: Dict[str, Any] | None = None
    vector_columns: List[str] = ["vector"]
    data_path: str
    metric: str | None = None
    index_config: Optional[Dict[Any, Any]] = None
    # schema_dict is a byte string
    schema_dict_str: str | None = None
    model_config = ConfigDict(protected_namespaces=())


class VDFMeta(BaseModel):
    version: str
    file_structure: List[str]
    author: str
    exported_from: str
    indexes: Dict[str, List[NamespaceMeta]]
    exported_at: str
    id_column: Optional[str] = None
