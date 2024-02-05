from pydantic import BaseModel, ConfigDict
from typing import Dict, List


class NamespaceMeta(BaseModel):
    namespace: str
    index_name: str
    total_vector_count: int
    exported_vector_count: int
    dimensions: int
    model_name: str
    vector_columns: List[str] = ["vector"]
    data_path: str
    metric: str
    model_config = ConfigDict(protected_namespaces=())


class VDFMeta(BaseModel):
    version: str
    file_structure: List[str]
    author: str
    exported_from: str
    indexes: Dict[str, List[NamespaceMeta]]
    exported_at: str
