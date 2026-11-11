import argparse
import os
import sys
import tempfile

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from vdf_io.export_vdf.weaviate_export import (
    ExportWeaviate,
    generate_python_class_for_collection,
    map_vdf_distance_to_weaviate,
    map_weaviate_distance_to_vdf,
)
from vdf_io.import_vdf.weaviate_import import ImportWeaviate


class _MockObj:
    def __init__(self, uuid: str, properties: dict, vector):
        self.uuid = uuid
        self.properties = properties
        self.vector = vector


class _MockConfig:
    def __init__(self, distance="cosine", properties=None):
        self.distance = distance
        self.properties = properties or [{"name": "title", "dataType": ["text"]}]

    def get(self):
        return self


class _MockCollection:
    def __init__(self, name: str, objs=None, distance="cosine"):
        self.name = name
        self._objs = list(objs or [])
        self.config = _MockConfig(distance=distance)
        self.inserted = []

    def iterator(self, include_vector: bool = True):
        return iter(self._objs)

    def insert(self, uuid: str, properties: dict, vector):
        self.inserted.append({"uuid": uuid, "properties": properties, "vector": vector})


class _MockCollectionsNamespace:
    def __init__(self, initial=None):
        self._cols = dict(initial or {})

    def list_all(self):
        return self._cols

    def get(self, name: str):
        return self._cols[name]

    def create(self, name: str, distance_metric: str = "cosine", vector_columns=None):
        self._cols[name] = _MockCollection(name, distance=distance_metric)
        return self._cols[name]


class _MockWeaviateClient:
    def __init__(self, collections=None):
        self.collections = _MockCollectionsNamespace(collections)


def test_distance_metric_bidirectional_mapping():
    assert map_weaviate_distance_to_vdf("cosine") == "Cosine"
    assert map_weaviate_distance_to_vdf("l2-squared") == "Euclid"
    assert map_weaviate_distance_to_vdf("dot") == "Dot"
    assert map_vdf_distance_to_weaviate("Cosine") == "cosine"
    assert map_vdf_distance_to_weaviate("Euclid") == "l2-squared"
    assert map_vdf_distance_to_weaviate("Manhattan") == "manhattan"


def test_automatic_python_class_generation():
    code = generate_python_class_for_collection(
        "Article",
        [
            {"name": "title", "dataType": ["text"]},
            {"name": "views", "dataType": ["int"]},
        ],
    )
    assert "class ArticleRecord:" in code
    assert "title: Optional[str] = None" in code
    assert "views: Optional[int] = None" in code


def test_export_and_import_single_and_multivector_collections():
    with tempfile.TemporaryDirectory() as tmpdir:
        col = _MockCollection(
            "Documents",
            objs=[
                _MockObj(
                    "doc-1",
                    {"title": "Hello Weaviate"},
                    {"title_vec": [0.1, 0.2, 0.3], "body_vec": [0.4, 0.5, 0.6]},
                ),
                _MockObj(
                    "doc-2",
                    {"title": "Second Doc"},
                    {"title_vec": [0.7, 0.8, 0.9], "body_vec": [0.1, 0.1, 0.1]},
                ),
            ],
            distance="dot",
        )
        src_client = _MockWeaviateClient({"Documents": col})
        exporter = ExportWeaviate.export_vdb(
            {
                "url": "http://localhost:8080",
                "vdf_directory": tmpdir,
                "classes": "Documents",
            },
            client_override=src_client,
        )
        assert "Documents" in exporter.index_metas
        meta = exporter.index_metas["Documents"][0]
        assert meta["total_vector_count"] == 2
        assert meta["metric"] == "Dot"
        assert set(meta["vector_columns"]) == {"vector_title_vec", "vector_body_vec"}
        assert os.path.exists(meta["schema_model_path"])

        dst_client = _MockWeaviateClient()
        importer = ImportWeaviate.import_vdb(
            {"url": "http://localhost:8080", "dir": tmpdir},
            client_override=dst_client,
        )
        assert importer.imported_counts["Documents"] == 2
        dst_col = dst_client.collections.get("Documents")
        assert len(dst_col.inserted) == 2
        assert dst_col.inserted[0]["vector"]["title_vec"] == [0.1, 0.2, 0.3]
        assert dst_col.inserted[0]["vector"]["body_vec"] == [0.4, 0.5, 0.6]


def test_export_defaults_to_all_classes_when_unspecified():
    with tempfile.TemporaryDirectory() as tmpdir:
        src_client = _MockWeaviateClient(
            {
                "ClassA": _MockCollection(
                    "ClassA", [_MockObj("a1", {"k": "v"}, [1.0, 2.0])]
                ),
                "ClassB": _MockCollection(
                    "ClassB", [_MockObj("b1", {"k": "w"}, [3.0, 4.0])]
                ),
            }
        )
        exporter = ExportWeaviate.export_vdb(
            {"url": "http://localhost:8080", "vdf_directory": tmpdir},
            client_override=src_client,
        )
        assert set(exporter.index_metas.keys()) == {"ClassA", "ClassB"}


def test_cli_make_parser_registers_weaviate_options():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="db")
    ExportWeaviate.make_parser(sub)
    parsed = parser.parse_args(
        ["weaviate", "--url", "http://localhost:8080", "--classes", "Docs", "--local"]
    )
    assert parsed.url == "http://localhost:8080"
    assert parsed.classes == "Docs"
    assert parsed.local is True
