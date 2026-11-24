import argparse
import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qdrant_client.http.models import Distance

from vdf_io.export_vdf.weaviate_export import (
    ExportWeaviate,
    connect_weaviate_client,
    generate_weaviate_dataclass_code,
)
from vdf_io.import_vdf.weaviate_import import (
    ImportWeaviate,
    normalize_weaviate_class_name,
    to_weaviate_uuid,
)
from vdf_io.names import DBNames
from vdf_io.util import standardize_metric, standardize_metric_reverse


class FakeCollectionData:
    def __init__(self):
        self.inserted_batches = []

    def insert_many(self, objs):
        self.inserted_batches.append(objs)


class FakeCollection:
    def __init__(self, items=None, distance="cosine", named_metrics=None, props=None):
        self._items = items or []
        self._distance = distance
        self._named_metrics = named_metrics or {}
        self._props = props or []
        self.data = FakeCollectionData()
        self.aggregate = SimpleNamespace(
            over_all=lambda total_count=True: SimpleNamespace(
                total_count=len(self._items)
            )
        )
        self.config = SimpleNamespace(get=self._get_config)

    def _get_config(self):
        vec_cfg = {
            k: SimpleNamespace(
                vector_index_config=SimpleNamespace(
                    distance_metric=SimpleNamespace(value=v)
                )
            )
            for k, v in self._named_metrics.items()
        }
        prop_objs = [
            SimpleNamespace(name=p["name"], data_type=SimpleNamespace(value=p["type"]))
            for p in self._props
        ]
        return SimpleNamespace(
            vector_index_config=SimpleNamespace(
                distance_metric=SimpleNamespace(value=self._distance)
            ),
            vector_config=vec_cfg if vec_cfg else None,
            properties=prop_objs,
        )

    def iterator(self, include_vector=True):
        return iter(self._items)


class FakeWeaviateClient:
    def __init__(self, collections_map=None):
        self._collections = dict(collections_map or {})
        self.created_specs = []
        self.closed = False
        self.collections = SimpleNamespace(
            list_all=lambda: dict(self._collections),
            get=lambda name: self._collections[name],
            create=self._create,
        )

    def _create(self, name, **kwargs):
        col = FakeCollection()
        self._collections[name] = col
        self.created_specs.append((name, kwargs))
        return col

    def close(self):
        self.closed = True


def test_weaviate_distance_metric_mapping():
    assert standardize_metric("cosine", DBNames.WEAVIATE) == Distance.COSINE
    assert standardize_metric("COSINE", DBNames.WEAVIATE) == Distance.COSINE
    assert standardize_metric("l2-squared", DBNames.WEAVIATE) == Distance.EUCLID
    assert standardize_metric("dot", DBNames.WEAVIATE) == Distance.DOT
    assert standardize_metric_reverse(Distance.COSINE, DBNames.WEAVIATE) == "cosine"
    assert standardize_metric_reverse(Distance.EUCLID, DBNames.WEAVIATE) == "l2-squared"
    assert standardize_metric_reverse(Distance.DOT, DBNames.WEAVIATE) == "dot"


def test_connect_weaviate_local_and_cloud_routing():
    fake_mod = MagicMock()
    fake_mod.auth.AuthApiKey.side_effect = lambda k: f"key:{k}"
    with patch("vdf_io.export_vdf.weaviate_export.weaviate", fake_mod):
        connect_weaviate_client("http://localhost:8080", api_key=None, grpc_port=50051)
        fake_mod.connect_to_local.assert_called_once()
        connect_weaviate_client("https://cluster.weaviate.network", api_key="wcs")
        fake_mod.connect_to_weaviate_cloud.assert_called_once()
        connect_weaviate_client(
            "https://internal.corp:8443", api_key="k", grpc_port=50052
        )
        fake_mod.connect_to_custom.assert_called_once()


def test_export_and_import_single_and_multi_vector_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    single_items = [
        SimpleNamespace(
            uuid="11111111-1111-1111-1111-111111111111",
            properties={"title": "Doc A", "views": 10},
            vector=[0.1, 0.2, 0.3],
        ),
        SimpleNamespace(
            uuid="22222222-2222-2222-2222-222222222222",
            properties={"title": "Doc B", "views": 25},
            vector={"default": [0.4, 0.5, 0.6]},
        ),
    ]
    multi_items = [
        SimpleNamespace(
            uuid="33333333-3333-3333-3333-333333333333",
            properties={"sku": "SKU-1"},
            vector={"title_vec": [1.0, 0.0], "body_vec": [0.0, 1.0, 0.5]},
        ),
        SimpleNamespace(
            uuid="44444444-4444-4444-4444-444444444444",
            properties={"sku": "SKU-2"},
            vector={"title_vec": [0.5, 0.5], "body_vec": [0.2, 0.8, 0.1]},
        ),
    ]
    src_client = FakeWeaviateClient(
        {
            "Article": FakeCollection(
                items=single_items,
                distance="dot",
                props=[
                    {"name": "title", "type": "text"},
                    {"name": "views", "type": "int"},
                ],
            ),
            "MultiVecProduct": FakeCollection(
                items=multi_items,
                named_metrics={"title_vec": "cosine", "body_vec": "l2-squared"},
                props=[{"name": "sku", "type": "text"}],
            ),
        }
    )

    exporter = ExportWeaviate(
        {
            "library_version": "0.1.0",
            "url": "http://localhost:8080",
            "classes": None,
            "batch_size": 10,
        },
        client=src_client,
    )
    assert exporter.get_index_names() == ["Article", "MultiVecProduct"]
    assert exporter.get_data() is True

    vdf_dir = os.path.join(tmp_path, exporter.vdf_directory)
    meta = json.loads(
        open(os.path.join(vdf_dir, "VDF_META.json"), encoding="utf-8").read()
    )
    assert meta["indexes"]["Article"][0]["vector_columns"] == ["vector"]
    assert meta["indexes"]["Article"][0]["metric"] == Distance.DOT
    assert meta["indexes"]["MultiVecProduct"][0]["vector_columns"] == [
        "title_vec",
        "body_vec",
    ]
    assert (
        meta["indexes"]["MultiVecProduct"][0]["model_map"]["body_vec"]["dimensions"]
        == 3
    )

    dst_client = FakeWeaviateClient()
    importer = ImportWeaviate(
        {
            "dir": vdf_dir,
            "library_version": "0.1.0",
            "url": "http://localhost:8080",
            "batch_size": 2,
        },
        client=dst_client,
    )
    assert importer.upsert_data() == 4
    assert len(dst_client.created_specs) == 2

    m_obj = dst_client._collections["MultiVecProduct"].data.inserted_batches[0][0]
    m_vec = m_obj["vector"] if isinstance(m_obj, dict) else getattr(m_obj, "vector")
    assert m_vec == {"title_vec": [1.0, 0.0], "body_vec": [0.0, 1.0, 0.5]}


def test_cli_parsers_and_helpers():
    parser = argparse.ArgumentParser()
    ExportWeaviate.make_parser(parser.add_subparsers())
    parsed = parser.parse_args(
        ["weaviate", "--url", "http://localhost:8080", "--classes", "Article"]
    )
    assert parsed.classes == "Article"

    parser_in = argparse.ArgumentParser()
    ImportWeaviate.make_parser(parser_in.add_subparsers())
    parsed_in = parser_in.parse_args(["weaviate", "--url", "http://localhost:8080"])
    assert parsed_in.url == "http://localhost:8080"

    code = generate_weaviate_dataclass_code(
        "Article", [{"name": "title", "data_type": "text"}]
    )
    assert "class Article:" in code
    assert normalize_weaviate_class_name("my-index") == "My_index"
    assert to_weaviate_uuid("doc-42") == to_weaviate_uuid("doc-42")
