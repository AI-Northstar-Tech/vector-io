import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vdf_io.export_vdf.weaviate_export import (
    ExportWeaviate,
    _unpack_object_vectors,
    connect_weaviate_client,
)
from vdf_io.import_vdf.weaviate_import import (
    ImportWeaviate,
    _normalize_weaviate_name,
    _resolve_weaviate_distance,
    _to_weaviate_uuid,
)
from vdf_io.names import DBNames
from vdf_io.util import standardize_metric, standardize_metric_reverse


class FakeWeaviateCollection:
    """In-memory fake Weaviate collection supporting iterator, aggregate, config, and bulk insertion."""

    def __init__(self, items, distance_value="cosine"):
        self._items = list(items)
        self._distance_value = distance_value
        self.inserted_batches = []
        self.aggregate = SimpleNamespace(
            over_all=lambda total_count=True: SimpleNamespace(
                total_count=len(self._items)
            )
        )
        self.config = SimpleNamespace(
            get=lambda: SimpleNamespace(
                vector_index_config=SimpleNamespace(
                    distance_metric=SimpleNamespace(value=self._distance_value)
                ),
                vector_config=None,
            )
        )
        self.data = SimpleNamespace(insert_many=self._insert_many)

    def iterator(self, include_vector=True):
        """Yields stored fake Weaviate objects during collection export."""
        return iter(self._items)

    def _insert_many(self, objects):
        """Records batches of DataObject instances inserted during VDF import."""
        self.inserted_batches.append(list(objects))


class FakeWeaviateClient:
    """In-memory fake Weaviate v4 client tracking created and queried collections."""

    def __init__(self, initial_collections=None):
        self._store = dict(initial_collections or {})
        self.created_configs = {}
        self.collections = SimpleNamespace(
            list_all=lambda: {name: {} for name in self._store},
            get=self._get_collection,
            create=self._create_collection,
        )

    def _get_collection(self, name: str):
        """Returns an existing fake collection or initializes an empty one on demand."""
        if name not in self._store:
            self._store[name] = FakeWeaviateCollection([])
        return self._store[name]

    def _create_collection(self, name: str, **kwargs):
        """Registers a newly created collection and records its vector index configuration."""
        self.created_configs[name] = kwargs
        coll = FakeWeaviateCollection([])
        self._store[name] = coll
        return coll


class TestWeaviateSupport(unittest.TestCase):
    """Verifies Weaviate connection routing, metric standardization, export, and import."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="vdf_weaviate_test_")
        self.orig_cwd = os.getcwd()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("vdf_io.export_vdf.weaviate_export.weaviate.connect_to_local")
    def test_connect_local_instance_when_url_is_localhost(self, mock_local):
        """Localhost URLs must route to connect_to_local without requiring an API key."""
        connect_weaviate_client({"url": "http://localhost:8080"})
        mock_local.assert_called_once_with(
            host="localhost",
            port=8080,
            grpc_port=50051,
            auth_credentials=None,
            skip_init_checks=True,
        )

    @patch("vdf_io.export_vdf.weaviate_export.weaviate.connect_to_wcs")
    def test_connect_cloud_instance_for_wcs_domain(self, mock_wcs):
        """Weaviate Cloud Services URLs must route to connect_to_wcs with API key auth."""
        connect_weaviate_client(
            {
                "url": "https://my-cluster.weaviate.network",
                "api_key": "wcs-secret-key",
            }
        )
        mock_wcs.assert_called_once()
        call_kwargs = mock_wcs.call_args.kwargs
        self.assertEqual(
            call_kwargs["cluster_url"], "https://my-cluster.weaviate.network"
        )
        self.assertIsNotNone(call_kwargs["auth_credentials"])

    def test_distance_metric_roundtrip_and_vector_unpacking(self):
        """Distance metrics and single/named vector payloads must map losslessly."""
        std_cos = standardize_metric("cosine", DBNames.WEAVIATE)
        self.assertEqual(
            standardize_metric_reverse(std_cos, DBNames.WEAVIATE), "cosine"
        )
        std_l2 = standardize_metric("l2-squared", DBNames.WEAVIATE)
        self.assertEqual(
            _resolve_weaviate_distance({"metric": std_l2}).value, "l2-squared"
        )
        self.assertEqual(
            _unpack_object_vectors({"default": [1, 2]}), {"vector": [1.0, 2.0]}
        )
        self.assertEqual(
            _unpack_object_vectors({"title": [0.5], "body": [0.25]}),
            {"title": [0.5], "body": [0.25]},
        )

    @patch("vdf_io.export_vdf.weaviate_export.connect_weaviate_client")
    def test_export_and_import_single_and_multi_vector_collections(
        self, mock_export_conn
    ):
        """End-to-end VDF export and import must preserve single and named vector schemas."""
        single_items = [
            SimpleNamespace(
                uuid="11111111-1111-1111-1111-111111111111",
                vector={"default": [0.1, 0.2, 0.3]},
                properties={"title": "Hello"},
            )
        ]
        multi_items = [
            SimpleNamespace(
                uuid="22222222-2222-2222-2222-222222222222",
                vector={"title_vec": [0.4, 0.5], "body_vec": [0.6, 0.7]},
                properties={"category": "news"},
            )
        ]
        source_client = FakeWeaviateClient(
            {
                "Article": FakeWeaviateCollection(single_items, "cosine"),
                "MultiDoc": FakeWeaviateCollection(multi_items, "dot"),
            }
        )
        mock_export_conn.return_value = source_client

        exporter = ExportWeaviate(
            {
                "url": "http://localhost:8080",
                "library_version": "0.1.247",
            }
        )
        exporter.get_data()

        meta_path = os.path.join(exporter.vdf_directory, "VDF_META.json")
        self.assertTrue(os.path.isfile(meta_path))
        with open(meta_path) as f:
            vdf_meta = json.load(f)
        self.assertEqual(
            vdf_meta["indexes"]["Article"][0]["vector_columns"], ["vector"]
        )
        self.assertEqual(
            vdf_meta["indexes"]["MultiDoc"][0]["vector_columns"],
            ["title_vec", "body_vec"],
        )

        target_client = FakeWeaviateClient()
        with patch(
            "vdf_io.import_vdf.weaviate_import.connect_weaviate_client",
            return_value=target_client,
        ):
            importer = ImportWeaviate(
                {
                    "dir": os.path.join(self.temp_dir, exporter.vdf_directory),
                    "cwd": self.temp_dir,
                    "url": "http://localhost:8080",
                    "library_version": "0.1.247",
                }
            )
            importer.upsert_data()

        self.assertEqual(importer.args["imported_count"], 2)
        self.assertIn("Article", target_client.created_configs)
        self.assertIn("MultiDoc", target_client.created_configs)
        self.assertEqual(_normalize_weaviate_name("my-index"), "My_index")
        self.assertEqual(len(_to_weaviate_uuid("row-42")), 36)


if __name__ == "__main__":
    unittest.main()
