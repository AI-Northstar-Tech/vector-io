import unittest
from types import SimpleNamespace

import pandas as pd

from vdf_io.export_vdf.weaviate_export import ExportWeaviate
from vdf_io.import_vdf.weaviate_import import ImportWeaviate
from vdf_io.weaviate_util import (
    generate_weaviate_uuid,
    normalize_vector_map,
    sanitize_properties,
    weaviate_metric_from_vdf,
)


class FakeBatch:
    def __init__(self):
        self.objects = []
        self.number_errors = 0
        self.failed_objects = []
        self.batch_size = None

    def fixed_size(self, batch_size):
        self.batch_size = batch_size
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_object(self, properties, uuid, vector):
        self.objects.append(
            {
                "properties": properties,
                "uuid": uuid,
                "vector": vector,
            }
        )


class FakeCollection:
    def __init__(self):
        self.batch = FakeBatch()


class WeaviateSupportTests(unittest.TestCase):
    def test_normalize_vector_map_handles_default_and_named_vectors(self):
        self.assertEqual(normalize_vector_map([1, 2]), {"vector": [1.0, 2.0]})
        self.assertEqual(
            normalize_vector_map({"default": [1, 2], "title": (3, 4)}),
            {"vector": [1.0, 2.0], "title": [3.0, 4.0]},
        )

    def test_weaviate_uuid_is_deterministic(self):
        first_uuid = generate_weaviate_uuid("Article", "row-1")
        second_uuid = generate_weaviate_uuid("Article", "row-1")
        self.assertEqual(first_uuid, second_uuid)
        self.assertEqual(
            generate_weaviate_uuid("Article", "00000000-0000-0000-0000-000000000001"),
            "00000000-0000-0000-0000-000000000001",
        )

    def test_metric_mapping_uses_weaviate_names(self):
        self.assertEqual(weaviate_metric_from_vdf("Cosine"), "cosine")
        self.assertEqual(weaviate_metric_from_vdf("Euclid"), "l2-squared")
        self.assertEqual(weaviate_metric_from_vdf("Dot"), "dot")
        self.assertEqual(weaviate_metric_from_vdf("Manhattan"), "manhattan")

    def test_sanitize_properties_excludes_vectors_and_ids(self):
        properties = sanitize_properties(
            {
                "id": "row-1",
                "title vector": [0.1, 0.2],
                "bad name": "value",
                "empty": None,
            },
            excluded_columns=["id", "title vector"],
        )
        self.assertEqual(properties, {"bad_name": "value"})

    def test_export_row_preserves_uuid_as_vdf_id(self):
        exporter = object.__new__(ExportWeaviate)
        vector_columns = []
        item = SimpleNamespace(
            uuid="00000000-0000-0000-0000-000000000001",
            properties={"id": "metadata-id", "title": "A"},
            vector={"default": [0.1, 0.2], "body_vector": [0.3, 0.4]},
        )

        row = exporter._item_to_row(item, vector_columns)

        self.assertEqual(row["id"], "00000000-0000-0000-0000-000000000001")
        self.assertEqual(vector_columns, ["vector", "body_vector"])
        self.assertEqual(row["vector"], [0.1, 0.2])
        self.assertEqual(row["body_vector"], [0.3, 0.4])
        self.assertEqual(row["title"], "A")

    def test_import_batch_uses_named_vectors(self):
        importer = object.__new__(ImportWeaviate)
        importer.args = {"batch_size": 2}
        importer.id_column = "id"
        importer.abnormal_vector_format = False
        collection = FakeCollection()
        df = pd.DataFrame(
            [
                {
                    "id": "row-1",
                    "title": "A",
                    "title_vector": [0.1, 0.2],
                    "body_vector": [0.3, 0.4],
                },
                {
                    "id": "row-2",
                    "title": "B",
                    "title_vector": [0.5, 0.6],
                    "body_vector": [0.7, 0.8],
                },
            ]
        )

        imported_count = importer.upsert_batch(
            collection,
            "Article",
            df,
            ["title_vector", "body_vector"],
        )

        self.assertEqual(imported_count, 2)
        self.assertEqual(collection.batch.batch_size, 2)
        self.assertEqual(
            collection.batch.objects[0]["vector"],
            {"title_vector": [0.1, 0.2], "body_vector": [0.3, 0.4]},
        )
        self.assertEqual(collection.batch.objects[0]["properties"], {"title": "A"})


if __name__ == "__main__":
    unittest.main()
