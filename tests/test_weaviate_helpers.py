import unittest
from types import SimpleNamespace
from uuid import UUID

from vdf_io.weaviate_helpers import (
    build_weaviate_properties,
    coerce_weaviate_uuid,
    normalize_weaviate_vectors,
    split_weaviate_object,
)


class WeaviateHelpersTest(unittest.TestCase):
    def test_normalizes_default_vector(self):
        self.assertEqual(
            normalize_weaviate_vectors([0.1, 0.2, 0.3]),
            {"vector": [0.1, 0.2, 0.3]},
        )

    def test_normalizes_named_vectors(self):
        raw_vectors = {
            "dense": (0.1, 0.2),
            "sparse": [],
            "rerank": [0.3, 0.4],
            "missing": None,
        }

        self.assertEqual(
            normalize_weaviate_vectors(raw_vectors),
            {"dense": [0.1, 0.2], "rerank": [0.3, 0.4]},
        )

    def test_normalizes_weaviate_default_named_vector_to_vdf_vector(self):
        self.assertEqual(
            normalize_weaviate_vectors({"default": [0.1, 0.2]}),
            {"vector": [0.1, 0.2]},
        )

    def test_split_weaviate_object_keeps_properties_and_vectors_separate(self):
        obj = SimpleNamespace(
            uuid="43c5d1fa-c3b7-4f19-b73f-3ca57f4ed7ac",
            properties={"title": "hello", "rank": 2},
            vector={"dense": [0.1, 0.2], "rerank": [0.3, 0.4]},
        )

        object_id, vectors, metadata = split_weaviate_object(obj)

        self.assertEqual(object_id, "43c5d1fa-c3b7-4f19-b73f-3ca57f4ed7ac")
        self.assertEqual(vectors, {"dense": [0.1, 0.2], "rerank": [0.3, 0.4]})
        self.assertEqual(
            metadata,
            {
                "id": "43c5d1fa-c3b7-4f19-b73f-3ca57f4ed7ac",
                "title": "hello",
                "rank": 2,
            },
        )

    def test_build_weaviate_properties_removes_id_and_vector_columns(self):
        row = {
            "id": "source-1",
            "vector": [0.1, 0.2],
            "rerank": [0.3, 0.4],
            "title": "hello",
        }

        self.assertEqual(
            build_weaviate_properties(row, "id", ["vector", "rerank"]),
            {"title": "hello"},
        )

    def test_coerce_weaviate_uuid_is_valid_and_stable(self):
        source_id = "source-1"

        coerced_once = coerce_weaviate_uuid(source_id)
        coerced_twice = coerce_weaviate_uuid(source_id)

        self.assertEqual(coerced_once, coerced_twice)
        self.assertEqual(str(UUID(coerced_once)), coerced_once)

    def test_coerce_weaviate_uuid_preserves_existing_uuid(self):
        existing_uuid = "43c5d1fa-c3b7-4f19-b73f-3ca57f4ed7ac"

        self.assertEqual(coerce_weaviate_uuid(existing_uuid), existing_uuid)


if __name__ == "__main__":
    unittest.main()
