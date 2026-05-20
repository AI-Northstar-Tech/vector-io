import ast
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


def class_methods(path, class_name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                item.name
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    raise AssertionError(f"{class_name} not found in {path}")


class WeaviateSupportStaticTests(unittest.TestCase):
    def test_export_weaviate_implements_export_contract(self):
        methods = class_methods(
            REPO_ROOT / "src" / "vdf_io" / "export_vdf" / "weaviate_export.py",
            "ExportWeaviate",
        )

        self.assertTrue(
            {
                "make_parser",
                "export_vdb",
                "get_all_index_names",
                "get_index_names",
                "get_data",
            }.issubset(methods)
        )

    def test_import_weaviate_implements_import_contract(self):
        import_path = REPO_ROOT / "src" / "vdf_io" / "import_vdf" / "weaviate_import.py"
        self.assertTrue(import_path.exists())

        methods = class_methods(import_path, "ImportWeaviate")
        self.assertTrue(
            {
                "make_parser",
                "import_vdb",
                "get_all_index_names",
                "upsert_data",
                "create_collection",
                "build_vector_payload",
            }.issubset(methods)
        )


if __name__ == "__main__":
    unittest.main()
