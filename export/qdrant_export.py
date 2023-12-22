from qdrant_client import QdrantClient
import os
from tqdm import tqdm
import pandas as pd
import sqlite3
from dotenv import load_dotenv

from export.vdb_export import ExportVDB

load_dotenv()


class ExportQdrant(ExportVDB):
    def __init__(self, args):
        """
        Initialize the class
        """
        self.args = args
        try:
            self.client = QdrantClient(
                url=self.args["url"], api_key=self.args["qdrant_api_key"]
            )
        except:
            self.client = QdrantClient(url=self.args["url"])

    def get_all_class_names(self):
        """
        Get all class names from Qdrant
        """
        collections = self.client.get_collections().collections
        class_names = [collection.name for collection in collections]
        return class_names

    def get_data(self):
        if (
            "collections" not in self.args
            or self.args["collections"] is None
            or self.args["collections"] == "all"
        ):
            collection_names = self.get_all_class_names()
        else:
            collection_names = self.args["collections"].split(",")
        for collection_name in collection_names:
            self.get_data_for_class(collection_name)
            print("Exported data from collection {}".format(collection_name))
        print("Exported data from {} collections".format(len(collection_names)))
        return True

    def get_data_for_class(self, class_name):
        print(self.client.get_collection(collection_name=class_name))
        total = self.client.get_collection(collection_name=class_name).points_count
        hash_value = extract_data_hash({"class_name": class_name, "total": total})
        print("Total number of vectors to export: {}".format(total))
        vectors = {}
        metadata = {}
        batch_ctr = 0
        timestamp_in_format = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        vdf_directory = f"vdf_{class_name}_{timestamp_in_format}_{hash_value}"
        vectors_directory = os.path.join(vdf_directory, "vectors_default")
        os.makedirs(vdf_directory, exist_ok=True)
        for offset in tqdm(range(0, total, MAX_FETCH_SIZE)):
            points = self.client.search(
                collection_name=class_name,
                query={"vector": {"top": MAX_FETCH_SIZE, "offset": offset}},
            ).result.points
            for point in points:
                vectors[point.id] = point.payload.vector
                metadata[point.id] = point.payload.metadata
                if len(vectors) >= MAX_PARQUET_FILE_SIZE:
                    batch_ctr += 1
                    num_vectors = self.save_vectors_to_parquet(
                        vectors, metadata, batch_ctr, vectors_directory
                    )
                    print("Saved {} vectors to parquet file".format(num_vectors))
        if len(vectors) > 0:
            batch_ctr += 1
            num_vectors = self.save_vectors_to_parquet(
                vectors, metadata, batch_ctr, vectors_directory
            )
            print("Saved {} vectors to parquet file".format(num_vectors))
        # Create and save internal metadata JSON
        self.file_structure.append(os.path.join(vectors_directory, "VDF_META.json"))
        internal_metadata = {
            "file_structure": self.file_structure,
            # author is from unix username
            "author": os.getlogin(),
            "dimensions": len(vectors[0]),
            "total_vector_count": total,
            "exported_vector_count": len(vectors),
            "exported_from": "qdrant",
            "model_name": "default",
        }
        with open(os.path.join(vectors_directory, "VDF_META.json"), "w") as json_file:
            json.dump(internal_metadata, json_file, indent=4)
        # print internal metadata properly
        print(json.dumps(internal_metadata, indent=4))
        return True
