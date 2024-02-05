from pydantic import BaseModel, ConfigDict
from typing import Dict, List
import os
import json
import datetime
from tqdm import tqdm

from pymilvus import connections, utility, Collection

from export_vdf.vdb_export_cls import ExportVDB
from util import standardize_metric
from names import DBNames


MAX_FETCH_SIZE = 1_000


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
    exported_from: str = "milvus"
    indexes: Dict[str, List[NamespaceMeta]]
    exported_at: str


class ExportMilvus(ExportVDB):
    DB_NAME_SLUG = DBNames.MILVUS

    def __init__(self, args: Dict):
        """
        Initialize the class.

        Keys in args:
            uri: optional, a string of connection uri
            token: a string of API key or token
            collections: optional, a string of collection names with comma as seperator
        """
        if args is None:
            args = {}
        assert isinstance(args, dict), "Invalid args."
        super().__init__(args)

        uri = self.args.get("uri", "http://localhost:19530")
        token = self.args.get("token", "")
        connections.connect(uri=uri, token=token)

    def get_data(self) -> bool:
        if self.args.get("collections") is None:
            collection_names = self.get_all_collection_names()
        else:
            collection_names = self.args.get("collections").split(",")

        index_metas = {}
        for collection_name in tqdm(collection_names, desc="Fetching indexes"):
            index_meta = self.get_data_for_collection(collection_name)
            index_metas[collection_name] = index_meta

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = VDFMeta(
            version=self.args.get("library_version"),
            file_structure=self.file_structure,
            author=os.environ.get("USER"),
            exported_from=self.DB_NAME_SLUG,
            indexes=index_metas,
            exported_at=datetime.datetime.now().astimezone().isoformat(),
        )
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json.dump(internal_metadata.dict(), json_file, indent=4)
        # print internal metadata properly
        print(json.dumps(internal_metadata.dict(), indent=4))
        return True

    def get_all_collection_names(self) -> List[str]:
        return utility.list_collections()

    def get_data_for_collection(self, collection_name: str) -> List[NamespaceMeta]:
        vectors_directory = os.path.join(self.vdf_directory, collection_name)
        os.makedirs(vectors_directory, exist_ok=True)

        try:
            collection = Collection(collection_name)
            collection.load()
        except Exception as e:
            raise RuntimeError(f"Load collection failed. \n{e}")

        total = collection.num_entities
        dim = None
        id_field = collection.primary_field.name
        vector_field = None
        all_fields = []
        for f in collection.schema.fields:
            all_fields.append(f.name)
            if f.dtype.value in [100, 101]:
                dim = f.params["dim"]
                vector_field = f.name

        num_vectors_exported = 0
        pbar = tqdm(total=total, desc=f"Exporting {collection_name}")
        query_iterator = collection.query_iterator(
            batch_size=MAX_FETCH_SIZE, output_fields=all_fields
        )

        while True:
            res = query_iterator.next()
            if len(res) == 0:
                query_iterator.close()
                break

            vectors = {}
            metadata = {}
            for res_i in res:
                k = res_i.pop(id_field)
                vectors[k] = res_i.pop(vector_field)
                metadata[k] = res_i
            self.save_vectors_to_parquet(vectors, metadata, vectors_directory)

            num_vectors_exported += len(res)
            pbar.update(len(res))

        namespace_meta = NamespaceMeta(
            namespace="",
            index_name=collection_name,
            total_vector_count=total,
            exported_vector_count=num_vectors_exported,
            vector_columns=[vector_field],
            dimensions=dim,
            model_name=self.args.get("model_name"),
            data_path="/".join(vectors_directory.split("/")[1:]),
            metric=standardize_metric(
                collection.indexes[0].params["metric_type"], self.DB_NAME_SLUG
            ),
        )

        return [namespace_meta]
