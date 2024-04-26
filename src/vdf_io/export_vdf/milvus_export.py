from typing import Dict, List
import os
import json
import datetime
from tqdm import tqdm
from pymilvus import connections, utility, Collection


from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.meta_types import NamespaceMeta, VDFMeta
from vdf_io.util import (
    get_author_name,
    set_arg_from_input,
    set_arg_from_password,
    standardize_metric,
)
from vdf_io.names import DBNames


MAX_FETCH_SIZE = 1_000


class ExportMilvus(ExportVDB):
    DB_NAME_SLUG = DBNames.MILVUS

    @classmethod
    def make_parser(cls, subparsers):
        # Milvus
        parser_milvus = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Milvus"
        )
        parser_milvus.add_argument(
            "-u", "--uri", type=str, help="Milvus connection URI"
        )
        parser_milvus.add_argument(
            "-t", "--token", type=str, required=False, help="Milvus connection token"
        )
        parser_milvus.add_argument(
            "-c", "--collections", type=str, help="Names of collections to export"
        )

    @classmethod
    def export_vdb(cls, args):
        """
        Export data from Milvus
        """
        set_arg_from_input(
            args,
            "uri",
            "Enter the uri of Milvus (hit return for 'http://localhost:19530'): ",
            str,
            "http://localhost:19530",
        )
        set_arg_from_input(
            args,
            "collections",
            "Enter the name of collection(s) to export (comma-separated) (hit return to export all):",
            str,
        )
        set_arg_from_password(
            args,
            "token",
            "Enter your Milvus/Zilliz Token (hit return to skip): ",
            "ZILLIZ_CLOUD_TOKEN",
        )
        milvus_export = ExportMilvus(args)
        milvus_export.get_data()
        return milvus_export

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

        index_metas: Dict[str, List[NamespaceMeta]] = {}
        for collection_name in tqdm(collection_names, desc="Fetching indexes"):
            index_meta = self.get_data_for_collection(collection_name)
            index_metas[collection_name] = index_meta

        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = VDFMeta(
            version=self.args.get("library_version"),
            file_structure=self.file_structure,
            author=get_author_name(),
            exported_from=self.DB_NAME_SLUG,
            indexes=index_metas,
            exported_at=datetime.datetime.now().astimezone().isoformat(),
        )
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json.dump(internal_metadata.model_dump(), json_file, indent=4)
        # print internal metadata properly
        print(json.dumps(internal_metadata.model_dump(), indent=4))
        return True

    def get_all_collection_names(self) -> List[str]:
        return utility.list_collections()

    def get_data_for_collection(self, collection_name: str) -> List[NamespaceMeta]:
        vectors_directory = self.create_vec_dir(collection_name)

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
        self.args["exported_count"] += num_vectors_exported

        return [namespace_meta]
