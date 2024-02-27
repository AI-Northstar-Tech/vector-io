from vdf_io.export_vdf.vdb_export_cls import ExportVDB
from vdf_io.names import DBNames

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import VectorStoreQuery

from vdf_io.util import set_arg_from_input


class ExportLlamaIndex(ExportVDB):
    DB_NAME_SLUG = DBNames.LLAMA_INDEX

    @classmethod
    def make_parser(cls, subparsers):
        parser_llamaindex = subparsers.add_parser(
            "llamaindex", help="Export data from LlamaIndex"
        )
        parser_llamaindex.add_argument(
            "--persist_dir", type=str, help="Location of LlamaIndex persistent storage"
        )

    @classmethod
    def export_vdb(cls, args):
        """
        Export data from LlamaIndex
        """
        set_arg_from_input(
            args,
            "persist_dir",
            "Enter the location of LlamaIndex persistent storage: ",
            str,
        )
        export_obj = ExportLlamaIndex(args)
        export_obj.get_data()
        return export_obj

    def __init__(self, args):
        """
        Initialize the class
        """
        super().__init__(args)

    def get_data(self) -> ExportVDB:
        # rebuild storage context
        storage_context = StorageContext.from_defaults(
            persist_dir=self.args["persist_dir"]
        )
        # load index
        index = load_index_from_storage(storage_context)
        self.index = index
        from rich import print

        docs = self.index._storage_context.docstore.docs
        # print(docs)
        print(
            self.index._storage_context.vector_store.query(
                query=VectorStoreQuery(node_ids=docs.keys())
            )
        )
        return self
