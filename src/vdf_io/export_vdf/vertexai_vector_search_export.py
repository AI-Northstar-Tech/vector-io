"""
Export data from vertex ai vector search index
"""

import os
import json
from tqdm import tqdm

import google.auth
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex  # as vs
from google.cloud.aiplatform import MatchingEngineIndexEndpoint  # as vsep

from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, standardize_metric
from vdf_io.export_vdf.vdb_export_cls import ExportVDB


class ExportVertexAIVectorSearch(ExportVDB):
    DB_NAME_SLUG = DBNames.VERTEXAI

    @classmethod
    def make_parser(cls, subparsers):
        parser_vertexai_vectorsearch = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Export data from Vertex AI Vector Search"
        )
        parser_vertexai_vectorsearch.add_argument(
            "-p", "--project-id", type=str, help="Google Cloud Project ID"
        )
        parser_vertexai_vectorsearch.add_argument(
            "-i", "--index", type=str, help="Name of the index or indexes to export"
        )
        parser_vertexai_vectorsearch.add_argument(
            "-c",
            "--gcloud-credentials-file",
            type=str,
            help="Path to Google Cloud service account credentials file",
            default=None,
        )
        parser_vertexai_vectorsearch.add_argument(
            "-m",
            "--max_vectors",
            type=str,
            help="Optional: max vectors to retrieve",
            default=None,
        )

    @classmethod
    def export_vdb(cls, args):
        """
        Export data from Vertex AI Vector Search
        """
        set_arg_from_input(args, "project_id", "Enter the Google Cloud Project ID: ")
        set_arg_from_input(
            args,
            "index",
            "Enter name of index to export (hit return to export all. Comma separated for multiple indexes): ",
        )
        set_arg_from_input(
            args,
            "gcloud_credentials_file",
            "Enter path to service account credentials file (hit return to use application default credentials): ",
        )
        # max_vectors
        set_arg_from_input(
            args,
            "max_vectors",
            "Optional: max_vectors to export; can be larger than actual vector count",
        )
        vertexai_vectorsearch_export = ExportVertexAIVectorSearch(args)
        vertexai_vectorsearch_export.get_data()
        return vertexai_vectorsearch_export

    def __init__(self, args):
        """
        Initialize the class
        """
        # call super class constructor
        super().__init__(args)
        max_vectors = args["max_vectors"]
        self.max_vectors = max_vectors if max_vectors is not None else None

        try:
            # set gcloud credentials by loading credentials file if provided
            # else default credentials will be ascertained from the environment
            self.credentials = None
            if self.args.get("gcloud_credentials_file"):
                self.credentials, _ = google.auth.load_credentials_from_file(
                    self.args.get("gcloud_credentials_file")
                )
            # initialize vertex ai client
            aiplatform.init(
                project=self.args.get("project_id"), credentials=self.credentials
            )
        except Exception as e:
            raise Exception("Failed to initialize Vertex AI Client") from e

    def get_all_index_r_names(self):
        # get all index resource_names
        index_names = [index.resource_name for index in MatchingEngineIndex.list()]
        return index_names

    def get_all_index_d_names(self):
        # get all index display_names
        index_names = [index.display_name for index in MatchingEngineIndex.list()]
        return index_names

    def get_data(self):
        index_names = []
        # index_endpoint_names = [] # todo - rm

        all_index_r_names = self.get_all_index_r_names()  # was all_index_names
        all_index_d_names = self.get_all_index_d_names()

        try:
            # find index from user input args
            if "index" not in self.args or self.args["index"] is None:
                index_names = all_index_r_names
                print(f"No index provided; exporting from all {index_names} indexes")
            else:
                i_arg = self.args["index"]
                if isinstance(i_arg, str):
                    indexes = i_arg.split(",")
                else:
                    indexes = i_arg

                print(f"indexes: {indexes}")
                for index_arg in indexes:
                    d_ids = []
                    if index_arg in all_index_r_names:
                        # resource_name given
                        index_names.append(index_arg)
                    elif index_arg in all_index_d_names:
                        # display name given
                        i_arg_d_list = [
                            index.resource_name
                            for index in MatchingEngineIndex.list(
                                filter=f"display_name={index_arg}",
                            )
                            if index.display_name == index_arg
                        ]
                        index_names.append(i_arg_d_list[0])
                    else:
                        for index_r in all_index_r_names:
                            test_index = MatchingEngineIndex(index_name=index_r)
                            if test_index.deployed_indexes:
                                # grabbing all deployed indexes
                                d_ids.extend(test_index.deployed_indexes)

                    # returning only those that match
                    indexes_deployed_test = [
                        d_id
                        for d_id in d_ids
                        if (
                            d_id.display_name == index_arg
                            or d_id.deployed_index_id == index_arg
                        )
                    ]
                    if indexes_deployed_test:
                        target_endpoint = MatchingEngineIndexEndpoint(
                            indexes_deployed_test[0].index_endpoint
                        )
                        for d in target_endpoint.deployed_indexes:
                            if d.id == index_arg or d.display_name == index_arg:
                                index_names.append(d.index)

        except ValueError as ve:
            print("Could not find given index value")
            raise ve

        index_metas = {}
        for index_name in tqdm(index_names, desc="Fetching indexes"):
            index_meta = self.get_data_for_index(index_name)
            index_metas[index_name] = index_meta

        # Create and save internal metadata JSON
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = {
            "version": self.args["library_version"],
            "file_structure": self.file_structure,
            "author": os.environ.get("USER"),
            "exported_from": self.DB_NAME_SLUG,
            "indexes": index_metas,
        }
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json.dump(internal_metadata, json_file, indent=4)
        # print internal metadata properly
        tqdm.write(json.dumps(internal_metadata, indent=4))
        return True

    def get_index_endpoint_name(self, index_name):
        index = MatchingEngineIndex(index_name=index_name)
        index_endpoints = [
            (deployed_index.index_endpoint, deployed_index.deployed_index_id)
            for deployed_index in index.deployed_indexes
        ]

        index_endpoint_name = None
        if len(index_endpoints) > 0:
            index_endpoint_name = index_endpoints[0][0]
            deployed_index_id = index_endpoints[0][1]
        else:
            raise Exception(
                "Index not deployed to an endpoint. Cannot export index data"
            )

        return index_endpoint_name, deployed_index_id

    def get_data_for_index(self, index_name):
        index_meta_list = []
        # get index endpoint resource id and deployed index id
        index_endpoint_name, deployed_index_id = self.get_index_endpoint_name(
            index_name=index_name
        )
        # define index and index endpoint
        index = MatchingEngineIndex(index_name=index_name)
        index_endpoint = MatchingEngineIndexEndpoint(index_endpoint_name)

        vectors_directory = self.create_vec_dir(index.display_name)

        # get index metadata
        index_meta = index.to_dict()
        total = int(index_meta.get("indexStats", {}).get("vectorsCount"))
        if self.max_vectors:
            total = self.max_vectors

        dim = int(index_meta.get("metadata", {}).get("config", {}).get("dimensions", 0))

        # start exporting
        pbar = tqdm(total=total, desc=f"Exporting {index.display_name}")
        # find nearest neighbors as proxy to export all datapoint ids
        neighbors = index_endpoint.find_neighbors(
            deployed_index_id=deployed_index_id,
            queries=[[0.0] * dim],
            num_neighbors=total,
            return_full_datapoint=False,
        )
        # get full datapoint including metadata
        datapoints = None
        if len(neighbors) > 0:
            datapoint_ids = [p.id for p in neighbors[0]]
            datapoints = index_endpoint.read_index_datapoints(
                deployed_index_id=deployed_index_id, ids=datapoint_ids
            )
        # print(f"# of neighbors = {len(neighbors)}")

        vectors = None
        metadata = None
        if len(datapoints) > 0:
            vectors = {pt.datapoint_id: list(pt.feature_vector) for pt in datapoints}
            metadata = {
                pt.datapoint_id: {
                    md.namespace: list(md.allow_list) for md in pt.restricts
                }
                for pt in datapoints
                if pt.restricts
            }
        # print(f"# of vectors = {len(vectors)}")
        num_vectors_exported = self.save_vectors_to_parquet(
            vectors,
            metadata,
            # self.file_ctr,
            vectors_directory,
        )
        pbar.update(len(vectors))

        namespace_meta = {
            "index_name": index.display_name,
            "namespace": "namespace",
            "total_vector_count": total,
            "exported_vector_count": num_vectors_exported,
            "metric": standardize_metric(
                index_meta.get("metadata", {})
                .get("config", {})
                .get("distanceMeasureType"),
                self.DB_NAME_SLUG,
            ),
            "dimensions": dim,
            "model_name": self.args["model_name"],
            "vector_columns": ["vector"],
            "data_path": vectors_directory,
        }
        index_meta_list.append(namespace_meta)
        self.args["exported_count"] += num_vectors_exported

        return index_meta_list
        # return {f"{index.display_name}": index_meta_list}
        # return {f"{index.display_name}": [namespace_meta]}
