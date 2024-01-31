""" 
Export data from vertex ai vector search index
"""

import json
from names import DBNames
from util import standardize_metric
from export_vdf.vdb_export_cls import ExportVDB

import google.auth
from google.cloud import aiplatform
from google.cloud.aiplatform import MatchingEngineIndex as vs
from google.cloud.aiplatform import MatchingEngineIndexEndpoint as vsep

import os
import json
import pandas as pd
from tqdm import tqdm


class ExportVertexAIVectorSearch(ExportVDB):
    DB_NAME_SLUG = DBNames.VERTEXAI

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

    def get_all_index_names(self):
        index_names = [index.resource_name for index in vs.list()]
        return index_names

    def get_data(self):
        index_names = []
        if "index" not in self.args or self.args["index"] is None:
            index_names = self.get_all_index_names()
        else:
            indexes = self.args["index"].split(",")
            for index in indexes:
                filter_by = f'display_name="{index}"'
                fetched_index = [
                    index.resource_name for index in vs.list(filter=filter_by)
                ]
                index_names.extend(fetched_index)

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
        index = vs(index_name=index_name)
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
        # print(f"index_endpoint_name = {index_endpoint_name}")
        # print(f"deployed_index_id   = {deployed_index_id}")

        # define index and index endpoint
        index = vs(index_name=index_name)
        index_endpoint = vsep(index_endpoint_name)

        vectors_directory = os.path.join(self.vdf_directory, index.display_name)
        os.makedirs(vectors_directory, exist_ok=True)

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

        return index_meta_list
        # return {f"{index.display_name}": index_meta_list}
        # return {f"{index.display_name}": [namespace_meta]}
