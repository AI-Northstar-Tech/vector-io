""" 
Import data to vertex ai vector search index
"""

from typing import Dict, List

from vdf_io.names import DBNames
from vdf_io.import_vdf.vdf_import_cls import ImportVDF


# gcloud config set project $PROJECT_ID - users
import os
import json
import itertools
import pandas as pd
from tqdm import tqdm
from google.cloud import aiplatform as aip
import google.cloud.aiplatform_v1 as aipv1

# SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]

# NEW
import uuid
import time
import numpy as np
from google.cloud import storage
from google.protobuf import struct_pb2
from google.cloud.aiplatform_v1.types.index import Index
from google.cloud.aiplatform_v1.types.index_endpoint import IndexEndpoint
from google.cloud.aiplatform_v1.types.index_endpoint import DeployedIndex
from ratelimit import limits, sleep_and_retry
from backoff import on_exception, expo
import google.api_core.exceptions as google_exceptions


# exceptions
class ResourceNotExistException(Exception):
    def __init__(self, resource: str, message="Resource Does Not Exist."):
        self.resource = resource
        self.message = message
        super().__init__(self.message)


class ImportVertexAIVectorSearch(ImportVDF):
    DB_NAME_SLUG = DBNames.VERTEXAI

    def __init__(self, args: Dict) -> None:
        super().__init__(args)
        self.DB_NAME_SLUG = DBNames.VERTEXAI
        self.project_id = self.args["project_id"]
        self.location = self.args["location"]
        self.batch_size = self.args.get("batch_size", 100)

        # optional
        self.create_new_index = self.args.get("create_new_index", False)

        self.deploy_new_index = self.args.get("deploy_new_index", False)

        # =========================================================
        # set index, endpoint, and SDK client
        # =========================================================
        self.parent = f"projects/{self.project_id}/locations/{self.location}"

        client_endpoint = f"{self.location}-aiplatform.googleapis.com"
        # set index client
        self.index_client = aipv1.IndexServiceClient(
            client_options=dict(api_endpoint=client_endpoint)
        )
        # set index endpoint client
        self.index_endpoint_client = aipv1.IndexEndpointServiceClient(
            client_options=dict(api_endpoint=client_endpoint)
        )
        # set SDK client
        aip.init(
            project=self.project_id,
            location=self.location,
        )
        # cloud storage client
        self.storage_client = storage.Client(project=self.project_id)

        # =========================================================
        # filters: restricts and crowding
        # =========================================================
        if self.args["filter_restricts"]:
            # String filters: allows and denies
            allows = []
            denies = []
            list_restrict_entries = []
            for name in self.args["filter_restricts"]:
                name_space_filter_entry = {}
                all_allows = []
                all_denies = []
                allows = []
                denies = []
                name_space_filter_entry["namespace"] = name.get("namespace")
                if name.get("allow_list") is not None:
                    allow_items = name.get("allow_list")
                    allows.append(allow_items)
                    # allows.append([a for a in allow_items])
                if name.get("deny_list") is not None:
                    deny_items = name.get("deny_list")
                    denies.append(deny_items)

                if allows:
                    all_allows = list(itertools.chain.from_iterable(allows))
                    name_space_filter_entry["allow_list"] = all_allows

                if denies:
                    all_denies = list(itertools.chain.from_iterable(denies))
                    name_space_filter_entry["deny_list"] = all_denies

                list_restrict_entries.append(name_space_filter_entry)

        self.list_restrict_entries = (
            list_restrict_entries if self.args["filter_restricts"] is not None else None
        )
        print(f"list_restrict_entries : {self.list_restrict_entries}")

        if self.args["numeric_restricts"]:
            # Numeric filters:
            list_of_numeric_entries = []
            for name in self.args["numeric_restricts"]:
                name_space_filter_entry = {}
                name_space_filter_entry["namespace"] = name.get("namespace")
                name_space_filter_entry["data_type"] = name.get("data_type")
                list_of_numeric_entries.append(name_space_filter_entry)

        self.list_of_numeric_entries = (
            list_of_numeric_entries
            if self.args["numeric_restricts"] is not None
            else None
        )
        print(f"list_of_numeric_entries : {self.list_of_numeric_entries}")

        # =========================================================
        # set index args
        # =========================================================
        self.index_names = []
        for index_name, index_meta in self.vdf_meta["indexes"].items():
            # load data
            print(f"Importing data for index: {index_name}")
            print(f"index_meta: {json.dumps(index_meta, indent=4)}")
            for namespace_meta in index_meta:
                all_indexes = [index.display_name for index in self.list_indexes()]
                # check if index exists
                index_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta["namespace"]
                    else ""
                )
                suffix = 2
                while index_name in all_indexes and self.args["create_new"] is True:
                    index_name = index_name + f"_{suffix}"
                    suffix += 1
                self.index_name = index_name
                self.index_names.append(self.index_name)
                if self.index_name not in all_indexes:
                    # create index
                    unique_id = uuid.uuid4()
                    if self.index_name is None:
                        self.index_name = f"my_vvs_index_{unique_id}"
                        print(f"Creating new index: {self.index_name} ...")
                    self.dimensions = namespace_meta["dimensions"]
                    print(f"dimensions: {self.dimensions}")

                    # optional; used if create_new_index = True
                    self.approx_nn_count = self.args.get("approx_nn_count", 150)
                    self.leaf_node_emb_count = self.args.get(
                        "leaf_node_emb_count", 1000
                    )
                    self.leaf_nodes_percent = self.args.get("leaf_nodes_percent", 7)
                    self.distance_measure = self.args.get(
                        "distance_measure", "DOT_PRODUCT_DISTANCE"
                    )
                    self.shard_size = self.args.get("shard_size", "SHARD_SIZE_MEDIUM")
                    self.gcs_bucket = self.args.get("gcs_bucket", None)

                    if self.gcs_bucket is None:
                        self.gcs_bucket = self.index_name
                        try:
                            print(f"Creating new gcs_bucket: {self.gcs_bucket} ...")
                            bucket_client = self.storage_client.bucket(self.gcs_bucket)
                            new_bucket = self.storage_client.create_bucket(
                                self.gcs_bucket, location=self.location
                            )
                            print(
                                f"Created bucket {new_bucket.name} in {new_bucket.location}"
                            )
                        except Exception as e:
                            print(f"{self.gcs_bucket} bucket already exists {e}")
                            pass
                    self.gcs_folder = "init_index"
                    self.local_file_name = "embeddings_0.json"
                    self.contents_delta_uri = (
                        f"gs://{self.gcs_bucket}/{self.gcs_folder}"
                    )

                    # dummy embedding - TODO: use input data from parquet(?)
                    init_embedding = {
                        "id": str(unique_id),
                        "embedding": list(np.zeros(self.dimensions)),
                    }

                    # dump embedding to a local file
                    with open(self.local_file_name, "w") as f:
                        json.dump(init_embedding, f)

                    # upload to GCS
                    bucket_client = self.storage_client.bucket(self.gcs_bucket)
                    blob = bucket_client.blob(
                        f"{self.gcs_folder}/{self.local_file_name}"
                    )
                    blob.upload_from_filename(f"{self.local_file_name}")

                    self.target_index = self._create_index(
                        index_name=self.index_name,
                        contents_delta_uri=self.contents_delta_uri,
                        dimensions=self.dimensions,
                        approximate_neighbors_count=self.approx_nn_count,
                        leaf_node_embedding_count=self.leaf_node_emb_count,
                        leaf_nodes_to_search_percent=self.leaf_nodes_percent,
                        distance_measure_type=self.distance_measure,
                        shard_size=self.shard_size,
                    )

                    if self.deploy_new_index:
                        # Note: Don't need to deploy index to upsert vectors to index

                        self.index_endpoint_name = f"{self.index_name}_endpoint"
                        self.index_endpoint = self._create_index_endpoint(
                            endpoint_name=self.index_endpoint_name
                        )

                        # optional; used if deploy_new_index = True
                        self.machine_type = self.args.get(
                            "machine_type", "e2-standard-16"
                        )
                        self.min_replicas = self.args.get("min_replicas", 1)
                        self.max_replicas = self.args.get("max_replicas", 1)

                        self._deploy_index(
                            index_name=self.index_name,
                            endpoint_name=self.index_endpoint.display_name,
                            machine_type=self.machine_type,
                            min_replicas=self.min_replicas,
                            max_replicas=self.max_replicas,
                        )
                else:
                    self.target_index = self._get_index()
                    if self.target_index is None:
                        raise ValueError(
                            f"Failed to set target_index: {self.index_name}"
                        )

                self.target_index_resource_name = self.target_index.name

                # init target index to import vectors to
                self.target_vertexai_index = aip.MatchingEngineIndex(
                    self.target_index_resource_name
                )
                print(f"Importing to index: {self.target_vertexai_index.display_name}")
                print(f"Full resource name: {self.target_vertexai_index.resource_name}")
                print("Target index config:")

                index_config_dict = self.target_vertexai_index.to_dict()
                _index_meta_config = index_config_dict["metadata"]["config"]
                tqdm.write(json.dumps(_index_meta_config, indent=4))

    # =========================================================
    # VectorSearch helpers
    # =========================================================
    def _set_index_name(self, index_name: str) -> None:
        """

        :param index_name:
        :return:
        """
        self.index_name = index_name

    def _set_index_endpoint_name(self, index_endpoint_name: str = None) -> None:
        """

        :param index_endpoint_name:
        :return:
        """
        if index_endpoint_name is not None:
            self.index_endpoint_name = index_endpoint_name
        elif self.index_name is not None:
            self.index_endpoint_name = f"{self.index_name}_endpoint"
        else:
            raise ResourceNotExistException("index")

    def _get_index(self) -> Index:
        """

        :return:
        """
        # Check if index exists
        d_ids = []
        indexes = []
        if self.index_name is not None:
            all_indexes = [index for index in self.list_indexes()]
            try:
                indexes = [
                    index.name
                    for index in all_indexes
                    if index.display_name == self.index_name
                    or index.name == self.index_name
                ]
            except Exception as e:
                print(
                    f"{self.index_name} not an existing display_name or resource_name: {e}"
                )
                pass
            if not indexes:
                try:
                    # checking deployed indexes
                    for test_index in all_indexes:
                        if test_index.deployed_indexes:
                            # grabbing all deployed indexes
                            d_ids.extend(test_index.deployed_indexes)

                    # returning only those that match
                    deployed_index_match = [
                        d_id
                        for d_id in d_ids
                        if (
                            d_id.display_name == self.index_name
                            or d_id.deployed_index_id == self.index_name
                        )
                    ]
                    # need to do this to return the index resoruce_name
                    if deployed_index_match:
                        target_endpoint = aip.MatchingEngineIndexEndpoint(
                            deployed_index_match[0].index_endpoint
                        )
                        for d in target_endpoint.deployed_indexes:
                            if (
                                d.id == self.index_name
                                or d.display_name == self.index_name
                            ):
                                indexes.append(d.index)
                except Exception as e:
                    print(f"not an existing deployed_index: {e}")
                    pass

        if len(indexes) == 0:
            print(f"Index {self.index_name} not found")
            return None
        else:
            index_id = indexes[0]
            # print("Found existing index")
            request = aipv1.GetIndexRequest(name=index_id)
            index = self.index_client.get_index(request=request)
            return index

    def _get_index_endpoint(self) -> IndexEndpoint:
        """

        :return:
        """
        # Check if index endpoint exists
        all_index_endpoints = [response for response in self.list_index_endpoints()]

        if self.index_endpoint_name is not None:
            print(f"Checking if {self.index_endpoint_name} already exists...")
            try:
                index_endpoints = [
                    response.name
                    for response in all_index_endpoints
                    if response.display_name == self.index_endpoint_name
                    or response.name == self.index_endpoint_name
                ]
            except Exception as e:
                print(f"{self.index_endpoint_name} not an existing index endpoint: {e}")
                pass
        else:
            raise ResourceNotExistException("index_endpoint")

        if len(index_endpoints) == 0:
            print(f"Could not find index endpoint: {self.index_endpoint_name}")
            return None
        else:
            index_endpoint_id = index_endpoints[0]
            index_endpoint = self.index_endpoint_client.get_index_endpoint(
                name=index_endpoint_id
            )
            return index_endpoint

    def list_indexes(self) -> List[Index]:
        """

        :return:
        """
        request = aipv1.ListIndexesRequest(parent=self.parent)
        page_result = self.index_client.list_indexes(request=request)
        indexes = [response for response in page_result]
        return indexes

    def list_index_endpoints(self) -> List[IndexEndpoint]:
        """

        :return:
        """
        request = aipv1.ListIndexEndpointsRequest(parent=self.parent)
        page_result = self.index_endpoint_client.list_index_endpoints(request=request)
        index_endpoints = [response for response in page_result]
        return index_endpoints

    def list_deployed_indexes(self, endpoint_name: str = None) -> List[DeployedIndex]:
        """

        :param endpoint_name:
        :return:
        """
        try:
            if endpoint_name is not None:
                self._set_index_endpoint_name(index_endpoint_name=endpoint_name)
            index_endpoint = self._get_index_endpoint()
            deployed_indexes = index_endpoint.deployed_indexes
        except ResourceNotExistException as rnee:
            raise rnee

        return list(deployed_indexes)

    def _build_index_config(
        self,
        contents_delta_uri: str,
        index_display_name: str,
        dimensions: int,
        approximate_neighbors_count: int,
        leaf_node_embedding_count: int,
        leaf_nodes_to_search_percent: int,
        description: str,
        distance_measure_type: str,
        shard_size: str,
    ):
        vector_search_index_config = {
            "index_display_name": index_display_name,
            "contents_delta_uri": contents_delta_uri,
            "dimensions": dimensions,
            "approximate_neighbors_count": approximate_neighbors_count,
            "distance_measure_type": distance_measure_type,
            "leaf_node_embedding_count": leaf_node_embedding_count,
            "leaf_nodes_to_search_percent": leaf_nodes_to_search_percent,
            "description": description,
            "labels": {
                "project": f"{self.project_id}",
                "tag": "vectorio-import",
            },
        }
        print("vector_search_index_config:")
        tqdm.write(json.dumps(vector_search_index_config, indent=4))

        tree_ah_config = struct_pb2.Struct(
            fields={
                "leafNodeEmbeddingCount": struct_pb2.Value(
                    number_value=vector_search_index_config["leaf_node_embedding_count"]
                ),
                "leafNodesToSearchPercent": struct_pb2.Value(
                    number_value=vector_search_index_config[
                        "leaf_nodes_to_search_percent"
                    ]
                ),
            }
        )
        algorithm_config = struct_pb2.Struct(
            fields={"treeAhConfig": struct_pb2.Value(struct_value=tree_ah_config)}
        )
        config = struct_pb2.Struct(
            fields={
                "dimensions": struct_pb2.Value(
                    number_value=vector_search_index_config["dimensions"]
                ),
                "approximateNeighborsCount": struct_pb2.Value(
                    number_value=vector_search_index_config[
                        "approximate_neighbors_count"
                    ]
                ),
                "distanceMeasureType": struct_pb2.Value(
                    string_value=vector_search_index_config["distance_measure_type"]
                ),
                "algorithmConfig": struct_pb2.Value(struct_value=algorithm_config),
                "shardSize": struct_pb2.Value(string_value=shard_size),
            }
        )
        metadata = struct_pb2.Struct(
            fields={
                "config": struct_pb2.Value(struct_value=config),
                "contentsDeltaUri": struct_pb2.Value(
                    string_value=vector_search_index_config["contents_delta_uri"]
                ),
            }
        )
        return metadata

    def _create_index(
        self,
        contents_delta_uri: str,
        dimensions: int,
        approximate_neighbors_count: int,
        leaf_node_embedding_count: int,
        leaf_nodes_to_search_percent: int,
        distance_measure_type: str,
        shard_size: str,
        index_name: str = None,
    ) -> Index:
        """

        :param contents_delta_uri:
        :param dimensions:
        :param approximate_neighbors_count:
        :param leaf_node_embedding_count:
        :param leaf_nodes_to_search_percent:
        :param distance_measure_type:
        :param shard_size:
        :param index_name:
        :return:
        """

        if index_name is not None:
            self._set_index_name(index_name=index_name)
        # Get index
        if self.index_name is None:
            raise ResourceNotExistException("index")
        index = self._get_index()
        # Create index if does not exists
        if index:
            print(f"Index {self.index_name} exists with resource_name:\n {index.name}")
        else:
            print(f"Index {self.index_name} does not exists. Creating index ...")

            invoke_time = time.strftime("%Y%m%d_%H%M%S")
            description = f"created during vectorio import at {invoke_time}"

            metadata = self._build_index_config(
                contents_delta_uri=contents_delta_uri,
                index_display_name=index_name,
                dimensions=dimensions,
                description=description,
                approximate_neighbors_count=approximate_neighbors_count,
                leaf_node_embedding_count=leaf_node_embedding_count,
                leaf_nodes_to_search_percent=leaf_nodes_to_search_percent,
                distance_measure_type=distance_measure_type,
                shard_size=shard_size,
            )

            index_request = {
                "display_name": self.index_name,
                "description": description,
                "metadata": struct_pb2.Value(struct_value=metadata),
                "index_update_method": aipv1.Index.IndexUpdateMethod.STREAM_UPDATE,
            }

            r = self.index_client.create_index(
                parent=self.parent, index=Index(index_request)
            )

            # Poll the operation until its done successfully.
            print("Poll the operation to create index ...")
            while True:
                if r.done():
                    break
                time.sleep(5)
                print(".", end="")

            index = r.result()
            print(
                f"\nIndex {self.index_name} created with resource_name:\n {index.name}"
            )

        return index

    def _create_index_endpoint(
        self,
        endpoint_name: str = None,
    ) -> IndexEndpoint:
        """

        :param endpoint_name:
        :return:
        """
        try:
            if endpoint_name is not None:
                self._set_index_endpoint_name(index_endpoint_name=endpoint_name)

            # Get index endpoint if exists
            index_endpoint = self._get_index_endpoint()

            # Create Index Endpoint if does not exists
            if index_endpoint is not None:
                print("Index endpoint already exists")
            else:
                print(
                    f"Index endpoint {self.index_endpoint_name} does not exists. Creating index endpoint..."
                )

                index_endpoint_request = {"display_name": self.index_endpoint_name}

                index_endpoint = IndexEndpoint(index_endpoint_request)
                index_endpoint.public_endpoint_enabled = True

                r = self.index_endpoint_client.create_index_endpoint(
                    parent=self.parent, index_endpoint=index_endpoint
                )

                print("Poll the operation to create index endpoint ...")
                while True:
                    if r.done():
                        break
                    time.sleep(10)
                    print(".", end="")

                index_endpoint = r.result()

        except Exception as e:
            print(f"Failed to create index endpoint {self.index_endpoint_name}")
            raise e

        print(f"\nReturning index endpoint: {index_endpoint}")
        return index_endpoint

    def _deploy_index(
        self,
        index_name: str = None,
        endpoint_name: str = None,
        machine_type: str = "e2-standard-16",
        min_replicas: int = 2,
        max_replicas: int = 2,
    ) -> IndexEndpoint:
        """

        :param endpoint_name:
        :param index_name:
        :param machine_type:
        :param min_replicas:
        :param max_replicas:
        :return:
        """

        index = self._get_index()
        index_endpoint = self._get_index_endpoint()

        # Deploy Index to endpoint
        try:
            # Check if index is already deployed to the endpoint
            if index_endpoint.deployed_indexes:
                if index.name in index_endpoint.deployed_indexes:
                    print(
                        f"Skipping Index deployment. Index {self.index_name}"
                        + f"already deployed {index.name} to endpoint {self.index_endpoint_name}"
                    )
                    return index_endpoint

            invoke_time = time.strftime("%Y%m%d_%H%M%S")
            deployed_index_id = f"{self.index_name.replace('-', '_')}_{invoke_time}"
            deploy_index_config = {
                "id": deployed_index_id,
                "display_name": deployed_index_id,
                "index": index.name,
                "dedicated_resources": {
                    "machine_spec": {
                        "machine_type": machine_type,
                    },
                    "min_replica_count": min_replicas,
                    "max_replica_count": max_replicas,
                },
            }
            print("Deploying index with request:")
            tqdm.write(json.dumps(deploy_index_config, indent=4))
            r = self.index_endpoint_client.deploy_index(
                index_endpoint=index_endpoint.name,
                deployed_index=DeployedIndex(deploy_index_config),
            )

            # Poll the operation until its done successfullly.
            print("Poll the operation to deploy index ...")
            while True:
                if r.done():
                    break
                time.sleep(60)
                print(".", end="")

            print(
                f"\nDeployed {self.index_name} to endpoint {self.index_endpoint_name}"
            )

        except Exception as e:
            print(
                f"Failed to deploy {self.index_name} to endpoint: {self.index_endpoint_name}"
            )
            raise e

        return index_endpoint

    def upsert_data(self):
        MINUTE = 60
        CALLS_PER_PRD = self.args.get("requests_per_minute", 6000)

        # rate is 1 QPS
        @sleep_and_retry  # If there are more requests than rate, sleep shortly
        @on_exception(
            expo, google_exceptions.ResourceExhausted, max_tries=10
        )  # if we receive exceptions from Google API, retry
        @limits(calls=CALLS_PER_PRD, period=MINUTE)
        def upsert_in_rate(self, upsert_request):
            """
            Upserts vectors and applies rate limits.
            """
            return self.index_client.upsert_datapoints(request=upsert_request)

        # upsert for each index
        for new_index_name, (index_name, index_meta) in tqdm(
            zip(self.index_names, self.vdf_meta["indexes"].items()),
            desc="Iterating over indexes",
        ):
            self.index_name = new_index_name
            self.target_index = self._get_index()
            self.target_index_resource_name = self.target_index.name
            # init target index to import vectors to
            self.target_vertexai_index = aip.MatchingEngineIndex(
                self.target_index_resource_name
            )
            # load data
            tqdm.write(
                f"Importing data from {index_name} to {self.target_vertexai_index.display_name}"
            )
            tqdm.write(f"index_meta: {json.dumps(index_meta, indent=4)}")

            for namespace_meta in tqdm(index_meta, desc="Iterating over namespaces"):
                # get data path
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                print(f"data_path: {data_path}")

                # get col names
                vector_metadata_names, vector_column_name = self.get_vector_column_name(
                    namespace_meta["vector_columns"], namespace_meta
                )
                print(f"vector_column_name    : {vector_column_name}")
                print(f"vector_metadata_names : {vector_metadata_names}")

                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(final_data_path)

                total_ids = []
                for file in tqdm(parquet_files, desc="Iterating over parquet files"):
                    file_path = os.path.join(final_data_path, file)
                    df = pd.read_parquet(file_path)
                    df["id"] = df["id"].apply(lambda x: str(x))

                    insert_datapoints_payload = []

                    for idx, row in tqdm(
                        df.iterrows(), desc="Iterating over rows", total=len(df)
                    ):
                        row = json.loads(row.to_json())

                        total_ids.append(row["id"])
                        row[vector_column_name] = [
                            float(emb) for emb in row[vector_column_name]
                        ]
                        numeric_restrict_entry_list = []
                        restrict_entry_list = []
                        allow_values = []
                        deny_values = []
                        crowding_tag_val = None

                        # if idx == 10:
                        #     # sanity check
                        #     print(f"row['id'] : {row['id']}")

                        if self.list_restrict_entries:
                            for entry in self.list_restrict_entries:
                                restrict_entry = {}

                                restrict_entry["namespace"] = entry.get("namespace")

                                if entry.get("allow_list"):
                                    for col in entry.get("allow_list"):
                                        allow_values.append(row[col])
                                        restrict_entry["allow_list"] = [
                                            str(a) for a in allow_values
                                        ]

                                if entry.get("deny_list"):
                                    for col in entry.get("deny_list"):
                                        deny_values.append(row[col])
                                        restrict_entry["deny_list"] = [
                                            str(d) for d in deny_values
                                        ]

                                restrict_entry_list.append(restrict_entry)

                                # if idx == 10:
                                #     print(f"restrict_entry_list : {restrict_entry_list}")

                        if self.list_of_numeric_entries:
                            # numeric_restrict_entry_list = []
                            for entry in self.list_of_numeric_entries:
                                numeric_restrict_entry = {}

                                data_type = entry.get("data_type")
                                col_name = entry.get("namespace")
                                numeric_restrict_entry["namespace"] = entry.get(
                                    "namespace"
                                )
                                numeric_restrict_entry[data_type] = row[col_name]
                                numeric_restrict_entry_list.append(
                                    numeric_restrict_entry
                                )

                            # if idx == 10:
                            #     # sanity check
                            #     print(f"numeric_restrict_entry_list : {numeric_restrict_entry_list}")

                        if self.args["crowding_tag"]:
                            crowding_tag_col = self.args["crowding_tag"]
                            crowding_tag_val = str(row[crowding_tag_col])

                            # if idx == 10:
                            #     # sanity check
                            #     print(f"crowding_tag_col : {crowding_tag_col}")
                            #     print(f"crowding_tag_val : {crowding_tag_val}")

                        insert_datapoints_payload.append(
                            aipv1.IndexDatapoint(
                                datapoint_id=row["id"],
                                feature_vector=row[vector_column_name],
                                restricts=restrict_entry_list,
                                numeric_restricts=numeric_restrict_entry_list,
                                crowding_tag=aipv1.IndexDatapoint.CrowdingTag(
                                    crowding_attribute=crowding_tag_val
                                ),
                            )
                        )

                        if idx % self.batch_size == 0:
                            upsert_request = aipv1.UpsertDatapointsRequest(
                                index=self.target_vertexai_index.resource_name,
                                datapoints=insert_datapoints_payload,
                            )
                            # self.index_client.upsert_datapoints(request=upsert_request)
                            upsert_in_rate(self, upsert_request=upsert_request)
                            insert_datapoints_payload = []

                        if len(total_ids) % CALLS_PER_PRD == 0:
                            print(f"{len(total_ids)} total vectors upserted")

                    if len(insert_datapoints_payload) > 0:
                        upsert_request = aipv1.UpsertDatapointsRequest(
                            index=self.target_vertexai_index.resource_name,
                            datapoints=insert_datapoints_payload,
                        )

                        # self.index_client.upsert_datapoints(request=upsert_request)
                        upsert_in_rate(self, upsert_request=upsert_request)

        print("Index import complete")
        print(
            f"Updated {self.target_vertexai_index.display_name} with {len(total_ids)} vectors"
        )
