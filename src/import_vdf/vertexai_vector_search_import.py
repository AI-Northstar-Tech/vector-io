""" 
import data to vertex ai vector search index
"""
import google.auth
import google.auth.transport.requests

from typing import Dict, List
from src.names import DBNames
from os import listdir

from src.import_vdf.vdf_import_cls import ImportVDF
from src.util import db_metric_to_standard_metric

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# gcloud config set project $PROJECT_ID - users
import os
import json
import pandas as pd
from tqdm import tqdm
from google.cloud import aiplatform as aip
import google.cloud.aiplatform_v1 as aipv1

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class ImportVertexAIVectorSearch(ImportVDF):
    DB_NAME_SLUG = DBNames.VERTEXAI

    def __init__(self, args: Dict) -> None:
        super().__init__(args)
        self.DB_NAME_SLUG = DBNames.VERTEXAI
        self.project_id = args["project_id"]
        self.project_num = args["project_num"]
        self.location = args["location"]
        self.batch_size = args['batch_size']
        self.target_index_id = args["target_index_id"]
        self.target_index_resource_name = f"projects/{self.project_num}/locations/{self.location}/indexes/{self.target_index_id}"
        
        # clients
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        self.client = self._get_client()

        # set index client
        client_endpoint = f"{self.location}-aiplatform.googleapis.com"
        self.index_client = aipv1.IndexServiceClient(
            client_options=dict(api_endpoint=client_endpoint)
        )
        aip.init(
            project=self.project_id,
            location=self.location,
        )
        
        # init target index to import vectors to
        self.target_vertexai_index = aip.MatchingEngineIndex(self.target_index_resource_name)
        print(f"Importing to index : {self.target_vertexai_index.display_name}")
        print(f"Full resource name : {self.target_vertexai_index.resource_name}")
        print(f"Target index config:")
        
        index_config_dict = self.target_vertexai_index.to_dict()
        _index_meta_config = index_config_dict['metadata']['config']
        tqdm.write(json.dumps(_index_meta_config, indent=4))

    def _get_client(self):
        """Gets the Vertex AI Vector Search client.
        Returns:
            The Vertex AI Vector Search service.

        Note this uses the default credentials from the environment.
        https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default

            To enable application default credentials with the Cloud SDK run:

            gcloud auth application-default login
            If the Cloud SDK has an active project, the project ID is returned. 
            The active project can be set using:

            gcloud config set project

        """
        creds, _ = google.auth.default()

        try:
            service = build("aiplatform", "v1", credentials=creds)
            return service
        except HttpError as err:
            raise ConnectionError(
                "Error getting Vertex AI Vector Search client"
            ) from err
            
    def upsert_data(self):
        
        for index_name, index_meta in self.vdf_meta["indexes"].items():
            
            # load data
            print(f"Importing data from: {index_name}")
            print(f"index_meta: {index_meta}")
                
            for namespace_meta in index_meta:
                
                # get data path
                data_path = namespace_meta["data_path"]
                print(f"data_path: {data_path}")
                
                # get col names
                vector_metadata_names, vector_column_name = self.get_vector_column_name(
                    namespace_meta["vector_columns"], namespace_meta
                )
                print(f"vector_column_name    : {vector_column_name}")
                print(f"vector_metadata_names : {vector_metadata_names}")
                
                # Load the data from the parquet files
                parquet_files = self.get_parquet_files(data_path)
                
                total_ids = []
                for file in tqdm(parquet_files, desc="Inserting data"):
                    file_path = os.path.join(data_path, file)
                    df = pd.read_parquet(file_path)
                    df["id"] = df["id"].apply(lambda x: str(x))
                    
                    data_rows = []
                    insert_datapoints_payload = []
                    
                    for idx, row in df.iterrows():
                        row = json.loads(row.to_json())
                        
                        total_ids.append(row["id"])
                        
                        row[vector_column_name] = [float(emb) for emb in row[vector_column_name]]
                        insert_datapoints_payload.append(
                            aipv1.IndexDatapoint(
                                datapoint_id=row["id"],
                                feature_vector=row[vector_column_name],
                            )
                        )
                        if idx % self.batch_size == 0:
                            upsert_request = aipv1.UpsertDatapointsRequest(
                                index=self.target_vertexai_index.resource_name,
                                datapoints=insert_datapoints_payload,
                            )
                            self.index_client.upsert_datapoints(request=upsert_request)
                            insert_datapoints_payload = []
                            
                    if len(insert_datapoints_payload) > 0:
                            
                        upsert_request = aipv1.UpsertDatapointsRequest(
                            index=self.target_vertexai_index.resource_name, 
                            datapoints=insert_datapoints_payload
                        )
                        
                        self.index_client.upsert_datapoints(request=upsert_request)
                    
        print(f"Index import complete")
        print(f"Updated {self.target_vertexai_index.display_name} with {len(total_ids)} vectors")
                
    def upsert_data_jw(self, index_names: str, data: List[Dict]) -> None:
        """
        Upserts vector data to a Vertex AI Vector Search index.
        """
        datapoints = []
        for datapoint in data:
            dp = {}
            dp.update(
                {
                    "datapointId": datapoint["datapointId"],
                    "featureVector": datapoint["featureVector"],
                }
            )
            try:
                dp.update({"restricts": datapoint["restricts"]})
            except KeyError:
                pass
            try:
                dp.update({"numericRestricts": datapoint["numericRestricts"]})
            except KeyError:
                pass
            try:
                dp.update({"crowdingTag": datapoint["crowdingTag"]})
            except KeyError:
                pass
            datapoints.append(dp)

        datapoints = {"datapoints": datapoints}
        index_to_upsert = f"{self.parent}/indexes/{index_names}"
        upsert_client = (
            self.client.projects()
            .locations()
            .indexes()
            .upsertDatapoints(index=index_to_upsert, body=datapoints)
        )
        # fix the uri
        upsert_client.uri = upsert_client.uri.replace("aip", f"{self.location}-aip")

        try:
            response = upsert_client.execute()
            print(f"Upserted datapoints")
        except HttpError as err:
            raise ConnectionError("Error upserting to index") from err

    def create_index_endpoint(self):
        """
        https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.indexEndpoints/create
        """
        create_index_endpoint_client = (
            self.client.projects()
            .locations()
            .indexEndpoints()
            .create(parent=self.parent)
        )
        # fix the uri
        create_index_endpoint_client.uri = create_index_endpoint_client.uri.replace(
            "aip", f"{self.location}-aip"
        )

        try:
            response = create_index_endpoint_client.execute()
            print(f"Created Index Endpoint: {response['name']}")
        except HttpError as err:
            raise ConnectionError("Error deleting index") from err

    def deploy_index_to_endpoint(
        self, deployment_name: str, index_name: str, endpoint_name: str
    ):
        """
        https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.indexEndpoints/deployIndex
        """
        fqn_index = f"{self.parent}/indexes/{index_name}"
        deployed_index = {"deployedIndex": {"id": deployment_name, "index": fqn_index}}
        index_endpoint = f"{self.parent}/indexEndpoints/{endpoint_name}"
        deploy_index_endpoint_client = (
            self.client.projects()
            .locations()
            .indexEndpoints()
            .deployIndex(indexEndpoint=index_endpoint, body=deployed_index)
        )
        # fix the uri
        deploy_index_endpoint_client.uri = deploy_index_endpoint_client.uri.replace(
            "aip", f"{self.location}-aip"
        )

        try:
            response = deploy_index_endpoint_client.execute()
            print(f"Created Index Endpoint: {response['name']}")
        except HttpError as err:
            raise ConnectionError("Error deleting index") from err
            
    # Create index if none exists
    def create_index(
        self,
        name: str,
        display_name: str,
        description: str,
        gcs_data_path: str,
        dimensions: int,
        approximate_neighbors_count: int,
        distance_measure_type: str = "dotproduct",
        leaf_node_embedding_count: int = 1000,
        leaf_nodes_to_search_percent: int = 10,
        index_type="streaming",
    ) -> Dict:
        """creates a new index

        Args:
            display_name: The display name of the index.
            description: The description of the index.
            gcs_data_path: The Cloud Storage path to the data.
            dimensions: The number of dimensions of the vectors.
            approximate_neighbors_count: The approximate number of neighbors to return.
            distance_measure_type: The distance measure type to use.
            leaf_node_embedding_count: The number of embeddings in each leaf node.
            leaf_nodes_to_search_percent: The percentage of leaf nodes to search.
            metadata: Additional metadata to associate with the index.
            index_type: The type of index to create.

        Index interface:
        https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.indexes
        https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.indexes/create
            interface Index {
            namespace: string;
            total_vector_count: number;
            exported_vector_count: number;
            dimensions: number;
            model_name: string;
            vector_columns: string[];
            data_path: string;
            metric: 'Euclid' | 'Cosine' | 'Dot';
            }
        """
        metric_dict = db_metric_to_standard_metric[DBNames.VERTEXAI]
        distance = metric_dict[distance_measure_type]
        index = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "metadata": {
                "contentsDeltaUri": gcs_data_path,
                "config": {
                    "dimensions": dimensions,
                    "approximateNeighborsCount": approximate_neighbors_count,
                    "distanceMeasureType": distance,
                    "algorithm_config": {
                        "treeAhConfig": {
                            "leafNodeEmbeddingCount": leaf_node_embedding_count,
                            "leafNodesToSearchPercent": leaf_nodes_to_search_percent,
                        }
                    },
                },
            },
        }

        if index_type == "streaming":
            index["indexUpdateMethod"] = "STREAM_UPDATE"

        create_client = (
            self.client.projects()
            .locations()
            .indexes()
            .create(parent=self.parent, body=index)
        )
        # fix the uri
        create_client.uri = create_client.uri.replace("aip", f"{self.location}-aip")

        try:
            response = create_client.execute()
            print(f"Index created: {response['name']}")
            return {
                "namespace": response["name"],
                "total_vector_count": 0,
                "exported_vector_count": 0,
                "dimensions": dimensions,
                "model_name": DBNames.VERTEXAI,
                "vector_columns": [],
                "data_path": gcs_data_path,
                "metric": distance_measure_type,
            }

        except HttpError as err:
            raise ConnectionError("Error creating index") from err
            
    def delete_index(self, index_name: str) -> None:
        """deletes an index

        Args:
            name: The name of the index to delete.
        """

        index_to_delete = f"{self.parent}/indexes/{index_name}"
        delete_client = (
            self.client.projects().locations().indexes().delete(name=index_to_delete)
        )
        # fix the uri
        delete_client.uri = delete_client.uri.replace("aip", f"{self.location}-aip")

        try:
            response = delete_client.execute()
            print(f"Index deleted: {response['name']}")
        except HttpError as err:
            raise ConnectionError("Error deleting index") from err
