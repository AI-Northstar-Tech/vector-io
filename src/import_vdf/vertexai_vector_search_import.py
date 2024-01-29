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

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class ImportVertexVectorSearch(ImportVDF):
    DB_NAME_SLUG = DBNames.VERTEX_VECTOR_SEARCH

    def __init__(self, args: Dict) -> None:
        # super duper call
        super().__init__(args)
        self.project_id = args["project_id"]
        self.location = args["location"]
        self.DB_NAME_SLUG = DBNames.VERTEX_VECTOR_SEARCH
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        self.client = self._get_client()
        # self.vdf_meta #TODO - this is where the vdf metadata sits, we will need to map this to the test import

    def _get_client(self):
        """Gets the Vertex AI Vector Search client.
        Returns:
            The Vertex AI Vector Search service.

        Note this uses the default credentials from the environment.
        https://google-auth.readthedocs.io/en/latest/reference/google.auth.html#google.auth.default

            To enable application default credentials with the Cloud SDK run:

            gcloud auth application-default login
            If the Cloud SDK has an active project, the project ID is returned. The active project can be set using:

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
        metric_dict = db_metric_to_standard_metric[DBNames.VERTEX_VECTOR_SEARCH]
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
                "model_name": DBNames.VERTEX_VECTOR_SEARCH,
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

    def upsert_data(self, index_names: str, data: List[Dict]) -> None:
        """deletes an index

        Args:
            name: The name of the index to delete.
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
            raise ConnectionError("Error deleting index") from err
        
    def create_index_endpoint(self):
        '''https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.indexEndpoints/create

        '''
        create_index_endpoint_client = (
            self.client.projects()
            .locations()
            .indexEndpoints()
            .create(parent=self.parent)
        )
        # fix the uri
        create_index_endpoint_client.uri = create_index_endpoint_client.uri.replace("aip", f"{self.location}-aip")

        try:
            response = create_index_endpoint_client.execute()
            print(f"Created Index Endpoint: {response['name']}")
        except HttpError as err:
            raise ConnectionError("Error deleting index") from err

    def deploy_index_to_endpoint(self, deployment_name: str, index_name: str, endpoint_name: str):
        '''https://cloud.google.com/vertex-ai/docs/reference/rest/v1/projects.locations.indexEndpoints/deployIndex

        '''
        fqn_index = f"{self.parent}/indexes/{index_name}"
        deployed_index = {"deployedIndex": {
            "id": deployment_name,
            "index": fqn_index
        }}
        index_endpoint = f"{self.parent}/indexEndpoints/{endpoint_name}"
        deploy_index_endpoint_client = (
            self.client.projects()
            .locations()
            .indexEndpoints()
            .deployIndex(indexEndpoint=index_endpoint, body=deployed_index)
        )
        # fix the uri
        deploy_index_endpoint_client.uri = deploy_index_endpoint_client.uri.replace("aip", f"{self.location}-aip")

        try:
            response = deploy_index_endpoint_client.execute()
            print(f"Created Index Endpoint: {response['name']}")
        except HttpError as err:
            raise ConnectionError("Error deleting index") from err

    