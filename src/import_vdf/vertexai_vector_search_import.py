import google.auth
import google.auth.transport.requests

from typing import Dict
from src.names import DBNames
from os import listdir


from src.import_vdf.vdf_import_cls import ImportVDF
from src.util import db_metric_to_standard_metric

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# gcloud config set project $PROJECT_ID - users

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


class ImportVertexVectorSearch():
    DB_NAME_SLUG = DBNames.VERTEX_VECTOR_SEARCH

    def __init__(self, project_id: str, location: str) -> None:
        # super duper call
        super().__init__()
        self.project_id = project_id
        self.location = location
        self.DB_NAME_SLUG = DBNames.VERTEX_VECTOR_SEARCH
        # super().__init__(args={})
        self.parent = f"projects/{self.project_id}/locations/{self.location}"
        self.client = self._get_client()

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
        distance_measure_type: str = db_metric_to_standard_metric[DB_NAME_SLUG]["euclidean"],
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

        index = {
            "name": name,
            "display_name": display_name,
            "description": description,
            "metadata": {
                "contentsDeltaUri": gcs_data_path,
                "config": {
                    "dimensions": dimensions,
                    "approximateNeighborsCount": approximate_neighbors_count,
                    "distanceMeasureType": db_metric_to_standard_metric[
                        distance_measure_type
                    ],
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

        try:
            response = create_client.execute()
            print(f"Index created: {response['name']}")
            return {
                "namespace": response["name"],
                "total_vector_count": 0,
                "exported_vector_count": 0,
                "dimensions": dimensions,
                "model_name": DB_NAME_SLUG,
                "vector_columns": [],
                "data_path": gcs_data_path,
                "metric": distance_measure_type,
            }

        except HttpError as err:
            raise ConnectionError("Error creating index") from err

    # def upsert_data(self):
    #     client = self.get_client()
    #     for index_name, index_meta in self.vdf_meta["indexes"].items():
    #         print(f"Importing data for index '{index_name}'")
    #         for namespace_meta in index_meta:
    #             print(f"Importing data for namespace '{namespace_meta['namespace']}'")
    #             data_path = namespace_meta["data_path"]
    #             parquet_files = self.get_parquet_files(data_path)
    #             for file in parquet_files:
    #       x          print(f"Importing data from file '{file}'")
    #                 with open(file, "rb") as f:
    #                     try:
    #                         client.projects().locations().indexEndpoints().importData(
    #                             name=f"projects/{self.args['project_id']}/locations/{self.args['region']}/indexEndpoints/{index_name}",
    #                             body={"inputConfig": {"gcsSource": {"uris": [file]}}},
    #                         ).execute()
    #                     except HttpError as e:
    #                         print(f"Error importing data from file '{file}'", e)
    #                         raise e
