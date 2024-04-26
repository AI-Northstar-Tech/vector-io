# import os
# from azure.search.documents import SearchClient
# from azure.search.documents.indexes.models import (
#     CorsOptions,
#     ScoringProfile,
#     SearchField,
#     SemanticConfiguration,
#     VectorSearch,
# )
# from azure.core.credentials import AzureKeyCredential
# from azure.core.exceptions import ResourceNotFoundError
# from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
# from azure.search.documents import SearchClient
# from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
# from azure.search.documents.indexes.models import (
#     ExhaustiveKnnAlgorithmConfiguration,
#     ExhaustiveKnnParameters,
#     HnswAlgorithmConfiguration,
#     HnswParameters,
#     SearchIndex,
#     SemanticConfiguration,
#     SemanticField,
#     SemanticPrioritizedFields,
#     SemanticSearch,
#     VectorSearch,
#     VectorSearchAlgorithmKind,
#     VectorSearchAlgorithmMetric,
#     VectorSearchProfile,
# )
# # from langchain_core.utils import get_from_env

# # AZURE_SEARCH_SERVICE_ENDPOINT
# # AZURE_SEARCH_INDEX_NAME
# # AZURE_SEARCH_API_KEY
# # doc count: https://learn.microsoft.com/en-us/python/api/azure-search-documents/azure.search.documents.searchclient?view=azure-python#azure-search-documents-searchclient-get-document-count
# # upsert: https://learn.microsoft.com/en-us/python/api/azure-search-documents/azure.search.documents.searchclient?view=azure-python#azure-search-documents-searchclient-merge-or-upload-documents
# # search, skip: https://learn.microsoft.com/en-us/python/api/azure-search-documents/azure.search.documents.searchclient?view=azure-python#azure-search-documents-searchclient-search
# # FIELDS_ID = get_from_env(
# #     key="AZURESEARCH_FIELDS_ID", env_key="AZURESEARCH_FIELDS_ID", default="id"
# # )
# # FIELDS_CONTENT = get_from_env(
# #     key="AZURESEARCH_FIELDS_CONTENT",
# #     env_key="AZURESEARCH_FIELDS_CONTENT",
# #     default="content",
# # )
# # FIELDS_CONTENT_VECTOR = get_from_env(
# #     key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
# #     env_key="AZURESEARCH_FIELDS_CONTENT_VECTOR",
# #     default="content_vector",
# # )
# # FIELDS_METADATA = get_from_env(
# #     key="AZURESEARCH_FIELDS_TAG", env_key="AZURESEARCH_FIELDS_TAG", default="metadata"
# # )
# index_name = "azuresearch"
# vector_search = VectorSearch(
#     algorithms=[
#         HnswAlgorithmConfiguration(
#             name="default",
#             kind=VectorSearchAlgorithmKind.HNSW,
#             parameters=HnswParameters(
#                 m=4,
#                 ef_construction=400,
#                 ef_search=500,
#                 metric=VectorSearchAlgorithmMetric.COSINE,
#             ),
#         ),
#         ExhaustiveKnnAlgorithmConfiguration(
#             name="default_exhaustive_knn",
#             kind=VectorSearchAlgorithmKind.EXHAUSTIVE_KNN,
#             parameters=ExhaustiveKnnParameters(
#                 metric=VectorSearchAlgorithmMetric.COSINE
#             ),
#         ),
#     ],
#     profiles=[
#         VectorSearchProfile(
#             name="myHnswProfile",
#             algorithm_configuration_name="default",
#         ),
#         VectorSearchProfile(
#             name="myExhaustiveKnnProfile",
#             algorithm_configuration_name="default_exhaustive_knn",
#         ),
#     ],
# )
# key = os.environ.get("AZURE_SEARCH_API_KEY")
# endpoint = os.environ.get("AZURE_SEARCH_SERVICE_ENDPOINT")
# credential = AzureKeyCredential(key) if key else DefaultAzureCredential()
# user_agent = "vector-io"
# # create new index
# index_client: SearchIndexClient = SearchIndexClient(
#     endpoint=endpoint, credential=credential, user_agent=user_agent
# )

# if index_name in [x.name for x in index_client.list_indexes()]:
#     index = index_client.get_index(index_name)
# else:
#     index = SearchIndex(
#         name=index_name,
#         fields=fields,
#         vector_search=vector_search,
#         semantic_search=semantic_search,
#         scoring_profiles=scoring_profiles,
#         default_scoring_profile=default_scoring_profile,
#         cors_options=cors_options,
#     )
#     index_client.create_index(index)
# # upsert data
# search_client: SearchClient = SearchClient(
#     endpoint=endpoint,
#     index_name=index_name,
#     credential=credential,
#     user_agent=user_agent,
# )
# documents = []
# search_client.merge_or_upload_documents(documents=documents)
