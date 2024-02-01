""" 
import data to vertex ai vector search index
"""
import google.auth
import google.auth.transport.requests

from typing import Dict, List, Optional
from names import DBNames
from os import listdir

from import_vdf.vdf_import_cls import ImportVDF
from util import db_metric_to_standard_metric

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# gcloud config set project $PROJECT_ID - users
import os
import json
import itertools
import pandas as pd
from tqdm import tqdm
from google.cloud import aiplatform as aip
import google.cloud.aiplatform_v1 as aipv1

from dataclasses import dataclass, field

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
        # =========================================================
        # Find index: check by display name for deployed/undeployed
        # =========================================================
        target_index_id = args["target_index_id"]
        
        print(f"Checking undeployed indexes...\n")
        indexes_display_test = [
            # search by index display name
            index.resource_name for index in aip.MatchingEngineIndex.list(
                filter=f'display_name={target_index_id}',
            )
            if index.display_name == target_index_id
        ]
        if indexes_display_test:
            print(f"Found undeployed index:")
            target_index = aip.MatchingEngineIndex(index_name=indexes_display_test[0])
            print(f"target_index: {target_index.display_name}\n")

        if not indexes_display_test:
            print(f"No undeployed indexes named: {target_index_id}\n")
            print(f"Checking deployed indexes...\n")
            all_index_names = [index.resource_name for index in aip.MatchingEngineIndex.list()]
            d_ids = []
            for index in all_index_names:
                test_index = aip.MatchingEngineIndex(index_name=index)
                if test_index.deployed_indexes:
                    d_ids.extend(test_index.deployed_indexes)

            indexes_deployed_test = [
                d_id for d_id in d_ids if (
                    d_id.display_name == target_index_id 
                    or d_id.deployed_index_id == target_index_id
                )
            ]
            if indexes_deployed_test:
                target_index_endpoint = aip.MatchingEngineIndexEndpoint(indexes_deployed_test[0].index_endpoint)
                for d in target_index_endpoint.deployed_indexes:
                    if d.id == target_index_id:
                        target_index = aip.MatchingEngineIndex(index_name=d.index)
                        print(f"Found target_index: {target_index.display_name}")
                        print(f"currently deployed to {target_index_endpoint.display_name}")
            else:
                raise Exception(
                    f"{target_index_id} not found. "
                    "Please provide a valid index name for your project"
                )
        
        self.target_index_resource_name = target_index.resource_name
        
        # =========================================================
        # filters: restricts and crowding
        # =========================================================
        filter_restricts = args["filter_restricts"]
        numeric_restricts = args["numeric_restricts"]
        crowding_tag = args["crowding_tag"]
        self.filter_restricts = filter_restricts if filter_restricts is not None else None
        self.numeric_restricts = numeric_restricts if numeric_restricts is not None else None
        self.crowding_tag = crowding_tag if crowding_tag is not None else None
        
        if self.filter_restricts:
            # String filters: allows and denies
            allows = []
            denies = []
            list_of_ns_restrict_entries = []
            for name in self.filter_restricts:
                name_space_filter_entry = {}
                all_allows = []
                all_denies = []
                allows = []
                denies = []
                name_space_filter_entry['namespace'] = name.get('namespace')
                if name.get("allow_list") is not None:
                    allow_items = name.get("allow_list")
                    allows.append(allow_items)
                    # allows.append([a for a in allow_items])
                if name.get("deny_list") is not None:
                    deny_items = name.get("deny_list")
                    denies.append(deny_items)

                if allows:
                    all_allows = list(itertools.chain.from_iterable(allows))
                    name_space_filter_entry['allow_list'] = all_allows

                if denies:
                    all_denies = list(itertools.chain.from_iterable(denies))
                    name_space_filter_entry['deny_list'] = all_denies

                list_of_ns_restrict_entries.append(name_space_filter_entry)
        
        self.list_of_ns_restrict_entries = list_of_ns_restrict_entries if self.filter_restricts is not None else None
        print(f"list_of_ns_restrict_entries : {self.list_of_ns_restrict_entries}")
        
        if self.numeric_restricts:
            # Numeric filters:
            list_of_numeric_entries = []
            for name in self.numeric_restricts:
                name_space_filter_entry = {}
                name_space_filter_entry['namespace'] = name.get('namespace')
                name_space_filter_entry['data_type'] = name.get('data_type')
                list_of_numeric_entries.append(name_space_filter_entry)
            
        self.list_of_numeric_entries = list_of_numeric_entries if self.numeric_restricts is not None else None
        print(f"list_of_numeric_entries : {self.list_of_numeric_entries}")
        
        # =========================================================
        # Index Client
        # =========================================================
        self.parent = f"projects/{self.project_id}/locations/{self.location}"

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
                        row[vector_column_name] = [
                            float(emb) for emb in row[vector_column_name]
                        ]
                        
                        restrict_entry_list = []
                        allow_values = []
                        deny_values = []
                        
                        # if idx == 10:
                        #     # sanity check
                        #     print(f"row['id'] : {row['id']}")
                        
                        if self.list_of_ns_restrict_entries:
                            for entry in self.list_of_ns_restrict_entries:
                                restrict_entry = {}

                                restrict_entry["namespace"] = entry.get("namespace")

                                if entry.get("allow_list"):
                                    for col in entry.get("allow_list"):
                                        allow_values.append(row[col])
                                        restrict_entry["allow_list"] = [str(a) for a in allow_values]

                                if entry.get("deny_list"):
                                    for col in entry.get("deny_list"):
                                        deny_values.append(row[col])
                                        restrict_entry["deny_list"] = [str(d) for d in deny_values]

                                restrict_entry_list.append(restrict_entry)
                                
                                # if idx == 10:
                                #     print(f"restrict_entry_list : {restrict_entry_list}")

                        if self.list_of_numeric_entries:
                            numeric_restrict_entry_list = []
                            for entry in self.list_of_numeric_entries:
                                numeric_restrict_entry = {}

                                data_type = entry.get("data_type")
                                col_name = entry.get("namespace")
                                numeric_restrict_entry["namespace"] = entry.get("namespace")
                                numeric_restrict_entry[data_type] = row[col_name]
                                numeric_restrict_entry_list.append(numeric_restrict_entry)
                                
                            # if idx == 10:
                            #     # sanity check
                            #     print(f"numeric_restrict_entry_list : {numeric_restrict_entry_list}")
                            
                        if self.crowding_tag:
                            crowding_tag_col = self.crowding_tag
                            crowding_tag_val = row[crowding_tag_col]
                            
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
                                    crowding_attribute=str(crowding_tag_val)
                                )
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
        print(
            f"Updated {self.target_vertexai_index.display_name} with {len(total_ids)} vectors"
        )