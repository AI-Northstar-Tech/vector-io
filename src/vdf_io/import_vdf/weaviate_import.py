from typing import List

from dotenv import load_dotenv
from halo import Halo
from tqdm import tqdm
import weaviate

from vdf_io.constants import INT_MAX
from vdf_io.import_vdf.vdf_import_cls import ImportVDB
from vdf_io.names import DBNames
from vdf_io.util import set_arg_from_input, set_arg_from_password
from vdf_io.weaviate_helpers import (
    build_weaviate_properties,
    coerce_weaviate_uuid,
    get_vector_payload,
    sanitize_weaviate_collection_name,
)

load_dotenv()


class ImportWeaviate(ImportVDB):
    DB_NAME_SLUG = DBNames.WEAVIATE

    @classmethod
    def make_parser(cls, subparsers):
        parser_weaviate = subparsers.add_parser(
            cls.DB_NAME_SLUG, help="Import data to Weaviate"
        )
        parser_weaviate.add_argument("--url", type=str, help="URL of Weaviate instance")
        parser_weaviate.add_argument("--api_key", type=str, help="Weaviate API key")
        parser_weaviate.add_argument(
            "--connection_type",
            type=str,
            choices=["auto", "cloud", "local", "custom"],
            default="auto",
            help="Weaviate connection type (default: auto)",
        )
        parser_weaviate.add_argument(
            "--host",
            type=str,
            default="localhost",
            help="Host for local/custom Weaviate connections",
        )
        parser_weaviate.add_argument(
            "--http_port",
            type=int,
            default=8080,
            help="HTTP port for local/custom Weaviate connections",
        )
        parser_weaviate.add_argument(
            "--grpc_host",
            type=str,
            help="gRPC host for custom Weaviate connections",
        )
        parser_weaviate.add_argument(
            "--grpc_port",
            type=int,
            default=50051,
            help="gRPC port for local/custom Weaviate connections",
        )

    @classmethod
    def import_vdb(cls, args):
        set_arg_from_input(
            args,
            "connection_type",
            "Enter Weaviate connection type (auto/cloud/local/custom) (default: auto): ",
            str,
            "auto",
            choices=["auto", "cloud", "local", "custom"],
        )
        if args.get("connection_type") in {"auto", "cloud", "custom"}:
            set_arg_from_input(
                args,
                "url",
                "Enter the URL of Weaviate instance (leave empty for local): ",
                str,
            )
        set_arg_from_password(
            args,
            "api_key",
            "Enter the Weaviate API key (leave empty if unused): ",
            "WEAVIATE_API_KEY",
        )
        weaviate_import = ImportWeaviate(args)
        try:
            weaviate_import.upsert_data()
        except KeyboardInterrupt:
            tqdm.write(
                f"Data import interrupted. {weaviate_import.total_imported_count} rows imported."
            )
        return weaviate_import

    def __init__(self, args):
        super().__init__(args)
        self.client = self._connect()

    def _api_key_auth(self):
        if not self.args.get("api_key"):
            return None
        try:
            from weaviate.classes.init import Auth

            return Auth.api_key(self.args["api_key"])
        except Exception:
            return weaviate.auth.AuthApiKey(self.args["api_key"])

    def _connect(self):
        connection_type = self.args.get("connection_type", "auto") or "auto"
        url = self.args.get("url")
        auth_credentials = self._api_key_auth()

        if connection_type == "auto":
            if url and ("weaviate.cloud" in url or "weaviate.network" in url):
                connection_type = "cloud"
            elif url:
                connection_type = "custom"
            else:
                connection_type = "local"

        if connection_type == "cloud":
            connector = getattr(weaviate, "connect_to_weaviate_cloud", None)
            if connector is None:
                connector = getattr(weaviate, "connect_to_wcs")
            return connector(
                cluster_url=url,
                auth_credentials=auth_credentials,
                skip_init_checks=True,
            )

        if connection_type == "local":
            kwargs = {
                "host": self.args.get("host", "localhost"),
                "port": self.args.get("http_port", 8080),
                "grpc_port": self.args.get("grpc_port", 50051),
            }
            if auth_credentials is not None:
                kwargs["auth_credentials"] = auth_credentials
            return weaviate.connect_to_local(**kwargs)

        from urllib.parse import urlparse

        parsed_url = urlparse(url if "://" in url else f"http://{url}")
        http_secure = parsed_url.scheme == "https"
        http_port = parsed_url.port or (
            443 if http_secure else self.args.get("http_port", 8080)
        )
        grpc_host = self.args.get("grpc_host") or parsed_url.hostname
        grpc_port = self.args.get("grpc_port") or (443 if http_secure else 50051)
        kwargs = {
            "http_host": parsed_url.hostname,
            "http_port": http_port,
            "http_secure": http_secure,
            "grpc_host": grpc_host,
            "grpc_port": grpc_port,
            "grpc_secure": http_secure,
        }
        if auth_credentials is not None:
            kwargs["auth_credentials"] = auth_credentials
        return weaviate.connect_to_custom(**kwargs)

    def get_all_index_names(self) -> List[str]:
        all_collections = self.client.collections.list_all()
        if isinstance(all_collections, dict):
            return list(all_collections.keys())
        return [
            getattr(collection, "name", str(collection))
            for collection in all_collections
        ]

    def _get_collection(self, collection_name):
        if hasattr(self.client.collections, "use"):
            return self.client.collections.use(collection_name)
        return self.client.collections.get(collection_name)

    def _self_provided_vector_config(self, vector_column_names):
        from weaviate.classes.config import Configure

        if hasattr(Configure, "Vectors"):
            if len(vector_column_names) == 1 and vector_column_names[0] == "vector":
                return Configure.Vectors.self_provided()
            return [
                Configure.Vectors.self_provided(name=vector_column_name)
                for vector_column_name in vector_column_names
            ]

        if len(vector_column_names) == 1 and vector_column_names[0] == "vector":
            return Configure.Vectorizer.none()
        return [
            Configure.NamedVectors.none(name=vector_column_name)
            for vector_column_name in vector_column_names
        ]

    def _create_collection(self, collection_name, vector_column_names):
        vector_config = self._self_provided_vector_config(vector_column_names)
        try:
            self.client.collections.create(
                name=collection_name,
                vector_config=vector_config,
            )
        except TypeError:
            self.client.collections.create(
                name=collection_name,
                vectorizer_config=vector_config,
            )

    def upsert_data(self):
        max_hit = False
        self.total_imported_count = 0
        existing_collections = self.get_all_index_names()
        for index_name, index_meta in tqdm(
            self.vdf_meta["indexes"].items(), desc="Importing indexes"
        ):
            tqdm.write(f"Importing data for index '{index_name}'")
            for namespace_meta in tqdm(index_meta, desc="Importing namespaces"):
                self.set_dims(namespace_meta, index_name)
                data_path = namespace_meta["data_path"]
                final_data_path = self.get_final_data_path(data_path)
                new_collection_name = index_name + (
                    f"_{namespace_meta['namespace']}"
                    if namespace_meta["namespace"]
                    else ""
                )
                new_collection_name = sanitize_weaviate_collection_name(
                    new_collection_name
                )
                new_collection_name = self.create_new_name(
                    new_collection_name,
                    existing_collections,
                    delimiter="_",
                )
                vector_column_names, _ = self.get_vector_column_name(
                    new_collection_name, namespace_meta, multi_vector_supported=True
                )
                if new_collection_name not in existing_collections:
                    self._create_collection(new_collection_name, vector_column_names)
                    existing_collections.append(new_collection_name)

                collection = self._get_collection(new_collection_name)
                parquet_files = self.get_parquet_files(final_data_path)
                for file in tqdm(parquet_files, desc="Iterating parquet files"):
                    remaining_rows = (
                        self.args.get("max_num_rows") or INT_MAX
                    ) - self.total_imported_count
                    if remaining_rows <= 0:
                        max_hit = True
                        break

                    file_path = self.get_file_path(final_data_path, file)
                    df = self.read_parquet_progress(
                        file_path,
                        max_num_rows=remaining_rows,
                    )
                    imported_count = self._import_dataframe(
                        collection, df, vector_column_names
                    )
                    self.total_imported_count += imported_count
                    if self.total_imported_count >= (
                        self.args.get("max_num_rows") or INT_MAX
                    ):
                        max_hit = True
                        break
                if max_hit:
                    break
            if max_hit:
                tqdm.write(
                    f"Max rows to be imported {self.args['max_num_rows']} hit. Exiting"
                )
                break
        tqdm.write("Data import completed successfully.")
        self.args["imported_count"] = self.total_imported_count

    def _import_dataframe(self, collection, df, vector_column_names):
        imported_count = 0
        batch_size = self.args.get("batch_size", 200) or 200
        with collection.batch.fixed_size(batch_size=batch_size) as batch:
            with Halo(text="Uploading vectors to Weaviate", spinner="dots"):
                for _, row in tqdm(
                    df.iterrows(), desc="Preparing objects", total=len(df)
                ):
                    row_dict = row.to_dict()
                    object_id = row_dict.get(self.id_column)
                    if object_id is None:
                        continue
                    vectors = get_vector_payload(row_dict, vector_column_names)
                    if not vectors:
                        continue
                    vector_arg = (
                        vectors[vector_column_names[0]]
                        if len(vector_column_names) == 1
                        else vectors
                    )
                    batch.add_object(
                        properties=build_weaviate_properties(
                            row_dict, self.id_column, vector_column_names
                        ),
                        vector=vector_arg,
                        uuid=coerce_weaviate_uuid(object_id),
                    )
                    imported_count += 1
                    if batch.number_errors > 10:
                        tqdm.write("Batch import stopped due to excessive errors.")
                        break
        failed_objects = collection.batch.failed_objects
        if failed_objects:
            tqdm.write(f"Number of failed imports: {len(failed_objects)}")
            tqdm.write(f"First failed object: {failed_objects[0]}")
        return imported_count

    def cleanup(self):
        if hasattr(self, "client") and hasattr(self.client, "close"):
            self.client.close()
        super().cleanup()
