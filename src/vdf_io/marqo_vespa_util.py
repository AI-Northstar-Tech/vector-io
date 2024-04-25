import ssl
from typing import Any, Dict, List, Optional
import httpx
from pydantic import BaseModel, Field
from rich import print as rprint


class Document(BaseModel):
    id: str
    fields: Dict[str, Any]


class VisitDocumentsResponse(BaseModel):
    path_id: str = Field(alias="pathId")
    documents: List[Document]
    document_count: int = Field(alias="documentCount")
    continuation: Optional[str]


class VespaClient:
    def __init__(
        self,
        config_url: str,
        document_url: str,
        query_url: str,
        content_cluster_name: str,
        cert_file: Optional[str] = None,
        pk_file: Optional[str] = None,
        pool_size: int = 10,
        feed_pool_size: int = 10,
        get_pool_size: int = 10,
        delete_pool_size: int = 10,
        partial_update_pool_size: int = 10,
    ):
        """
        Create a VespaClient object.
        Args:
            config_url: Vespa Deploy API base URL
            document_url: Vespa Document API base URL
            query_url: Vespa Query API base URL
            pool_size: Number of connections to keep in the connection pool
            feed_pool_size: Number of connections to keep in batch feed requests connection pool to Vespa
            get_pool_size: Number of connections to keep in batch get requests connection pool to Vespa
            delete_pool_size: Number of connections to keep batch delete requests connection pool to Vespa
            partial_update_pool_size: Number of connections to keep batch partial update requests connection pool to Vespa
        """
        self.config_url = config_url.strip("/")
        self.document_url = document_url.strip("/")
        self.query_url = query_url.strip("/")
        try:
            self.http_client = httpx.Client(
                limits=httpx.Limits(
                    max_keepalive_connections=pool_size, max_connections=pool_size
                ),
                verify=True if cert_file else False,
                cert=(cert_file, pk_file) if cert_file else None,
            )
        except ssl.SSLError as e:  # noqa: F821
            raise VespaError("Failed to create http client due to SSL error", cause=e)
        self.content_cluster_name = content_cluster_name
        self.feed_pool_size = feed_pool_size
        self.get_pool_size = get_pool_size
        self.delete_pool_size = delete_pool_size
        self.partial_pool_size = partial_update_pool_size

    def get_all_documents(
        self, schema: str, stream=False, continuation: Optional[str] = None
    ) -> VisitDocumentsResponse:
        """
        Get all documents in a schema.
        Args:
            schema: Schema to get from
            stream: Whether to stream the response
            continuation: Continuation token for pagination

        Returns:
            BatchGetDocumentResponse object
        """
        try:
            query_params = {
                "stream": str(stream).lower(),
                "continuation": continuation,
            }
            query_string = "&".join(
                [f"{key}={value}" for key, value in query_params.items() if value]
            )
            url = f"{self.document_url}/document/v1/{schema}/{schema}/docid"
            url = f'{url.strip("?")}?{query_string}'
            print(f"{url=}")
            resp = self.http_client.get(url)
        except httpx.HTTPError as e:
            raise VespaError(e) from e
        self._raise_for_status(resp)
        rprint(resp.json())
        return VisitDocumentsResponse(**resp.json())

    def _raise_for_status(self, resp: httpx.Response) -> None:
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            response = e.response
            try:
                json = response.json()
                error_code = json["error-code"]
                message = json["message"]
            except Exception:
                raise VespaStatusError(message=response.text, cause=e) from e

            self._raise_for_error_code(error_code, message, e)


class MarqoErrorMeta(type):
    """
    This metaclass adds a default __init__ method that takes an optional `message: str` and
    an optional `cause: Exception` to subclasses of `MarqoError`.
    """

    def __new__(cls, name, bases, attrs):
        if name != "MarqoError":
            if not any(issubclass(base, MarqoError) for base in bases):
                raise TypeError(
                    f"Class {name} must inherit from {MarqoError.__name__}. "
                    f"Do not use this metaclass directly. Inherit from {MarqoError.__name__} instead."
                )

        return super().__new__(cls, name, bases, attrs)

    def __init__(cls, name, bases, attrs):
        if "__init__" not in attrs:

            def __init__(
                self, message: Optional[str] = None, cause: Optional[Exception] = None
            ):
                super(cls, self).__init__(message, cause)

            setattr(cls, "__init__", __init__)
        super().__init__(name, bases, attrs)


class MarqoError(Exception, metaclass=MarqoErrorMeta):
    """
    Base class for all Marqo errors.
    """

    def __init__(
        self, message: Optional[str] = None, cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.cause = cause


class VespaError(MarqoError):
    pass


class VespaStatusError(VespaError):
    @property
    def status_code(self) -> int:
        try:
            return self.cause.response.status_code
        except Exception as e:
            raise Exception(f"Could not get status code from {self.cause}") from e

    def __str__(self) -> str:
        try:
            return f"{self.status_code}: {self.message}"
        except Exception:
            return super().__str__()
