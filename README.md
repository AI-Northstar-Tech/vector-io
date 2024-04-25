# Vector IO

<p>
  <a href="https://pypi.org/project/vdf-io/"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/vdf-io"></a>
  <a href="https://pypi.org/project/vdf-io/"><img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/vdf-io?style=flat&link=https%3A%2F%2Fpypi.org%2Fproject%2Fvdf-io%2F"></a>
  <a href="https://discord.gg/HGxDZxNt9G"><img alt="Discord" src="https://img.shields.io/discord/1223707915827937321?style=flat&logo=discord&link=https%3A%2F%2Fdiscord.gg%2FHGxDZxNt9G"></a>
</p>

<p align=center>
<!-- include photo -->
<img src="assets/vector-io-logo.png" width="200"/>
</p>

This library uses a universal format for vector datasets to easily export and import data from all vector databases.

Request support for a VectorDB by voting/commenting on [this poll](https://github.com/AI-Northstar-Tech/vector-io/discussions/38)

See the [Contributing](#contributing) section to add support for your favorite vector database.

## Supported Vector Databases

<details open>
  <summary>Fully Supported</summary>

| Vector Database                | Import | Export |
|--------------------------------|--------|--------|
| Pinecone                       | ✅     | ✅     |
| Qdrant                         | ✅     | ✅     |
| Milvus                         | ✅     | ✅     |
| GCP Vertex AI Vector Search    | ✅     | ✅     |
| KDB.AI                         | ✅     | ✅     |
| LanceDB                        | ✅     | ✅     |

</details>

-----

<details open>

  <summary>Partial</summary>
  
| Vector Database                | Import | Export |
|--------------------------------|--------|--------|

</details>
<!-- line break -->

-----

<details>
  <summary>In Progress</summary>

| Vector Database                | Import | Export |
|--------------------------------|--------|--------|
| DataStax Astra DB              | ❌     | ❌     |
| txtai                          | ❌     | ✅    (pending) |
</details>

-----

<details>
  <summary>Not Supported</summary>

| Vector Database                | Import | Export |
|--------------------------------|--------|--------|
| Azure AI Search                | ❌     | ❌     |
| Rockset                        | ❌     | ❌     |
| MongoDB Atlas                  | ❌     | ❌     |
| Weaviate                       | ❌     | ❌     |
| Epsilla                        | ❌     | ❌     |
| Redis Search                   | ❌     | ❌     |
| OpenSearch                     | ❌     | ❌     |
| Vespa                          | ❌     | ❌     |
| Marqo                          | ❌     | ❌     |
| Activeloop Deep Lake           | ❌     | ❌     |
| Apache Cassandra               | ❌     | ❌     |
| ApertureDB                     | ❌     | ❌     |
| ClickHouse                     | ❌     | ❌     |
| CrateDB                        | ❌     | ❌     |
| Elasticsearch                  | ❌     | ❌     |
| Meilisearch                    | ❌     | ❌     |
| MyScale                        | ❌     | ❌     |
| Neo4j                          | ❌     | ❌     |
| Nuclia DB                      | ❌     | ❌     |
| OramaSearch                    | ❌     | ❌     |
| pgvector                       | ❌     | ❌     |
| Turbopuffer                    | ❌     | ❌     |
| Typesense                      | ❌     | ❌     |
| USearch                        | ❌     | ❌     |
| Anari AI                       | ❌     | ❌     |
| Vald                           | ❌     | ❌     |
| Apache Solr                    | ❌     | ❌     |
</details>

## Installation

### Using pip

```bash
pip install vdf-io
```

### From source

```bash
git clone https://github.com/AI-Northstar-Tech/vector-io.git
cd vector-io
pip install -r requirements.txt
```

## Universal Vector Dataset Format (VDF) specification

1. VDF_META.json: It is a json file with the following schema VDFMeta defined in [src/vdf_io/meta_types.py](src/vdf_io/meta_types.py):

```python
class NamespaceMeta(BaseModel):
    namespace: str
    index_name: str
    total_vector_count: int
    exported_vector_count: int
    dimensions: int
    model_name: str | None = None
    vector_columns: List[str] = ["vector"]
    data_path: str
    metric: str | None = None
    index_config: Optional[Dict[Any, Any]] = None
    schema_dict: Optional[Dict[str, Any]] = None


class VDFMeta(BaseModel):
    version: str
    file_structure: List[str]
    author: str
    exported_from: str
    indexes: Dict[str, List[NamespaceMeta]]
    exported_at: str
    id_column: Optional[str] = None

```

2. Parquet files/folders for metadata and vectors.

## Export Script

```bash
export_vdf --help
usage: export_vdf [-h] [-m MODEL_NAME]
                  [--max_file_size MAX_FILE_SIZE]
                  [--push_to_hub | --no-push_to_hub]
                  [--public | --no-public]
                  {pinecone,qdrant,kdbai,milvus,vertexai_vectorsearch}
                  ...

Export data from various vector databases to the VDF format for vector datasets

options:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        Name of model used
  --max_file_size MAX_FILE_SIZE
                        Maximum file size in MB (default:
                        1024)
  --push_to_hub, --no-push_to_hub
                        Push to hub
  --public, --no-public
                        Make dataset public (default:
                        False)

Vector Databases:
  Choose the vectors database to export data from

  {pinecone,qdrant,kdbai,milvus,vertexai_vectorsearch}
    pinecone            Export data from Pinecone
    qdrant              Export data from Qdrant
    kdbai               Export data from KDB.AI
    milvus              Export data from Milvus
    vertexai_vectorsearch
                        Export data from Vertex AI Vector
                        Search
```

## Import script

```bash
import_vdf --help
usage: import_vdf [-h] [-d DIR] [-s | --subset | --no-subset]
                  [--create_new | --no-create_new]
                  {milvus,pinecone,qdrant,vertexai_vectorsearch,kdbai}
                  ...

Import data from VDF to a vector database

options:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Directory to import
  -s, --subset, --no-subset
                        Import a subset of data (default: False)
  --create_new, --no-create_new
                        Create a new index (default: False)

Vector Databases:
  Choose the vectors database to export data from

  {milvus,pinecone,qdrant,vertexai_vectorsearch,kdbai}
    milvus              Import data to Milvus
    pinecone            Import data to Pinecone
    qdrant              Import data to Qdrant
    vertexai_vectorsearch
                        Import data to Vertex AI Vector Search
    kdbai               Import data to KDB.AI
```

## Re-embed script

This Python script is used to re-embed a vector dataset. It takes a directory of vector dataset in the VDF format and re-embeds it using a new model. The script also allows you to specify the name of the column containing text to be embedded.

```bash
reembed_vdf --help
usage: reembed_vdf [-h] -d DIR [-m NEW_MODEL_NAME]
                  [-t TEXT_COLUMN]

Reembed a vector dataset

options:
  -h, --help            show this help message and exit
  -d DIR, --dir DIR     Directory of vector dataset in
                        the VDF format
  -m NEW_MODEL_NAME, --new_model_name NEW_MODEL_NAME
                        Name of new model to be used
  -t TEXT_COLUMN, --text_column TEXT_COLUMN
                        Name of the column containing
                        text to be embedded
```

## Examples

```bash
export_vdf -m hkunlp/instructor-xl --push_to_hub pinecone --environment gcp-starter

import_vdf -d /path/to/vdf/dataset milvus

reembed_vdf -d /path/to/vdf/dataset -m sentence-transformers/all-MiniLM-L6-v2 -t title
```

Follow the prompt to select the index and id range to export.

## Contributing

### Adding a new vector database

If you wish to add an import/export implementation for a new vector database, you must also implement the other side of the import/export for the same database.
Please fork the repo and send a PR for both the import and export scripts.

Steps to add a new vector database (ABC):

1. Add your database name in [src/vdf_io/names.py](src/vdf_io/names.py) in the DBNames enum class.
2. Create new files `src/vdf_io/export_vdf/export_abc.py` and `src/vdf_io/import_vdf/import_abc.py` for the new DB.

**Export**:

1. In your export file, define a class ExportABC which inherits from ExportVDF.
2. Specify a DB_NAME_SLUG for the class
3. The class should implement:
   1. make_parser() function to add database specific arguments to the export_vdf CLI
   2. export_vdb() function to prompt user for info not provided in the CLI. It should then call the get_data() function.
   3. get_data() function to download points (in a batched manner) with all the metadata from the specified index of the vector database. This data should be stored in a series of parquet files/folders. The metadata should be stored in a json file with the [schema above](#universal-vector-dataset-format-vdf-specification).
4. Use the script to export data from an example index of the vector database and verify that the data is exported correctly.

**Import**:

1. In your import file, define a class ImportABC which inherits from ImportVDF.
2. Specify a DB_NAME_SLUG for the class
3. The class should implement:
   1. make_parser() function to add database specific arguments to the import_vdf CLI, such as the url of the database, any authentication tokens, etc.
   2. import_vdb() function to prompt user for info not provided in the CLI. It should then call the upsert_data() function.
   3. upsert_data() function to upload points from a vdf dataset (in a batched manner) with all the metadata to the specified index of the vector database. All metadata about the dataset should be read from the VDF_META.json file in the vdf folder.
4. Use the script to import data from the example vdf dataset exported in the previous step and verify that the data is imported correctly.

### Changing the VDF specification

If you wish to change the VDF specification, please open an issue to discuss the change before sending a PR.

### Efficiency improvements

If you wish to improve the efficiency of the import/export scripts, please fork the repo and send a PR.

## Telemetry

Running the scripts in the repo will send anonymous usage data to AI Northstar Tech to help improve the library.

You can opt out this by setting the environment variable `DISABLE_TELEMETRY_VECTORIO` to `1`.

## Questions

If you have any questions, please open an issue on the repo or message Dhruv Anand on [LinkedIn](https://www.linkedin.com/in/dhruv-anand-ainorthstartech/)
