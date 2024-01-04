# Vector IO

## **Vector DB companies don't have an incentive to create an interoperable format for vector datasets. So we've built this library to help the community avoid vendor lock-in and ease migrations and restructuring**

### Use the universal VDF format for vector datasets to easily export and import data from all vector databases

**[Universal Vector Dataset Format (VDF) specification](https://docs.google.com/document/d/1SaZ0nsBw8ZZCCcPXoc2nwTY5A3KBJkTEnmZZvFxHAu4/edit#heading=h.32if60hafsdt)**

## Motivation

Each vector database has their own way of storing vectors and metadata, making it hard for people to transfer a dataset from one into the other.
Existing Alternatives:

1. Qdrant (<https://qdrant.tech/documentation/cloud/backups/>) and Weaviate (<https://weaviate.io/developers/weaviate/configuration/backups>) have their own backup formats, which are not portable to other DBs.
2. Pinecone has a datasets library: <https://pinecone-io.github.io/pinecone-datasets/pinecone_datasets.html> but it is not used outside of their demo public datasets. Their backups are stored on their own servers, with their own proprietary format, without the ability to move the data out of their cluster.
3. Txt-ai: they have a proprietary format that can only be loaded via their library: <https://huggingface.co/NeuML/txtai-wikipedia/tree/main>. It was used for just one wikipedia embeddings dump.
4. Cohere (<https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings>) released their model's wikipedia embeddings in an ad-hoc schema in parquet format.
5. Macrocosm/Alexandria: They provide various embedding dumps. They distribute them as zip files, containing multiple parquet files. The parquet file contains both the embedding as well as metadata like title and doi. They provide an ad-hoc script to read the data, and a params.txt file to record the version and the model+prompt used.

A collection of utility functions and scripts to import and export vector datasets between various vector databases.

The representation that we export to is:

1. parquet file for the vectors (to be changed to Parquet)
2. SQLite file for metadata
3. json for meta information about the dataset (author, description, model used, statistics, licensing)

Feel free to send a PR to add an import/export implementation for your favorite vector DB.

## Export

    ./export.py --help
usage: export.py [-h] [-m MODEL_NAME] [--max_file_size MAX_FILE_SIZE]
                 {pinecone,weaviate,qdrant} ...

Export data from Pinecone, Weaviate and Qdrant to sqlite database and parquet
file

options:
  -h, --help            show this help message and exit
  -m MODEL_NAME, --model_name MODEL_NAME
                        Name of model used
  --max_file_size MAX_FILE_SIZE
                        Maximum file size in MB (default: 1024)

Vector Databases:
  Choose the vectors database to export data from

  {pinecone,weaviate,qdrant}
    pinecone            Export data from Pinecone
    weaviate            Export data from Weaviate
    qdrant              Export data from Qdrant

./export.py pinecone --help
usage: export.py pinecone [-h] [-e ENVIRONMENT] [-i INDEX] [-s ID_RANGE_START]
                          [--id_range_end ID_RANGE_END] [-f ID_LIST_FILE]
                          [--modify_to_search MODIFY_TO_SEARCH]

options:
  -h, --help            show this help message and exit
  -e ENVIRONMENT, --environment ENVIRONMENT
                        Environment of Pinecone instance
  -i INDEX, --index INDEX
                        Name of index to export
  -s ID_RANGE_START, --id_range_start ID_RANGE_START
                        Start of id range
  --id_range_end ID_RANGE_END
                        End of id range
  -f ID_LIST_FILE, --id_list_file ID_LIST_FILE
                        Path to id list file
  --modify_to_search MODIFY_TO_SEARCH
                        Allow modifying data to search


## Import

WIP

# Compare Vector DBs easily

Check out the [Feature Matrix](https://docs.google.com/spreadsheets/d/e/2PACX-1vTw7znhJYkkJ_EM7ZPMPRPPuAE8kjUDfvi9STzvq1sXaeqei4LSGL_Qpfe-MooQZPHROhdzgJcY8ZXF/pubhtml) which lists all the vector DBs and their attributes.

You can chat with [VectorDB Guide GPT](https://chat.openai.com/g/g-OS6d9grY0-vectordb-guide) to figure out which DB you should to use. It uses the data fro mthe above table.

## Other VectorDB Feature Matrices

- <https://objectbox.io/vector-database/>
- [Vector Database Comparison Cheatsheet](https://docs.google.com/spreadsheets/d/1oAeF4Q7ILxxfInGJ8vTsBck3-2U9VV8idDf3hJOozNw/edit?pli=1#gid=0)
