Vector IO
=========

This library uses a universal format for vector datasets to easily
export and import data from all vector databases.

See the `Contributing <#contributing>`__ section to add support for your
favorite vector database.

Universal Vector Dataset Format (VDF) specification
---------------------------------------------------

1. VDF_META.json: It is a json file with the following schema:

::

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

   interface VDFMeta {
     version: string;
     file_structure: string[];
     author: string;
     exported_from: 'pinecone' | 'qdrant'; // others when they are added
     indexes: {
       [key: string]: Index[];
     };
     exported_at: string;
   }

2. Parquet files/folders for metadata and vectors.

Installation
------------

.. code:: bash

   git clone https://github.com/AI-Northstar-Tech/vector-io.git
   cd vector-io
   pip install -r requirements.txt

Export Script
-------------

.. code:: bash

   ./export_vdf.py --help

   usage: export.py [-h] [-m MODEL_NAME] [--max_file_size MAX_FILE_SIZE]
                    [--push_to_hub | --no-push_to_hub]
                    {pinecone,qdrant} ...

   Export data from a vector database to VDF

   options:
     -h, --help            show this help message and exit
     -m MODEL_NAME, --model_name MODEL_NAME
                           Name of model used
     --max_file_size MAX_FILE_SIZE
                           Maximum file size in MB (default: 1024)
     --push_to_hub, --no-push_to_hub
                           Push to hub

   Vector Databases:
     Choose the vectors database to export data from

     {pinecone,qdrant}
       pinecone            Export data from Pinecone
       qdrant              Export data from Qdrant

.. code:: bash

   ./export_vdf.py pinecone --help
   usage: export.py pinecone [-h] [-e ENVIRONMENT] [-i INDEX]
                             [-s ID_RANGE_START]
                             [--id_range_end ID_RANGE_END]
                             [-f ID_LIST_FILE]
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

.. code:: bash

   ./export_vdf.py qdrant --help
   usage: export.py qdrant [-h] [-u URL] [-c COLLECTIONS]

   options:
     -h, --help            show this help message and exit
     -u URL, --url URL     Location of Qdrant instance
     -c COLLECTIONS, --collections COLLECTIONS
                           Names of collections to export

Import script
-------------

.. code:: bash

   ./import_vdf.py --help
   usage: import_vdf.py [-h] [-d DIR] {pinecone,qdrant} ...

   Import data from VDF to a vector database

   options:
     -h, --help         show this help message and exit
     -d DIR, --dir DIR  Directory to import

   Vector Databases:
     Choose the vectors database to export data from

     {pinecone,qdrant}
       pinecone         Import data to Pinecone
       qdrant           Import data to Qdrant

   ./import_vdf.py pinecone --help
   usage: import_vdf.py pinecone [-h] [-e ENVIRONMENT]

   options:
     -h, --help            show this help message and exit
     -e ENVIRONMENT, --environment ENVIRONMENT
                           Pinecone environment

   ./import_vdf.py qdrant --help  
   usage: import_vdf.py qdrant [-h] [-u URL]

   options:
     -h, --help         show this help message and exit
     -u URL, --url URL  Qdrant url

Examples
--------

.. code:: bash

   ./export_vdf.py -m hkunlp/instructor-xl --push_to_hub pinecone --environment gcp-starter

Follow the prompt to select the index and id range to export.

Contributing
------------

Adding a new vector database
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to add an import/export implementation for a new vector
database, you must also implement the other side of the import/export
for the same database. Please fork the repo and send a PR for both the
import and export scripts.

Changing the VDF specification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you wish to change the VDF specification, please open an issue to
discuss the change before sending a PR.

Efficiency improvements
~~~~~~~~~~~~~~~~~~~~~~~

If you wish to improve the efficiency of the import/export scripts,
please fork the repo and send a PR.

Questions
---------

If you have any questions, please open an issue on the repo or message
Dhruv Anand on
`LinkedIn <https://www.linkedin.com/in/dhruv-anand-ainorthstartech/>`__
