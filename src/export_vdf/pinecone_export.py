import datetime
from names import DBNames
from util import standardize_metric
from export_vdf.vdb_export_cls import ExportVDB
from pinecone import Pinecone, Vector
import os
import json
import numpy as np
import json
import pandas as pd
from tqdm import tqdm

PINECONE_MAX_K = 10_000
MAX_TRIES_OVERALL = 150
MAX_FETCH_SIZE = 1_000
THREAD_POOL_SIZE = 30


class ExportPinecone(ExportVDB):
    DB_NAME_SLUG = DBNames.PINECONE

    def __init__(self, args):
        """
        Initialize the class
        """
        # call super class constructor
        super().__init__(args)
        self.pc = Pinecone(api_key=args["pinecone_api_key"])
        self.collected_ids_by_modifying = False

    def get_all_index_names(self):
        return self.pc.list_indexes().names()

    def get_ids_from_vector_query(
        self, index, input_vector, namespace, all_ids, hash_value
    ):
        if self.args.get("modify_to_search"):
            marker_key = "exported_vectorio_" + hash_value
            results = index.query(
                vector=input_vector,
                filters={marker_key: {"$ne": True}},
                top_k=PINECONE_MAX_K,
                namespace=namespace,
            )
            if len(results["matches"]) == 0:
                tqdm.write("No vectors found that have not been exported yet.")
                return []
            # mark the vectors as exported
            ids = [result["id"] for result in results["matches"]]
            ids_to_mark = list(set(ids) - all_ids)
            tqdm.write(
                f"Found {len(ids_to_mark)} vectors that have not been exported yet."
            )
            # fetch the vectors and upsert them with the exported_vectorio flag with MAX_FETCH_SIZE at a time
            mark_pbar = tqdm(total=len(ids_to_mark), desc="Step 1/3: Marking vectors")
            mark_batch_size = MAX_FETCH_SIZE
            i = 0
            while i < len(ids_to_mark):
                batch_ids = ids_to_mark[i : i + mark_batch_size]
                try:
                    data = index.fetch(batch_ids)
                except Exception as e:
                    tqdm.write(
                        f"Error fetching vectors: {e}. Trying with a smaller batch size (--batch_size)"
                    )
                    mark_batch_size = mark_batch_size * 3 // 4
                    if mark_batch_size < MAX_FETCH_SIZE / 100:
                        raise Exception("Could not fetch vectors")
                    continue
                batch_vectors = data["vectors"]
                # verify that the ids are the same
                assert set(batch_ids) == set(batch_vectors.keys())
                # add exported_vectorio flag to metadata
                # Format the vectors for upsert
                upsert_data = []
                for id, vector_data in batch_vectors.items():
                    if "metadata" not in vector_data:
                        vector_data["metadata"] = {}
                    vector_data["metadata"][marker_key] = True
                    cur_vec = Vector(
                        id=id,
                        values=vector_data["values"],
                        metadata=vector_data["metadata"],
                    )
                    if vector_data.get("sparseValues"):
                        cur_vec.sparse_values = vector_data["sparseValues"]
                    upsert_data.append(cur_vec)
                # upsert the vectors
                try:
                    resp = index.upsert(vectors=upsert_data, namespace=namespace)
                except Exception as e:
                    tqdm.write(
                        f"Error upserting vectors: {e}. Trying with a smaller batch size (--batch_size)"
                    )
                    mark_batch_size = mark_batch_size * 3 // 4
                    if mark_batch_size < MAX_FETCH_SIZE / 100:
                        raise Exception("Could not upsert vectors")
                    continue
                i += resp["upserted_count"]
                mark_pbar.update(len(batch_ids))
            self.collected_ids_by_modifying = True
            tqdm.write(f"Marked {len(ids_to_mark)} vectors as exported.")
        else:
            results = index.query(
                vector=input_vector,
                include_values=False,
                top_k=PINECONE_MAX_K,
                namespace=namespace,
            )
        ids = set(result["id"] for result in results["matches"])
        return ids

    def get_all_ids_from_index(
        self, index, num_dimensions, namespace="", hash_value=""
    ):
        if (
            self.args["id_range_start"] is not None
            and self.args["id_range_end"] is not None
        ):
            tqdm.write(
                "Using id range {} to {}".format(
                    self.args["id_range_start"], self.args["id_range_end"]
                )
            )
            return [
                str(x)
                for x in range(
                    int(self.args["id_range_start"]),
                    int(self.args["id_range_end"]) + 1,
                )
            ]
        if self.args["id_list_file"]:
            with open(self.args["id_list_file"]) as f:
                return [line.strip() for line in f.readlines()]
        num_vectors = index.describe_index_stats()["namespaces"][namespace][
            "vector_count"
        ]
        # do small random search and check if ids are int
        random_results = index.query(
            vector=np.random.rand(num_dimensions).tolist(),
            include_values=False,
            top_k=100,
            namespace=namespace,
        )
        random_results_ids_strs = [x["id"] for x in random_results["matches"]]
        all_ids = set()
        if not all(x.isdigit() for x in random_results_ids_strs):
            tqdm.write(
                "The ids are not integers. Please provide a range of ids using --id_list_file if you want to export a subset of vectors."
            )
        else:
            random_results_ids = [
                int(x) for x in random_results_ids_strs if x.isdigit()
            ]
            # keep querying out past the range of ids to get all ids
            all_ids.update(random_results_ids)
            ids_checked = set(all_ids)
            with tqdm(
                total=num_vectors, desc="Collecting IDs using fetch on integer ids"
            ) as pbar:
                fetch_size = MAX_FETCH_SIZE
                while len(all_ids) < num_vectors:
                    range_min = min(all_ids) - fetch_size
                    range_max = max(all_ids) + 10 * fetch_size
                    range_obj = range(range_min, range_max)
                    tqdm.write(
                        "Checking ids in range {} to {}".format(range_min, range_max)
                    )
                    ids_to_fetch = [
                        x
                        for x in list(range_obj)
                        if x not in all_ids.union(ids_checked)
                    ]
                    # in increments of fetch size
                    i = 0
                    try_count = 0
                    while (
                        i < len(ids_to_fetch)
                        and len(all_ids) < num_vectors
                        and try_count < MAX_TRIES_OVERALL
                        and len(ids_to_fetch) > 0
                    ):
                        ids_to_fetch_strs = [
                            str(x) for x in ids_to_fetch[i : i + fetch_size]
                        ]
                        newly_fetched = index.fetch(
                            ids_to_fetch_strs, namespace=namespace
                        )
                        pbar.update(len(newly_fetched["vectors"]))
                        all_ids.update(
                            [int(x) for x in newly_fetched["vectors"].keys()]
                        )
                        i += fetch_size
                        try_count += 1
                        ids_checked.update([int(x) for x in ids_to_fetch_strs])
                    if try_count >= MAX_TRIES_OVERALL:
                        tqdm.write(
                            f"Could not collect all ids after {MAX_TRIES_OVERALL} tries. Please provide range of ids instead. Exporting the ids collected so far."
                        )
                    else:
                        tqdm.write(
                            f"Collected {len(all_ids)} ids out of {num_vectors} vectors in {try_count} fetches."
                        )
                    return [str(x) for x in all_ids]
        # random search method
        max_tries = max((num_vectors // PINECONE_MAX_K) * 3, MAX_TRIES_OVERALL)
        try_count = 0
        # -1s in each dimension are the min values
        vector_range_min = np.array([-1] * num_dimensions)
        vector_range_max = np.array([1] * num_dimensions)
        with tqdm(
            total=num_vectors, desc="Collecting IDs using random vector search"
        ) as pbar:
            while len(all_ids) < num_vectors:
                # fetch 10 random vectors from all_ids
                if len(all_ids) > 10:
                    vector_range_min, vector_range_max = self.update_range(
                        index, all_ids, vector_range_min, vector_range_max
                    )
                input_vector = (
                    np.random.rand(num_dimensions)
                    * (vector_range_max - vector_range_min)
                    + vector_range_min
                )
                ids = self.get_ids_from_vector_query(
                    index, input_vector.tolist(), namespace, all_ids, hash_value
                )
                prev_size = len(all_ids)
                all_ids.update(ids)
                if len(ids) == 0:
                    tqdm.write("No new ids found, exiting...")
                    break
                new_ids = all_ids - set(ids)
                if len(new_ids) > 0:
                    # fetch 1 random vector from new_ids
                    self.update_range_from_new_ids(
                        index, vector_range_min, vector_range_max, new_ids
                    )
                curr_size = len(all_ids)
                if curr_size > prev_size:
                    tqdm.write(
                        f"updating ids set with {curr_size - prev_size} new ids..."
                    )
                try_count += 1
                if try_count > max_tries and len(all_ids) < num_vectors:
                    tqdm.write(
                        f"Could not collect all ids after {max_tries} random searches."
                        " Please provide range of ids instead. Exporting the ids collected so far."
                    )
                    break
                pbar.update(curr_size - prev_size)
        tqdm.write(
            f"Collected {len(all_ids)} ids out of {num_vectors} vectors in {try_count} tries."
        )
        return all_ids

    def update_range_from_new_ids(
        self, index, vector_range_min, vector_range_max, new_ids
    ):
        # use update_range to update the range of the vectors
        self.update_range(index, new_ids, vector_range_min, vector_range_max, size=1)

    def update_range(self, index, all_ids, vector_range_min, vector_range_max, size=10):
        random_ids = np.random.choice(list(all_ids), size=size).tolist()
        random_vectors = [
            x["values"] for x in index.fetch(random_ids)["vectors"].values()
        ]
        # extend the range of the vectors
        random_vectors_np = np.array(random_vectors)

        # Initialize vector_range_min and vector_range_max if they are not set
        if vector_range_min is None:
            vector_range_min = np.min(random_vectors_np, axis=0)
        else:
            vector_range_min = np.minimum(
                vector_range_min, np.min(random_vectors_np, axis=0)
            )

        if vector_range_max is None:
            vector_range_max = np.max(random_vectors_np, axis=0)
        else:
            vector_range_max = np.maximum(
                vector_range_max, np.max(random_vectors_np, axis=0)
            )

        return vector_range_min, vector_range_max

    def unmark_vectors_as_exported(self, all_ids, namespace, hash_value):
        if (
            self.args.get("modify_to_search") == False
            or not self.collected_ids_by_modifying
        ):
            return

        # unmark the vectors as exported
        marker_key = "exported_vectorio_" + hash_value
        for i in tqdm(
            range(0, len(all_ids), MAX_FETCH_SIZE), desc="Step 2/3: Unmarking vectors"
        ):
            batch_ids = all_ids[i : i + MAX_FETCH_SIZE]
            data = self.index.fetch(batch_ids)
            batch_vectors = data["vectors"]
            # verify that the ids are the same
            assert set(batch_ids) == set(batch_vectors.keys())
            # add exported_vectorio flag to metadata
            # Format the vectors for upsert
            upsert_data = []
            for id, vector_data in batch_vectors.items():
                if "metadata" in vector_data:
                    del vector_data["metadata"][marker_key]
                cur_vec = Vector(
                    id=id,
                    values=vector_data["values"],
                    metadata=vector_data["metadata"],
                )
                if vector_data.get("sparseValues"):
                    cur_vec.sparse_values = vector_data["sparseValues"]
                upsert_data.append(cur_vec)
            # upsert the vectors
            resp = self.index.upsert(vectors=upsert_data, namespace=namespace)
        tqdm.write(f"Unmarked {len(all_ids)} vectors as exported.")

    def get_data(self):
        if "index" not in self.args or self.args["index"] is None:
            index_names = self.get_all_index_names()
        else:
            index_names = self.args["index"].split(",")
            # check if index exists
            for index_name in index_names:
                if index_name not in self.get_all_index_names():
                    tqdm.write(f"Index {index_name} does not exist, skipping...")
        index_metas = {}
        for index_name in tqdm(index_names, desc="Fetching indexes"):
            index_meta = self.get_data_for_index(index_name)
            for index_meta_elem in index_meta:
                index_meta_elem["metric"] = standardize_metric(
                    self.pc.describe_index(index_name).metric, self.DB_NAME_SLUG
                )
            index_metas[index_name] = index_meta

        # Create and save internal metadata JSON
        self.file_structure.append(os.path.join(self.vdf_directory, "VDF_META.json"))
        internal_metadata = {
            "version": self.args["library_version"],
            "file_structure": self.file_structure,
            # author is from unix username
            "author": os.environ.get("USER"),
            "exported_from": self.DB_NAME_SLUG,
            "indexes": index_metas,
            # timestamp with timezone
            "exported_at": datetime.datetime.now().astimezone().isoformat(),
        }
        with open(os.path.join(self.vdf_directory, "VDF_META.json"), "w") as json_file:
            json.dump(internal_metadata, json_file, indent=4)
        # print internal metadata properly
        tqdm.write(json.dumps(internal_metadata, indent=4))
        return True

    def get_data_for_index(self, index_name):
        self.index = self.pc.Index(index_name)
        index_info = self.index.describe_index_stats()
        # Fetch the actual data from the Pinecone index
        index_meta = []
        for namespace in tqdm(index_info["namespaces"], desc="Fetching namespaces"):
            namespace_info = index_info["namespaces"][namespace]
            tqdm.write(f"Iterating namespace '{namespace}'")
            vectors_directory = os.path.join(
                self.vdf_directory,
                index_name + ("_" + namespace if namespace else ""),
                f"i{self.file_ctr}.parquet",
            )
            os.makedirs(vectors_directory, exist_ok=True)

            all_ids = list(
                self.get_all_ids_from_index(
                    index=self.pc.Index(
                        index_name,
                    ),
                    num_dimensions=index_info["dimension"],
                    namespace=namespace,
                    hash_value=self.hash_value,
                )
            )
            # unmark the vectors as exported
            self.unmark_vectors_as_exported(all_ids, namespace, self.hash_value)
            # vectors is a dict of string to dict with keys id, values, metadata
            vectors = {}
            metadata = {}
            batch_ctr = 1
            total_size = 0
            prev_total_size = 0
            i = 0
            fetch_size = MAX_FETCH_SIZE
            pbar = tqdm(total=len(all_ids), desc="Final Step: Fetching vectors")
            while i < len(all_ids):
                batch_ids = all_ids[i : i + fetch_size]
                import signal

                # Define your exception for the timeout event
                class TimeoutException(Exception):
                    pass

                # Define your signal handler
                def signal_handler(signum, frame):
                    raise TimeoutException()

                # Set the signal handler
                signal.signal(signal.SIGALRM, signal_handler)

                TIMEOUT = 10  # Set your timeout value here

                try:
                    # Start the timer
                    signal.alarm(TIMEOUT)

                    # Try to fetch the data
                    data = self.index.fetch(batch_ids)

                    # If the fetch is successful, cancel the timer
                    signal.alarm(0)
                except Exception as e:
                    tqdm.write(
                        f"Error fetching vectors: {e}. Trying with a smaller batch size (--batch_size): {fetch_size}"
                    )
                    fetch_size = fetch_size * 3 // 4
                    continue
                batch_vectors = data["vectors"]
                # verify that the ids are the same
                # commenting out as some ids in range might not be present in DB
                # assert set(batch_ids) == set(batch_vectors.keys())
                metadata.update(
                    {
                        k: v["metadata"] if "metadata" in v else {}
                        for k, v in batch_vectors.items()
                    }
                )
                vectors.update({k: v["values"] for k, v in batch_vectors.items()})
                # if size of vectors is greater than 1GB, save the vectors to a parquet file
                if (vectors.__sizeof__() + metadata.__sizeof__()) > self.args[
                    "max_file_size"
                ] * 1024 * 1024:
                    prev_total_size = total_size
                    total_size += self.save_vectors_to_parquet(
                        vectors, metadata, vectors_directory
                    )
                i += fetch_size
                pbar.update(total_size - prev_total_size)
            total_size += self.save_vectors_to_parquet(
                vectors, metadata, vectors_directory
            )
            namespace_meta = {
                "namespace": namespace,
                "total_vector_count": namespace_info["vector_count"],
                "exported_vector_count": total_size,
                "dimensions": index_info["dimension"],
                "model_name": self.args["model_name"],
                "vector_columns": ["vector"],
                "data_path": "/".join(vectors_directory.split("/")[1:]),
            }
            index_meta.append(namespace_meta)
        return index_meta
