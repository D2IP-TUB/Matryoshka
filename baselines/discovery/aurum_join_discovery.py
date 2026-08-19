"""Aurum-based join discovery, used only by the paper's baseline augmenters.

Three variants are kept:

``AurumJoinDiscoveryLakeBench``
    MinHash-index variant following the LakeBench harness.
``AurumJoinDiscovery``
    Local index variant reading the pickled MinHash/HNSW graphs under
    ``baselines/discovery/Aurum/graphs``.
``AurumJoinDiscoveryOG``
    Thin client for the original Aurum service (Elasticsearch + REST API).

These classes were split out of ``matryoshka.retrieval`` so that the core
package does not depend on ``datasketch``, ``hnswlib`` or a running Aurum
service. See ``baselines/README.md`` for provenance.
"""
import csv
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import polars as pl
import polars.selectors as cs
import requests
from datasketch import MinHash

from .Aurum.hnsw_search import HNSWSearcher


class AurumJoinDiscoveryLakeBench:
    def __init__(self, index_file: str, separator: str = '\t', num_perm: int = 128, scale: float = 1.0, random_seed: int = 42) -> None:
        """
        Initialize Aurum-based join discovery using local MinHash index.
        
        Parameters:
        ----------
        index_file: str
            Path to the MinHash embeddings pickle file created by build_hash.py
        hnsw_index_file: str
            Path to precomputed HNSW index file (optional). If provided and exists,
            the index will be loaded. If not provided or doesn't exist, index will
            be built on-the-fly (and saved if path is provided).
        separator: str
            CSV separator used for data lake tables (default: '\t')
        num_perm: int
            Number of MinHash permutations used in index (default: 128)
        scale: float
            Percentage of index to use for scalability experiments (default: 1.0)
        random_seed: int
            Random seed for deterministic table sampling (default: 42)
        """
        self.index_file = f'{index_file.split(".")[0]}_embeddings.pkl'
        self.hnsw_index_file = f'{index_file.split(".")[0]}_index.bin'
        self.separator = separator.encode().decode('unicode_escape')
        self.num_perm = num_perm
        self.scale = scale
        self.random_seed = random_seed
        lake_name = self.index_file.split('/')[-1].split('_embeddings.pkl')[0]
        params_map = {
            'nyc': {
                'K': 20,
                'N': 25,
                'threshold': 0.5,
                'max_join_cols': 3
            },
            'canada_us_uk_open_data': {
                'K': 20,
                'N': 75,
                'threshold': 0.5,
                'max_join_cols': 3
            },
            'gittables': {
                'K': 20,
                'N': 200,
                'threshold': 0.5,
                'max_join_cols': 3
            },
        }
        self.params = params_map[lake_name]
        
        if not os.path.exists(self.index_file):
            raise ValueError(f"Index file does not exist: {self.index_file}")
        if self.hnsw_index_file and not os.path.exists(self.hnsw_index_file):
            print(f"HNSW index file does not exist: {self.hnsw_index_file}. The index will be built on-the-fly.")
            self.hnsw_index_file = None


    def find_joinable_tables(self, query_table_path: str, features: list[str], query_separator: str = ',', output_path: str = None) -> pl.DataFrame:
        """
        Find joinable tables for a query table using MinHash-based similarity search.
        
        Parameters:
        ----------
        query_table_path: str
            Path to the query table CSV file
        query_separator: str
            CSV separator for query table (default: ',')
        K: int
            Number of top candidate tables to return (default: 10)
        N: int
            Number of nearest neighbors per column (default: 10)
        threshold: float
            Similarity threshold for column matching (default: 0.7)
        max_join_cols: int
            Maximum number of matching columns for join mode (default: 3)
        output_path: str
            Optional path to save join paths CSV
            
        Returns:
        -------
        pl.DataFrame: Join paths with schema (from_id, to_id, from_column, to_column, weight)
        """
        query_separator = query_separator.encode().decode('unicode_escape')
        
        if not os.path.exists(query_table_path):
            raise ValueError(f"Query table does not exist: {query_table_path}")
        query_table_splits_path = f'{"/".join(query_table_path.split("/")[:-1])}/splits.json'
        with open(query_table_splits_path, 'r') as f:
            splits_info = json.load(f)
        query_col = splits_info[0]['query_col']
        query_table = pl.read_csv(query_table_path, separator=query_separator, columns=[query_col])
        query_table_path = f'{query_table_path}_aurum_query.csv'
        query_table.write_csv(query_table_path, separator=query_separator)
        
        # Build query embeddings
        query = self._build_query_embeddings(query_table_path, query_separator)
        
        # Initialize searcher with join mode (will load precomputed index if available)
        searcher = HNSWSearcher(self.index_file, self.hnsw_index_file, self.scale, search_mode='join', random_seed=self.random_seed)
        
        # Execute search
        results, num_candidates = searcher.topk(
            'aurum',
            query,
            **self.params
        )
        
        # Read query table header
        with open(query_table_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=query_separator)
            query_header = next(reader)
        
        # Collect all join path records in a list for efficient batch creation
        join_records = []
        
        for result in results:
            score = result[0]
            column_pairs = result[1]
            candidate_table = result[2]
            
            if len(column_pairs) > 0:
                # Sort column_pairs for deterministic ordering
                sorted_pairs = sorted(column_pairs, key=lambda x: (x[0], x[1], -x[2]))
                for query_col_idx, cand_col_idx, similarity in sorted_pairs:
                    query_col_name = query_header[query_col_idx] if query_col_idx < len(query_header) else f'col_{query_col_idx}'
                    cand_col_name = f'col_{cand_col_idx}'
                    
                    join_records.append({
                        'from_id': query_table_path,
                        'to_id': f'{candidate_table}.csv',
                        'from_column': query_col_name,
                        'to_column': cand_col_name,
                        'weight': similarity
                    })
        
        # Create DataFrame from all records at once
        join_paths = pl.DataFrame(join_records, schema={
            'from_id': pl.String,
            'to_id': pl.String,
            'from_column': pl.String,
            'to_column': pl.String,
            'weight': pl.Float64
        })
        
        # Remove duplicates and sort with all keys for deterministic output
        join_paths = join_paths.filter(pl.col('from_column').is_in(features))
        join_paths = join_paths.unique(maintain_order=True)
        join_paths = join_paths.sort('weight', descending=True).head(50)
        join_paths = join_paths.sort(['from_column', 'to_id', 'to_column', 'weight'], descending=[False, False, False, True])
        
        if output_path:
            join_paths.write_csv(output_path)
        
        return join_paths


    def _build_query_embeddings(self, query_table_path: str, separator: str) -> tuple:
        """
        Build MinHash embeddings for a query table.
        
        Parameters:
        ----------
        query_table_path: str
            Path to query CSV file
        separator: str
            CSV delimiter
            
        Returns:
        -------
        tuple: (table_name, column_embeddings_array)
        """
        data_array = []
        try:
            with open(query_table_path, encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=separator)
                for idx, row in enumerate(csv_reader):
                    if idx == 0:
                        header = row
                    elif row:
                        data_array.append(row)
        except Exception as e:
            raise ValueError(f"Error reading query table: {e}")
        
        if len(data_array) == 0:
            raise ValueError("Query table is empty")
        
        # Build MinHash for each column
        column_embeddings = []
        for col_idx in range(len(data_array[0])):
            m = MinHash(num_perm=self.num_perm)
            unique_vals = list(set([row[col_idx] for row in data_array if col_idx < len(row)]))
            for val in unique_vals:
                m.update(val.encode('utf-8'))
            column_embeddings.append(list(m.hashvalues))
        
        return (query_table_path, np.array(column_embeddings))


def _to_snake_case(name: str) -> str:
    """Canonical snake_case for a column name. Lower-cases, splits camelCase,
    replaces whitespace and other non-alphanumeric runs with single
    underscores, and trims trailing underscores. Idempotent."""
    if name is None:
        return name
    # Split camelCase boundaries: aB -> a_B.
    out = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', '_', name)
    # Split ABCxyz -> ABC_xyz to handle acronym + word transitions.
    out = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', '_', out)
    out = out.lower()
    # Collapse non-alphanumeric runs (spaces, punctuation, dashes) into one underscore.
    out = re.sub(r'[^a-z0-9]+', '_', out)
    return out.strip('_')


class AurumJoinDiscovery:
    def __init__(self, index_dir: str, separator: str = '\t', num_perm: int = 256, scale: float = 1.0, random_seed: int = 42) -> None:
        """
        Initialize LSH Ensemble-based join discovery.

        Parameters:
        ----------
        index_dir: str
            Path to the LSH Ensemble index directory (containing lsh_ensemble.pkl,
            column_map.pkl, and metadata.pkl built by augmentation/LSH/build_index.py)
        separator: str
            CSV separator used for data lake tables (default: '\\t')
        num_perm: int
            Kept for API compatibility; actual value is read from index metadata.
        scale: float
            Kept for API compatibility.
        random_seed: int
            Kept for API compatibility.
        """
        self.index_dir = index_dir.replace('.pkl', '')
        self.separator = separator.encode().decode('unicode_escape')
        self.scale = scale
        self.random_seed = random_seed

        # Load LSH Ensemble index (also sets self._LSHMinHash)
        self._load_index()

        # Lake-specific params
        lake_name = os.path.basename(index_dir).split('_')[0]
        params_map = {
            'nyc': {'top_k': 20},
            'canada': {'top_k': 20},
            'gittables': {'top_k': 20},
        }
        self.params = params_map.get(lake_name, {'top_k': 20})

    def _load_index(self) -> None:
        """
        Load LSH Ensemble index files from disk.

        Temporarily swaps sys.modules to use the local datasketch package
        (augmentation/LSH/datasketch/) for correct unpickling, since the
        index was built with that package rather than the pip-installed one.
        """
        for fname in ('lsh_ensemble.pkl', 'column_map.pkl', 'metadata.pkl'):
            path = os.path.join(self.index_dir, fname)
            if not os.path.exists(path):
                raise ValueError(f"Index file not found: {path}")

        lsh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LSH')

        # Save pip-installed datasketch modules and swap in local ones for unpickling
        saved_modules = {
            key: sys.modules.pop(key)
            for key in list(sys.modules)
            if key == 'datasketch' or key.startswith('datasketch.')
        }
        sys.path.insert(0, lsh_dir)
        try:
            import datasketch as _local_ds
            self._LSHMinHash = _local_ds.MinHash

            with open(os.path.join(self.index_dir, 'lsh_ensemble.pkl'), 'rb') as f:
                self.lsh_ensemble = pickle.load(f)
            with open(os.path.join(self.index_dir, 'column_map.pkl'), 'rb') as f:
                self.column_map = pickle.load(f)
            with open(os.path.join(self.index_dir, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
        finally:
            sys.path.remove(lsh_dir)
            # Remove local datasketch modules and restore pip ones
            for key in list(sys.modules):
                if key == 'datasketch' or key.startswith('datasketch.'):
                    del sys.modules[key]
            sys.modules.update(saved_modules)

        self.num_perm = self.metadata['num_perm']
        self.seed = self.metadata.get('seed', 42)
        self.threshold = self.metadata['threshold']
        # Prefer separator from index metadata (what the lake was built with)
        if 'separator' in self.metadata:
            self.separator = self.metadata['separator']

    def find_joinable_tables(self, query_table_path: str, query_col: str, features: list[str],
                             query_separator: str = ',', output_path: str = None,
                             min_containment: float = 0.1) -> pl.DataFrame:
        """
        Find joinable tables using LSH Ensemble containment search.

        Parameters:
        ----------
        query_table_path: str
            Path to the query table CSV file
        features: list[str]
            List of column names to consider for join discovery
        query_separator: str
            CSV separator for query table (default: ',')
        output_path: str
            Optional path to save join paths CSV
        min_containment: float
            Effective containment threshold at query time. The index was
            built with ``threshold=self.threshold`` (typically 0.5), but we
            scale the query size by this factor so LSH Ensemble surfaces
            candidates whose containment is at least ``min_containment``.
            The exact-containment loop below still filters by this value.
        """
        query_separator = query_separator.encode().decode('unicode_escape')

        if not os.path.exists(query_table_path):
            raise ValueError(f"Query table does not exist: {query_table_path}")

        # Read splits info to get query column
        query_table_splits_path = os.path.join(os.path.dirname(query_table_path), 'splits.json')
        with open(query_table_splits_path, 'r') as f:
            splits_info = json.load(f)
        query_col = splits_info[0]['query_col']

        # Read query table. Load every feature column so the per-feature
        # loop below can actually inspect non-query columns (was a no-op
        # before, since only ``query_col`` survived the read).
        _all_cols = pl.read_csv(query_table_path, separator=query_separator, n_rows=0).columns
        _wanted = [c for c in {query_col, *features} if c in _all_cols]
        query_table = pl.read_csv(query_table_path, separator=query_separator, columns=_wanted)

        # Note: an earlier version of this method snake_cased the query
        # column names. That was empirically a no-op for MinHash matching
        # (which is value-based) but it propagated through to the
        # ``from_column`` field of the emitted join_paths.csv, breaking
        # downstream consumers that look up the base-table column by name
        # (notably AutoFeat at autofeat.py:277 which crashes on
        # ``previous_join['arrest.csv.age']`` when the actual column is
        # ``arrest.csv.Age``). Column names are now left in their canonical
        # form on both sides.

        # Defensive filter: avoid target leakage when the lake folder accidentally
        # contains the base table (or its sibling metadata). Match by realpath
        # AND by basename, since the lake may hold a separate copy of the base
        # table on disk (different inode, same content as query_table_path).
        query_basename = os.path.basename(query_table_path)
        try:
            query_realpath = os.path.realpath(query_table_path)
        except OSError:
            query_realpath = query_table_path
        _excluded_basenames = {query_basename, 'connections.csv', 'tables.json'}

        def _is_excluded(file_path: str) -> bool:
            if os.path.basename(file_path) in _excluded_basenames:
                return True
            try:
                return os.path.realpath(file_path) == query_realpath
            except OSError:
                return False

        join_records = []

        # Query LSH Ensemble for each feature column
        for col_name in features:
            if col_name not in query_table.columns:
                continue

            # Get unique non-null string values
            values = query_table[col_name].drop_nulls().cast(pl.String).unique().to_list()
            values = [v.strip() for v in values if v and v.strip()]

            if not values:
                continue

            # Build MinHash using the same params as the index
            mh = self._LSHMinHash(num_perm=self.num_perm, seed=self.seed)
            mh.update_batch(values)
            query_size = len(values)
            query_set = set(values)

            # Lie to LSH Ensemble about the query size so it returns columns
            # with at least ``min_containment`` overlap rather than the
            # ``self.threshold`` (typically 0.5) baked into the index. The
            # downstream exact-containment loop still filters precisely.
            effective_size = max(1, int(round(query_size * min_containment / max(self.threshold, 1e-6))))
            candidates = self.lsh_ensemble.query(mh, effective_size)

            if not candidates:
                continue

            # Group candidates by file for efficient IO
            file_candidates = {}
            for key in candidates:
                if key not in self.column_map:
                    continue
                info = self.column_map[key]
                if _is_excluded(info['file_path']):
                    continue
                file_candidates.setdefault(info['file_path'], []).append(info['column_name'])

            # Compute exact containment for each candidate
            for file_path, cand_cols in file_candidates.items():
                try:
                    cand_df = pl.read_csv(
                        file_path, separator=self.separator,
                        columns=cand_cols, infer_schema=False
                    )
                except Exception:
                    continue

                for cand_col in cand_cols:
                    try:
                        target_values = set(
                            cand_df[cand_col].drop_nulls()
                            .str.strip_chars().unique().to_list()
                        )
                        containment = len(query_set & target_values) / query_size
                    except Exception:
                        continue

                    if containment >= min_containment:
                        join_records.append({
                            'from_id': query_table_path,
                            'to_id': file_path,
                            'from_column': col_name,
                            'to_column': cand_col,
                            'weight': containment
                        })

        # Build output DataFrame
        join_paths = pl.DataFrame(join_records, schema={
            'from_id': pl.String,
            'to_id': pl.String,
            'from_column': pl.String,
            'to_column': pl.String,
            'weight': pl.Float64
        })

        # Deduplicate and sort
        join_paths = join_paths.unique(maintain_order=True)
        join_paths = join_paths.sort('weight', descending=True).head(self.params['top_k'])
        join_paths = join_paths.sort(
            ['from_column', 'to_id', 'to_column', 'weight'],
            descending=[False, False, False, True]
        )
        join_paths = join_paths.with_columns(pl.col('to_id').str.split('/').list[-1].alias('to_id'))

        if output_path:
            join_paths.write_csv(output_path)

        return join_paths


class AurumJoinDiscoveryOG:
    def __init__(self, api_host: str, api_port: int, es_host: str = "aurum-datadiscovery-elasticsearch-1") -> None:
        self.api_host = api_host
        self.api_port = api_port
        self.base_url = f'http://{self.api_host}:{self.api_port}'
        self.es_host = es_host


    def find_joinable_tables(self, template_path: str, table_path: str, source_name: str, baseline: str, lake: str, output_path: str = None):
        with open(template_path, 'r') as f:
            template = f.read()
        payload = {
            "table_path": table_path,
            "template": template,
            "es_host": self.es_host,
            "lake": lake
        }
        request = requests.post(f'{self.base_url}/update_index', json=payload)
        response = request.json()
        new_id_info = list(response['new_id_info'].keys())
        new_table_ids = list(response['new_table_ids'].values())[0]

        payload = {
            "table_path": source_name,
            "lake": lake
        }
        request = requests.post(f'{self.base_url}/similar_tables', json=payload)
        response = request.json()
        join_paths = self._prepare_join_paths(response, source_name, baseline, output_path)

        return join_paths, new_id_info, new_table_ids, response


    def cleanup(self, new_id_info, new_table_ids, response, lake):
        payload = {
            "id_info": new_id_info,
            "table_ids": new_table_ids,
            "join_paths_dict": response,
            "lake": lake
        }
        request = requests.post(f'{self.base_url}/remove_from_index', json=payload)


    def _prepare_join_paths(self, response: dict, source_name: str, baseline: str, output_path: str):
        match baseline:
            case 'metam':
                join_paths = pl.DataFrame(schema={'tbl1': pl.String, 'col1': pl.String, 'tbl2': pl.String, 'col2': pl.String})
                for pair in response['similar_tables']['edges']:
                    if pair[0]['source_name'] != pair[1]['source_name']:
                        if pair[0]['source_name'] == source_name:
                            tbl1 = source_name#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[0]['field_name']
                            tbl2 = pair[1]['source_name']
                            col2 = pair[1]['field_name']
                        if pair[1]['source_name'] == source_name:
                            tbl1 = pair[1]['field_name']#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[1]['field_name']
                            tbl2 = pair[0]['source_name']
                            col2 = pair[0]['field_name']
                        join_paths = pl.concat([join_paths, pl.DataFrame({'tbl1': [tbl1], 'col1': [col1], 'tbl2': [tbl2], 'col2': [col2]})], how='vertical')
            case 'arda':
                join_paths = pl.DataFrame(schema={'from_id': pl.String, 'to_id': pl.String, 'from_column': pl.String, 'to_column': pl.String, 'weight': pl.Float64})
                for pair in response['similar_tables']['edges']:
                    if pair[0]['source_name'] != pair[1]['source_name']:
                        if pair[0]['source_name'] == source_name:
                            tbl1 = source_name#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[0]['field_name']
                            tbl2 = pair[1]['source_name']
                            col2 = pair[1]['field_name']
                        if pair[1]['source_name'] == source_name:
                            tbl1 = pair[1]['field_name']#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[1]['field_name']
                            tbl2 = pair[0]['source_name']
                            col2 = pair[0]['field_name']
                        weight = np.nan
                        join_paths = pl.concat([join_paths, pl.DataFrame({'from_id': [tbl1], 'to_id': [tbl2], 'from_column': [col1], 'to_column': [col2], 'weight': [weight]})], how='vertical')
        join_paths = join_paths.unique()
        join_paths.write_csv(output_path)

        return join_paths