"""
Query interface for Joise joinability search.

This script searches for joinable columns in a data lake using the Joise algorithm.

Usage:
    python query.py --index_dir ./index --query_table /path/to/query.csv --K 10
"""

import argparse
import csv
import json
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from josie import search_joise, read_query_id

# Set CSV field size limit
csv.field_size_limit(sys.maxsize)


# Process-local cache of fully-loaded indexes, keyed by ``index_dir``. A
# single ``load_index`` call populates this; subsequent calls in the same
# process (or in a forked child) return the cached tuple in constant time.
_INDEX_CACHE: Dict[str, Tuple[Dict, Dict, Dict, Dict]] = {}


def _load_json_fast(path: str):
    """Parse a JSON document via ``orjson`` when available, with a fallback
    to the stdlib parser. ``orjson.loads`` is roughly 5x faster than
    ``json.load`` on the large posting-list / raw-dict / integer-set files
    used here, with identical Python object outputs."""
    try:
        import orjson
        with open(path, "rb") as f:
            return orjson.loads(f.read())
    except ImportError:
        with open(path, "r") as f:
            return json.load(f)


def load_index(index_dir: str) -> Tuple[Dict, Dict, Dict, Dict]:
    """Load JOSIE index artifacts from disk with two-level caching.

    Layer 1: ``_INDEX_CACHE`` — a module-level dictionary. Repeated calls
    against the same ``index_dir`` (e.g., from forked child processes that
    inherit the parent's address space) return the cached tuple without
    touching disk.

    Layer 2: ``<outputs>/load_cache.pkl`` — a single pickle file
    consolidating the four artifacts. Built lazily on the first
    JSON-parsing load; subsequent process starts (without inherited
    memory) reuse it. Pickle protocol 5 is roughly 5-10x faster than the
    JSON parsing path for the same Python objects because pickle preserves
    types directly instead of re-parsing text. Stale caches (older than
    any source file) are ignored and rebuilt.
    """
    if index_dir in _INDEX_CACHE:
        return _INDEX_CACHE[index_dir]

    outpath = os.path.join(index_dir, "outputs")
    map_path = os.path.join(index_dir, "setMap.pkl")
    integer_set_path = os.path.join(outpath, "integerSet.json")
    posting_lists_path = os.path.join(outpath, "PLs.json")
    raw_dict_path = os.path.join(outpath, "rawDict.json")
    cache_path = os.path.join(outpath, "load_cache.pkl")

    source_paths = [map_path, integer_set_path, posting_lists_path, raw_dict_path]
    if os.path.exists(cache_path):
        try:
            cache_mtime = os.path.getmtime(cache_path)
            cache_fresh = all(
                os.path.exists(p) and os.path.getmtime(p) <= cache_mtime
                for p in source_paths
            )
        except OSError:
            cache_fresh = False
        if cache_fresh:
            print(f"Loading index from pickle cache: {cache_path}")
            with open(cache_path, "rb") as f:
                tup = pickle.load(f)
            _INDEX_CACHE[index_dir] = tup
            return tup

    print("Loading index files (JSON parse, slow path)...")
    with open(map_path, "rb") as f:
        set_map = pickle.load(f)
    print(f"  Loaded set map: {len(set_map)} columns")

    integer_set = _load_json_fast(integer_set_path)
    print(f"  Loaded integer sets: {len(integer_set)} sets")

    posting_lists = _load_json_fast(posting_lists_path)
    print(f"  Loaded posting lists: {len(posting_lists)} tokens")

    raw_dict = _load_json_fast(raw_dict_path)
    print(f"  Loaded raw dictionary: {len(raw_dict)} entries")

    tup = (integer_set, posting_lists, raw_dict, set_map)
    _INDEX_CACHE[index_dir] = tup

    try:
        tmp_path = cache_path + ".tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(tup, f, protocol=5)
        os.replace(tmp_path, cache_path)
        print(f"  Wrote pickle cache for future runs: {cache_path}")
    except OSError as e:
        print(f"  Could not write pickle cache (continuing): {e!r}")

    return tup


def get_query_tokens(query_path: str, separator: str, column_name: str = None) -> Tuple[List[str], str]:
    """
    Extract tokens from a query table column.
    
    Args:
        query_path: Path to query CSV file
        separator: CSV delimiter
        column_name: Name of column to use (default: first column)
        
    Returns:
        Tuple of (list of unique tokens, column name)
    """
    df = pd.read_csv(query_path, sep=separator, engine='python', 
                     on_bad_lines='skip', encoding='utf-8')
    
    if column_name is None:
        column_name = df.columns[0]
    elif column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in query table. "
                        f"Available columns: {list(df.columns)}")
    
    # Get unique non-null string tokens
    raw_tokens = list(set(df[column_name].dropna().astype(str).tolist()))
    
    # Filter out empty strings and pure numbers
    tokens = [t for t in raw_tokens if t.strip() and not t.isdigit()]
    
    return tokens, column_name


def search_all_columns(query_path: str, separator: str, 
                       integer_set: Dict, posting_lists: Dict, 
                       raw_dict: Dict, set_map: Dict, k: int,
                       ignore_self: bool = True) -> List[Tuple]:
    """
    Search for joinable columns for all columns in the query table.
    
    Args:
        query_path: Path to query CSV file
        separator: CSV delimiter
        integer_set: Integer set dictionary
        posting_lists: Posting lists dictionary
        raw_dict: Raw token dictionary
        set_map: Set ID to table/column mapping
        k: Number of top results per column
        ignore_self: Whether to ignore self-matches
        
    Returns:
        List of (query_table, query_column, candidate_table, candidate_column, overlap) tuples
    """
    table_name = os.path.basename(query_path)
    df = pd.read_csv(query_path, sep=separator, engine='python',
                     on_bad_lines='skip', encoding='utf-8')
    
    all_results = []
    
    for column_name in tqdm(df.columns, desc="Searching columns"):
        # Get query tokens
        raw_tokens = list(set(df[column_name].dropna().astype(str).tolist()))
        tokens = [t for t in raw_tokens if t.strip() and not t.isdigit()]
        
        if len(tokens) == 0:
            continue
        
        # Check if query is in the index
        query_id = read_query_id(set_map, table_name, column_name)
        
        # Search
        results = search_joise(
            integer_set, posting_lists, tokens, raw_dict, set_map,
            k, ignore_self=(ignore_self and query_id >= 0), query_id=query_id
        )
        
        # Format results
        for set_id in results:
            if set_id + 1 in set_map:
                info = set_map[set_id + 1]
                all_results.append((
                    query_path,
                    column_name,
                    info.get('table_name', f'set_{set_id}'),
                    info.get('column_name', f'col_{set_id}'),
                    None  # Overlap score not easily available
                ))
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Search for joinable tables using Joise algorithm')
    parser.add_argument("--index_dir", type=str, required=True,
                        help="Path to index directory created by build_index.py")
    parser.add_argument("--query_table", type=str, required=True,
                        help="Path to query table CSV file")
    parser.add_argument("--output_file", type=str, default="joise_results.csv",
                        help="Output CSV file path (default: joise_results.csv)")
    parser.add_argument("--separator", type=str, default=",",
                        help="CSV separator for data lake tables (default: ',')")
    parser.add_argument("--query_separator", type=str, default=None,
                        help="CSV separator for query table (default: same as --separator)")
    parser.add_argument("--query_column", type=str, default=None,
                        help="Specific column in query table to use (default: search all columns)")
    parser.add_argument("--K", type=int, default=10,
                        help="Number of top results to return per column (default: 10)")
    parser.add_argument("--ignore_self", action="store_true", default=True,
                        help="Ignore self-matches if query table is in the index (default: True)")
    parser.add_argument("--no_ignore_self", action="store_false", dest="ignore_self",
                        help="Do not ignore self-matches")
    
    args = parser.parse_args()
    
    # Decode escape sequences in separators
    args.separator = args.separator.encode().decode('unicode_escape')
    if args.query_separator is None:
        args.query_separator = args.separator
    else:
        args.query_separator = args.query_separator.encode().decode('unicode_escape')
    
    # Validate paths
    if not os.path.exists(args.index_dir):
        raise ValueError(f"Index directory does not exist: {args.index_dir}")
    if not os.path.exists(args.query_table):
        raise ValueError(f"Query table does not exist: {args.query_table}")
    
    print(f"Index directory: {args.index_dir}")
    print(f"Query table: {args.query_table}")
    print(f"Output file: {args.output_file}")
    print(f"K: {args.K}")
    print()
    
    # Load index
    integer_set, posting_lists, raw_dict, set_map = load_index(args.index_dir)
    
    start_time = time.time()
    
    if args.query_column is not None:
        # Search single column
        print(f"\nSearching for joinable columns for: {args.query_column}")
        
        tokens, col_name = get_query_tokens(args.query_table, args.query_separator, 
                                            args.query_column)
        print(f"Query tokens: {len(tokens)}")
        
        table_name = os.path.basename(args.query_table)
        query_id = read_query_id(set_map, table_name, col_name)
        
        results = search_joise(
            integer_set, posting_lists, tokens, raw_dict, set_map,
            args.K, ignore_self=(args.ignore_self and query_id >= 0), query_id=query_id
        )
        
        # Format results
        all_results = []
        for set_id in results:
            if set_id + 1 in set_map:
                info = set_map[set_id + 1]
                all_results.append((
                    args.query_table,
                    col_name,
                    info.get('table_name', f'set_{set_id}'),
                    info.get('column_name', f'col_{set_id}')
                ))
    else:
        # Search all columns
        print(f"\nSearching for joinable columns for all columns in query table...")
        all_results = search_all_columns(
            args.query_table, args.query_separator,
            integer_set, posting_lists, raw_dict, set_map,
            args.K, args.ignore_self
        )
    
    search_time = time.time() - start_time
    print(f"\nSearch completed in {search_time:.2f} seconds")
    print(f"Found {len(all_results)} matches")
    
    # Save results
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\nWriting results to: {args.output_file}")
    
    with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['from_id', 'to_id', 'from_column', 'to_column'])
        
        for result in all_results:
            if len(result) >= 4:
                writer.writerow([result[0], result[2], result[1], result[3]])
    
    print(f"Wrote {len(all_results)} results")


if __name__ == '__main__':
    main()
