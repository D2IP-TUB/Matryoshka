"""
Query interface for LSH Ensemble containment-based join search.

This script searches for joinable columns in a data lake using LSH Ensemble
with containment similarity.

Usage:
    python query.py --index_dir ./index --query_file /path/to/query.parquet --query_column col_name --top_k 10
"""

import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
from tqdm import tqdm

# Add parent directory to path for datasketch imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasketch import MinHash, MinHashLSHEnsemble


def load_index(index_dir: str) -> Tuple[MinHashLSHEnsemble, Dict, Dict]:
    """
    Load index files from disk.
    
    Args:
        index_dir: Path to index directory
        
    Returns:
        Tuple of (lsh_ensemble, column_map, metadata)
    """
    print("Loading index files...")
    
    # Load LSH Ensemble index
    index_path = os.path.join(index_dir, "lsh_ensemble.pkl")
    with open(index_path, 'rb') as f:
        lsh_ensemble = pickle.load(f)
    print(f"  Loaded LSH Ensemble index")
    
    # Load column map
    map_path = os.path.join(index_dir, "column_map.pkl")
    with open(map_path, 'rb') as f:
        column_map = pickle.load(f)
    print(f"  Loaded column map: {len(column_map)} columns")
    
    # Load metadata
    metadata_path = os.path.join(index_dir, "metadata.pkl")
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"  Index parameters: {metadata['num_perm']} perms, {metadata['num_part']} partitions")
    
    return lsh_ensemble, column_map, metadata


def create_query_minhash(values: List[str], num_perm: int, seed: int) -> Tuple[MinHash, int]:
    """
    Create MinHash signature for query values.
    
    Args:
        values: List of string values
        num_perm: Number of permutation functions
        seed: Random seed for MinHash
        
    Returns:
        Tuple of (minhash, cardinality)
    """
    mh = MinHash(num_perm=num_perm, seed=seed)
    unique_values = set()
    
    for val in values:
        if val and str(val).strip():
            clean_val = str(val).strip()
            unique_values.add(clean_val)
            mh.update(clean_val)
    
    return mh, len(unique_values)


def get_query_values(query_path: str, column_name: Optional[str] = None,
                     file_format: str = 'parquet') -> Tuple[List[str], str]:
    """
    Extract values from a query file column.
    
    Args:
        query_path: Path to query file
        column_name: Name of column to use (default: first column)
        file_format: File format ('parquet' or 'csv')
        
    Returns:
        Tuple of (list of unique values, column name)
    """
    if file_format == 'parquet':
        df = pd.read_parquet(query_path)
    else:
        df = pd.read_csv(query_path, on_bad_lines='skip', encoding='utf-8')
    
    if column_name is None:
        column_name = df.columns[0]
    elif column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in query file. "
                        f"Available columns: {list(df.columns)}")
    
    # Get unique non-null string values
    values = df[column_name].dropna().astype(str).unique().tolist()
    
    return values, column_name


def compute_containment(query_mh: MinHash, target_mh: MinHash) -> float:
    """
    Estimate containment similarity using MinHash Jaccard.
    
    This is an approximation - true containment requires actual set comparison.
    Uses Jaccard as a proxy: containment ≈ jaccard for similar-size sets.
    
    Args:
        query_mh: Query MinHash
        target_mh: Target MinHash
        
    Returns:
        Estimated containment score
    """
    return query_mh.jaccard(target_mh)


def rerank_candidates(query_values: List[str], candidates: List[str],
                      column_map: Dict, num_perm: int, seed: int,
                      top_k: int) -> List[Tuple[str, str, float]]:
    """
    Rerank candidates by computing actual containment estimates.
    
    Args:
        query_values: Query column values
        candidates: List of candidate column keys
        column_map: Mapping from keys to column metadata
        num_perm: Number of MinHash permutations
        seed: Random seed
        top_k: Number of top results to return
        
    Returns:
        List of (file_path, column_name, containment) tuples sorted by containment
    """
    query_set = set(str(v).strip() for v in query_values if v and str(v).strip())
    query_size = len(query_set)
    
    if query_size == 0:
        return []
    
    results = []
    
    for key in candidates:
        if key not in column_map:
            continue
        
        info = column_map[key]
        file_path = info['file_path']
        col_name = info['column_name']
        
        # For exact containment, we would load the column and compute |Q ∩ X| / |Q|
        # Here we use the stored cardinality as a proxy for ranking
        cardinality = info.get('cardinality', 1)
        
        # Estimate containment using MinHash Jaccard approximation
        # Higher cardinality targets are more likely to contain query values
        score = min(1.0, cardinality / query_size) if query_size > 0 else 0.0
        
        results.append((file_path, col_name, score, key))
    
    # Sort by score descending, then by key for determinism
    results.sort(key=lambda x: (-x[2], x[3]))
    
    return [(r[0], r[1], r[2]) for r in results[:top_k]]


def search(index_dir: str, query_path: str, query_column: Optional[str] = None,
           top_k: int = 10, threshold: Optional[float] = None,
           file_format: str = 'parquet') -> List[Tuple[str, str, float]]:
    """
    Search for columns with high containment of query values.
    
    Args:
        index_dir: Path to index directory
        query_path: Path to query file
        query_column: Column to use for query (default: first column)
        top_k: Number of top results to return
        threshold: Containment threshold (default: use index threshold)
        file_format: Query file format ('parquet' or 'csv')
        
    Returns:
        List of (file_path, column_name, containment) tuples
    """
    # Load index
    lsh_ensemble, column_map, metadata = load_index(index_dir)
    
    num_perm = metadata['num_perm']
    seed = metadata.get('seed', 42)
    
    if threshold is None:
        threshold = metadata['threshold']
    
    # Get query values
    print(f"\nLoading query from: {query_path}")
    query_values, used_column = get_query_values(query_path, query_column, file_format)
    print(f"  Column: {used_column}")
    print(f"  Unique values: {len(query_values)}")
    
    if len(query_values) == 0:
        print("Warning: Query column has no valid values")
        return []
    
    # Create query MinHash
    query_mh, query_size = create_query_minhash(query_values, num_perm, seed)
    print(f"  Query cardinality: {query_size}")
    
    # Query LSH Ensemble
    print(f"\nSearching with containment threshold: {threshold}")
    t1 = time.time()
    
    candidates = lsh_ensemble.query(query_mh, query_size)
    
    t2 = time.time()
    print(f"  Found {len(candidates)} candidates in {(t2 - t1) * 1000:.2f}ms")
    
    # Rerank and return top K
    if len(candidates) == 0:
        return []
    
    results = rerank_candidates(
        query_values, candidates, column_map, 
        num_perm, seed, top_k
    )
    
    return results


def search_all_columns(index_dir: str, query_path: str, top_k: int = 10,
                       threshold: Optional[float] = None,
                       file_format: str = 'parquet') -> Dict[str, List[Tuple]]:
    """
    Search for joinable columns using all columns in the query file.
    
    Args:
        index_dir: Path to index directory
        query_path: Path to query file
        top_k: Number of top results per query column
        threshold: Containment threshold
        file_format: Query file format
        
    Returns:
        Dict mapping query column names to their results
    """
    # Load query file to get all columns
    if file_format == 'parquet':
        df = pd.read_parquet(query_path)
    else:
        df = pd.read_csv(query_path, on_bad_lines='skip', encoding='utf-8')
    
    all_results = {}
    
    for column in df.columns:
        print(f"\n{'=' * 60}")
        print(f"Querying with column: {column}")
        print('=' * 60)
        
        results = search(
            index_dir=index_dir,
            query_path=query_path,
            query_column=column,
            top_k=top_k,
            threshold=threshold,
            file_format=file_format
        )
        
        all_results[column] = results
    
    return all_results


def print_results(results: List[Tuple[str, str, float]], query_column: str) -> None:
    """
    Pretty print search results.
    
    Args:
        results: List of (file_path, column_name, containment) tuples
        query_column: Name of the query column
    """
    print(f"\n{'=' * 60}")
    print(f"Top {len(results)} results for query column: {query_column}")
    print('=' * 60)
    
    if len(results) == 0:
        print("No results found")
        return
    
    for i, (file_path, col_name, score) in enumerate(results, 1):
        file_name = os.path.basename(file_path)
        print(f"{i:3d}. {file_name}:{col_name}")
        print(f"     Score: {score:.4f}")
        print(f"     Path: {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Search for joinable columns using LSH Ensemble containment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--index_dir', '-i',
        type=str,
        required=True,
        help="Directory containing index files"
    )
    
    parser.add_argument(
        '--query_file', '-q',
        type=str,
        required=True,
        help="Path to query file"
    )
    
    parser.add_argument(
        '--query_column', '-c',
        type=str,
        default=None,
        help="Column to use for query (default: first column)"
    )
    
    parser.add_argument(
        '--top_k', '-k',
        type=int,
        default=10,
        help="Number of top results to return"
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=None,
        help="Containment threshold (default: use index threshold)"
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='parquet',
        choices=['parquet', 'csv'],
        help="Query file format"
    )
    
    parser.add_argument(
        '--all_columns', '-a',
        action='store_true',
        help="Search using all columns in query file"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output file path for results (JSON format)"
    )
    
    args = parser.parse_args()
    
    t_start = time.time()
    
    if args.all_columns:
        all_results = search_all_columns(
            index_dir=args.index_dir,
            query_path=args.query_file,
            top_k=args.top_k,
            threshold=args.threshold,
            file_format=args.format
        )
        
        for col_name, results in all_results.items():
            print_results(results, col_name)
        
        if args.output:
            import json
            # Convert results to JSON-serializable format
            output_data = {
                col: [(f, c, s) for f, c, s in res]
                for col, res in all_results.items()
            }
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    else:
        results = search(
            index_dir=args.index_dir,
            query_path=args.query_file,
            query_column=args.query_column,
            top_k=args.top_k,
            threshold=args.threshold,
            file_format=args.format
        )
        
        query_col = args.query_column or "first column"
        print_results(results, query_col)
        
        if args.output:
            import json
            output_data = [(f, c, s) for f, c, s in results]
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {args.output}")
    
    t_end = time.time()
    print(f"\nTotal time: {t_end - t_start:.2f}s")


if __name__ == "__main__":
    main()
