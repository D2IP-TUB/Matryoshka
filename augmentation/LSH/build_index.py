"""
Build MinHash LSH Ensemble index for containment-based join search.

This script processes a data lake directory and creates LSH Ensemble index
files for efficient containment queries.

Usage:
    python build_index.py --data_dir /path/to/parquet/files --output_dir ./index
"""

import argparse
import gc
import os
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for datasketch imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasketch import MinHash, MinHashLSHEnsemble


def process_file(args: Tuple[str, int, int, str, str]) -> List[Tuple[str, str, bytes, int]]:
    """
    Process a single file and extract MinHash signatures for each column.

    Returns compact results: hashvalues as raw bytes instead of full MinHash
    objects. This reduces IPC overhead by ~66% since the permutation arrays
    (identical for all MinHashes) are not sent through pipes.

    Args:
        args: Tuple of (file_path, num_perm, seed, file_format, separator)

    Returns:
        List of (file_path, column_name, hashvalues_bytes, cardinality) tuples
    """
    file_path, num_perm, seed, file_format, separator = args
    results = []

    try:
        # Read file based on format
        if file_format == 'parquet':
            df = pd.read_parquet(file_path)
        elif file_format == 'csv':
            df = pd.read_csv(file_path, sep=separator, on_bad_lines='skip', encoding='utf-8', low_memory=False)
        else:
            return results

        for column_name in df.columns:
            # Get unique non-null string values, strip whitespace vectorized
            series = df[column_name].dropna().astype(str).str.strip()
            values = series[series != ''].unique()

            if len(values) == 0:
                continue

            # Create MinHash signature with batch update (vectorized)
            mh = MinHash(num_perm=num_perm, seed=seed)
            mh.update_batch(values)

            # Store hashvalues as bytes (compact) instead of full MinHash object
            if not mh.is_empty():
                results.append((file_path, column_name, mh.digest().tobytes(), len(values)))

    except Exception:
        pass  # Skip problematic files

    return results


def collect_files(data_dir: str, file_format: str) -> List[str]:
    """
    Collect all files of the specified format from the data directory.

    Ensures deterministic ordering by sorting all paths.

    Args:
        data_dir: Root directory to search
        file_format: File extension to look for ('parquet' or 'csv')

    Returns:
        Sorted list of file paths
    """
    extension = f'.{file_format}'
    file_list = []

    for root, dirs, files in os.walk(data_dir):
        dirs.sort()  # Sort subdirectories for consistent traversal order
        for file in sorted(files):  # Sort files within directory
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))

    file_list.sort()  # Final sort for full determinism
    return file_list


def build_index(data_dir: str, output_dir: str, num_perm: int = 256,
                threshold: float = 0.5, num_part: int = 32,
                workers: int = 4, file_format: str = 'parquet',
                seed: int = 42, separator: str = ',') -> None:
    """
    Build the LSH Ensemble index for the data lake.

    Args:
        data_dir: Directory containing data files
        output_dir: Directory to save index files
        num_perm: Number of MinHash permutation functions
        threshold: Containment threshold for optimization
        num_part: Number of LSH Ensemble partitions
        workers: Number of parallel workers
        file_format: File format ('parquet' or 'csv')
        seed: Random seed for reproducibility
        separator: CSV delimiter character
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60, flush=True)
    print("LSH Ensemble Index Builder", flush=True)
    print("=" * 60, flush=True)
    print(f"Data directory: {data_dir}", flush=True)
    print(f"Output directory: {output_dir}", flush=True)
    print(f"Parameters:", flush=True)
    print(f"  - Permutations: {num_perm}", flush=True)
    print(f"  - Threshold: {threshold}", flush=True)
    print(f"  - Partitions: {num_part}", flush=True)
    print(f"  - Workers: {workers}", flush=True)
    print(f"  - File format: {file_format}", flush=True)
    print(f"  - Random seed: {seed}", flush=True)
    if file_format == 'csv':
        print(f"  - CSV separator: {repr(separator)}", flush=True)
    print("=" * 60, flush=True)

    # Step 1: Collect files
    print("\nStep 1: Collecting files...", flush=True)
    file_list = collect_files(data_dir, file_format)

    if len(file_list) == 0:
        raise ValueError(f"No {file_format} files found in {data_dir}")

    print(f"Found {len(file_list)} {file_format} files", flush=True)

    # Step 2: Generate MinHash signatures
    print("\nStep 2: Generating MinHash signatures...", flush=True)
    t1 = time.time()

    # Prepare arguments for parallel processing
    args_list = [(f, num_perm, seed, file_format, separator) for f in file_list]

    # Collect results incrementally — workers return compact hashvalues bytes
    # instead of full MinHash objects to minimize IPC pipe traffic.
    column_map = {}
    hashvalues_list = []    # Store raw hashvalue arrays
    cardinalities = []
    key_id = 0

    # Use smaller chunksize to avoid a single slow file blocking progress
    chunksize = max(1, min(256, len(args_list) // (workers * 4)))
    with Pool(processes=workers) as pool:
        for file_results in tqdm(
            pool.imap_unordered(process_file, args_list, chunksize=chunksize),
            total=len(args_list),
            desc="Processing files"
        ):
            for file_path, column_name, hv_bytes, cardinality in file_results:
                key = str(key_id)
                hashvalues_list.append(np.frombuffer(hv_bytes, dtype=np.uint64).copy())
                cardinalities.append(cardinality)
                column_map[key] = {
                    'file_path': file_path,
                    'column_name': column_name,
                    'cardinality': cardinality
                }
                key_id += 1

    t2 = time.time()
    num_sigs = len(hashvalues_list)
    print(f"Generated {num_sigs} MinHash signatures in {t2 - t1:.2f}s", flush=True)

    if num_sigs == 0:
        raise ValueError("No valid columns found in data files")

    # Step 3: Build LSH Ensemble index
    # Reconstruct MinHash objects on the fly from stored hashvalues.
    # This is memory-efficient: only one MinHash exists at a time in the generator,
    # though LSH Ensemble materializes them internally.
    print(f"\nStep 3: Building LSH Ensemble index ({num_sigs} signatures)...", flush=True)
    t1 = time.time()

    # Create a template MinHash to get shared permutations (same seed → same perms)
    template_mh = MinHash(num_perm=num_perm, seed=seed)
    permutations = template_mh.permutations

    def _make_signatures():
        """Generator that reconstructs MinHash objects from stored hashvalues."""
        for i in range(num_sigs):
            mh = MinHash(
                num_perm=num_perm,
                seed=seed,
                hashvalues=hashvalues_list[i],
                permutations=permutations,
            )
            yield (str(i), mh, cardinalities[i])

    lsh_ensemble = MinHashLSHEnsemble(
        threshold=threshold,
        num_perm=num_perm,
        num_part=num_part
    )

    # Index all signatures
    lsh_ensemble.index(_make_signatures())

    t2 = time.time()
    print(f"Built index in {t2 - t1:.2f}s", flush=True)
    print(f"Partitions: {len(lsh_ensemble.partitions)}", flush=True)
    for i, (lower, upper) in enumerate(lsh_ensemble.partitions):
        print(f"  Partition {i}: size range [{lower}, {upper}]", flush=True)

    # Free intermediate data no longer needed
    del hashvalues_list, cardinalities, permutations, template_mh
    gc.collect()

    # Step 4: Save index files
    print("\nStep 4: Saving index files...", flush=True)

    # Save LSH Ensemble index
    index_path = os.path.join(output_dir, "lsh_ensemble.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump(lsh_ensemble, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save column map
    map_path = os.path.join(output_dir, "column_map.pkl")
    with open(map_path, 'wb') as f:
        pickle.dump(column_map, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    metadata = {
        'num_perm': num_perm,
        'threshold': threshold,
        'num_part': num_part,
        'num_columns': len(column_map),
        'seed': seed,
        'file_format': file_format,
        'separator': separator,
        'partitions': lsh_ensemble.partitions
    }
    metadata_path = os.path.join(output_dir, "metadata.pkl")
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nIndex files saved to: {output_dir}", flush=True)
    print(f"  - lsh_ensemble.pkl: LSH Ensemble index", flush=True)
    print(f"  - column_map.pkl: Column metadata", flush=True)
    print(f"  - metadata.pkl: Index configuration", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Index Building Complete", flush=True)
    print("=" * 60, flush=True)
    print(f"Total columns indexed: {len(column_map)}", flush=True)
    print(f"Index size: {os.path.getsize(index_path) / (1024 * 1024):.2f} MB", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build LSH Ensemble index for containment-based join search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        required=True,
        help="Directory containing data files to index"
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help="Directory to save index files"
    )
    
    parser.add_argument(
        '--num_perm', '-p',
        type=int,
        default=256,
        help="Number of MinHash permutation functions"
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=0.5,
        help="Containment threshold for index optimization"
    )
    
    parser.add_argument(
        '--num_part', '-n',
        type=int,
        default=32,
        help="Number of LSH Ensemble partitions"
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel workers"
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='parquet',
        choices=['parquet', 'csv'],
        help="Input file format"
    )
    
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        '--separator',
        type=str,
        default=',',
        help="CSV delimiter character"
    )
    
    args = parser.parse_args()
    
    build_index(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_perm=args.num_perm,
        threshold=args.threshold,
        num_part=args.num_part,
        workers=args.workers,
        file_format=args.format,
        seed=args.seed,
        separator=args.separator
    )


if __name__ == "__main__":
    main()
