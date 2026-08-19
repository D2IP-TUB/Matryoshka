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
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path for datasketch imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasketch import MinHash, MinHashLSHEnsemble

# Module-level temp dir, set by main process before spawning workers
_TEMP_DIR = None


def process_file_batch(args: Tuple[List[str], int, int, str, str, str]) -> str:
    """
    Process a batch of files and write results to a temporary pickle file.

    Writing to disk instead of returning through IPC pipes avoids the
    pipe-buffer deadlock that occurs with many workers and large results.

    Args:
        args: Tuple of (file_paths, num_perm, seed, file_format, separator, temp_dir)

    Returns:
        Path to the temporary pickle file containing results.
        Each result is (file_path, column_name, hashvalues_bytes, cardinality).
    """
    file_paths, num_perm, seed, file_format, separator, temp_dir = args
    results = []

    for file_path in file_paths:
        try:
            # Read file based on format
            if file_format == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_format == 'csv':
                # Use dtype=str to skip type inference entirely, avoiding
                # segfaults in pandas_parser.cpython on malformed files.
                # All values are strings anyway for MinHash hashing.
                df = pd.read_csv(file_path, sep=separator, on_bad_lines='skip',
                                 encoding='utf-8', engine='python', dtype=str)
            else:
                continue

            for column_name in df.columns:
                # Get unique non-null string values, strip whitespace vectorized
                series = df[column_name].dropna().astype(str).str.strip()
                values = series[series != ''].unique()

                if len(values) == 0:
                    continue

                # Create MinHash signature with batch update (vectorized)
                mh = MinHash(num_perm=num_perm, seed=seed)
                mh.update_batch(values)

                # Store hashvalues as bytes (compact)
                if not mh.is_empty():
                    results.append((file_path, column_name,
                                    mh.digest().tobytes(), len(values)))

        except Exception:
            pass  # Skip problematic files

    # Write results to a temp file — avoids IPC pipe limits
    fd, tmp_path = tempfile.mkstemp(dir=temp_dir, suffix='.pkl')
    with os.fdopen(fd, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    return tmp_path


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
    # Skip lake-metadata files that should never be returned as join candidates
    # (they describe the schema or the base table itself, not joinable data).
    # The base-table file is matched by name == basename(data_dir) + extension.
    base_table_basename = os.path.basename(os.path.normpath(data_dir)) + extension
    excluded_basenames = {'connections.csv', 'tables.json', base_table_basename}
    file_list = []

    for root, dirs, files in os.walk(data_dir):
        dirs.sort()  # Sort subdirectories for consistent traversal order
        for file in sorted(files):  # Sort files within directory
            if file in excluded_basenames:
                continue
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

    # Workers write results to temp files on disk instead of returning
    # through IPC pipes. This completely avoids the pipe-buffer deadlock
    # that occurs with many workers and large result sets.
    temp_dir = os.path.join(output_dir, '_tmp_signatures')
    os.makedirs(temp_dir, exist_ok=True)

    column_map = {}
    hashvalues_list = []
    cardinalities = []
    key_id = 0

    # Split files into batches for workers (each worker gets ~100 files)
    files_per_worker = max(1, min(100, len(file_list) // (workers * 2)))
    batches = []
    for i in range(0, len(file_list), files_per_worker):
        batch_files = file_list[i:i + files_per_worker]
        batches.append((batch_files, num_perm, seed, file_format, separator, temp_dir))

    # Check for already-completed batches from a previous interrupted run
    existing_tmp = set(os.listdir(temp_dir))
    if existing_tmp:
        print(f"  Found {len(existing_tmp)} completed batches from previous run", flush=True)

    # Each batch writes a file named by its index: batch_XXXXX.pkl
    # Re-create batches with deterministic temp file names so we can skip done ones
    pending_batches = []
    completed_tmp_paths = []
    for idx, batch in enumerate(batches):
        tmp_name = f"batch_{idx:05d}.pkl"
        tmp_path = os.path.join(temp_dir, tmp_name)
        if os.path.exists(tmp_path):
            completed_tmp_paths.append(tmp_path)
        else:
            pending_batches.append((idx, batch, tmp_path))

    print(f"  {len(completed_tmp_paths)} batches already done, {len(pending_batches)} pending", flush=True)

    # Submit pending batches and collect temp file paths.
    # If a worker segfaults, catch the error and retry failed batches sequentially.
    tmp_paths = list(completed_tmp_paths)
    failed_batches = []

    if pending_batches:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_file_batch, batch): (idx, tmp_expected)
                       for idx, batch, tmp_expected in pending_batches}
            try:
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc="Processing files"):
                    try:
                        result_path = future.result()
                        idx, tmp_expected = futures[future]
                        # Rename to deterministic name for resume support
                        if result_path != tmp_expected:
                            os.rename(result_path, tmp_expected)
                            result_path = tmp_expected
                        tmp_paths.append(result_path)
                    except Exception as e:
                        idx, tmp_expected = futures[future]
                        print(f"\n  Batch {idx} failed: {e}. Will retry.", flush=True)
                        failed_batches.append((idx, batches[idx], tmp_expected))
            except Exception as e:
                # BrokenProcessPool — collect indices of incomplete batches
                print(f"\n  Pool broken: {e}. Retrying failed batches...", flush=True)
                for fut, (idx, tmp_expected) in futures.items():
                    if not fut.done() or (fut.done() and fut.exception() is not None):
                        failed_batches.append((idx, batches[idx], tmp_expected))

    # Retry failed batches sequentially (no pool, no crash propagation)
    if failed_batches:
        print(f"  Retrying {len(failed_batches)} failed batches sequentially...", flush=True)
        for idx, batch, tmp_expected in tqdm(failed_batches, desc="Retrying"):
            try:
                result_path = process_file_batch(batch)
                if result_path != tmp_expected:
                    os.rename(result_path, tmp_expected)
                    result_path = tmp_expected
                tmp_paths.append(result_path)
            except Exception as e:
                print(f"  Retry failed for batch {idx}: {e}", flush=True)

    # Read all temp files and build column map
    print("  Collecting results from disk...", flush=True)
    for tmp_path in tqdm(tmp_paths, desc="Loading results"):
        with open(tmp_path, 'rb') as f:
            file_results = pickle.load(f)
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
        os.unlink(tmp_path)  # Clean up temp file immediately

    # Remove temp directory
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

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
