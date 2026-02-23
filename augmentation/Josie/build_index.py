"""
Build inverted index for Joise joinability search.

This script processes a data lake directory and creates the inverted index
files needed for the Joise algorithm.

Usage:
    python build_index.py --input_dir /path/to/datalake --output_dir ./index
"""

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import time
from multiprocessing import Process, Manager
from typing import Dict, List, Tuple, Any

import pandas as pd
import psutil
from tqdm import tqdm

# Set CSV field size limit
csv.field_size_limit(sys.maxsize)


def create_raw_tokens_worker(file_list: List[str], queue, result_queue, 
                             worker_id: int, separator: str):
    """
    Worker function to extract raw tokens from CSV files.
    
    Args:
        file_list: List of CSV file paths to process
        queue: Queue for progress updates
        result_queue: Queue for results
        worker_id: Worker identifier
        separator: CSV delimiter
    """
    extracted_data = []
    set_map = {}
    set_id = worker_id * 1000000  # Offset to avoid ID collisions
    
    for table_path in file_list:
        try:
            table_name = os.path.basename(table_path)
            df = pd.read_csv(table_path, sep=separator, engine='python', 
                           on_bad_lines='skip', encoding='utf-8')
            
            for column_name in df.columns:
                # Get unique tokens from column
                raw_tokens = list(set(df[column_name].dropna().astype(str).tolist()))
                pos = 0
                
                # Filter tokens: only non-empty strings that are not purely numeric
                valid_tokens = []
                for token in raw_tokens:
                    if isinstance(token, str) and token.strip() and not token.isdigit():
                        pos += 1
                        valid_tokens.append((token, set_id, pos))
                
                if pos > 0:
                    extracted_data.extend(valid_tokens)
                    set_id += 1
                    set_map[set_id] = {'table_name': table_name, 'column_name': column_name}
                    
        except Exception as e:
            pass  # Skip problematic files
        
        queue.put(1)
    
    result_queue.put((extracted_data, set_map))
    queue.put((-1, worker_id))


def create_raw_tokens(input_dir: str, output_dir: str, separator: str,
                      num_processes: int) -> bool:
    """
    Extract raw tokens from all CSV files in the data lake.
    
    Creates:
    - rawTokens.csv: Token, SetID, Position
    - setMap.pkl: Mapping from SetID to table/column names
    
    Args:
        input_dir: Path to data lake directory
        output_dir: Path to output directory
        separator: CSV delimiter
        num_processes: Number of parallel workers
        
    Returns:
        True on success
    """
    print("Step 1: Extracting raw tokens from data lake...")
    
    raw_tokens_path = os.path.join(output_dir, "rawTokens.csv")
    map_path = os.path.join(output_dir, "setMap.pkl")
    
    t1 = time.time()
    
    # Collect all CSV files (sorted for deterministic ordering)
    file_list = []
    for root, dirs, files in os.walk(input_dir):
        dirs.sort()  # Sort subdirectories for consistent traversal order
        for file in sorted(files):  # Sort files within directory
            if file.endswith('.csv'):
                file_list.append(os.path.join(root, file))
    file_list.sort()  # Sort all paths for reproducibility
    
    if len(file_list) == 0:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    print(f"Found {len(file_list)} CSV files")
    
    # Split files across workers
    def split_list(lst, n):
        avg = len(lst) // n
        remainder = len(lst) % n
        result = []
        start = 0
        for i in range(n):
            end = start + avg + (1 if i < remainder else 0)
            result.append(lst[start:end])
            start = end
        return result
    
    sub_lists = split_list(file_list, num_processes)
    
    # Create processes
    manager = Manager()
    queues = [manager.Queue() for _ in range(num_processes)]
    result_queues = [manager.Queue() for _ in range(num_processes)]
    processes = []
    finished = [False] * num_processes
    
    bars = [tqdm(total=len(sub_lists[i]), desc=f"Worker-{i}", position=i) 
            for i in range(num_processes)]
    
    for i in range(num_processes):
        p = Process(target=create_raw_tokens_worker, 
                   args=(sub_lists[i], queues[i], result_queues[i], i, separator))
        processes.append(p)
        p.start()
    
    # Monitor progress
    while not all(finished):
        for i in range(num_processes):
            try:
                res = queues[i].get_nowait()
                if isinstance(res, tuple) and res[0] == -1:
                    finished[i] = True
                else:
                    bars[i].update(res)
            except:
                continue
    
    for p in processes:
        p.join()
    
    for bar in bars:
        bar.close()
    
    # Collect results
    all_data = []
    all_maps = {}
    
    while not all(q.empty() for q in result_queues):
        for q in result_queues:
            try:
                data, smap = q.get_nowait()
                all_data.extend(data)
                all_maps.update(smap)
            except:
                continue
    
    # Renumber set IDs sequentially
    old_to_new = {}
    new_set_map = {}
    new_set_id = 0
    
    for old_id in sorted(all_maps.keys()):
        new_set_id += 1
        old_to_new[old_id] = new_set_id
        new_set_map[new_set_id] = all_maps[old_id]
    
    # Update data with new IDs
    updated_data = [(token, old_to_new.get(sid, sid), pos) 
                    for token, sid, pos in all_data]
    
    # Save results
    print("\nSaving raw tokens...")
    df = pd.DataFrame(updated_data, columns=["RawToken", "SetID", "Position"])
    df.to_csv(raw_tokens_path, index=False)
    
    with open(map_path, "wb") as f:
        pickle.dump(new_set_map, f)
    
    t2 = time.time()
    print(f"Extracted {len(updated_data)} tokens from {len(new_set_map)} columns")
    print(f"Time: {(t2-t1)/60:.2f} minutes")
    
    return True


def is_equal(x: List, y: List) -> bool:
    """Check if two posting lists are equal based on freq and hash."""
    return x[2] == y[2] and x[3] == y[3]


def create_index(output_dir: str) -> bool:
    """
    Create inverted index from raw tokens.
    
    Creates:
    - outputs/integerSet.json: Token IDs for each set
    - outputs/PLs.json: Posting lists
    - outputs/rawDict.json: Raw token dictionary
    
    Args:
        output_dir: Path to output directory (containing rawTokens.csv)
        
    Returns:
        True on success
    """
    print("\nStep 2: Building inverted index...")
    
    raw_tokens_path = os.path.join(output_dir, "rawTokens.csv")
    outpath = os.path.join(output_dir, "outputs")
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    t1 = time.time()
    
    # Read raw tokens
    print("Loading raw tokens...")
    raw_tokens_df = pd.read_csv(raw_tokens_path)
    
    # Group by RawToken to create initial posting lists
    print("Creating initial posting lists...")
    pl_list = []
    grouped = raw_tokens_df[['RawToken', 'SetID']].groupby('RawToken')
    
    for group_name, group_df in tqdm(grouped, desc="Grouping tokens"):
        token_data = group_df["SetID"].values.tolist()
        pl_list.append([
            group_name,
            token_data,
            len(token_data),
            hashlib.sha256(str(token_data).encode()).hexdigest()
        ])
    
    t2 = time.time()
    print(f"Initial PL creation time: {(t2-t1)/60:.2f} minutes")
    
    # Sort by frequency and hash to generate TokenIDs
    print("Sorting and generating token IDs...")
    pl_list_sorted = sorted(pl_list, key=lambda x: (x[2], x[3]), reverse=False)
    del pl_list
    
    token_ids = [[pl[0], i] for i, pl in enumerate(pl_list_sorted)]
    
    # Generate Group IDs (GIDs) for duplicate posting lists
    gids = [0] * len(pl_list_sorted)
    group_id = 0
    for i in range(len(pl_list_sorted)):
        gids[i] = group_id
        if i == len(pl_list_sorted) - 1:
            break
        if not is_equal(pl_list_sorted[i], pl_list_sorted[i + 1]):
            group_id += 1
    
    del pl_list_sorted
    
    # Create token table
    token_table = [[tid[0], tid[1], gids[i]] for i, tid in enumerate(token_ids)]
    del token_ids, gids
    
    token_table_df = pd.DataFrame(token_table, columns=["RawToken", "TokenID", "GroupID"])
    del token_table
    
    t3 = time.time()
    print(f"Token table creation time: {(t3-t2)/60:.2f} minutes")
    
    # Create integer sets
    print("Creating integer sets...")
    integer_set = {}
    merged_df = pd.merge(raw_tokens_df, token_table_df, on='RawToken', how='left')
    del raw_tokens_df, token_table_df
    
    set_len = {}
    grouped1 = merged_df.groupby('SetID')
    
    for group_name, group_df in tqdm(grouped1, desc="Building integer sets"):
        tids = group_df["TokenID"].values.tolist()
        integer_set[str(group_name)] = tids
        set_len[group_name] = len(tids)
    
    t4 = time.time()
    print(f"Integer set creation time: {(t4-t3)/60:.2f} minutes")
    
    # Save integer sets
    with open(os.path.join(outpath, "integerSet.json"), "w") as f:
        json.dump(integer_set, f)
    
    # Create posting lists and raw dictionary
    print("Creating posting lists...")
    pls = {}
    raw_dict = {}
    grouped2 = merged_df.groupby('TokenID')
    
    for group_name, group_df in tqdm(grouped2, desc="Building PLs"):
        set_ids = group_df["SetID"].values.tolist()
        positions = group_df["Position"].values.tolist()
        gid = int(group_df["GroupID"].values.tolist()[0])
        raw = group_df["RawToken"].values.tolist()[0]
        tid = int(group_name)
        freq = len(set_ids)
        
        raw_dict[raw] = [tid, gid, freq]
        
        pl = []
        for i in range(freq):
            pl.append([set_ids[i], positions[i], set_len[set_ids[i]]])
        pls[str(tid)] = pl
    
    # Save posting lists and raw dictionary
    with open(os.path.join(outpath, "PLs.json"), "w") as f:
        json.dump(pls, f)
    
    with open(os.path.join(outpath, "rawDict.json"), "w") as f:
        json.dump(raw_dict, f)
    
    t5 = time.time()
    print(f"Total index creation time: {(t5-t1)/60:.2f} minutes")
    
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024 / 1024:.2f} GB")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Build Joise inverted index for data lake')
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to data lake directory containing CSV files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save index files")
    parser.add_argument("--separator", type=str, default=",",
                        help="CSV separator/delimiter (default: ',')")
    parser.add_argument("--num_processes", type=int, default=4,
                        help="Number of parallel processes (default: 4)")
    
    args = parser.parse_args()
    
    # Decode escape sequences in separator
    args.separator = args.separator.encode().decode('unicode_escape')
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input path is not a directory: {args.input_dir}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Separator: {repr(args.separator)}")
    print(f"Processes: {args.num_processes}")
    print()
    
    # Step 1: Extract raw tokens
    create_raw_tokens(args.input_dir, args.output_dir, args.separator, args.num_processes)
    
    # Step 2: Build inverted index
    create_index(args.output_dir)
    
    print("\nIndex building complete!")


if __name__ == '__main__':
    main()
