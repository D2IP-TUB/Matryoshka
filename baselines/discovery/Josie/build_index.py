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
import polars as pl
import psutil
import pyarrow.parquet as pq_pa
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
                # Match upstream `createRawTokens.py`: keep only tokens whose
                # original pandas dtype is `str` and that are not purely
                # numeric. This skips int/float columns entirely (they would
                # otherwise be cast to str and inflate the index by 2-10x).
                raw_tokens = list(set(df[column_name].tolist()))
                pos = 0
                valid_tokens = []
                for token in raw_tokens:
                    if (type(token) is str
                            and token.strip()
                            and not token.isdigit()):
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


def create_index_polars(output_dir: str) -> bool:
    """Streaming Step 2 (polars + pyarrow).

    The pandas implementation in :func:`create_index` reads the entire
    ``rawTokens.csv`` into memory and builds two large in-memory dicts before
    serializing them — this OOMs on lakes whose rawTokens.csv exceeds a few
    GB (CUK at ~20 GB, GitTables larger still).

    This variant:
      1. Computes the token table lazily with ``polars.scan_csv +
         group_by + sort`` and writes it to a parquet sidecar.
      2. Computes ``set_len`` (rows per SetID) the same way.
      3. Joins raw tokens with the token table and sinks two sorted parquets
         (by SetID for the integer-set output, by TokenID for the PL / rawDict
         output).
      4. Reads each sorted parquet with ``pyarrow.parquet.iter_batches`` and
         streams one JSON entry per group key — never holds the full
         ``pls`` / ``integer_set`` dict in memory.

    Output files are byte-compatible with :func:`create_index`.
    """
    raw_tokens_path = os.path.join(output_dir, "rawTokens.csv")
    outpath = os.path.join(output_dir, "outputs")
    os.makedirs(outpath, exist_ok=True)

    print("\nStep 2: Building inverted index (polars streaming)...")
    t0 = time.time()

    # --- Phase 1: token table (one row per unique RawToken) -----------------
    print("Phase 1: per-token aggregates...")
    t = time.time()
    rt = pl.scan_csv(raw_tokens_path, infer_schema_length=10_000)
    token_table_path = os.path.join(outpath, "_token_table.parquet")
    (
        rt.group_by("RawToken")
          .agg([
              pl.col("SetID").sort().alias("SetIDs_sorted"),
              pl.len().cast(pl.Int64).alias("freq"),
          ])
          # Hash key = comma-joined sorted SetIDs. Identical sets get identical
          # strings, which is what `is_equal` needs for GroupID dedup.
          .with_columns(
              pl.col("SetIDs_sorted")
                .list.eval(pl.element().cast(pl.String))
                .list.join(",")
                .alias("hash")
          )
          .drop("SetIDs_sorted")
          .sort(["freq", "hash"])
          .with_row_index("TokenID")
          .with_columns(
              ((pl.col("freq") != pl.col("freq").shift(1)) |
               (pl.col("hash") != pl.col("hash").shift(1)))
              .fill_null(False).cum_sum().alias("GroupID")
          )
          .select(["RawToken", "TokenID", "GroupID", "freq"])
          .sink_parquet(token_table_path)
    )
    n_tokens = pl.scan_parquet(token_table_path).select(pl.len()).collect().item()
    print(f"  unique tokens: {n_tokens}  ({time.time()-t:.1f}s)")

    # --- Phase 2: set_len ---------------------------------------------------
    print("Phase 2: set_len (rows per SetID)...")
    t = time.time()
    set_len_df = (
        rt.group_by("SetID")
          .agg(pl.len().cast(pl.Int64).alias("set_len"))
          .collect(streaming=True)
    )
    set_len_dict = dict(zip(
        set_len_df["SetID"].to_list(),
        set_len_df["set_len"].to_list(),
    ))
    n_sets = len(set_len_dict)
    print(f"  total sets: {n_sets}  ({time.time()-t:.1f}s)")

    # --- Phase 3: merged dataframe, sorted twice ----------------------------
    print("Phase 3a: merge + sort by SetID -> parquet...")
    t = time.time()
    merged_by_set = os.path.join(outpath, "_merged_by_setid.parquet")
    (
        pl.scan_csv(raw_tokens_path, infer_schema_length=10_000)
          .join(pl.scan_parquet(token_table_path), on="RawToken", how="left")
          .select(["SetID", "TokenID", "Position", "GroupID", "RawToken"])
          .sort("SetID")
          .sink_parquet(merged_by_set)
    )
    print(f"  -> {merged_by_set}  ({time.time()-t:.1f}s)")

    print("Phase 3b: re-sort by TokenID -> parquet...")
    t = time.time()
    merged_by_tok = os.path.join(outpath, "_merged_by_tokenid.parquet")
    (
        pl.scan_parquet(merged_by_set)
          .sort("TokenID")
          .sink_parquet(merged_by_tok)
    )
    print(f"  -> {merged_by_tok}  ({time.time()-t:.1f}s)")

    # --- Phase 4: write integerSet.json (streaming, one entry per SetID) ----
    print("Phase 4: writing integerSet.json...")
    t = time.time()
    integer_set_path = os.path.join(outpath, "integerSet.json")
    with open(integer_set_path, "w") as f:
        f.write("{")
        first = True
        cur_set = None
        cur_tids: list = []
        reader = pq_pa.ParquetFile(merged_by_set)
        for batch in tqdm(
            reader.iter_batches(batch_size=200_000, columns=["SetID", "TokenID"]),
            total=reader.metadata.num_rows // 200_000 + 1,
            desc="integerSet"):
            sids = batch.column("SetID").to_pylist()
            tids = batch.column("TokenID").to_pylist()
            for sid, tid in zip(sids, tids):
                if sid != cur_set:
                    if cur_set is not None:
                        sep = "" if first else ","
                        f.write(f'{sep}"{cur_set}":')
                        json.dump(cur_tids, f)
                        first = False
                    cur_set = sid
                    cur_tids = []
                if tid is not None:
                    cur_tids.append(int(tid))
        if cur_set is not None:
            sep = "" if first else ","
            f.write(f'{sep}"{cur_set}":')
            json.dump(cur_tids, f)
        f.write("}")
    print(f"  done in {time.time()-t:.1f}s")

    # --- Phase 5: write PLs.json + rawDict.json (one entry per TokenID) -----
    print("Phase 5: writing PLs.json + rawDict.json...")
    t = time.time()
    pls_path = os.path.join(outpath, "PLs.json")
    rd_path = os.path.join(outpath, "rawDict.json")
    with open(pls_path, "w") as fp, open(rd_path, "w") as fr:
        fp.write("{")
        fr.write("{")
        first_p = True
        first_r = True
        cur_tid = None
        cur_gid = None
        cur_raw = None
        cur_pl: list = []
        reader = pq_pa.ParquetFile(merged_by_tok)
        for batch in tqdm(
            reader.iter_batches(batch_size=200_000),
            total=reader.metadata.num_rows // 200_000 + 1,
            desc="PLs"):
            sids = batch.column("SetID").to_pylist()
            tids = batch.column("TokenID").to_pylist()
            poss = batch.column("Position").to_pylist()
            gids = batch.column("GroupID").to_pylist()
            raws = batch.column("RawToken").to_pylist()
            for sid, tid, pos, gid, raw in zip(sids, tids, poss, gids, raws):
                if tid is None:
                    continue
                if tid != cur_tid:
                    if cur_tid is not None:
                        sep_p = "" if first_p else ","
                        fp.write(f'{sep_p}"{cur_tid}":')
                        json.dump(cur_pl, fp)
                        first_p = False
                        sep_r = "" if first_r else ","
                        fr.write(f"{sep_r}{json.dumps(cur_raw)}:")
                        json.dump([int(cur_tid), int(cur_gid), len(cur_pl)], fr)
                        first_r = False
                    cur_tid = tid
                    cur_gid = gid
                    cur_raw = raw
                    cur_pl = []
                cur_pl.append([int(sid), int(pos), int(set_len_dict.get(sid, 0))])
        if cur_tid is not None:
            sep_p = "" if first_p else ","
            fp.write(f'{sep_p}"{cur_tid}":')
            json.dump(cur_pl, fp)
            sep_r = "" if first_r else ","
            fr.write(f"{sep_r}{json.dumps(cur_raw)}:")
            json.dump([int(cur_tid), int(cur_gid), len(cur_pl)], fr)
        fp.write("}")
        fr.write("}")
    print(f"  done in {time.time()-t:.1f}s")

    # Cleanup intermediates
    for p in (token_table_path, merged_by_set, merged_by_tok):
        try:
            os.remove(p)
        except OSError:
            pass

    print(f"Total index creation time: {(time.time()-t0)/60:.2f} minutes")
    rss_gb = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    print(f"Memory usage: {rss_gb:.2f} GB")
    return True


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
    
    # Create integer sets — streamed JSON write so the dict never lives in RAM.
    # On NYC the in-memory dict accumulation hit MemoryError around 18% of
    # tokens; streaming sidesteps it without changing the file format.
    print("Creating integer sets...")
    merged_df = pd.merge(raw_tokens_df, token_table_df, on='RawToken', how='left')
    del raw_tokens_df, token_table_df

    set_len = {}
    grouped1 = merged_df.groupby('SetID')

    integer_set_path = os.path.join(outpath, "integerSet.json")
    with open(integer_set_path, "w") as f:
        f.write("{")
        first = True
        for group_name, group_df in tqdm(grouped1, desc="Building integer sets"):
            tids = group_df["TokenID"].values.tolist()
            set_len[group_name] = len(tids)
            sep = "" if first else ","
            f.write(f'{sep}"{group_name}":')
            json.dump(tids, f)
            first = False
        f.write("}")
    del grouped1

    t4 = time.time()
    print(f"Integer set creation time: {(t4-t3)/60:.2f} minutes")

    # Create posting lists and raw dictionary — also streamed.
    print("Creating posting lists...")
    grouped2 = merged_df.groupby('TokenID')

    pls_path = os.path.join(outpath, "PLs.json")
    raw_dict_path = os.path.join(outpath, "rawDict.json")
    with open(pls_path, "w") as pls_f, open(raw_dict_path, "w") as rd_f:
        pls_f.write("{")
        rd_f.write("{")
        first_pl = True
        first_rd = True
        for group_name, group_df in tqdm(grouped2, desc="Building PLs"):
            set_ids = group_df["SetID"].values.tolist()
            positions = group_df["Position"].values.tolist()
            gid = int(group_df["GroupID"].values.tolist()[0])
            raw = group_df["RawToken"].values.tolist()[0]
            tid = int(group_name)
            freq = len(set_ids)

            pl = [
                [int(set_ids[i]), int(positions[i]), int(set_len[set_ids[i]])]
                for i in range(freq)
            ]
            sep = "" if first_pl else ","
            pls_f.write(f'{sep}"{tid}":')
            json.dump(pl, pls_f)
            first_pl = False

            sep = "" if first_rd else ","
            rd_f.write(f"{sep}{json.dumps(raw)}:")
            json.dump([tid, gid, freq], rd_f)
            first_rd = False
        pls_f.write("}")
        rd_f.write("}")
    del grouped2, merged_df
    
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

    # Step 2: Build inverted index (polars streaming variant; the pandas
    # `create_index` is kept for reference but OOMs on lakes > ~10 GB).
    create_index_polars(args.output_dir)
    
    print("\nIndex building complete!")


if __name__ == '__main__':
    main()
