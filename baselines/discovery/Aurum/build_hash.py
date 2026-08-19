import logging

import numpy as np
import csv
import os
import pickle
import time
from datasketch import MinHash
from multiprocessing import Process, Queue
from tqdm import tqdm
import sys
import pandas as pd
import multiprocessing
import argparse
import hnswlib

# Set CSV field size limit
csv.field_size_limit(sys.maxsize)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Build MinHash index for data lake tables')
parser.add_argument("--input_dir", type=str, required=True, 
                    help="Path to the data lake directory containing CSV files")
parser.add_argument("--output_file", type=str, default="hnsw_output.pkl",
                    help="Output pickle file path (default: hnsw_output.pkl)")
parser.add_argument("--index_file", type=str, default="hnsw_index.bin",
                    help="Output HNSW index file path (default: hnsw_index.bin)")
parser.add_argument("--separator", type=str, default=",",
                    help="CSV separator/delimiter (default: ',')")
parser.add_argument("--num_processes", type=int, default=30,
                    help="Number of parallel processes (default: 30)")
parser.add_argument("--num_perm", type=int, default=128,
                    help="Number of MinHash permutations (default: 128)")
parser.add_argument("--index_threads", type=int, default=1,
                    help="Number of threads for HNSW index creation (default: 1)")

args = parser.parse_args()

# Decode escape sequences in separator (e.g., "\t" -> tab character)
args.separator = args.separator.encode().decode('unicode_escape')

# Validate input directory
if not os.path.exists(args.input_dir):
    raise ValueError(f"Input directory does not exist: {args.input_dir}")
if not os.path.isdir(args.input_dir):
    raise ValueError(f"Input path is not a directory: {args.input_dir}")


hnsw = []

def build_hash(file_ls, queue, queue_hnsw, idx, separator, num_perm):
    """
    Build MinHash signatures for columns in CSV files.
    
    Args:
        file_ls: List of file paths to process
        queue: Queue for progress updates
        queue_hnsw: Queue for results
        idx: Process index
        separator: CSV delimiter
        num_perm: Number of MinHash permutations
    """
    all_file_target = []
    data_embadding = []
    # Store all .csv file paths
    for j in range(0, len(file_ls)):
        if file_ls[j].endswith(".csv"):
            all_file_target.append(file_ls[j])

    head_array = []
    # Build index
    for i in range(0, len(all_file_target)):
        # Read data
        data_array_target = []
        try:
            with open(all_file_target[i], encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=separator)
                flag = 0
                for idx, row in enumerate(csv_reader):
                    if flag == 0:
                        head_array = row
                        flag = 1
                    elif row:
                        data_array_target.append(row)
        except Exception as e:
            print(f"Error processing {all_file_target[i]}: {e}")
            queue.put(1)
            continue

        # Extract table name from file path
        str0 = os.path.basename(all_file_target[i])
        str0 = str0[:-4]  # Remove .csv extension
        
        table_temp = []
        # Build minhash index for each column
        if len(data_array_target) > 0 and len(data_array_target[0]) > 0:
            for j in range(0, len(data_array_target[0])):
                m1 = MinHash(num_perm=num_perm)
                temp = list(set([row[j] for row in data_array_target if j < len(row)]))
                for str_i in temp:
                    m1.update(str_i.encode('utf-8'))
                table_temp.append(list(m1.hashvalues))
            temp_array = np.array(table_temp)
            data_embadding.append((str0, temp_array))
        queue.put(1)
    queue_hnsw.put(data_embadding)
    queue.put((-1, "test-pid"))


def split_list(lst, num_parts):
    avg = len(lst) // num_parts
    remainder = len(lst) % num_parts

    result = []
    start = 0
    for i in range(num_parts):
        if i < remainder:
            end = start + avg + 1
        else:
            end = start + avg
        result.append(lst[start:end])
        start = end

    return result


start_time = time.time()
file_ls = []

# Scan input directory for CSV files
print(f"Scanning directory: {args.input_dir}")
for root, dirs, files in os.walk(args.input_dir):
    root_file_ls = [os.path.join(root, file) for file in files if file.endswith('.csv')]
    file_ls.extend(root_file_ls)

print(f"Found {len(file_ls)} CSV files")

if len(file_ls) == 0:
    raise ValueError(f"No CSV files found in {args.input_dir}")

# Split files across processes
sub_file_ls = split_list(file_ls, args.num_processes)

process_list = []
queue_hnsw = multiprocessing.Manager().Queue()

# Create a queue for each process
queues = [multiprocessing.Manager().Queue() for i in range(args.num_processes)]
# Array to track finished processes
finished = [False for i in range(args.num_processes)]

# Create progress bar for each process
bars = [tqdm(total=len(sub_file_ls[i]), desc=f"Process-{i}", position=i) for i in range(args.num_processes)]
# Store results from each process
results = [None for i in range(args.num_processes)]

print(f"Starting {args.num_processes} processes...")
for i in range(args.num_processes):
    process = Process(target=build_hash, args=(sub_file_ls[i], queues[i], queue_hnsw, i, args.separator, args.num_perm))
    process_list.append(process)
    process.start()

while True:
    for i in range(args.num_processes):
        queue = queues[i]
        bar = bars[i]
        try:
            # Get data from queue (non-blocking)
            res = queue.get_nowait()
            if isinstance(res, tuple) and res[0] == -1:
                # Process finished
                finished[i] = True
                results[i] = res[1]
                continue
            bar.update(res)
        except Exception as e:
            continue

    # Check if all processes finished
    if all(finished):
        break


for process in process_list:
    process.join()

# Collect results from all processes
while not queue_hnsw.empty():
    try:
        k = queue_hnsw.get_nowait()
        hnsw.extend(k)
    except Exception as e:
        continue

# Save to output file
print(f"\nSaving index to {args.output_file}...")
output_dir = os.path.dirname(args.output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(args.output_file, 'wb') as f:
    pickle.dump(hnsw, f)

print(f"MinHash embeddings saved successfully! Total tables indexed: {len(hnsw)}")

# Build HNSW index from the embeddings
print(f"\nBuilding HNSW index...")
index_start_time = time.time()

# Preprocess tables to extract all columns
all_columns = []
col_table_ids = []
for idx, table in enumerate(hnsw):
    for col in table[1]:
        all_columns.append(col)
        col_table_ids.append(idx)

print(f"Total columns to index: {len(all_columns)}")

if len(all_columns) > 0:
    vec_dim = len(all_columns[0])
    print(f"Vector dimension: {vec_dim}")
    
    # Initialize HNSW index
    index = hnswlib.Index(space='cosine', dim=vec_dim)
    index.set_num_threads(args.index_threads)
    index.init_index(max_elements=len(all_columns), ef_construction=100, M=32, random_seed=42)
    index.set_ef(10)
    
    # Add all column vectors to the index
    index.add_items(all_columns)
    
    # Save the HNSW index
    index_output_dir = os.path.dirname(args.index_file)
    if index_output_dir and not os.path.exists(index_output_dir):
        os.makedirs(index_output_dir)
    
    index.save_index(args.index_file)
    print(f"HNSW index saved to {args.index_file}")
    print(f"Index building time: {time.time() - index_start_time:.2f}s")
else:
    print("Warning: No columns found, skipping HNSW index creation")

end_time = time.time()
run_time = round(end_time - start_time)
hour = run_time // 3600
minute = (run_time - 3600 * hour) // 60
second = run_time - 3600 * hour - 60 * minute

print(f'\nTotal runtime: {hour}h {minute}m {second}s ({run_time}s total)')
