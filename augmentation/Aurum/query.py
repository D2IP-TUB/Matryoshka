import pickle
import argparse
import time
import numpy as np
from hnsw_search import HNSWSearcher
import csv
import sys
import os
from tqdm import tqdm
from datasketch import MinHash

# Set CSV field size limit
csv.field_size_limit(sys.maxsize)


def build_query_embeddings(query_table_path, separator, num_perm):
    """
    Build MinHash embeddings for a query table.
    
    Args:
        query_table_path: Path to query CSV file
        separator: CSV delimiter
        num_perm: Number of MinHash permutations (must match index)
    
    Returns:
        Tuple of (table_name, column_embeddings_array)
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
        m = MinHash(num_perm=num_perm)
        unique_vals = list(set([row[col_idx] for row in data_array if col_idx < len(row)]))
        for val in unique_vals:
            m.update(val.encode('utf-8'))
        column_embeddings.append(list(m.hashvalues))
    
    table_name = os.path.basename(query_table_path)
    return (query_table_path, np.array(column_embeddings))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search for joinable/unionable tables in data lake')
    parser.add_argument("--index_file", type=str, required=True,
                        help="Path to the index pickle file created by build_hash.py")
    parser.add_argument("--query_table", type=str, required=True,
                        help="Path to the query table CSV file")
    parser.add_argument("--output_file", type=str, default="search_results.csv",
                        help="Output CSV file path (default: search_results.csv)")
    parser.add_argument("--separator", type=str, default=",",
                        help="CSV separator for data lake tables (default: ',')")
    parser.add_argument("--query_separator", type=str, default=None,
                        help="CSV separator for query table (default: same as --separator)")
    parser.add_argument("--K", type=int, default=10,
                        help="Number of top candidate tables to return (default: 10)")
    parser.add_argument("--N", type=int, default=10,
                        help="Number of nearest neighbors per column (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold for column matching (default: 0.7)")
    parser.add_argument("--search_mode", type=str, default='join', choices=['join', 'union'],
                        help="Search mode: 'join' for joinable tables, 'union' for unionable tables")
    parser.add_argument("--max_join_cols", type=int, default=3,
                        help="Maximum number of matching columns for join mode (default: 3)")
    parser.add_argument("--num_perm", type=int, default=128,
                        help="Number of MinHash permutations (must match build_hash.py, default: 128)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Percentage of index to use (default: 1.0 = 100%%)")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for deterministic sampling (default: 42)")

    args = parser.parse_args()
    
    # Decode escape sequences in separators
    args.separator = args.separator.encode().decode('unicode_escape')
    if args.query_separator is None:
        args.query_separator = args.separator
    else:
        args.query_separator = args.query_separator.encode().decode('unicode_escape')
    
    # Validate input files
    if not os.path.exists(args.index_file):
        raise ValueError(f"Index file does not exist: {args.index_file}")
    if not os.path.exists(args.query_table):
        raise ValueError(f"Query table does not exist: {args.query_table}")
    
    print(f"Loading index from: {args.index_file}")
    print(f"Query table: {args.query_table}")
    print(f"Search mode: {args.search_mode}")
    print(f"Data lake separator: {repr(args.separator)}")
    print(f"Query table separator: {repr(args.query_separator)}")
    
    # Build query embeddings
    print("Building query embeddings...")
    query = build_query_embeddings(args.query_table, args.query_separator, args.num_perm)
    
    # Initialize searcher (note: no actual index_path needed, just for compatibility)
    searcher = HNSWSearcher(args.index_file, "", args.scale, search_mode=args.search_mode, random_seed=args.random_seed)
    
    # Execute search
    print(f"Searching for top-{args.K} {args.search_mode}able tables...")
    start_time = time.time()

    # Execute search
    print(f"Searching for top-{args.K} {args.search_mode}able tables...")
    start_time = time.time()
    
    results, num_candidates = searcher.topk(
        'aurum',  # encoder type
        query, 
        args.K, 
        N=args.N, 
        threshold=args.threshold, 
        max_join_cols=args.max_join_cols
    )
    
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.2f} seconds")
    print(f"Found {len(results)} results from {num_candidates} candidates")
    
    # Prepare output
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    
    # Read query table header
    with open(args.query_table, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=args.query_separator)
        query_header = next(reader)
    
    # Write results to CSV
    print(f"Writing results to: {args.output_file}")
    
    # Collect all matches and sort by similarity for strict K cutoff
    all_matches = []
    for result in results:
        score = result[0]  # Overall score
        column_pairs = result[1]  # List of (query_col_idx, candidate_col_idx, similarity)
        candidate_table = result[2]  # Table name
        
        if len(column_pairs) > 0:
            for query_col_idx, cand_col_idx, similarity in column_pairs:
                query_col_name = query_header[query_col_idx] if query_col_idx < len(query_header) else f'col_{query_col_idx}'
                cand_col_name = f'col_{cand_col_idx}'
                all_matches.append((
                    args.query_table,
                    candidate_table,
                    query_col_name,
                    cand_col_name,
                    similarity
                ))
    
    # Sort by similarity descending and take top K
    all_matches.sort(key=lambda x: -x[4])
    top_matches = all_matches[:args.K]
    
    with open(args.output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        # Write header
        writer.writerow(['from_id', 'to_id', 'from_column', 'to_column', 'weight'])
        
        for match in top_matches:
            writer.writerow([
                match[0],
                match[1],
                match[2],
                match[3],
                f'{match[4]:.4f}'
            ])
    
    print(f"\nWrote {len(top_matches)} matches (from {len(all_matches)} total)")
    print(f"Total runtime: {time.time() - start_time:.2f}s")