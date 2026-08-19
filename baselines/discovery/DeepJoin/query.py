"""Query a DeepJoin index for joinable tables.

Given a query column from the user's table, returns the top-k joinable
columns/tables from the indexed lake, ranked by cosine similarity in the
shared Sentence-Transformer embedding space.

Usage:
    python query.py --index_dir ./index \
                    --query_file /path/to/query.parquet \
                    --query_column col_name \
                    --top_k 20
"""
import argparse
import os
import pickle
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd


def _column_string(col_name: str, values: List[str], sample_size: int,
                   seed: int) -> str:
    """Mirror build_index.py's column-string format."""
    rng = random.Random(seed)
    if not values:
        return f'{col_name}:'
    sample = rng.sample(values, min(sample_size, len(values)))
    snippets = [v[:128] for v in sample]
    return f'{col_name}: ' + ', '.join(snippets)


def load_index(index_dir: str):
    import hnswlib
    with open(os.path.join(index_dir, 'metadata.pkl'), 'rb') as f:
        meta = pickle.load(f)
    with open(os.path.join(index_dir, 'column_map.pkl'), 'rb') as f:
        column_map = pickle.load(f)
    index = hnswlib.Index(space='cosine', dim=meta['dim'])
    index.load_index(os.path.join(index_dir, 'hnsw.bin'),
                     max_elements=meta['n_columns'])
    index.set_ef(max(64, meta.get('hnsw_M', 32) * 2))
    return index, column_map, meta


def _read_query_column(query_file: str, query_column: str,
                       separator: str) -> List[str]:
    if query_file.endswith('.parquet'):
        df = pd.read_parquet(query_file, columns=[query_column])
    else:
        df = pd.read_csv(query_file, sep=separator, on_bad_lines='skip',
                         encoding='utf-8', engine='python', dtype=str,
                         usecols=[query_column])
    series = df[query_column].dropna().astype(str).str.strip()
    values = series[series != ''].unique().tolist()
    return values


def query(index_dir: str, query_file: str, query_column: str,
          top_k: int, separator: str = ',',
          seed: int = 42) -> List[Tuple[float, str, str]]:
    """Run a single query. Returns [(score, file_path, col_name), ...]."""
    from sentence_transformers import SentenceTransformer

    index, column_map, meta = load_index(index_dir)
    values = _read_query_column(query_file, query_column, separator)
    text = _column_string(query_column, values, meta['sample_size'], seed)

    model = SentenceTransformer(meta['model_name'])
    emb = model.encode([text], convert_to_numpy=True,
                       normalize_embeddings=True).astype(np.float32)

    # hnswlib returns distance = 1 - cosine_sim (cosine space)
    labels, distances = index.knn_query(emb, k=top_k)
    out: List[Tuple[float, str, str]] = []
    for lbl, dist in zip(labels[0], distances[0]):
        fp, col = column_map[int(lbl)]
        out.append((1.0 - float(dist), fp, col))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--index_dir', required=True)
    p.add_argument('--query_file', required=True)
    p.add_argument('--query_column', required=True)
    p.add_argument('--top_k', type=int, default=20)
    p.add_argument('--separator', default=',', help='CSV separator')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    t0 = time.time()
    results = query(args.index_dir, args.query_file, args.query_column,
                    args.top_k, args.separator, args.seed)
    dt = time.time() - t0
    print(f'top-{args.top_k} results ({dt:.3f}s):')
    for score, fp, col in results:
        print(f'  {score:.4f}  {os.path.basename(fp)}  ::  {col}')


if __name__ == '__main__':
    main()
