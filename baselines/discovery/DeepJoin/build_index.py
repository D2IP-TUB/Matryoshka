"""Build a DeepJoin index for embedding-based joinable-column search.

Adapted from LakeBench (BIT-DataLab/LakeBench /join/Deepjoin), simplified for
the Matryoshka R2D6 ablation:
  * uses the pretrained Sentence-Transformer `all-mpnet-base-v2` directly
    (no contrastive fine-tuning step from the original DeepJoin paper),
  * encodes each lake column as `<column_name>: v1, v2, ...` and indexes it
    with an HNSW index (cosine distance),
  * stores embeddings + a column<->table-and-column map alongside the index
    so `query.py` can map a top-k neighbour set back to tables.

Usage:
    python build_index.py \
        --data_dir /path/to/parquet/files \
        --output_dir ./index \
        --file_format parquet \
        --model_name sentence-transformers/all-mpnet-base-v2 \
        --sample_size 20 --batch_size 64
"""
import argparse
import gc
import os
import pickle
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Column-string construction (per-file, runs in worker processes; no torch)
# ---------------------------------------------------------------------------

def _make_column_string(col_name: str, values: List[str], sample_size: int,
                        rng: random.Random) -> str:
    """Render a column as a text snippet for the encoder.

    Format: ``"<col_name>: v1, v2, ..."`` (capped at `sample_size` values).
    Long values are truncated to 128 chars so a few outliers don't dominate
    the model's 384-token context window.
    """
    if not values:
        return f'{col_name}:'
    sample = rng.sample(values, min(sample_size, len(values)))
    snippets = [v[:128] for v in sample]
    return f'{col_name}: ' + ', '.join(snippets)


def _process_file_batch(args: Tuple[List[str], int, str, str, int]):
    """Read a batch of files, return per-column text snippets ready to encode.

    Worker returns a list of (file_path, col_name, col_text) tuples.
    """
    file_paths, sample_size, file_format, separator, seed = args
    rng = random.Random(seed)
    out: List[Tuple[str, str, str]] = []
    for fp in file_paths:
        try:
            if file_format == 'parquet':
                df = pd.read_parquet(fp)
            elif file_format == 'csv':
                df = pd.read_csv(fp, sep=separator, on_bad_lines='skip',
                                 encoding='utf-8', engine='python', dtype=str)
            else:
                continue
        except Exception:
            continue
        for col in df.columns:
            try:
                series = df[col].dropna().astype(str).str.strip()
                values = series[series != ''].unique().tolist()
                if not values:
                    continue
                out.append((fp, str(col),
                            _make_column_string(str(col), values, sample_size, rng)))
            except Exception:
                continue
    return out


def collect_files(data_dir: str, file_format: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(data_dir):
        for n in names:
            if n.endswith(f'.{file_format}'):
                files.append(os.path.join(root, n))
    return sorted(files)


def _batch(items: List, n: int) -> List[List]:
    return [items[i:i + n] for i in range(0, len(items), n)]


# ---------------------------------------------------------------------------
# Embedding (main process; loads SentenceTransformer once)
# ---------------------------------------------------------------------------

def _encode_all(snippets: List[Tuple[str, str, str]], model_name: str,
                batch_size: int) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Encode all per-column snippets with one model instance.

    Returns (embeddings, column_map) where column_map[i] = (file_path, col_name).
    """
    # Import here so the worker processes (which don't need torch) skip the
    # heavy import cost.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    column_map: List[Tuple[str, str]] = [(fp, col) for fp, col, _ in snippets]
    texts = [t for _, _, t in snippets]
    print(f'Encoding {len(texts)} columns with {model_name} '
          f'(batch_size={batch_size})...')
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine-distance HNSW expects unit vectors
    )
    return embeddings.astype(np.float32), column_map


# ---------------------------------------------------------------------------
# HNSW index build
# ---------------------------------------------------------------------------

def _build_hnsw(embeddings: np.ndarray, M: int = 32,
                ef_construction: int = 200):
    import hnswlib
    n, dim = embeddings.shape
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    index.add_items(embeddings, ids=np.arange(n))
    index.set_ef(64)
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True,
                   help='Directory containing the data lake')
    p.add_argument('--output_dir', required=True,
                   help='Directory to write the index files')
    p.add_argument('--file_format', default='parquet', choices=['parquet', 'csv'])
    p.add_argument('--separator', default=',', help='CSV separator')
    p.add_argument('--model_name',
                   default='sentence-transformers/all-mpnet-base-v2',
                   help='Sentence-Transformer model name or local path')
    p.add_argument('--sample_size', type=int, default=20,
                   help='Number of values per column to include in the snippet')
    p.add_argument('--batch_size', type=int, default=64,
                   help='Encoder batch size')
    p.add_argument('--n_workers', type=int, default=max(1, cpu_count() // 2),
                   help='Number of worker processes for file reading')
    p.add_argument('--files_per_batch', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hnsw_M', type=int, default=32)
    p.add_argument('--hnsw_ef_construction', type=int, default=200)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Collecting *.{args.file_format} files under {args.data_dir}...')
    files = collect_files(args.data_dir, args.file_format)
    print(f'  found {len(files)} files')

    # 1) Parallel file reading -> per-column text snippets
    t0 = time.time()
    snippets: List[Tuple[str, str, str]] = []
    batches = _batch(files, args.files_per_batch)
    worker_args = [
        (b, args.sample_size, args.file_format, args.separator,
         args.seed + i)
        for i, b in enumerate(batches)
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as ex:
        futures = [ex.submit(_process_file_batch, wa) for wa in worker_args]
        for f in tqdm(as_completed(futures), total=len(futures),
                      desc='Reading files'):
            snippets.extend(f.result())
    print(f'  collected {len(snippets)} columns in {time.time() - t0:.1f}s')
    gc.collect()

    if not snippets:
        print('No columns extracted; aborting.', file=sys.stderr)
        sys.exit(1)

    # 2) Encode (single process, batched on whatever device torch finds)
    t0 = time.time()
    embeddings, column_map = _encode_all(snippets, args.model_name,
                                         args.batch_size)
    print(f'  encoded in {time.time() - t0:.1f}s; shape={embeddings.shape}')

    # 3) HNSW index
    t0 = time.time()
    index = _build_hnsw(embeddings, M=args.hnsw_M,
                        ef_construction=args.hnsw_ef_construction)
    print(f'  built HNSW index in {time.time() - t0:.1f}s')

    # 4) Persist
    np.save(os.path.join(args.output_dir, 'embeddings.npy'), embeddings)
    with open(os.path.join(args.output_dir, 'column_map.pkl'), 'wb') as f:
        pickle.dump(column_map, f, protocol=pickle.HIGHEST_PROTOCOL)
    index.save_index(os.path.join(args.output_dir, 'hnsw.bin'))
    with open(os.path.join(args.output_dir, 'metadata.pkl'), 'wb') as f:
        pickle.dump({
            'model_name': args.model_name,
            'sample_size': args.sample_size,
            'dim': int(embeddings.shape[1]),
            'n_columns': int(embeddings.shape[0]),
            'hnsw_M': args.hnsw_M,
            'hnsw_ef_construction': args.hnsw_ef_construction,
        }, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Index written to {args.output_dir}')


if __name__ == '__main__':
    main()
