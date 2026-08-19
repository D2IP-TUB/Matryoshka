# DeepJoin

Embedding-based joinable-column search using a Sentence-Transformer model and
HNSW nearest-neighbor index. Adapted from
[LakeBench](https://github.com/BIT-DataLab/LakeBench/tree/main/join/Deepjoin)
for use as a retrieval-only baseline in the Matryoshka experiments (R2D6).

## What's different from the original DeepJoin

The original DeepJoin paper fine-tunes a Sentence-Transformer on a labelled
column-pair similarity dataset (Sato / WebTables) using a contrastive loss
before indexing. For our ablation we drop the fine-tuning step and use the
pretrained `sentence-transformers/all-mpnet-base-v2` model directly. The
hypothesis under test (R2D6) is that downstream improvement is driven mostly
by the *feature-selection* stage, not the retrieval engine, so using
DeepJoin's retrieval mechanism without fine-tuning is a deliberately
conservative comparison.

The Hungarian-matching verification stage from `hnsw_search.py:_verify` is
also omitted; we use the raw HNSW top-k.

## Install

```bash
pip install -r requirements.txt
```

`torch` and `sentence-transformers` are heavy. On CPU expect ~50–200
columns/sec encoding throughput. The first run will also download the
all-mpnet-base-v2 weights (~420 MB) into the Hugging Face cache.

## Usage

### Build an index over a data lake

```bash
python build_index.py \
  --data_dir /path/to/parquet/files \
  --output_dir ./nyc_index \
  --file_format parquet \
  --batch_size 64 \
  --sample_size 20
```

Outputs (under `--output_dir`):

| File           | Contents                                               |
| -------------- | ------------------------------------------------------ |
| `embeddings.npy` | (n_columns, 768) float32 column embeddings             |
| `column_map.pkl` | list of `(file_path, col_name)` per embedding row     |
| `hnsw.bin`       | hnswlib HNSW index file                                |
| `metadata.pkl`   | model name, sample size, dim, n_columns, HNSW params  |

### Query for joinable columns

```bash
python query.py \
  --index_dir ./nyc_index \
  --query_file /path/to/user_table.csv \
  --query_column zip_code \
  --top_k 20
```

Returns the top-k `(score, file_path, col_name)` matches ranked by cosine
similarity in the embedding space.

## Notes

* Each column is rendered as the text snippet `"<col_name>: v1, v2, ..."`
  using `sample_size` random values, truncated to 128 chars each. Increasing
  `sample_size` lets the model see more of the value distribution but costs
  more tokens per encode.
* HNSW parameters (`M`, `ef_construction`) are exposed on `build_index.py`
  but the defaults (32 / 200) match the original LakeBench port.
* If GPU is available, `sentence-transformers` will use it automatically
  via `torch.cuda.is_available()`. Set `CUDA_VISIBLE_DEVICES` to pick the
  device.
