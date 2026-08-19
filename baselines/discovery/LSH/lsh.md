# LSH Ensemble Module

This module implements **LSH Ensemble** for containment-based join search in data lakes. It uses MinHash signatures combined with partitioned Locality-Sensitive Hashing to find tables where a query column is likely contained within indexed columns.

## Overview

LSH Ensemble is designed for **containment similarity** queries, making it ideal for finding joinable tables where the query set might be a subset of indexed sets. Unlike Jaccard similarity (which requires similar set sizes), containment works well for asymmetric relationships.

The algorithm:
1. **Offline Phase**: Build MinHash signatures for each column, then index them using LSH Ensemble with size-based partitioning
2. **Online Phase**: Query the index with a MinHash to find columns with high containment

## Source

Adapted from [LakeBench LSH](https://github.com/BIT-DataLab/LakeBench/tree/main/join/LSH)

## Files

- `build_index.py` - Offline index building (MinHash generation + LSH Ensemble construction)
- `query.py` - Online containment-based querying
- `datasketch/` - Core LSH Ensemble implementation:
  - `minhash.py` - MinHash signature generation
  - `lshensemble.py` - LSH Ensemble index for containment queries
  - `lshensemble_partition.py` - Optimal partitioning for set sizes
  - `lsh.py` - MinHash LSH for Jaccard queries
  - `storage.py` - Backend storage (dict, Redis, Cassandra)

## Usage

### Offline Indexing

```bash
python build_index.py --data_dir /path/to/parquet/files \
                      --output_dir /path/to/index \
                      --num_perm 256 \
                      --threshold 0.5 \
                      --num_part 32 \
                      --workers 4
```

**Arguments:**
- `--data_dir`: Directory containing parquet files to index
- `--output_dir`: Directory to save the index files
- `--num_perm`: Number of MinHash permutation functions (default: 256)
- `--threshold`: Containment threshold for query optimization (default: 0.5)
- `--num_part`: Number of LSH Ensemble partitions (default: 32)
- `--workers`: Number of parallel workers (default: 4)

### Online Querying

```bash
python query.py --index_dir /path/to/index \
                --query_file /path/to/query.parquet \
                --query_column column_name \
                --top_k 10 \
                --threshold 0.5
```

**Arguments:**
- `--index_dir`: Directory containing the index files
- `--query_file`: Parquet file containing the query column
- `--query_column`: Name of the column to query
- `--top_k`: Number of top results to return (default: 10)
- `--threshold`: Containment threshold for filtering (default: 0.5)

## Key Parameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `num_perm` | Number of MinHash permutations | 128-512 |
| `threshold` | Containment threshold | 0.3-0.8 |
| `num_part` | Number of partitions | 8-64 |
| `m` | Memory multiplier | 2-8 |

## Algorithm Details

### Containment Similarity

For sets A and B, containment of A in B is:
$$\text{containment}(A, B) = \frac{|A \cap B|}{|A|}$$

### LSH Ensemble Partitioning

Sets are partitioned by size to handle the asymmetric nature of containment:
- Smaller sets in earlier partitions
- Larger sets in later partitions
- Each partition uses optimized LSH parameters

### Complexity

- **Index Building**: O(n × num_perm) for MinHash, O(n × num_part) for indexing
- **Query Time**: O(num_part × num_perm / r) where r is rows per band
- **Space**: O(n × num_perm + num_part × buckets)

## Output Format

Query results are returned as a list of tuples:
```python
[
    (file_path, column_name, containment_score),
    ...
]
```

Sorted by containment score in descending order.
