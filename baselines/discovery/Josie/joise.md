```markdown
<div>
    <h1>Joise</h1>
</div>

<br>

<h2>Folder Structure</h2>

```
.
├─── build_index.py            # Build inverted index for joinability search
├─── query.py                  # Query for joinable tables
├─── josie.py                  # Core Joise algorithm
├─── josie_util.py             # Candidate entry and utility functions
├─── heap.py                   # Min-heap for top-k results
├─── cost.py                   # Cost model functions
├─── common.py                 # Common utility functions
└─── joise.md
```

<br>

<h2>Quick Start</h2>

**Step 1: Check your environment**

You need to properly install python packages first. Please check packages you have installed via `pip list`.
Required packages: pandas, tqdm, numpy

**Step 2: Build Index (Offline)**

Build an inverted index for the data lake. This creates:
- `rawTokens.csv` - Token extraction from all columns
- `setMap.pkl` - Mapping from set IDs to table/column names
- `outputs/integerSet.json` - Integer representation of sets
- `outputs/PLs.json` - Posting lists (inverted index)
- `outputs/rawDict.json` - Token dictionary

Parameters:
> --input_dir [path to data lake directory containing CSV files]
> --output_dir [directory to save index files]
> --separator [CSV delimiter, default: ","]
> --num_processes [number of parallel processes, default: 4]

```sh
python build_index.py --input_dir /path/to/datalake --output_dir ./index
```

**Step 3: Querying (Online)**

Search for joinable tables using the Joise algorithm.

Parameters:
> --index_dir [path to index directory created in step 2]
> --query_table [path to query CSV file]
> --output_file [output CSV file path]
> --K [number of top results to return, default: 10]
> --separator [CSV delimiter for data lake tables]
> --query_separator [CSV delimiter for query table]

```sh
python query.py --index_dir ./index --query_table /path/to/query.csv --K 10
```

<h2>Algorithm Description</h2>

Joise (Join Set Intersection Estimation) is an efficient algorithm for finding 
joinable tables in a data lake. It uses an inverted index over column values 
and employs cost-based optimization to balance between:
- Reading more posting lists (to discover more candidates)
- Probing candidate sets (to compute exact overlap)

Key features:
- Set intersection-based joinability estimation
- Cost model for I/O optimization
- Prefix filtering for early termination
- Top-k result retrieval using a min-heap

<h2>References</h2>

Based on the Joise implementation from LakeBench:
https://github.com/BIT-DataLab/LakeBench/tree/main/join/Joise
```
