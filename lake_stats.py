"""
Compute average number of rows and columns for CSV tables in each data lake.
Uses fast Python I/O with multiprocessing for per-file stats, DuckDB for aggregation.
"""
import duckdb
import os
import glob
import time
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_file_stats(args):
    """Fast: count rows (lines-1) and columns (fields in header) for a CSV."""
    fpath, delimiter = args
    try:
        nrows = 0
        ncols = 0
        with open(fpath, "r", errors="replace") as f:
            # Read header to count columns
            header = f.readline()
            if not header.strip():
                return None
            # Use csv.reader with the appropriate delimiter
            ncols = len(next(csv.reader([header], delimiter=delimiter)))
            # Count remaining lines (data rows)
            # Buffered read is fastest for line counting
            nrows = sum(1 for _ in f)
        return (fpath, nrows, ncols)
    except Exception:
        return None


LAKES = {
    "nyc": ("/mnt/data1/lakes/nyc/extracted", "\t"),
    "canada_us_uk_open_data": ("/mnt/data1/lakes/canada_us_uk_open_data/extracted", ","),
    "gittables": ("/mnt/data1/lakes/gittables/extracted", ","),
}

N_WORKERS = os.cpu_count() or 8

for lake_name, (lake_path, delimiter) in LAKES.items():
    print(f"\n{'='*60}")
    print(f"Processing: {lake_name} (delimiter={'TAB' if delimiter == chr(9) else repr(delimiter)})")

    csv_pattern = os.path.join(lake_path, "*.csv")
    all_files = glob.glob(csv_pattern)
    n_files = len(all_files)
    print(f"Total CSV files: {n_files}")

    start = time.time()

    # Collect per-file stats in parallel
    results = []  # list of (filename, nrows, ncols)
    errors = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(get_file_stats, (f, delimiter)): f for f in all_files}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            res = future.result()
            if res is not None:
                results.append(res)
            else:
                errors += 1
            if done_count % 50000 == 0:
                print(f"  Progress: {done_count}/{n_files} ({time.time()-start:.1f}s)")

    elapsed_collect = time.time() - start
    print(f"  Collected stats for {len(results)} files in {elapsed_collect:.1f}s (errors: {errors})")

    # Use DuckDB to aggregate
    con = duckdb.connect()
    con.execute("CREATE TABLE stats (filename VARCHAR, nrows BIGINT, ncols BIGINT)")
    con.executemany("INSERT INTO stats VALUES (?, ?, ?)", results)

    agg = con.execute("""
        SELECT 
            count(*) as n_tables,
            avg(nrows) as avg_rows,
            avg(ncols) as avg_cols,
            median(nrows) as median_rows,
            median(ncols) as median_cols,
            min(nrows) as min_rows,
            max(nrows) as max_rows,
            min(ncols) as min_cols,
            max(ncols) as max_cols
        FROM stats
    """).fetchone()

    con.close()

    elapsed = time.time() - start
    print(f"\nResults for {lake_name}:")
    print(f"  Tables processed: {agg[0]} (errors: {errors})")
    print(f"  Avg rows:    {agg[1]:.2f}")
    print(f"  Avg cols:    {agg[2]:.2f}")
    print(f"  Median rows: {agg[3]:.2f}")
    print(f"  Median cols: {agg[4]:.2f}")
    print(f"  Rows range:  [{agg[5]}, {agg[6]}]")
    print(f"  Cols range:  [{agg[7]}, {agg[8]}]")
    print(f"  Total time:  {elapsed:.1f}s")
