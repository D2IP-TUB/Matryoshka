#!/bin/bash

PARQUET_DIR="/mnt/data1/lakes/gittables/extracted_parquet"
CSV_DIR="/mnt/data1/lakes/gittables/extracted"
mkdir -p "$CSV_DIR"

convert_file() {
    parquet_file="$1"
    csv_file="$CSV_DIR/$(basename "$parquet_file" .parquet).csv"
    duckdb -c "COPY (SELECT * FROM read_parquet('$parquet_file')) TO '$csv_file' (HEADER, DELIMITER ',');"
    echo "Converted $parquet_file -> $csv_file"
}

export -f convert_file
export CSV_DIR

find "$PARQUET_DIR" -maxdepth 1 -name '*.parquet' -print0 | \
parallel -0 -j 64 convert_file {}
