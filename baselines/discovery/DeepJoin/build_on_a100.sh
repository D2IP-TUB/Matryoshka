#!/usr/bin/env bash
# Build DeepJoin indexes for NYC, CUK, and GitTables on an A100 GPU host.
#
# Companion to the workflow documented in README_TRANSFER.md. Run this
# script on the A100 server AFTER the lake tarballs have been transferred
# and extracted. Outputs land under augmentation/DeepJoin/<lake>_index/.
#
# Approximate wall-clock on an A100 (40-80 GB):
#   NYC        : <1 min
#   CUK        : 1-2 min
#   GitTables  : 30-60 min  (file I/O is the bottleneck on 975k tiny CSVs)
set -euo pipefail

if [[ "${1:-}" == "" ]]; then
  echo "usage: $0 <LAKES_ROOT>"
  exit 1
fi
LAKES_ROOT="$1"

cd "$(dirname "$0")/../.."
ROOT="$PWD"

# Pick the GPU; override with CUDA_VISIBLE_DEVICES=... before invoking.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python -c "import torch; print(f'cuda={torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"-\"}')"

# Each (lake_alias, source_subdir, csv_separator) triple. The separator must
# match what the rest of the pipeline uses (see lake_table_sep in
# experiments/downstream/experiments.csv): NYC is tab-separated, CUK and
# GitTables are comma-separated. Passing the wrong separator silently
# collapses every row into one column and yields meaningless embeddings.
declare -a LAKES=(
  $'nyc|nyc/extracted|\t'
  'cuk|canada_us_uk_open_data/extracted|,'
  'gittables|gittables/extracted|,'
)

for entry in "${LAKES[@]}"; do
  IFS='|' read -r alias subdir sep <<< "$entry"
  data_dir="$LAKES_ROOT/$subdir"
  out_dir="$ROOT/augmentation/DeepJoin/${alias}_index"
  echo
  echo "==== $alias ===="
  echo "  source    : $data_dir"
  echo "  output    : $out_dir"
  printf '  separator : %q\n' "$sep"
  if [[ ! -d "$data_dir" ]]; then
    echo "  SKIP (source dir missing)"
    continue
  fi
  mkdir -p "$out_dir"
  python -u augmentation/DeepJoin/build_index.py \
    --data_dir "$data_dir" \
    --output_dir "$out_dir" \
    --file_format csv \
    --separator "$sep" \
    --batch_size 256 \
    --sample_size 20 \
    --n_workers 16 \
    2>&1 | tee "$out_dir/build.log"
done

echo
echo "All builds done. Rsync back with:"
echo "  rsync -avh augmentation/DeepJoin/{nyc,cuk,gittables}_index/ \\"
echo "    workstation:/home/fedor/Fast_Data_Discovery/augmentation/DeepJoin/"
