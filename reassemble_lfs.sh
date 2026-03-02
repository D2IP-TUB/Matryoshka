#!/usr/bin/env bash
# Reassemble files that were split to stay under GitHub LFS 2 GB limit.
# Run once after cloning / pulling.
set -euo pipefail

reassemble() {
    local target="$1"
    local pattern="${target}.part*.lfs"
    
    if [[ -f "$target" ]]; then
        echo "SKIP  $target (already exists)"
        return
    fi

    parts=( $(ls $pattern 2>/dev/null | sort) )
    if [[ ${#parts[@]} -eq 0 ]]; then
        echo "WARN  No parts found for $target"
        return
    fi

    echo "JOIN  ${#parts[@]} parts -> $target"
    cat "${parts[@]}" > "$target"
    echo "  OK  $(du -h "$target" | cut -f1)"
}

cd "$(dirname "$0")"

reassemble "augmentation/Aurum/graphs/gittables/lsh_ensemble.pkl"
reassemble "experiments/downstream/logs/nyc_fire_backward/augmented_nyc_fire_backward.csv"
reassemble "experiments/downstream/logs3/nyc_fire_backward/augmented_nyc_fire_backward.csv"
reassemble "experiments/downstream/logs4/nyc_fire_backward/augmented_nyc_fire_backward.csv"
reassemble "experiments/downstream/logs5/nyc_fire_backward/augmented_nyc_fire_backward.csv"

echo "Done."
