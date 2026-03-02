import json
import os
from pathlib import Path

import polars as pl

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs3"

STEPS_FORWARD = ["Retrieval", "Pruning", "Sketching", "Selection", "Augmentation"]
STEPS_BACKWARD = ["Retrieval", "Pruning", "Sketching", "Collinearity", "Selection", "Augmentation"]

# Log files use "Ranking"; we rename to "Sketching" for the paper.
STEP_RENAMES = {"Ranking": "Sketching"}


def parse_log(log_path: str) -> dict[str, float]:
    """Parse a single log file and return {step_name: runtime}."""
    runtimes = {}
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = entry.get("message", "")
            step = STEP_RENAMES.get(step, step)
            rt = entry.get("runtime")
            if rt is not None:
                runtimes[step] = rt
    return runtimes


def collect_runtimes(logs_dir: Path) -> pl.DataFrame:
    """Scan forward/backward log dirs and return a tidy DataFrame of runtimes."""
    records = []
    for dirname in sorted(os.listdir(logs_dir)):
        if not (dirname.endswith("_forward") or dirname.endswith("_backward")):
            continue
        log_path = logs_dir / dirname / f"{dirname}.log"
        if not log_path.exists():
            continue

        strategy = "forward" if dirname.endswith("_forward") else "backward"
        dataset = dirname.rsplit("_", 1)[0]

        runtimes = parse_log(log_path)
        total = 0.0
        for step, rt in runtimes.items():
            records.append({
                "dataset": dataset,
                "strategy": strategy,
                "step": step,
                "runtime": rt,
            })
            total += rt
        records.append({
            "dataset": dataset,
            "strategy": strategy,
            "step": "Total",
            "runtime": total,
        })

    return pl.DataFrame(records)


def summary_per_step(df: pl.DataFrame) -> pl.DataFrame:
    """Average runtime per step across all datasets, split by strategy."""
    return (
        df.filter(pl.col("step") != "Total")
        .group_by("strategy", "step")
        .agg(
            pl.col("runtime").mean().alias("mean"),
            pl.col("runtime").min().alias("min"),
            pl.col("runtime").max().alias("max"),
            pl.col("runtime").count().alias("count"),
        )
        .sort("strategy", "step")
    )


def pivot_table(df: pl.DataFrame, strategy: str) -> pl.DataFrame:
    """Pivot per-dataset runtimes for a given strategy into a wide table."""
    steps = STEPS_FORWARD if strategy == "forward" else STEPS_BACKWARD
    steps_with_total = steps + ["Total"]
    sub = df.filter(
        (pl.col("strategy") == strategy) & pl.col("step").is_in(steps_with_total)
    )
    wide = sub.pivot(on="step", index="dataset", values="runtime").sort("dataset")
    # Reorder columns
    ordered_cols = ["dataset"] + [s for s in steps_with_total if s in wide.columns]
    return wide.select(ordered_cols)


def print_table(title: str, table: pl.DataFrame):
    """Pretty-print a polars DataFrame as an aligned text table."""
    print(f"\n{'=' * 80}")
    print(title)
    print("=" * 80)
    cols = table.columns
    header = f"{'Dataset':<25}" + "".join(f" | {c:>12}" for c in cols[1:])
    print(header)
    print("-" * len(header))
    for row in table.iter_rows(named=True):
        line = f"{row['dataset']:<25}"
        for c in cols[1:]:
            v = row[c]
            line += f" | {v:>12.3f}" if v is not None else f" | {'—':>12}"
        print(line)


def totals_side_by_side(df: pl.DataFrame) -> pl.DataFrame:
    """Total runtime per dataset, forward vs backward side by side."""
    totals = df.filter(pl.col("step") == "Total")
    fwd = (
        totals.filter(pl.col("strategy") == "forward")
        .select(["dataset", "runtime"])
        .rename({"runtime": "forward"})
    )
    bwd = (
        totals.filter(pl.col("strategy") == "backward")
        .select(["dataset", "runtime"])
        .rename({"runtime": "backward"})
    )
    return fwd.join(bwd, on="dataset", how="full", coalesce=True).sort("dataset")


if __name__ == "__main__":
    df = collect_runtimes(LOGS_DIR)

    # --- 1. Summary statistics per step ---
    summary = summary_per_step(df)
    print("\n" + "=" * 80)
    print("SUMMARY: Average runtime per step (seconds)")
    print("=" * 80)
    for strategy in ["forward", "backward"]:
        sub = summary.filter(pl.col("strategy") == strategy)
        print(f"\n  {strategy.upper()}")
        print(f"  {'Step':<20} {'Mean':>8} {'Min':>8} {'Max':>8} {'N':>5}")
        print(f"  {'-' * 55}")
        for row in sub.iter_rows(named=True):
            print(f"  {row['step']:<20} {row['mean']:>8.3f} {row['min']:>8.3f} {row['max']:>8.3f} {row['count']:>5}")

    # --- 2. Detailed per-dataset tables ---
    fwd_table = pivot_table(df, "forward")
    bwd_table = pivot_table(df, "backward")
    print_table("DETAILED: Forward Selection — Runtime per step (s)", fwd_table)
    print_table("DETAILED: Backward Elimination — Runtime per step (s)", bwd_table)

    # --- 3. Total runtime comparison ---
    totals = totals_side_by_side(df)
    print(f"\n{'=' * 80}")
    print("TOTAL RUNTIME PER DATASET (seconds)")
    print("=" * 80)
    print(f"{'Dataset':<25} | {'Forward':>12} | {'Backward':>12}")
    print("-" * 55)
    for row in totals.iter_rows(named=True):
        fwd_str = f"{row['forward']:.3f}" if row.get("forward") is not None else "—"
        bwd_str = f"{row['backward']:.3f}" if row.get("backward") is not None else "—"
        print(f"{row['dataset']:<25} | {fwd_str:>12} | {bwd_str:>12}")

    # --- 4. Export tidy CSV ---
    out_path = Path(__file__).resolve().parent / "runtimes.csv"
    df.write_csv(out_path)
    print(f"\nTidy data saved to {out_path}")
