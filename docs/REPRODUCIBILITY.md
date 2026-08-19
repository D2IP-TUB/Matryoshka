# Reproducing the paper

This document states what the published evaluation needs, what this repository
provides, and what has to be obtained or rebuilt elsewhere. It is deliberately
explicit about the gaps: the full evaluation is a multi-terabyte, multi-day
undertaking and cannot be packaged in a source repository.

## What this repository provides

- The Matryoshka implementation itself, installable and runnable
  (`src/matryoshka/`).
- The competing augmenters and discovery indexes, as source
  (`baselines/`).
- One complete, self-contained end-to-end run on a small lake
  (`examples/covertype/`).

## What it does not provide

**The evaluation harness.** The sweep driver, the per-lake experiment
configuration, the downstream trainers and the result aggregation scripts live
alongside the experiment data rather than in this repository. They are strongly
coupled to absolute paths on the machines the experiments ran on. Their shape:

| Component | Role |
|---|---|
| `config.yml` | per lake: data path, index table names, and for each query table the join key, target, task and cached join paths |
| `generate_config.sh`, `experiment_config_generator.py` | expand the configuration into the cross product of (lake, query table, algorithm) |
| `run.py` | Phase 1 runs discovery for every combination and writes `augmented_<combo>.csv` plus `augmentation_plan_<combo>.pkl`; Phase 2 trains the downstream model on each and aggregates |
| `experiment_executor.py` | routes each row of the configuration to either `JoinSelection.find_best_joins` (Matryoshka) or a baseline adapter |
| `anytime_orchestrator.py`, `run_anytime.py` | the anytime experiment: run under a wall-clock budget, then score every emitted checkpoint |
| `ablation/` | top-*k*, `n_jobs`, runtime breakdown, leave-one-out and proxy-model ablations |

**The data lakes.** NYC Open Data (1,069 tables, 52 GB extracted), Canada/US/UK
Open Data, GitTables, the Kitana lake, and the eight AutoFeat lakes. Only the
AutoFeat `covertype` lake ships here, as the example.

**The pre-built indexes.** Their sizes are the reason:

| Lake | Index size | Build time (16 workers, EPYC 7702P) |
|---|---|---|
| NYC | 89 GB, 5,676,630 rows over 780 tables | 4 h 54 m |
| GitTables | 1.32 TiB | 107 h |
| Canada/US/UK | 2.34 TiB | 95 h |
| AutoFeat lakes (8) | ~72 GB total | minutes to hours each |

An index is a deterministic function of the lake, so rebuilding reproduces it
up to the order in which `table_index` values are assigned.

## Rebuilding an index

```bash
export MATRYOSHKA_DSN=postgresql://user:password@host:5432/matryoshka
matryoshka index build nyc /path/to/nyc --workers 16
matryoshka index stats nyc
```

The equivalent of the original `offline/main.py` plus its `lakes_config.json`
entry. Archive lakes are handled automatically: a directory of `.tar.gz`, `.tar`
or `.zip` files is indexed archive by archive with a continuing table index; a
directory of tables is indexed in one pass.

Two flags matter for fidelity to the paper's setup:

- `--max-cols 100` (the default) skips tables wider than 100 columns, as the
  published builds did.
- `--update-support` additionally writes the `_meta`, `_num_state` and
  `_cat_state` side tables. They are required by `ExhaustiveIndex.update_table`
  and `find_union_peer`, and were **not** written for the published builds.
  Enabling them roughly doubles the index size.

To resume an interrupted build, pass `--start-table-index N` where `N` is one
past the last `Processed table N` line in the log.

Note that the overlap table is empty by design in the published configuration.
Retrieval runs against the btree on the feature-selection index's `key` column,
and that index embeds the `qcr_term_positive` / `qcr_term_negative` columns used
for pruning. An empty overlap table is not a failed build.

## Reproducing a single query

The paper's Matryoshka configuration for one query table, expressed through the
library API:

```python
import matryoshka as mk

index = mk.LakeIndex('nyc')
augmenter = mk.Augmenter(
    index,
    task='regression',
    top_k=20,
    metric='mse',          # the published proxy; the library default is
    tol=0.001,             # the conditional criterion, see README
    corr_threshold=0.1,    # 0.3 for classification, None for the AutoFeat lakes
    n_jobs=16,
    budget_seconds=1800,
)
result = augmenter.augment(query, key='zip_code', target='total_incident_duration')
```

Discovery is capped at 1800 s per query in the published runs. For the NYC
categorical regime the reported mean discovery time is 77 s; the downstream
random-forest grid search dominates end-to-end wall-clock.

## Headline numbers

For comparison when rerunning:

| Quantity | Value |
|---|---|
| Downstream quality, average over 11 query tables | +19.3 % |
| Random forest, join-key-only features | +36.1 % |
| Random forest, categorical features | +18.5 % |
| Random forest, binned features | +17.9 % |
| Anytime, categorical | mean +18.57 % at 77 s |
| Runtime breakdown | retrieval 43.6 %, sketching 9.8 %, selection 44.4 %, pruning and augmentation < 3 % |

Absolute runtimes depend on the machine. The published figures are from an
EPYC 7702P with 512 GB of RAM. Judge a rerun on the per-step shares and orders
of magnitude rather than on absolute seconds.

## Environment

The published runs used Python 3.12.3, PostgreSQL 18, Neo4j 2026.04.0 (baselines
only) and Ray 2.44.1. The library has since been verified against PostgreSQL 16,
polars 1.43, numpy 2.5, Ray 2.57 and scikit-learn 1.9; see
`docs/MIGRATION.md`. Neo4j is needed only by the AutoFeat, CAAFE and Kitana
baselines, for their join-path discovery.
