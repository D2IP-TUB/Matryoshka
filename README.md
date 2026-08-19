# Matryoshka

Matryoshka discovers and selects relevant features from a data lake to augment a
query table for a downstream machine learning task. It replaces join
materialisation with compact Gram matrix sketches and selects features by
incrementally fitting linear proxy models over those sketches, so the cost of
evaluating a candidate feature set does not grow with the number of rows in the
joined relations.

This repository holds the reference implementation of

> F. Turchenko, R. Zhang, B. Chen, M. Boehm, B. Salimi, A. Shaikhha and
> Z. Abedjan. **Matryoshka: Uncovering Relevant Features in Data Lakes to
> Enhance Machine Learning Applications.** PVLDB, Vol. 19, 2026.

Reported over eleven query tables and three data lakes, Matryoshka improves
downstream prediction quality by 19.3 % on average while achieving the lowest
geometric mean runtime and up to 120x faster execution on join-intensive
workloads.

---

## How it works

Matryoshka runs in two phases.

**Offline.** Each lake table is scanned once. Columns unlikely to serve as join
keys or as features are pruned. For every remaining key candidate `k`, the table
is aggregated with `GROUP BY k` using max/min/mean/median over numeric columns
and counts over low-cardinality categorical columns. Pre-aggregating this way
makes augmentation cardinality preserving: the query table gains columns, never
rows. For each aggregated table a Gram matrix sketch is computed, holding the
per-key count, column sums, diagonal, cofactor terms, and a QCR term used for
correlation-based pruning. Sketches and an inverted index from cell value to
(table, column, row) are stored in one PostgreSQL relation, so a single
structure answers both index queries: a grouped aggregation for joinable-table
discovery and an indexed lookup for sketch retrieval.

**Online.** Given a query table `Q`, a join column `k` and a target `y`,
Matryoshka looks up the values of `k` in the inverted index and takes the top-*k*
lake tables ranked by join-key coverage. It retrieves their sketches and
assembles the candidate feature pool. Correlation-based pruning drops candidates
with low predictive signal. Greedy forward selection then builds the feature set
`F*`, joining sketches and incrementally refitting a linear proxy — ordinary
least squares for regression, linear discriminant analysis for classification —
and stops when no remaining candidate improves the criterion by more than `tol`.
The selected features are materialised into `Q ⊕ F*`.

The proxy only has to *rank* candidates, not predict the target, which is a far
weaker requirement than fitting it accurately. This is why the compact,
redundancy-free feature sets it selects transfer to non-linear downstream models.
Purely non-linear signal is the one blind spot: low-order polynomial sketches
capture the univariate case, but richer non-linear interactions do not reduce to
the second-order statistics the sketches store.

## Installation

Matryoshka needs Python 3.11 or newer and a reachable PostgreSQL instance
(version 14 or newer).

```bash
git clone https://github.com/D2IP-TUB/Matryoshka.git
cd Matryoshka
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Optional extras: `l1` for the L1 proxy models, `stepwise` for the batched
incremental selector, `ssh` to reach the database over a tunnel, `dev` for the
test suite.

A throwaway database for trying the system out:

```bash
docker run -d --name matryoshka-pg \
  -e POSTGRES_PASSWORD=matryoshka -e POSTGRES_DB=matryoshka \
  -p 5432:5432 postgres:16
export MATRYOSHKA_DSN=postgresql://postgres:matryoshka@localhost:5432/matryoshka
```

Connection settings are resolved lazily, in this order: an explicit argument,
`MATRYOSHKA_DSN`, the `MATRYOSHKA_DB_*` variables, the standard `PG*` variables,
a `db_config.yaml` file, then defaults. Nothing is read at import time, so
`import matryoshka` works on a machine with no database configured. Run
`matryoshka info` to see what resolves.

## Quick start

```python
import polars as pl
import matryoshka as mk

# --- offline: index the lake once ---
index = mk.LakeIndex('covertype')
index.build('examples/covertype/data', max_workers=4)
print(index.stats())

# --- online: augment a query table ---
raw = pl.read_csv('examples/covertype/data/table_0_0.csv')
query = mk.prepare_query_table(raw, key='Key_0_0', target='class',
                               task='classification')

augmenter = mk.Augmenter(index, task='classification', top_k=20, n_jobs=8)
result = augmenter.augment(query, key='Key_0_0', target='class')

print(result.n_selected, 'features in', round(result.runtime_seconds), 's')
print(result.plan)          # ['table_1_2.csv.horizontaldistancetohydrology_min', ...]
print(result.table.shape)   # the augmented query table
```

The same pipeline from the shell:

```bash
matryoshka index build covertype examples/covertype/data --workers 4
matryoshka index stats covertype
matryoshka augment covertype query.csv --key Key_0_0 --target class \
    --task classification --raw -o augmented.csv
```

### The query table contract

The discovery pipeline builds the query table's Gram matrix sketch directly from
its columns, so it requires a specific layout: the join key first, as a `String`
column normalised with `process_key`; numeric, null-free feature columns next;
and the target last, ordinal-encoded for classification.
`mk.prepare_query_table` produces that layout from an arbitrary table using
ordinal encoding and median/mode imputation. Any encoder that yields numeric,
null-free columns is a valid substitute — the paper's experiments use a heavier
AutoGluon-based pipeline. Passing a table in any other shape raises
`UserTableNotProcessed`.

**If the query table is itself a member of the lake**, pass
`exclude_tables=['its_name.csv']` to `Augmenter`. Otherwise it retrieves itself
and its own target reappears as a candidate feature.

## Configuration

`Augmenter` covers the common case; `DiscoveryConfig` and `JoinSelection` expose
the full surface.

| Parameter | Default | Effect |
|---|---|---|
| `task` | `classification` | Selects the proxy model and default criterion. |
| `top_k` | 20 | Joinable lake tables retrieved per query. |
| `metric` | `conditional_mahalanobis` / `conditional_gcv` | Scoring criterion. |
| `tol` | 0.05 | Minimum relative improvement for a candidate to be accepted. |
| `corr_threshold` | 0.3 / 0.1 | Correlation pruning threshold; `None` disables pruning. |
| `budget_seconds` | `None` | Wall-clock cap on discovery, for anytime operation. |
| `exclude_tables` | `()` | Lake tables never retrieved. |
| `n_jobs` | 1 | Ray parallelism for candidate evaluation. |

The defaults are the **conditional** criteria, which score a candidate *given*
the already-selected features and so suppress redundant additions more
aggressively. The paper's published numbers use the marginal criteria:
`metric='average_mahalanobis', tol=0.5` for classification and
`metric='mse', tol=0.001` for regression.

Supported criteria are listed in `mk.REGRESSION_METRICS` and
`mk.CLASSIFICATION_METRICS`. Extra behaviour goes through `params`, for example
`params={'polynomial_features': {'enabled': True, 'degree': 2}}` to consider
low-order polynomials of each candidate, `{'dummify_features': ...}` for
quantile-bin indicators, or `{'n_hops': 2}` for multi-hop join graphs.

## Repository layout

```
src/matryoshka/          the installable library
  api.py                 LakeIndex and Augmenter, the high-level facade
  index.py               offline indexing (ExhaustiveIndex)
  retrieval.py           joinable-table discovery (JoinDiscovery)
  join_selection.py      online orchestration and augmentation (JoinSelection)
  planner.py             logical and physical plan construction
  preprocessing.py       query table preparation
  selection/             sketch processing, proxy models, search algorithms
  db/                    PostgreSQL schema and lazy connection settings
  lakes/                 archive readers and parallel table processors
examples/covertype/      a complete, runnable example
baselines/               competing systems, for reproducing the evaluation
tests/                   unit tests that need no database
docs/                    design notes and the migration record
```

## Example

`examples/covertype/` runs the whole pipeline on the AutoFeat `covertype`
benchmark — a 13-table lake of 423,680 rows each — and reports the downstream
effect of the augmentation. See its README for numbers and instructions.

## Baselines

`baselines/` holds the augmenters and discovery indexes Matryoshka is compared
against: ARDA, AutoFeat, CAAFE, Kitana, QCR, COCOA, Metam, and the Aurum, JOSIE,
LSH and DeepJoin indexes. They sit outside the installed package because they
pull in Neo4j, PyTorch, AutoGluon and an OpenAI client, none of which Matryoshka
itself needs. They are reached lazily by name:

```python
mk.available_baselines()          # names
mk.resolve_baseline('ArdaAugmenter')   # imports on demand
```

See `baselines/README.md` for provenance and dependencies.

## Reproducing the paper

The full evaluation harness, the lakes, and the pre-built indexes are not part
of this repository; several of the indexes are in the terabyte range. See
`docs/REPRODUCIBILITY.md` for what is required and what is available.

## Development

```bash
pip install -e ".[dev]"
pytest                   # unit tests, no database required
ruff check src tests
```

## Citation

```bibtex
@article{matryoshka2026,
  title   = {Matryoshka: Uncovering Relevant Features in Data Lakes to
             Enhance Machine Learning Applications},
  author  = {Turchenko, Fedor and Zhang, Runjie and Chen, Binger and
             Boehm, Matthias and Salimi, Babak and Shaikhha, Amir and
             Abedjan, Ziawasch},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {19},
  year    = {2026}
}
```
