# Example: the AutoFeat `covertype` lake

A complete Matryoshka run, from an empty database to a measured improvement in
downstream prediction quality. It is the smallest lake used in the paper, which
makes it the one example that runs end to end on a laptop in minutes rather than
on a server in hours.

## The benchmark

`covertype` comes from the AutoFeat benchmark (Ionescu et al., *AutoFeat:
Transitive Feature Discovery over Join Paths*, ICDE 2024). OpenML dataset 44159
was split into a base table and twelve auxiliary tables linked by synthetic
foreign keys:

```
table_0_0.csv    class, Soil_Type20/32/33/36, Key_0_0     <- the query table
table_1_1..1_3   4 features each, Key_0_0 and Key_1_x     <- one hop away
table_2_4..2_12  4 features each (6 in table_2_12), Key_1_x   <- two hops away
```

All thirteen tables have 423,680 rows. The base table's four `Soil_Type`
indicators carry almost no signal; the informative features live in the lake.
That is the point of the benchmark, and it is what makes the augmentation effect
easy to read off.

Two properties are worth stating because they shape the run:

- **The query table is a member of the lake.** `table_0_0.csv` is indexed along
  with everything else, so it must be excluded from retrieval. Otherwise the
  query table retrieves itself and `class` reappears as a candidate feature.
  The example passes `exclude_tables=['table_0_0.csv']`.
- **All key columns share one value space.** `Key_0_0`, `Key_1_1`, `Key_1_2` and
  `Key_1_3` contain the same 423,680 tokens. Every lake table is therefore
  directly joinable on the query key by value overlap, and single-hop retrieval
  reaches the level-2 tables as well. The "two hops" in the table names describe
  the benchmark's intended join graph, not a constraint on discovery here.

## Running it

```bash
# 1. a database
docker run -d --name matryoshka-pg \
  -e POSTGRES_PASSWORD=matryoshka -e POSTGRES_DB=matryoshka \
  -p 5432:5432 postgres:16
export MATRYOSHKA_DSN=postgresql://postgres:matryoshka@localhost:5432/matryoshka

# 2. the data (212 MB download from Zenodo, ~110 MB on disk)
python examples/covertype/prepare_data.py

# 3. the run
python examples/covertype/run_example.py --jobs 8
```

`prepare_data.py` fetches the archive, verifies its MD5, and prefixes the key
columns with `k`. The published key columns hold bare integers, which are
inferred as numeric rather than as join-key candidates; the paper's runs used the
prefixed form and so does this example.

Add `--sample 50000` to `run_example.py` for a run that finishes in about a
minute. Add `--rebuild` to drop and rebuild the index.

## What it does

1. Indexes the 13-table lake into PostgreSQL, skipping the build if the index
   already exists.
2. Prepares the query table with `mk.prepare_query_table`: normalised string key
   first, numeric null-free features, ordinal-encoded target last.
3. Runs retrieval, correlation pruning and greedy forward selection, excluding
   the query table from the lake.
4. Trains a random forest (100 trees, `min_samples_leaf=2`) on the original and
   on the augmented query table over the same 80/20 split, and reports both.

## Measured results

PostgreSQL 16 in Docker, 48 cores, `--jobs 16`, `top_k=20`, default conditional
Fisher proxy (`metric='conditional_mahalanobis'`, `tol=0.05`).

**Offline.**

| | |
|---|---|
| Tables indexed | 13 |
| Index rows | 6,794,266 |
| Index size | 6.8 GiB |
| Build time, 4 workers | 2 min 40 s |

The index is large relative to the 110 MB lake because every table has 423,680
distinct keys, and the index stores one Gram matrix sketch per key per table.
Index size scales with key cardinality, not with the byte size of the lake.

**Online.** 8 features selected in 323 s from the 12 candidate tables.

```
table_1_2.csv.horizontaldistancetohydrology_min
table_2_8.csv.horizontaldistancetoroadways_min
table_1_2.csv.aspect_min
table_2_8.csv.horizontaldistancetofirepoints_min
table_1_1.csv.hillshade3pm_min
table_1_2.csv.wildernessarea1_median
table_2_7.csv.hillshadenoon_min
table_1_3.csv.soiltype31_median
```

**Downstream**, random forest on a held-out 20 %:

| | features | accuracy | weighted F1 |
|---|---|---|---|
| base | 4 | 0.5048 | 0.4589 |
| augmented | 12 | 0.5940 | 0.5940 |
| **delta** | **+8** | **+0.0892** | **+0.1351 (+29.4 %)** |

For reference, the `--sample 50000` run selects 7 features in 55 s and moves
weighted F1 from 0.3545 to 0.5493.

The selected set spans five distinct lake tables and contains no two aggregates
of the same base column of the same key. Two mechanisms produce that. Once a
candidate is accepted, `_drop_aggregation_siblings` removes every other
aggregate of the same base column from the pool, so `aspect_min` and
`aspect_max` cannot both be selected. Beyond that, the conditional Fisher
criterion scores each remaining candidate *given* what is already selected, so a
correlated feature from another table stops clearing `tol` once its signal is
accounted for.

### Reading the plan

A plan entry is `<lake table>.<column>_<aggregate>`. That name omits the key
column, so the same lake column aggregated over two different key columns of the
same table maps to one name. Those are distinct candidates, and the second
carries a `__<table_index>_<key_col_index>_<feature_index>` suffix. The
50,000-row run shows both forms:

```
table_1_2.csv.aspect_min          # Aspect grouped by one key column of table_1_2
table_1_2.csv.aspect_min__2_5_1   # Aspect grouped by its other key column
```

`table_1_2.csv` carries both `Key_0_0` and `Key_1_2`, and the query key matches
either, because all key columns in this benchmark share one value space.

## Reproducibility

The numbers above come from a seeded run (`--seed 42`) on a freshly cloned
repository and a freshly built index. The index is a deterministic function of
the lake, and selection is deterministic given the index, so the eight selected
features are identical across repeated runs. The downstream scores are not
quite: the random forest is fitted with `n_jobs=-1`, and repeated runs of the
full table were observed to move weighted F1 between 0.5930 and 0.5940. Absolute
runtimes depend on hardware.

## Trying other settings

```bash
# the marginal criterion used for the paper's published numbers
python examples/covertype/run_example.py --sample 50000   # then edit run_example.py:
#   mk.Augmenter(index, task='classification', metric='average_mahalanobis', tol=0.5)

# a looser tolerance accepts more features
#   mk.Augmenter(index, task='classification', tol=0.01)

# consider low-order polynomials of each candidate
#   mk.Augmenter(index, task='classification',
#                params={'polynomial_features': {'enabled': True, 'degree': 2}})
```

## Licence

The `covertype` data is redistributed by the AutoFeat authors under CC BY 4.0
(Zenodo record 12755408). It is downloaded by `prepare_data.py`, not committed
to this repository.
