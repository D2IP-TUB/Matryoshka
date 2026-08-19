# Migration record

The implementation grew inside the research repository that produced the paper.
This document records what changed when it was restructured into an installable
library, so that a reader of the original code can locate anything that moved,
and so the behavioural changes are on the record rather than buried in a diff.

## Package layout

The top-level package was renamed `augmentation` to `matryoshka` and moved under
`src/`. Every module below is unchanged in behaviour unless noted in the next
section.

| Before | After |
|---|---|
| `augmentation/index.py` | `matryoshka/index.py` |
| `augmentation/retrieval.py` | `matryoshka/retrieval.py` (Aurum classes moved out) |
| `augmentation/join_selection.py` | `matryoshka/join_selection.py` |
| `augmentation/planner.py` | `matryoshka/planner.py` |
| `augmentation/join_graph.py` | `matryoshka/join_graph.py` |
| `augmentation/utils/config.py` | `matryoshka/config.py` |
| `augmentation/utils/exceptions.py` | `matryoshka/exceptions.py` |
| `augmentation/utils/database/query_processing.py` | `matryoshka/db/handler.py` |
| `augmentation/utils/database/string_iterator.py` | `matryoshka/db/string_iterator.py` |
| `augmentation/utils/lakes/archives.py` | `matryoshka/lakes/archives.py` |
| `augmentation/utils/lakes/multiprocessing_utils.py` | `matryoshka/lakes/parallel.py` |
| `augmentation/utils/logging/logger_config.py` | `matryoshka/utils/logging.py` |
| `augmentation/utils/{common,f32_numpy,sampling,system_monitoring}.py` | `matryoshka/utils/` |
| `augmentation/feature_selection/**` | `matryoshka/selection/**` |
| `augmentation/feature_selection/algorithms/{greedy,stepwise,lasso}.py` | `matryoshka/selection/algorithms/` |
| `augmentation/feature_selection/algorithms/{arda,autofeat,caafe,kitana,qcr,cocoa,metam}*.py` | `baselines/augmenters/` |
| `augmentation/utils/lakes/{qcr,cocoa}_index.py` | `baselines/augmenters/` |
| `augmentation/{Aurum,Josie,LSH,DeepJoin}/` | `baselines/discovery/` |
| `augmentation/testing/test_*.py` | `tests/` |
| `offline/main.py` + `lakes_config.json` | `matryoshka index build` (`matryoshka/cli.py`) |

New modules with no predecessor: `api.py` (the `LakeIndex` / `Augmenter`
facade), `cli.py`, `preprocessing.py`, `db/settings.py`, `baselines.py` (the
baseline registry), `selection/registry.py` (the model and strategy registry).

## Behavioural changes

These are the changes that alter what the code does, not just where it lives.

**Connection settings are resolved lazily.** `query_processing.py` opened
`db_config.yaml` at module import and raised `FileNotFoundError` if it was
absent, so no module in the package could be imported without a configured
database. Settings now resolve on first construction of a `DBHandler`, through
`matryoshka.db.settings.resolve_settings`, from an explicit argument, then
`MATRYOSHKA_DSN`, then `MATRYOSHKA_DB_*`, then `PG*`, then a YAML file, then
defaults. The original YAML schema still works.

**Baselines are no longer imported eagerly.** `join_selection.py` began with
`from .feature_selection.algorithms.{arda,arda_og,autofeat,autofeat_og,caafe,kitana,qcr} import *`,
so importing Matryoshka imported Neo4j, PyTorch, AutoGluon, an OpenAI client,
and `experiments.base_tables.base_table_preprocessing` from the evaluation
harness. `run_baseline` resolved the strategy class out of the module globals
those star imports had populated. It now calls
`matryoshka.baselines.resolve_baseline`, which imports the adapter by name on
demand. Failure to import one names the missing dependency.

**Proxy models and strategies resolve through a registry.** The same
`globals()[name]` pattern selected proxy models and search strategies. They now
resolve through `matryoshka.selection.registry`, which makes the supported set
introspectable (`available_models`, `available_strategies`), extensible
(`register_model`, `register_strategy`), and keeps optional dependencies
optional: `skglm` and `scikit-optimize` are imported only if an L1 model is
actually selected.

**Logs no longer land inside the package.** `ExhaustiveIndex`, `JoinDiscovery`
and `JoinSelection` defaulted their log directory to
`augmentation/utils/logging/logs`, that is, inside the source tree, which fails
outright for an installed package. The default is now `$MATRYOSHKA_LOG_DIR`, or
`./.matryoshka/logs`, and the directory is created with its parents.

**Table exclusion is by name, not by hardcoded index.** `JoinDiscovery` carried
`self.from_lake = {'trees': 560, 'fire': 550, 'jobs': 11304, 'hospital': 2061}`,
a map of four query-table names to the `table_index` those tables happened to
have in two specific lake builds. It was also unreachable from the forward path,
because `JoinSelection` never passed `table_name` through. Exclusion is now
`exclude_tables=[...]` on `JoinDiscovery`, `JoinSelection` and `Augmenter`,
resolved against `table_name` in the index and cached per instance. This matters
whenever the query table is itself a member of the lake: without it, the table
retrieves itself and its own target reappears as a candidate feature.

**The augmentation plan is returned in readable form.** `find_best_joins`
returned internal `<table>_<key_col>_<feature>` triples such as `2_0_2`, while
the human-readable names (`table_1_2.csv.horizontaldistancetohydrology_min`)
were computed and sent only to the log. The readable names are returned when
available, so the plan matches the columns of the augmented table.

**`table_index` continuation across archives was off by one.**
`ExhaustiveIndex.index_lake` returns the next free table index, but
`offline/main.py` added one to it before the next archive, leaving a gap in
`table_index` for every archive after the first. `LakeIndex.build` continues
from the returned value directly.

**Unimplemented configuration was removed.** `DiscoveryConfig` advertised the
strategy `CofactorLASSO` with model `RegressionCofactorLASSO`; neither class
exists anywhere in the tree. Both were removed from the allowed sets so that a
configuration which validates can also run.

**Metric allowlists were unified.** `DiscoveryConfig` listed metric names that
predated the conditional criteria, so `conditional_mahalanobis` and
`conditional_gcv` were accepted only because the validator did not enforce the
list. The allowlists are now derived from `REGRESSION_METRICS` and
`CLASSIFICATION_METRICS`, which match the dispatch tables in
`selection/base/model.py` and `selection/models.py`.

**A stale unit test was repaired.** `test_index_update_helpers.py` constructed
`ExhaustiveIndex` through `__new__` and set only `feature_extraction`; it had
been failing with `AttributeError: '_median_backend'` since the pluggable median
backend was introduced. It now also sets `_median_backend`.

**`clean_csv_value` escaping.** `value.replace(f'{separator}', f'\{separator}')`
relied on an invalid escape sequence, which Python 3.12 warns about and a future
version will reject. Rewritten as an explicit backslash concatenation; the
output is unchanged.

**A dead module-level side channel was removed.** `find_best_joins` assigned
`globals()['physical_plan']`; nothing read it.

## Code removed

Dead or environment-specific code dropped from the working tree. All of it
remains in the repository's git history.

| Removed | Reason |
|---|---|
| `augmentation/feature_selection/imputation.py` | imports `base.statistics`, a module that no longer exists; nothing imports it; its dispatch point `JoinSelection.run_imputation` is `pass` |
| `augmentation/testing/{algo,data,feature_selection,models,test}.py` | import the same missing modules |
| `backups/` | dated copies of four source files taken mid-refactor in February 2026 |
| `augmentation/feature_selection/algorithms/autofeat_og_utils/` | a 462 MB checkout of the upstream AutoFeat repository, with its own `.git`, datasets and model artefacts |
| `augmentation/Aurum/graphs/`, `augmentation/DeepJoin/*_index/` | 13 GB and 74 GB of generated index artefacts |
| `experiments/{plots,plots_style,utils}.py`, `experiments/run_all.sh` | fragments of an evaluation harness whose other parts are not in the repository; they import `experiments.downstream.*` and hardcode `~/Fast_Data_Discovery` |
| `offline/` | the Docker indexer setup, superseded by `matryoshka index build`; its `lakes_config.json` entries are now CLI arguments |
| `lake_stats.py`, `sample_lakes_analysis.py`, `sampled_lakes_headers.json`, `ai_analysis_results.md`, `reassemble_lfs.sh` | one-off analysis scripts and their outputs |

The `imputation` task remains in `DiscoveryConfig` and in the planner, but has
no implementation behind it. It is documented as such rather than silently
accepted.

## Verified environment

The library was reinstalled from `pyproject.toml` into a clean virtual
environment and exercised end to end against PostgreSQL 16. Resolved versions,
all substantially newer than those the paper's runs used:

| Package | Paper | Verified |
|---|---|---|
| Python | 3.12.3 | 3.12.3 |
| PostgreSQL | 18 | 16.15 |
| polars | 1.36.1 | 1.43.2 |
| numpy | 2.1.3 | 2.5.2 |
| ray | 2.44.1 | 2.57.0 |
| scikit-learn | 1.7.2 | 1.9.0 |
| pyarrow | 20.0.0 | 25.0.1 |
| scipy | 1.16.3 | 1.18.0 |

`connectorx`, reached through `polars.read_database_uri` in the retrieval path,
was missing from every requirements file in the original repository and is now
a declared dependency.

## Known rough edges

- `DiscoveryConfig(task='imputation')` validates but does not run.
- `JoinSelection.find_best_joins` swallows exceptions by default and returns the
  unmodified query table with an empty plan; pass `debug=True` to re-raise.
  `Augmenter.augment` forwards the flag.
- The `find_best_joins` step wiring binds step parameters by matching names
  against `locals()`, which is concise but makes the data flow between planner
  steps hard to follow.
- `polars.from_arrow` on an ADBC stream emits a `FutureWarning` under polars
  1.43; the call site is inside `adbc_driver_manager`, reached through
  `cursor.fetch_arrow`.
