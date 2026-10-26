# Paper benchmark configurations

The queries, Matryoshka settings and evaluation protocol of the experiments in
Section 7 of the paper.

| File | Content |
|---|---|
| `itw_queries.json` | The 11 in-the-wild (ITW) query tables of Table 2: lake, task, join key, target, categorical and numeric features. From-Lake queries list their lake copy under `exclude_tables`. |
| `autofeat_queries.json` | The base tables of the eight AutoFeat lakes. |
| `kitana_loo_benchmark.json` | The Kitana leave-one-out benchmark: 309 lake tables, of which the 258 with `skip: false` and a target are queries. |
| `settings.json` | Matryoshka settings per experiment (top-k, pruning threshold, proxy criterion, tolerance, budget) and the evaluation protocol. |
| `run_query.py` | Runs one query with these settings and scores it with the paper's protocol. |

## Running a query

The evaluation protocol lives in `matryoshka.evaluation` and needs the `paper`
extra:

```bash
pip install -e ".[paper]"
matryoshka index build cuk path/to/cuk_lake --workers 16
python benchmarks/paper/run_query.py jobs --query-dir path/to/query_tables \
    --index-name cuk --setup categorical
```

`--setup` picks one of the feature setups of Table 3 (`join_key`, `categorical`,
`binned`). The script prepares the query table with
`prepare_query_table_paper`, augments it, and scores the base and the augmented
table with `evaluate_paper_protocol`: imputation, the target-leakage filter
and a tuned random forest under 5-fold cross-validation.

## Settings that differ from the package defaults

`Augmenter` defaults to the conditional criteria (`conditional_gcv`,
`conditional_mahalanobis`, tolerance 0.05). The ITW and leave-one-out numbers
use the marginal ones, `mse` and `average_mahalanobis`; `settings.json` lists
them with their tolerances. The AutoFeat lakes run without correlation
pruning.
