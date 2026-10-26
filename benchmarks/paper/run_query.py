"""Run one of the paper's queries with the paper's settings and protocol.

Reads the query from ``itw_queries.json`` or ``autofeat_queries.json`` and the
Matryoshka settings from ``settings.json``, prepares the query table under one
of the feature setups, augments it from an existing lake index, and scores the
base and the augmented table with the paper's evaluation protocol
(:mod:`matryoshka.evaluation`, installed with ``pip install -e ".[paper]"``).

Usage::

    export MATRYOSHKA_DSN=postgresql://postgres:matryoshka@localhost:5432/matryoshka
    python benchmarks/paper/run_query.py jobs --query-dir path/to/query_tables \\
        --index-name cuk --setup categorical

``--query-dir`` holds the query tables as ``<name>.csv``. The index is the one
built from the query's lake with ``matryoshka index build``; From-Lake queries
exclude their lake copy as listed in ``exclude_tables``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

import matryoshka as mk
from matryoshka.evaluation import PAPER_SETUPS, evaluate_paper_protocol, prepare_query_table_paper

HERE = Path(__file__).parent
EXPERIMENTS = {'itw': 'itw_queries.json', 'autofeat': 'autofeat_queries.json'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('query', help='query name, e.g. jobs or covertype')
    parser.add_argument('--experiment', choices=sorted(EXPERIMENTS), default='itw')
    parser.add_argument('--query-dir', required=True, help='directory holding <query>.csv')
    parser.add_argument('--index-name', required=True, help='lake name of the index to query')
    parser.add_argument('--setup', choices=PAPER_SETUPS, default='categorical')
    parser.add_argument('--dsn', default=None, help='postgresql:// URI; defaults to MATRYOSHKA_DSN')
    parser.add_argument('--jobs', type=int, default=None, help='Ray parallelism; defaults to the paper setting')
    parser.add_argument('--no-leakage-filter', action='store_true',
                        help='score without the target-leakage filter')
    parser.add_argument('--no-intelex', action='store_true',
                        help="score with scikit-learn's random forests instead of scikit-learn-intelex's")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queries = json.loads((HERE / EXPERIMENTS[args.experiment]).read_text())['queries']
    spec = next((q for q in queries if q['name'] == args.query), None)
    if spec is None:
        raise SystemExit(f'unknown query {args.query!r}; known: {", ".join(q["name"] for q in queries)}')
    settings = json.loads((HERE / 'settings.json').read_text())[args.experiment]
    task, key, target = spec['task'], spec['key'], spec['target']

    raw = pl.read_csv(Path(args.query_dir) / spec['query_table'], ignore_errors=True)
    query = prepare_query_table_paper(raw, key, target, categorical_features=spec['categorical_features'],
                                      numeric_features=spec['numeric_features'], task=task, setup=args.setup)
    print(f'query    {spec["dataset"]}: {query.height:,} rows, {query.width - 2} features, setup {args.setup}')

    kwargs = {**settings['common'], **settings[task]}
    if args.jobs is not None:
        kwargs['n_jobs'] = args.jobs
    augmenter = mk.Augmenter(mk.LakeIndex(args.index_name, settings=args.dsn), task=task,
                             exclude_tables=spec.get('exclude_tables'), **kwargs)
    print(f'augment  {augmenter}')
    result = augmenter.augment(query, key=key, target=target, debug=True)
    print(f'         {result.n_selected} features selected in {result.runtime_seconds:.0f} s')
    for feature in result.plan:
        print(f'           {feature}')

    protocol = {'leakage_filter': not args.no_leakage_filter, 'intelex': not args.no_intelex}
    base = evaluate_paper_protocol(query, key, target, task, **protocol)
    augmented = evaluate_paper_protocol(result.table, key, target, task, **protocol)
    if augmented['leaky_columns']:
        print(f'leakage  dropped {augmented["leaky_columns"]}')
    if task == 'regression':
        delta = (base['rmse'] - augmented['rmse']) / base['rmse'] * 100
        print(f'RMSE     base {base["rmse"]:.4f}  augmented {augmented["rmse"]:.4f}  improvement {delta:+.1f}%')
    else:
        metric = 'f1_weighted' if 'f1_weighted' in base else 'f1'
        delta = (augmented[metric] - base[metric]) / abs(base[metric]) * 100
        print(f'F1       base {base[metric]:.4f}  augmented {augmented[metric]:.4f}  improvement {delta:+.1f}%')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
