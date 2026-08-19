"""End-to-end Matryoshka run on the AutoFeat ``covertype`` benchmark.

The script executes the full pipeline and reports the downstream effect of the
augmentation:

1. index the 13-table lake into PostgreSQL (skipped when the index exists);
2. prepare the query table into the layout the discovery pipeline expects;
3. run retrieval, correlation pruning and greedy forward selection;
4. train a random forest on the original and on the augmented query table over
   the same train/test split, and report the difference.

Usage::

    export MATRYOSHKA_DSN=postgresql://postgres:matryoshka@localhost:5432/matryoshka
    python examples/covertype/run_example.py --data examples/covertype/data

Add ``--sample 50000`` for a run that finishes in about a minute; the numbers
in the example README are from the full 423,680-row table.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import matryoshka as mk

# The base table of the benchmark. It is also a member of the lake, so it must
# be excluded from retrieval or the query table would rediscover its own target.
QUERY_TABLE = 'table_0_0.csv'
KEY = 'Key_0_0'
TARGET = 'class'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=str(Path(__file__).parent / 'data'),
                        help='directory holding the 13 covertype lake tables')
    parser.add_argument('--index-name', default='covertype',
                        help='lake name; determines the index table names')
    parser.add_argument('--dsn', default=None,
                        help='postgresql:// URI; defaults to MATRYOSHKA_DSN')
    parser.add_argument('--rebuild', action='store_true',
                        help='drop and rebuild the index even if it exists')
    parser.add_argument('--workers', type=int, default=4,
                        help='Ray workers for the offline build (default: 4)')
    parser.add_argument('--jobs', type=int, default=8,
                        help='Ray parallelism for selection (default: 8)')
    parser.add_argument('--top-k', type=int, default=20,
                        help='joinable lake tables retrieved (default: 20)')
    parser.add_argument('--sample', type=int, default=None,
                        help='use only this many query rows, for a quick run')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def evaluate(train: pl.DataFrame, test: pl.DataFrame, target: str, key: str,
             seed: int) -> dict[str, float]:
    """Train a random forest on every column except the key and report test scores."""
    features = [c for c in train.columns if c not in (target, key)]
    model = RandomForestClassifier(
        n_estimators=100, n_jobs=-1, random_state=seed, min_samples_leaf=2
    )
    model.fit(train.select(features).to_numpy(), train.get_column(target).to_numpy())
    predicted = model.predict(test.select(features).to_numpy())
    actual = test.get_column(target).to_numpy()
    return {
        'n_features': len(features),
        'accuracy': accuracy_score(actual, predicted),
        'f1_weighted': f1_score(actual, predicted, average='weighted'),
    }


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data)
    query_path = data_dir / QUERY_TABLE
    if not query_path.is_file():
        raise SystemExit(
            f'{query_path} not found. See examples/covertype/README.md for how '
            f'to obtain the AutoFeat covertype lake.'
        )

    index = mk.LakeIndex(args.index_name, settings=args.dsn)
    print(f'index    {index.feature_table} on {index.settings.host}:'
          f'{index.settings.port}/{index.settings.dbname}')

    # --- 1. offline phase ------------------------------------------------
    if args.rebuild and index.exists():
        index.drop()
    if not index.exists():
        print(f'building index from {data_dir} with {args.workers} workers ...')
        started = time.perf_counter()
        n_tables = index.build(data_dir, max_workers=args.workers)
        print(f'         indexed {n_tables} tables in {time.perf_counter() - started:.0f} s')
    stats = index.stats()
    print(f'         {stats.rows:,} rows over {stats.tables_indexed} tables, '
          f'{stats.total_size}')

    # --- 2. query table --------------------------------------------------
    raw = pl.read_csv(query_path)
    if args.sample:
        raw = raw.sample(n=min(args.sample, raw.height), seed=args.seed)
    query = mk.prepare_query_table(raw, key=KEY, target=TARGET, task='classification')
    print(f'query    {query.height:,} rows, {query.width - 2} features, target {TARGET!r}')

    train, test = mk.train_test_split_by_key(query, test_fraction=0.2, seed=args.seed)

    # --- 3. online phase -------------------------------------------------
    augmenter = mk.Augmenter(
        index,
        task='classification',
        top_k=args.top_k,
        n_jobs=args.jobs,
        exclude_tables=[QUERY_TABLE],
    )
    print(f'augment  {augmenter}')
    result = augmenter.augment(query, key=KEY, target=TARGET, debug=True)
    print(f'         {result.n_selected} features selected in '
          f'{result.runtime_seconds:.0f} s')
    for feature in result.plan:
        print(f'           {feature}')
    if result.n_selected == 0:
        print('no features selected; nothing to evaluate')
        return 1

    # Re-split the augmented table with the same seed, so the comparison below
    # holds the train/test partition fixed and varies only the feature set.
    augmented_train, augmented_test = mk.train_test_split_by_key(
        result.table, test_fraction=0.2, seed=args.seed
    )

    # --- 4. downstream comparison ----------------------------------------
    base = evaluate(train, test, TARGET, KEY, args.seed)
    aug = evaluate(augmented_train, augmented_test, TARGET, KEY, args.seed)
    delta = (aug['f1_weighted'] - base['f1_weighted']) / base['f1_weighted'] * 100

    print()
    print(f'{"":10s} {"features":>9s} {"accuracy":>9s} {"f1":>9s}')
    print(f'{"base":10s} {base["n_features"]:>9d} {base["accuracy"]:>9.4f} '
          f'{base["f1_weighted"]:>9.4f}')
    print(f'{"augmented":10s} {aug["n_features"]:>9d} {aug["accuracy"]:>9.4f} '
          f'{aug["f1_weighted"]:>9.4f}')
    print(f'{"delta":10s} {aug["n_features"] - base["n_features"]:>+9d} '
          f'{aug["accuracy"] - base["accuracy"]:>+9.4f} '
          f'{aug["f1_weighted"] - base["f1_weighted"]:>+9.4f}  ({delta:+.1f}%)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
