"""Command-line interface.

Three command groups mirror the library API:

``matryoshka index``
    build, inspect and drop the PostgreSQL index over a data lake;
``matryoshka augment``
    run discovery and feature selection for one query table;
``matryoshka info``
    report the resolved connection settings and the registered models,
    strategies and baselines.

Connection settings resolve as described in :mod:`matryoshka.db.settings`;
``--dsn`` overrides them for a single invocation.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__


def _add_dsn(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--dsn',
        default=None,
        metavar='URI',
        help='postgresql://user:password@host:port/dbname; overrides the '
             'environment and any db_config.yaml',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='matryoshka',
        description='Discover and select relevant features from a data lake.',
    )
    parser.add_argument('--version', action='version', version=f'matryoshka {__version__}')
    sub = parser.add_subparsers(dest='command', required=True)

    # -- index ------------------------------------------------------------
    index = sub.add_parser('index', help='manage the lake index')
    index_sub = index.add_subparsers(dest='index_command', required=True)

    build = index_sub.add_parser('build', help='index a data lake into PostgreSQL')
    build.add_argument('name', help='lake name; determines the index table names')
    build.add_argument('path', help='directory of tables, or of archives')
    build.add_argument('--workers', type=int, default=1,
                       help='Ray workers used for table processing (default: 1)')
    build.add_argument('--batch-size', type=int, default=1,
                       help='tables completed before flushing to PostgreSQL (default: 1)')
    build.add_argument('--max-cols', type=int, default=100,
                       help='skip lake tables wider than this (default: 100)')
    build.add_argument('--start-table-index', type=int, default=0,
                       help='first table index to assign; use to resume a build')
    build.add_argument('--update-support', action='store_true',
                       help='also write the per-table state tables needed by '
                            'update_table() and find_union_peer()')
    build.add_argument('--feature-extraction', action='store_true',
                       help='enable the extended per-key aggregate set')
    build.add_argument('--no-btree', action='store_true',
                       help='skip btree index creation; retrieval will be slow '
                            'until it is run')
    _add_dsn(build)

    stats = index_sub.add_parser('stats', help='report row counts and size')
    stats.add_argument('name')
    stats.add_argument('--tables', action='store_true',
                       help='also list every indexed lake table')
    _add_dsn(stats)

    drop = index_sub.add_parser('drop', help='drop every table of an index')
    drop.add_argument('name')
    drop.add_argument('--yes', action='store_true', help='do not prompt for confirmation')
    _add_dsn(drop)

    # -- augment ----------------------------------------------------------
    augment = sub.add_parser('augment', help='augment a query table from the lake')
    augment.add_argument('index_name', help='lake name used when the index was built')
    augment.add_argument('query', help='CSV file holding the query table')
    augment.add_argument('--key', required=True, help='join column')
    augment.add_argument('--target', required=True, help='prediction target column')
    augment.add_argument('--task', choices=['classification', 'regression'],
                         default='classification')
    augment.add_argument('--top-k', type=int, default=20,
                         help='joinable lake tables retrieved (default: 20)')
    augment.add_argument('--metric', default=None,
                         help='proxy scoring criterion; defaults to the '
                              'conditional criterion for the task')
    augment.add_argument('--tol', type=float, default=None,
                         help='minimum relative improvement for a candidate to be kept')
    augment.add_argument('--corr-threshold', type=float, default=None,
                         help='correlation pruning threshold; omit for the task default')
    augment.add_argument('--no-pruning', action='store_true',
                         help='disable correlation-based pruning')
    augment.add_argument('--budget-seconds', type=float, default=None,
                         help='wall-clock cap on discovery')
    augment.add_argument('--jobs', type=int, default=1, help='Ray parallelism (default: 1)')
    augment.add_argument('--raw', action='store_true',
                         help='run prepare_query_table() on the CSV first')
    augment.add_argument('-o', '--output', default=None, help='write the augmented table here')
    augment.add_argument('--verbose', action='store_true')
    _add_dsn(augment)

    # -- info -------------------------------------------------------------
    info = sub.add_parser('info', help='report configuration and registries')
    _add_dsn(info)
    return parser


def _cmd_index_build(args) -> int:
    from .api import LakeIndex

    index = LakeIndex(args.name, settings=args.dsn)
    print(f'building {index.feature_table} from {args.path}', file=sys.stderr)
    n = index.build(
        args.path,
        max_workers=args.workers,
        batch_size=args.batch_size,
        max_cols=args.max_cols,
        feature_extraction=args.feature_extraction,
        update_support=args.update_support,
        start_table_index=args.start_table_index,
        create_btree_indexes=not args.no_btree,
    )
    stats = index.stats()
    print(f'indexed {n} tables; {stats.rows} index rows over '
          f'{stats.tables_indexed} tables, {stats.total_size}')
    return 0


def _cmd_index_stats(args) -> int:
    from .api import LakeIndex

    index = LakeIndex(args.name, settings=args.dsn)
    stats = index.stats()
    print(f'feature table   {stats.feature_table}')
    print(f'overlap table   {stats.overlap_table}')
    print(f'index rows      {stats.rows}')
    print(f'tables indexed  {stats.tables_indexed}')
    print(f'total size      {stats.total_size}')
    if args.tables:
        print()
        for row in index.indexed_tables().iter_rows():
            print(f'  {row[0]:>6}  {row[1]}')
    return 0


def _cmd_index_drop(args) -> int:
    from .api import LakeIndex

    index = LakeIndex(args.name, settings=args.dsn)
    if not args.yes:
        answer = input(f'drop all tables of index {args.name!r} in '
                       f'{index.settings.dbname!r}? [y/N] ')
        if answer.strip().lower() not in ('y', 'yes'):
            print('aborted', file=sys.stderr)
            return 1
    index.drop()
    print(f'dropped {args.name}')
    return 0


def _cmd_augment(args) -> int:
    import polars as pl

    from .api import Augmenter, LakeIndex
    from .preprocessing import prepare_query_table

    query = pl.read_csv(args.query, ignore_errors=True)
    if args.raw:
        query = prepare_query_table(query, key=args.key, target=args.target, task=args.task)

    corr_threshold = None if args.no_pruning else args.corr_threshold
    index = LakeIndex(args.index_name, settings=args.dsn)
    augmenter = Augmenter(
        index,
        task=args.task,
        top_k=args.top_k,
        metric=args.metric,
        tol=args.tol,
        corr_threshold=corr_threshold if (args.no_pruning or args.corr_threshold is not None) else ...,
        n_jobs=args.jobs,
        budget_seconds=args.budget_seconds,
        verbose=args.verbose,
    )
    result = augmenter.augment(query, key=args.key, target=args.target)
    print(f'selected {result.n_selected} features in {result.runtime_seconds:.1f} s; '
          f'query table {query.shape} -> {result.table.shape}', file=sys.stderr)
    for feature in result.plan:
        print(f'  {feature}')
    if args.output:
        result.table.write_csv(args.output)
        print(f'wrote {args.output}', file=sys.stderr)
    return 0


def _cmd_info(args) -> int:
    from .baselines import available_baselines
    from .config import CLASSIFICATION_METRICS, REGRESSION_METRICS
    from .db.settings import resolve_settings
    from .selection.registry import available_models, available_strategies

    settings = resolve_settings(args.dsn)
    print(f'matryoshka {__version__}')
    print(f'database          {settings.user}@{settings.host}:{settings.port}/{settings.dbname}')
    print(f'strategies        {", ".join(available_strategies())}')
    print(f'proxy models      {", ".join(available_models())}')
    print(f'regression metrics      {", ".join(REGRESSION_METRICS)}')
    print(f'classification metrics  {", ".join(CLASSIFICATION_METRICS)}')
    print(f'baselines         {", ".join(available_baselines())}')
    return 0


_DISPATCH = {
    ('index', 'build'): _cmd_index_build,
    ('index', 'stats'): _cmd_index_stats,
    ('index', 'drop'): _cmd_index_drop,
    ('augment', None): _cmd_augment,
    ('info', None): _cmd_info,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    key = (args.command, getattr(args, 'index_command', None))
    return _DISPATCH[key](args)


if __name__ == '__main__':
    raise SystemExit(main())
