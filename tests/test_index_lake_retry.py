"""Recovery of `ExhaustiveIndex.index_lake` from dying Ray workers.

A worker killed during a build (e.g., by the out-of-memory killer) used to fail every table later assigned to it, and
the build still reported success. These tests run the real Ray scheduler with an indexing function that kills its
own worker, and pin that the worker is replaced, the table retried, and tables that keep killing their worker are
reported instead of silently dropped. The database is replaced by in-memory stubs.
"""
import os
import sys
from pathlib import Path

import polars as pl
import pytest

ray = pytest.importorskip('ray')

import matryoshka.index as index_module
from matryoshka.index import ExhaustiveIndex


class _NoPool:
    def __init__(self, *args, **kwargs):
        pass

    def open(self):
        pass

    def close(self):
        pass

    def closeall(self):
        pass


class CrashingIndex(ExhaustiveIndex):
    '''Kills its worker on `crash_always*` tables, and on the first attempt of `crash_once*` tables.'''

    def db_connect_pool(self):
        return _NoPool()

    def create_tables(self, conn, include_state_tables=False):
        pass

    def _flush_index_data(self, index_data, results, flush_num, flush_start, conn_pool=None):
        self.flushed.extend(pl.concat(index_data).get_column('table').to_list())

    def index_table(self, table, table_index, table_name=None, key_columns=None, debug=False):
        if table_name.startswith('crash_always'):
            os._exit(1)
        if table_name.startswith('crash_once'):
            marker = Path(self.marker_dir) / table_name
            if not marker.exists():
                marker.touch()
                os._exit(1)
        return [pl.DataFrame({'table': [table_name]})]


@pytest.fixture
def lake(tmp_path):
    directory = tmp_path / 'lake'
    directory.mkdir()
    for name in ['a', 'b', 'crash_always', 'crash_once', 'c', 'd']:
        pl.DataFrame({'key': ['x', 'y'], 'value': [1.0, 2.0]}).write_parquet(directory / f'{name}.parquet')
    return directory


def test_dead_workers_are_replaced_and_their_tables_retried_or_reported(lake, tmp_path, monkeypatch):
    # Ship the test classes to the workers by value: the test module is not importable there
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    monkeypatch.setattr(index_module, 'ConnectionPool', _NoPool)
    index = CrashingIndex(
        data_dir=str(lake), feature_selection_table_name='unused', overlap_table_name='unused_overlap',
        max_workers=2, batch_size=1, settings='postgresql://user:password@localhost:5432/unused',
        log_dir=str(tmp_path / 'logs'), silent=True, max_table_attempts=2,
    )
    index.flushed = []
    index.marker_dir = str(tmp_path)
    next_index = index.index_lake(0)

    assert next_index == 6
    assert sorted(index.flushed) == ['a.parquet', 'b.parquet', 'c.parquet', 'crash_once.parquet', 'd.parquet']
    assert [(name, reason.startswith('worker failed 2 time(s)')) for _, name, reason in index.failed_tables] == [('crash_always.parquet', True)]


def test_workers_are_replaced_while_the_connection_pool_is_open(tmp_path):
    import threading

    index = ExhaustiveIndex(
        feature_selection_table_name='unused', overlap_table_name='unused_overlap', max_workers=2,
        settings='postgresql://user:password@localhost:5432/unused', log_dir=str(tmp_path / 'logs'), silent=True,
    )
    pool = threading.Lock()  # unpicklable, like the open connection pool
    index._conn_pool = pool
    dead_worker = object()
    processors = [dead_worker]
    ray.init(num_cpus=1, include_dashboard=False)
    try:
        index._replace_worker(processors, 0, dead_worker)
        assert processors[0] is not dead_worker
        assert index._conn_pool is pool
        # a later failure of another task of the same (already replaced) worker leaves the new worker in place
        replacement = processors[0]
        index._replace_worker(processors, 0, dead_worker)
        assert processors[0] is replacement
    finally:
        ray.shutdown()


def test_max_table_attempts_must_be_positive():
    with pytest.raises(ValueError, match='max_table_attempts'):
        ExhaustiveIndex(max_table_attempts=0, settings='postgresql://user:password@localhost:5432/unused')
