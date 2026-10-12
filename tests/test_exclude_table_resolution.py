"""Resolving the lake tables excluded from an overlap query.

``table_name`` is not indexed, so matching it in SQL scans the whole index --
309 s on the 2,600-table open-data lake, paid by every run before a single
candidate is retrieved, to resolve the query table that leave-one-out
evaluation excludes. The resolution reads the (``table_index``, ``table_name``)
pairs through the ``table_index`` btree instead and matches names in Python,
which is the same answer in under a second. The tests pin the answer and the
property that makes it cheap.
"""
import polars as pl
import pytest

from matryoshka import retrieval
from matryoshka.retrieval import JoinDiscovery


INDEXED = {
    'covertype.csv': 1,
    'CAN_CSV0000000000009854.parquet': 2,
    'no_extension': 3,
    'wdc-1438042988250.59-6.parquet': 4,
}


@pytest.fixture
def discovery(monkeypatch):
    """A `JoinDiscovery` with its DB reads faked, built without touching one."""
    instance = JoinDiscovery.__new__(JoinDiscovery)
    instance.feature_selection_table_name = 'matryoshka_index_open'
    instance.conninfo = 'postgresql://user:pw@localhost:5432/beluga'
    instance._excluded_table_names = set()
    instance._excluded_index_cache = {}
    instance._table_index_name_cache = None

    queries = []

    def fake_read(query, conninfo):
        queries.append(query)
        return pl.DataFrame({
            'table_index': list(INDEXED.values()),
            'table_name': list(INDEXED.keys()),
        })

    monkeypatch.setattr(retrieval.pl, 'read_database_uri', fake_read)
    instance.queries = queries

    return instance


@pytest.mark.parametrize('name, expected', [
    ('covertype.csv', 1),                                 # exactly as indexed
    ('covertype', 1),                                     # without the extension
    ('/lake/open_data/covertype.csv', 1),                 # full path
    ('/lake/covertype', 1),                               # path, no extension
    ('CAN_CSV0000000000009854', 2),                        # the stem callers pass
    ('no_extension', 3),                                  # indexed without one
    ('wdc-1438042988250.59-6', 4),                         # dots inside the name
])
def test_resolves_the_name_variants(discovery, name, expected):
    assert discovery._resolve_excluded_indices([name]) == [expected]


def test_unindexed_names_are_ignored(discovery):
    # Excluding a table that was never indexed is a no-op, not an error.
    assert discovery._resolve_excluded_indices(['us_police_incidents']) == []
    assert discovery._resolve_excluded_indices(['us_police_incidents', 'covertype']) == [1]


def test_result_is_sorted_and_deduplicated(discovery):
    names = ['CAN_CSV0000000000009854.parquet', 'covertype', '/lake/covertype.csv', 'no_extension']
    assert discovery._resolve_excluded_indices(names) == [1, 2, 3]


def test_the_index_is_read_once_for_every_exclusion(discovery):
    discovery._resolve_excluded_indices(['covertype'])
    discovery._resolve_excluded_indices(['no_extension'])
    discovery._resolve_excluded_indices(['never_indexed'])
    assert len(discovery.queries) == 1


def test_nothing_to_resolve_reads_nothing(discovery):
    assert discovery._resolve_excluded_indices([]) == []
    assert discovery.queries == []


def test_the_scan_goes_through_table_index(discovery):
    discovery._resolve_excluded_indices(['covertype'])
    query = discovery.queries[0].lower()
    # The cheap path: driven by the indexed column, with no predicate or
    # expression on the unindexed `table_name`.
    assert 'table_index' in query
    assert 'where table_name' not in query
    assert 'table_name in' not in query
    assert 'regexp_replace' not in query


def test_a_failing_read_resolves_nothing(monkeypatch, discovery):
    def fail(query, conninfo):
        raise RuntimeError('connection refused')

    monkeypatch.setattr(retrieval.pl, 'read_database_uri', fail)
    discovery._table_index_name_cache = None
    assert discovery._resolve_excluded_indices(['covertype']) == []
