"""Resolving the lake tables excluded from an overlap query.

``table_name`` is not indexed, so matching it in SQL scans the whole index --
309 s on the 2,600-table open-data lake, paid by every run before a single
candidate is retrieved, to resolve the query table that leave-one-out
evaluation excludes. The resolution reads the (``table_index``, ``table_name``)
pairs through the ``table_index`` btree instead and matches names in Python,
which is the same answer in under a second. The tests pin the answer and the
property that makes it cheap.
"""
import warnings
from pathlib import Path

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


def test_unindexed_names_are_ignored_with_a_warning(discovery):
    # Excluding a table that was never indexed is a no-op, not an error, but
    # a misspelt name must not silently let the query table retrieve itself.
    with pytest.warns(RuntimeWarning, match='us_police_incidents'):
        assert discovery._resolve_excluded_indices(['us_police_incidents']) == []
    assert discovery._resolve_excluded_indices(['us_police_incidents', 'covertype']) == [1]


def test_result_is_sorted_and_deduplicated(discovery):
    names = ['CAN_CSV0000000000009854.parquet', 'covertype', '/lake/covertype.csv', 'no_extension']
    assert discovery._resolve_excluded_indices(names) == [1, 2, 3]


def test_the_index_is_read_once_for_every_exclusion(discovery):
    discovery._resolve_excluded_indices(['covertype'])
    discovery._resolve_excluded_indices(['no_extension'])
    with pytest.warns(RuntimeWarning):
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
    with pytest.warns(RuntimeWarning):
        assert discovery._resolve_excluded_indices(['covertype']) == []


def _fake_index(monkeypatch, discovery, fs_rows, meta_rows=None):
    """Serve `fs_rows` for the index scan and `meta_rows` for `<index>_meta`."""
    def fake_read(query, conninfo):
        discovery.queries.append(query)
        rows = meta_rows if '_meta' in query else fs_rows
        if rows is None:
            raise RuntimeError('relation does not exist')
        return pl.DataFrame({'table_index': list(rows.values()), 'table_name': list(rows.keys())},
                            schema={'table_index': pl.Int64, 'table_name': pl.String})

    monkeypatch.setattr(retrieval.pl, 'read_database_uri', fake_read)
    discovery._table_index_name_cache = None


ARCHIVED = {
    'datasets_UK/UK_CSV0000000000000890.csv': 10236,
    'datasets_CAN/CAN_CSV0000000000000890.csv': 5391,
    'datasets_USA/USA_CSV0000000000033613.csv': 3022,
}


@pytest.mark.parametrize('name, expected', [
    ('UK_CSV0000000000000890.csv', [10236]),                  # the file name callers pass
    ('UK_CSV0000000000000890', [10236]),                      # its stem
    ('datasets_UK/UK_CSV0000000000000890.csv', [10236]),      # the member path as stored
    ('/lakes/cuk/datasets_USA/USA_CSV0000000000033613.csv', [3022]),  # a path into the extracted lake
])
def test_resolves_tables_stored_under_their_archive_member_path(monkeypatch, discovery, name, expected):
    _fake_index(monkeypatch, discovery, ARCHIVED)
    assert discovery._resolve_excluded_indices([name]) == expected


def test_a_basename_shared_by_several_tables_excludes_all_of_them(monkeypatch, discovery):
    shared = {'2019/report.csv': 1, '2020/report.csv': 2, 'other.csv': 3}
    _fake_index(monkeypatch, discovery, shared)
    assert discovery._resolve_excluded_indices(['report.csv']) == [1, 2]
    # An exact stored name still picks out the one table.
    assert discovery._resolve_excluded_indices(['2020/report.csv']) == [2]


def test_falls_back_to_the_meta_table_when_the_index_has_no_names(monkeypatch, discovery):
    # Indexes written before `table_name` was a column: the scan fails.
    _fake_index(monkeypatch, discovery, None, meta_rows=ARCHIVED)
    assert discovery._resolve_excluded_indices(['UK_CSV0000000000000890.csv']) == [10236]
    assert any('_meta' in q for q in discovery.queries)


def test_falls_back_to_the_meta_table_when_the_names_are_null(monkeypatch, discovery):
    _fake_index(monkeypatch, discovery, {None: 1}, meta_rows={'covertype.csv': 1})
    assert discovery._resolve_excluded_indices(['covertype']) == [1]


def test_the_meta_table_is_not_read_when_the_index_has_names(monkeypatch, discovery):
    _fake_index(monkeypatch, discovery, ARCHIVED, meta_rows=ARCHIVED)
    discovery._resolve_excluded_indices(['UK_CSV0000000000000890'])
    assert not any('_meta' in q for q in discovery.queries)


@pytest.mark.parametrize('name', ['tm6d-hbzd', 'tm6d-hbzd.tsv', 'tm6d-hbzd.tsv.gz', 'tsv/tm6d-hbzd.tsv.gz'])
def test_resolves_tables_stored_compressed_in_a_tar(monkeypatch, discovery, name):
    _fake_index(monkeypatch, discovery, {'tsv/tm6d-hbzd.tsv.gz': 260, 'tsv/kyad-zm4j.tsv.gz': 560})
    assert discovery._resolve_excluded_indices([name]) == [260]


def test_a_dotted_name_keeps_its_dots_after_a_compression_suffix(monkeypatch, discovery):
    _fake_index(monkeypatch, discovery, {'wdc-1438042988250.59-6.gz': 4})
    # `.gz` goes, but `.59-6` is part of the name, not a second extension.
    assert discovery._resolve_excluded_indices(['wdc-1438042988250.59-6']) == [4]


def test_path_objects_are_resolved_and_cached_like_strings(discovery):
    assert discovery._resolve_excluded_indices([Path('/lake/covertype.csv')]) == [1]
    assert discovery._resolve_excluded_indices([Path('/lake/covertype.csv')]) == [1]
    with pytest.warns(RuntimeWarning):
        discovery._resolve_excluded_indices([Path('never_indexed')])
    # A cached miss does not warn again.
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        discovery._resolve_excluded_indices([Path('never_indexed')])
    assert len(discovery.queries) == 1
