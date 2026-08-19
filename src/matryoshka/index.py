import gc
import os
import re
import tempfile
import time
import warnings

import pgpq
import pyarrow as pa
import pyarrow.parquet as pq
import ray

warnings.filterwarnings('ignore', category=RuntimeWarning)
from itertools import combinations
from typing import Callable

import networkx as nx
import numpy as np
import polars as pl
import polars_hash as plh
from arrow_json import array_to_utf8_json_array
from polars.exceptions import ColumnNotFoundError
from psycopg_pool import ConnectionPool
from scipy import sparse as sp
from sklearn.preprocessing import OneHotEncoder

from matryoshka.db.handler import DBHandler
from matryoshka.lakes.archives import DirectoryLoader, TarArchiveLoader, ZipArchiveLoader
from matryoshka.lakes.parallel import RayTableProcessor, TableProcessor
from matryoshka.utils.common import process_key, semiring_aggregates
from matryoshka.utils.logging import default_log_dir, setup_logger
from matryoshka.utils.sampling import priority_sampling, reweight

# ---------------------------------------------------------------------------
# Pluggable median representation for {num_state}.values_
# ---------------------------------------------------------------------------
# The per-(table, key, col) state needs to answer "what's the median of all
# values seen so far?" The default backend ('exact') stores every raw value as
# a sorted float64 array — answers exactly at cost O(n) bytes per group.
#
# This interface exists so a sketch-based backend (T-digest, KLL) can drop in
# later: implement encode/decode/merge/median/size with bounded-size state and
# pass `median_backend='tdigest'` (or whatever) at construction. No call sites
# need to change. We ship only the exact backend in this revision; the paper's
# "MEDIAN via T-digest" sentence has an attachment point now.
#
# Storage format invariant: 'exact' produces bytes identical to the prior
# implementation (np.asarray(values, dtype=np.float64).tobytes()), so existing
# {num_state} rows in the database read back unchanged through the backend.

class ExactMedianBackend:
    '''Sorted np.float64 array packed as little-endian bytes. Exact median, O(n) space.'''

    def encode(self, values) -> bytes:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return b''
        return arr.tobytes()

    def decode(self, blob) -> np.ndarray:
        if not blob:
            return np.empty(0, dtype=np.float64)
        return np.frombuffer(bytes(blob), dtype=np.float64)

    def merge(self, a, b) -> np.ndarray:
        '''Combine two states. Result is sorted ascending.'''
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size == 0:
            return b if (b.size == 0 or _is_sorted(b)) else np.sort(b)
        if b.size == 0:
            return a
        return np.sort(np.concatenate([a, b]))

    def median(self, state) -> float:
        return float(np.median(state))

    def size(self, state) -> int:
        return int(np.asarray(state).size)


def _is_sorted(arr: np.ndarray) -> bool:
    return arr.size <= 1 or bool(np.all(arr[:-1] <= arr[1:]))


_MEDIAN_BACKENDS = {
    'exact': ExactMedianBackend,
    # 'tdigest': TDigestMedianBackend,  # not yet implemented — see R1W5 plan
    # 'kll':     KllMedianBackend,
}


class ExhaustiveIndex(DBHandler):
    def __init__(self, data_dir: str = None, feature_selection_table_name: str = None, overlap_table_name: str = None, tunnel: bool = False, batch_size: int = 1, max_workers: int = 1, feature_extraction: bool = False, prune_flag: bool = False, enable_sampling: bool = False, update_support: bool = False, max_cols: int = 100, median_backend: str = 'exact', settings=None, log_dir: str = None, silent: bool = False) -> None:
        super().__init__(feature_selection_table_name, overlap_table_name, tunnel, settings=settings)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.feature_extraction = feature_extraction
        self.prune_flag = prune_flag
        # When False (default), `_index_features` skips priority sampling and processes
        # the full table. Enable for the sampled-sketch baseline.
        self.enable_sampling = enable_sampling
        # When False (default), `index_table` skips writing per-table meta + raw
        # per-key state to the *_meta / *_num_state / *_cat_state side tables. Enable
        # only for runs that need `update_table` / `find_union_peer` afterwards.
        self.update_support = update_support
        # Tables with strictly more than `max_cols` columns are skipped by
        # `TableProcessor.process_table` (default 100; raise for wide tables).
        self.max_cols = max_cols
        # Backend for the median representation of {num_state}.values_. 'exact'
        # stores every raw value (current behaviour, exact median, O(n) space).
        # A sketch backend (e.g. 'tdigest') would trade some accuracy for space
        # — not implemented yet, see _MEDIAN_BACKENDS at module top.
        if median_backend not in _MEDIAN_BACKENDS:
            raise ValueError(
                f'unknown median_backend={median_backend!r}; '
                f'options: {list(_MEDIAN_BACKENDS)}'
            )
        self.median_backend_name = median_backend
        self._median_backend = _MEDIAN_BACKENDS[median_backend]()

        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        log_dir = str(log_dir) if log_dir is not None else str(default_log_dir())
        self.logger = setup_logger(name='exhaustive_index', log_dir=log_dir, log_file=f'exhaustive_index_{timestamp}.log', silent=silent)

        if feature_selection_table_name is not None and overlap_table_name is not None:
            self.feature_selection_table_name = feature_selection_table_name
            self.overlap_table_name = overlap_table_name


    def index_lake(self, table_index: int = None, from_tar_archive: bool = False, from_checkpoint: bool = False, checkpoint: str = None) -> None:
        '''
        Indexes all the tables in the data lake and stores the inverted index in the database. \n
        Wrapper for the `index_table` method.
        '''
        self.logger.info('Starting process')
        conn = self.db_connect_pool()
        self.create_tables(conn, include_state_tables=self.update_support)
        conn.closeall()

        if os.path.isdir(self.data_dir):
            loader = DirectoryLoader()
        elif from_tar_archive:
            loader = TarArchiveLoader()
        else:
            loader = ZipArchiveLoader()

        checkpoint_name = checkpoint if from_checkpoint else None

        # Flush cadence: flush to Postgres after this many tables have completed (or at end).
        flush_every = self.max_workers if self.batch_size is None else self.batch_size
        if table_index is None:
            table_index = 0

        # Create Ray actors BEFORE opening the connection pool — ConnectionPool holds
        # weakrefs which make `self` unpicklable, and actor creation ships `self.index_table`
        # (a bound method) which captures `self`.
        processors = None
        if self.max_workers > 1:
            if not ray.is_initialized():
                ray.init(_temp_dir='/tmp')
            processors = [RayTableProcessor.remote(self.index_table, max_cols=self.max_cols) for _ in range(self.max_workers)]

        # Persistent connection pool reused across all flushes in this run.
        self._conn_pool = ConnectionPool(self.conninfo, min_size=8, max_size=32, timeout=1800)
        self._conn_pool.open()

        try:
            if self.max_workers > 1:
                # Source-based scheduling: workers open files themselves (no driver materialization).
                use_sources = isinstance(loader, (DirectoryLoader, ZipArchiveLoader))
                if use_sources:
                    source_iter = loader.load_table_sources(self.data_dir, self.logger, checkpoint_name)
                    all_items = [(name, spec, table_index + i) for i, (name, spec) in enumerate(source_iter)]
                else:
                    all_items = [(name, lf, table_index + i) for i, (name, lf) in enumerate(loader.load_tables(self.data_dir, self.logger, checkpoint_name))]

                total = len(all_items)
                self.logger.info(f'Scheduling {total} tables across {self.max_workers} workers (sliding window)')

                window = 2 * self.max_workers
                next_i = 0
                pending = {}  # future -> (name, table_index)
                rr = 0

                def submit(item):
                    nonlocal rr
                    name, payload, idx = item
                    proc = processors[rr % len(processors)]
                    rr += 1
                    if use_sources:
                        fut = proc.process_source.remote((name, payload, idx))
                    else:
                        # Fallback: legacy path still materializes on driver
                        df = payload.collect() if isinstance(payload, pl.LazyFrame) else payload
                        fut = proc.process_table.remote((name, df, idx))
                    pending[fut] = (name, idx)

                # Prime the window
                while next_i < total and len(pending) < window:
                    submit(all_items[next_i])
                    next_i += 1

                index_data = []
                results_buffer = []  # (table_index, success)
                completed = 0
                flush_num = 0
                flush_start = time.perf_counter()

                while pending:
                    ready_futs, _ = ray.wait(list(pending.keys()), num_returns=1)
                    fut = ready_futs[0]
                    pending.pop(fut, None)
                    try:
                        data, idx, success, log_message = ray.get(fut)
                    except Exception as e:
                        self.logger.error(f'Task failed with exception: {e}')
                        data, idx, success, log_message = [], -1, False, ''

                    for line in log_message.split('\n') if log_message else []:
                        if 'Error' in line:
                            self.logger.error(line)
                        else:
                            self.logger.info(line)
                    if data:
                        index_data.extend(data)
                    results_buffer.append((idx, success))
                    completed += 1
                    if idx >= 0 and idx + 1 > table_index:
                        table_index = idx + 1

                    # Submit next task to keep window full
                    if next_i < total:
                        submit(all_items[next_i])
                        next_i += 1

                    # Flush periodically
                    if completed % flush_every == 0 and index_data:
                        flush_num += 1
                        self._flush_index_data(index_data, results_buffer, flush_num, flush_start)
                        index_data = []
                        results_buffer = []
                        flush_start = time.perf_counter()

                # Final flush
                if index_data:
                    flush_num += 1
                    self._flush_index_data(index_data, results_buffer, flush_num, flush_start)
            else:
                # Sequential path (unchanged behavior with existing batching)
                all_tables = list(loader.load_tables(self.data_dir, self.logger, checkpoint_name))
                batches = [all_tables[i:i + flush_every] for i in range(0, len(all_tables), flush_every)]
                processor = TableProcessor(self.index_table, max_cols=self.max_cols)
                for batch_num, tables in enumerate(batches, start=1):
                    batch_start = time.perf_counter()
                    table_data = [(name, table, table_index + i) for i, (name, table) in enumerate(tables)]
                    results = []
                    index_data = []
                    for data in table_data:
                        d, t_idx, success, log_message = processor.process_table(data)
                        for line in log_message.split('\n'):
                            if 'Error' in line:
                                self.logger.error(line)
                            else:
                                self.logger.info(line)
                        results.append((t_idx, success))
                        index_data.extend(d)
                    if len(index_data) == 0:
                        table_index += 1
                        continue
                    self._flush_index_data(index_data, results, batch_num, batch_start)
                    table_index = max(idx for idx, _ in results) + 1
        finally:
            if getattr(self, '_conn_pool', None) is not None:
                self._conn_pool.close()
                self._conn_pool = None
            if ray.is_initialized():
                ray.shutdown()

        self.logger.info('Finished indexing tables in the folder')
        self.logger.handlers.clear()

        return table_index


    def _flush_index_data(self, index_data: list, results: list, flush_num: int, flush_start: float) -> None:
        '''Concat accumulated index_data rows and write them to Postgres using the persistent pool.'''
        final_table = pl.concat(index_data, how='vertical_relaxed')
        rows_to_write = final_table.height
        write_start = time.perf_counter()
        self.write_to_db(final_table, self.feature_selection_table_name)
        write_elapsed = time.perf_counter() - write_start
        successful = sum(1 for _, success in results if success)
        failed = len(results) - successful
        total_elapsed = time.perf_counter() - flush_start
        self.logger.info(f'Flush {flush_num} write_time_sec={write_elapsed:.3f}, rows_written={rows_to_write}')
        self.logger.info(f'Flush {flush_num}: {successful} successful, {failed} failed')
        self.logger.info(f'Flush {flush_num} total_time_sec={total_elapsed:.3f}')


    def index_table(self, table: pl.LazyFrame, table_index: int, table_name: str = None, key_columns: list[str] = None, debug: bool = False) -> None:
        # normalize column names so that further prefixes and suffixes logic works
        new_colnames = [process_key(name) for name in table.collect_schema().keys()]
        duplicate_cols_indices = [i for i, name in enumerate(new_colnames) if new_colnames.count(name) > 1]
        for i in duplicate_cols_indices:
            new_colnames[i] += f'{i}v2'
        table = table.rename({name: new_col for name, new_col in zip(table.collect_schema().keys(), new_colnames)}).collect().lazy()
        if not key_columns:
            if self.prune_flag:
                useful_columns = self._prune_features(table)
                key_columns = self._identify_key_columns(table)
                if self.max_workers == 1:
                    intersection = list(set(useful_columns).intersection(set(key_columns)))
                    key_columns = set(key_columns).intersection(set(intersection))
                else:
                    intersection = key_columns
                table = table.select(intersection).collect().lazy()
            else:
                key_columns = table.columns
                useful_columns = key_columns
                table = table.select(key_columns).collect().lazy()
        else:
            useful_columns = self._prune_features(table)
            table = table.select([col for col in table.collect_schema().keys() if col in useful_columns+key_columns]).collect().lazy()

        table, numeric_cols, non_numeric_cols = self._split_columns_by_type(table)

        # Persist per-table state so we can later run `update_table` without re-reading
        # the source. Done once per table (independent of feature_extraction). Failures
        # here are logged but don't abort indexing. Skipped entirely unless
        # `update_support=True` to keep the default indexing path overhead-free.
        if self.update_support:
            try:
                self._write_table_meta(
                    table_index=table_index,
                    table_name=table_name,
                    numeric_cols=numeric_cols,
                    cat_cols=non_numeric_cols,
                    key_columns=list(key_columns),
                    useful_columns=list(useful_columns),
                    n_rows=table.select(pl.len()).collect().item(),
                )
                for gci, gc in enumerate(key_columns):
                    self._emit_state(table, gc, gci, numeric_cols, non_numeric_cols, table_index)
            except Exception as e:
                self.logger.error(f'Table {table_index} state emission failed: {e!r}')

        max_feature_index = 0
        group_col_index = 0
        if self.feature_extraction:
            query_funcs = [self._numeric_query, self._non_numeric_query]
        else:
            query_funcs = [self._combined_query]
        index_data = []
        i = 0
        for query_func in query_funcs:
            for group_col in key_columns:
                feature_selection_index, new_max_feature_index = self._index_features(table, query_func, group_col, group_col_index, numeric_cols, non_numeric_cols, table_index, table_name, max_feature_index, useful_columns, False)
                max_feature_index = new_max_feature_index
                group_col_index += 1
                if not debug:
                    if self.max_workers > 1:
                        if len(feature_selection_index) > 0:
                            index_data.extend(feature_selection_index)
                    else:
                        if len(feature_selection_index) > 0:
                            to_write = pl.concat(feature_selection_index, how='vertical_relaxed')
                            write_start = time.perf_counter()
                            rows_to_write = to_write.height
                            self.write_to_db(to_write, self.feature_selection_table_name)
                            write_elapsed = time.perf_counter() - write_start
                            self.logger.info(f'Table {table_index} chunk_write_time_sec={write_elapsed:.3f}, rows_written={rows_to_write}')
                else:
                    print(group_col, feature_selection_index)

        return index_data


    def _index_features(
        self,
        table: pl.LazyFrame,
        query_func: Callable,
        group_col: str,
        group_col_index: int,
        numeric_cols: list[str],
        non_numeric_cols: list[str],
        table_index: int,
        table_name: str,
        max_feature_index: int,
        useful_columns: list[str],
        feature_extraction: bool,
        sampling_threshold: int = 10_000
    ) -> list[tuple]:
        table = table.drop_nulls(subset=[group_col]).collect()
        table = table[[s.name for s in table if not (s.null_count() == table.height)]]
        numeric_cols = [c for c in numeric_cols if c in table.columns]
        non_numeric_cols = [c for c in non_numeric_cols if c in table.columns]
        if table.width <= 1:
            return [], max_feature_index

        if self.enable_sampling and table.height > sampling_threshold:
            ohe = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
            categorical_cols = [c for c in table.columns if c not in {group_col, *numeric_cols}]
            if categorical_cols:
                ohe_input = table.select(categorical_cols).to_numpy()
            else:
                ohe_input = None

            if numeric_cols:
                remaining_input = sp.csr_matrix(table.select(numeric_cols).to_numpy())
            else:
                remaining_input = None

            if ohe_input is not None:
                ohe_transformed = ohe.fit_transform(ohe_input)
                if remaining_input is not None and remaining_input.shape[1] > 0:
                    sample = priority_sampling(
                            sp.hstack(
                            [
                                ohe_transformed,
                                remaining_input
                            ],
                            format='csr'
                        ),
                        k=sampling_threshold,
                        seed=42
                    )
                else:
                    sample = priority_sampling(
                        ohe_transformed,
                        k=sampling_threshold,
                        seed=42
                    )
            elif remaining_input is not None and remaining_input.shape[1] > 0:
                # no non-numeric columns, sample based on numeric columns only
                sample = priority_sampling(
                    remaining_input,
                    k=sampling_threshold,
                    seed=42
                )
            else:
                return [], max_feature_index

            if isinstance(dict(table.schema)[group_col], pl.String):
                sample_mapping = pl.DataFrame({
                    group_col: table[sample.indices].select(pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)),
                    'row_norms_sq': sample.row_norms_sq
                })
            else:
                sample_mapping = pl.DataFrame({
                    group_col: table[sample.indices].select(pl.col(group_col)),
                    'row_norms_sq': sample.row_norms_sq
                })
            sample_mapping = sample_mapping.group_by(group_col).agg(pl.col('row_norms_sq').max().alias('row_norms_sq'))

            result = query_func(table.lazy(), group_col, numeric_cols, non_numeric_cols, useful_columns, feature_extraction)
            if result.width <= 1 or result.height <= 1:
                return [], max_feature_index

            result = reweight(result, sample, sample_mapping, group_col)
        else:
            result = query_func(table.lazy(), group_col, numeric_cols, non_numeric_cols, useful_columns, feature_extraction)
            if result.width <= 1 or result.height <= 1:
                return [], max_feature_index

        result = result.with_columns(pl.col(group_col).cast(pl.String).str.slice(0, 335))

        if self.feature_extraction:
            # populate the overlap table with unique keys from the result
            overlap_table = result.select(pl.col(group_col).alias('key'))
            overlap_table.insert_column(0, pl.Series(values=np.repeat([table_index], overlap_table.height, axis=0), name='table_index'))
            overlap_table.insert_column(1, pl.Series(values=np.repeat([group_col_index], overlap_table.height, axis=0), name='key_col_index'))
            overlap_table = overlap_table.with_row_index(name='row_index')
            self.write_to_db([overlap_table], self.overlap_table_name)

        grouped_columns = self._grouped_columns_mapping(result, group_col)

        feature_selection_index = []
        for col in list(grouped_columns.keys()):
            try:
                mapped_cols = grouped_columns[col]
            except KeyError:
                continue

            feature_slice = result.select([group_col] + mapped_cols)
            # find the base column
            for feature_slice_col in feature_slice.columns:
                if feature_slice_col.count('_') == 1:
                    base_col = feature_slice_col
                    break

            combs = self._combinations_with_element(mapped_cols, base_col)
            features_col_indices = [i for i in range(max_feature_index, len(combs) + max_feature_index)]
            max_feature_index = max(features_col_indices) + 1

            key_col = feature_slice.select(group_col).to_series()
            feature_slice = feature_slice.select(mapped_cols).to_numpy()
            feature_selection_index.extend(self._inverted_index_from_feature_matrix(feature_slice, key_col, table_index, table_name, group_col_index, combs, features_col_indices, mapped_cols))

            if len(mapped_cols) > 1:
                mapped_cols.remove(base_col)
                for k in mapped_cols:
                    try:
                        del grouped_columns[k]
                    except KeyError:
                        pass

        new_max_feature_index = max_feature_index

        return feature_selection_index, new_max_feature_index


    def _inverted_index_from_feature_matrix(self, feature_slice: np.ndarray, key_col: pl.Series, table_index: int, table_name: str, group_col_index: int, combs: list[list[int]], features_col_indices: list[str], mapped_cols: list[str]) -> list[tuple]:
        count, sum_, diag, cofactors = semiring_aggregates(group=feature_slice, features_locs=combs)
        if cofactors[0].shape[0] == 0:
            return []
        if np.isnan(sum_).any():
            return []
        feature_selection_index = []
        for f in range(len(features_col_indices)):
            row_count, row_sum, row_diag, row_cofactors = count[f], sum_[f], diag[f], cofactors[f]
            shape = [list(row_cofactors.shape[1:]) for _ in range(row_cofactors.shape[0])]
            row_cofactors = [row_cofactors[i].tobytes() for i in range(row_cofactors.shape[0])]
            df = pl.DataFrame(
                [
                    pl.Series(values=row_count, name='count'),
                    pl.Series(values=row_sum, name='sum'),
                    pl.Series(values=row_diag, name='diag'),
                    pl.Series(values=row_cofactors, name='cofactors'),
                    pl.Series(values=shape, name='shape', dtype=pl.List(pl.Int32))
                ]
            )
            del row_count, row_sum, row_diag, row_cofactors

            df = df.with_columns(pl.col(['sum', 'diag']).arr.to_list())
            df = df.fill_null('None')

            df = df.with_row_index()
            df_index = df.drop_in_place('index')
            df.insert_column(0, df_index.rename('row_index'))

            df = df.with_columns(pl.lit(group_col_index).alias('key_col_index'))
            df_index = df.drop_in_place('key_col_index')
            df.insert_column(0, df_index)
            df.insert_column(0, pl.Series(values=np.repeat([table_name], df.height, axis=0), name='table_name'))
            df.insert_column(0, pl.Series(values=np.repeat([table_index], df.height, axis=0), name='table_index'))
            df.insert_column(0, pl.Series(values=np.repeat([features_col_indices[f]], df.height, axis=0), name='feature_index'))
            df.insert_column(0, key_col.rename('key'))

            n_cols = df.select(pl.col('sum').list.len()).row(0)[0]
            query_col = 'key'
            sub_table = (
                df
                    .filter(pl.col(query_col).is_in(df.select('key').to_series()))
                    .sort(query_col)
                    .select(pl.col(query_col).repeat_by(n_cols).list.to_struct(upper_bound=n_cols).struct.unnest())
            )
            qcr_positive = self._build_qcr_candidate(sub_table, df, sub_table.columns, n_cols, True)
            qcr_negative = self._build_qcr_candidate(sub_table, df, sub_table.columns, n_cols, False)
            df = df.with_columns([qcr_positive, qcr_negative])

            headers = [mapped_cols[i] for i in combs[f]]
            df = df.with_columns(pl.Series(values=[headers] * df.height, name='column_headers', dtype=pl.List(pl.String)))

            feature_selection_index.append(df)

        return feature_selection_index


    def _build_qcr_candidate(self, sub_table: pl.DataFrame, group: pl.DataFrame, num_cols: list[str], n_cols: int, is_positive: bool, max_sha = 2**64 - 1) -> pl.Series:
        table = (
            pl.concat(
                [
                    group
                        .select(['key', 'sum'])
                        .with_columns(pl.col('sum').list.to_struct(upper_bound=n_cols).struct.unnest())
                        .drop(['sum'])
                ],
                how='horizontal'
            )
        )
        if is_positive:
            suffix = '_positive'
            table = table.with_columns(
                (pl.col(num_cols) - pl.col(num_cols).mean())
                .sign()
                .cast(pl.Int16)
                .cast(pl.String)
            )
        else:
            suffix = '_negative'
            table = table.with_columns(
                (pl.col(num_cols) - pl.col(num_cols).mean())
                .sign()
                .mul(-1)
                .cast(pl.Int16)
                .cast(pl.String)
            )
        qcr_candidate = pl.DataFrame().select(
            [
                (
                    plh.concat_str(table['key'], table[col]).chash.sha3_shake128(length=8).str.to_integer(base=16, dtype=pl.Int128)
                    / max_sha
                ).alias(f'qcr_col_{col}')
                for col in num_cols
            ]
        )
        qcr_candidate = qcr_candidate.select(pl.concat_list(pl.all()).alias(f'qcr_term{suffix}')).to_series()
        return qcr_candidate


    def _combined_query(self, table: pl.LazyFrame, group_col: str, numeric_cols: list[str], non_numeric_cols: list[str], useful_cols: list[str], feature_extraction: bool) -> pl.DataFrame:
        '''
        Combines numeric and non-numeric queries into a single query.
        '''
        numeric_result = self._numeric_query(table, group_col, numeric_cols, non_numeric_cols, useful_cols, feature_extraction)
        non_numeric_result = self._non_numeric_query(table, group_col, numeric_cols, non_numeric_cols, useful_cols, feature_extraction)

        combined_result = (
            numeric_result
                .join(non_numeric_result, on=group_col, how='inner')
        )
        feature_cols = [col for col in combined_result.columns if col != group_col]
        col_hashes = {
            col: combined_result[col].hash().sum()
            for col in feature_cols
        }
        unique_cols = list({v: k for k, v in col_hashes.items()}.values())
        combined_result = combined_result.select([group_col] + unique_cols)
        constant_check = (
            combined_result.select(
                (pl.all().exclude(group_col)-pl.all().exclude(group_col).mean())
                .sign()
                .var()
                != 0
            )
        )
        if constant_check.height == 0:
            return pl.DataFrame()
        non_constant_cols = [c for c, v in zip(constant_check.columns, constant_check.row(0)) if v]
        combined_result = combined_result.select([group_col] + non_constant_cols)

        return combined_result


    def _numeric_query(self, table: pl.LazyFrame, group_col: str, numeric_cols: list[str], non_numeric_cols: list[str], useful_cols: list[str], feature_extraction: bool) -> pl.DataFrame:
        pairs, numeric_cols_trunc = self._prepare_query_inputs(group_col, numeric_cols, useful_cols)

        if len(numeric_cols_trunc) == 0:
            if isinstance(table.collect_schema()[group_col], pl.String):
                result = (
                    table
                        .select(
                            pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)
                        )
                        .unique()
                        .collect()
                )
            else:
                result = table.select(group_col).unique().collect()
            return result

        imputing_query = (
            table
                .select([group_col]+numeric_cols_trunc)
                .with_columns(pl.col(numeric_cols_trunc).fill_null(strategy='mean'))
        )
        if feature_extraction:
            result = (
                imputing_query
                    .with_columns(
                        [
                            (pl.col(col1) * pl.col(col2)).alias(f'{col1}_mul_{col2}')
                            for col1, col2 in pairs
                        ]
                        +
                        [
                            pl.col(numeric_cols_trunc).pow(2).name.suffix('_^2'),
                            pl.col(numeric_cols_trunc).pow(3).name.suffix('_^3')
                        ]
                    )
            )
        else:
            result = imputing_query

        numeric_cols_trunc = [c for c in list(result.collect_schema().keys()) if c not in non_numeric_cols and not c.startswith(group_col)]
        if isinstance(table.collect_schema()[group_col], pl.String):
            result = (
                result
                    .with_columns(
                        pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)
                    )
                    .group_by(group_col)
                    .agg(
                        pl.col(numeric_cols_trunc).mean().name.suffix('_mean'),
                        pl.col(numeric_cols_trunc).median().name.suffix('_median'),
                        pl.col(numeric_cols_trunc).max().name.suffix('_max'),
                        pl.col(numeric_cols_trunc).min().name.suffix('_min')
                    )
            )
        else:
            result = (
                result
                    .group_by(group_col)
                    .agg(
                        pl.col(numeric_cols_trunc).mean().name.suffix('_mean'),
                        pl.col(numeric_cols_trunc).median().name.suffix('_median'),
                        pl.col(numeric_cols_trunc).max().name.suffix('_max'),
                        pl.col(numeric_cols_trunc).min().name.suffix('_min')
                    )
            )
        result = result.collect()

        return result


    def _non_numeric_query(self, table: pl.LazyFrame, group_col: str, numeric_cols: list[str], non_numeric_cols: list[str], useful_cols: list[str], feature_extraction = None) -> pl.DataFrame:
        _, non_numeric_cols_trunc = self._prepare_query_inputs(group_col, non_numeric_cols, useful_cols)

        if len(non_numeric_cols_trunc) == 0:
            if isinstance(table.collect_schema()[group_col], pl.String):
                result = (
                    table
                        .select(
                            pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)
                        )
                        .unique()
                        .collect()
                )
            else:
                result = table.select(group_col).unique().collect()
            return result

        imputing_plan = (
            table
                .select([group_col]+non_numeric_cols_trunc)
                .with_columns(
                    pl.col(non_numeric_cols_trunc).fill_null(pl.col(non_numeric_cols_trunc).drop_nulls().mode().head(1))
                )
                .collect()
        )

        unique_values_per_key = (
            imputing_plan
                .select((pl.selectors.string()).n_unique())
                .to_dict(as_series=False)
        )

        if isinstance(imputing_plan.collect_schema()[group_col], pl.String):
            imputing_plan = (
                imputing_plan
                    .with_columns(
                        pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)
                    )
            )

        imputing_plan_schema = list(imputing_plan.collect_schema().keys())
        imputing_plan_schema.remove(group_col)
        non_numeric_cols_trunc = [c for c in imputing_plan_schema if c not in numeric_cols and not c.startswith(group_col)]
        result = (
            imputing_plan
                .group_by(group_col)
                .agg(
                    pl.col(non_numeric_cols_trunc).unique_counts()
                )
                .with_columns(
                    [
                        pl.col(c) / unique_values_per_key[c][0]
                        for c in non_numeric_cols_trunc
                    ]
                )
                .group_by(group_col)
                .agg(
                    [el for c in non_numeric_cols_trunc for el in
                        [
                            (pl.col(c).list.sum()/unique_values_per_key[c][0]).first().name.suffix('_mean')
                        ]
                    ]
                    +
                    [
                        pl.col(non_numeric_cols_trunc).list.max().first().name.suffix('_max'),
                        (pl.col(non_numeric_cols_trunc).list.n_unique()).first().cast(pl.Int64).name.suffix('_nunique')
                    ]
                )
        )

        return result


    def _prepare_query_inputs(self, group_col: str, cols: list[str], useful_cols: list[str]):
        if self.feature_extraction:
            pairs = list(combinations(cols, 2))
        else:
            pairs = []
        cols_trunc = cols[:]
        if group_col in cols_trunc:
            cols_trunc.remove(group_col)
        cols_trunc = list(set(cols_trunc).intersection(set(useful_cols)))

        return pairs, cols_trunc


    def _grouped_columns_mapping(self, result: pl.DataFrame, group_col: str) -> dict:
        '''
        Maps the columns of the result DataFrame to their respective feature groups.
        The groups are determined based on the prefix and suffix of the column names for polynomial and interaction features.
        '''
        result_cols = result.columns
        result_cols.remove(group_col)
        grouped_columns = {}
        if self.feature_extraction:
            for col in result_cols:
                split_colname = re.split(r'[_^]', col, maxsplit=0)
                # prefix - the column name, suffix - the aggregation function
                # e.g. 'newcertifybydate_mean' -> prefix = 'newcertifybydate', suffix = 'mean'
                prefix, suffix = split_colname[0], split_colname[-1]
                if '^' in col:
                    # Polynomial columns - list keeps only the polynomials and base column
                    values = [c for c in result.columns if c.startswith(prefix) and c.endswith(suffix) and ('^' in c or '_mul_' not in c)]
                elif len(split_colname) == 2:
                    # Base column - list keeps only the base column
                    values = [col]
                else:
                    # Interaction columns - list keeps only the interaction columns and base column
                    values = [c for c in result.columns if c.startswith(prefix) and c.endswith(suffix) and '^' not in c]
                grouped_columns[col] = values
        else:
            col = result_cols[0]
            grouped_columns[col] = [col for col in result_cols]

        return grouped_columns


    def _combinations_with_element(self, strings, target_element):
        if self.feature_extraction:
            if len(strings) == 3:
                sizes = [2, 3]
            else:
                sizes = [2]
            target_element_idx = strings.index(target_element)
            results = [[target_element_idx]]
        else:
            sizes = [len(strings)]
            results = []
            target_element_idx = strings.index(target_element)
        for size in sizes:
            others = [strings.index(s) for s in strings if s != target_element]
            for combo in combinations(others, size - 1):
                results.append(list((target_element_idx, *combo)))

        return results


    def _write_batch_to_pg(self, batch: pa.RecordBatch, pool: ConnectionPool, pg_table_name: str) -> None:
        all_cols = batch.schema.names
        field_encoders = {
            col: pgpq.ArrowToPostgresBinaryEncoder.infer_encoder(batch.field(col))
            for col in all_cols
        }
        encoder = pgpq.ArrowToPostgresBinaryEncoder.new_with_encoders(batch.schema, field_encoders)
        batches = pa.Table.from_batches([batch]).combine_chunks().to_batches()

        with pool.connection() as conn:
            with conn.cursor() as cursor:
                with cursor.copy(f'COPY {pg_table_name} FROM STDIN WITH (FORMAT BINARY)') as copy:
                    copy.write(encoder.write_header())
                    for b in batches:
                        copy.write(encoder.write_batch(b))
                    copy.write(encoder.finish())


    def _preprocess_dataframe(self, df: pl.DataFrame, target_schema: pa.Schema, pg_table_name: str) -> pa.RecordBatch:
        if self.max_workers > 1:
            # --- Fast in-memory path (small tables) ---
            match pg_table_name:
                case self.feature_selection_table_name:
                    encode_cols = ['sum', 'diag', 'qcr_term_positive', 'qcr_term_negative']
                    arrow_table = df.to_arrow().combine_chunks()
                    batch = arrow_table.to_batches()[0]
                    del arrow_table
                    gc.collect()

                    all_cols = batch.schema.names
                    for col in encode_cols:
                        if col in all_cols:
                            idx = all_cols.index(col)
                            batch = batch.set_column(
                                idx,
                                col,
                                array_to_utf8_json_array(batch.column(col))
                            )
                case self.overlap_table_name:
                    df = df.unique(subset=['key']).select(['key', 'table_index', 'key_col_index', 'row_index'])
                    batch = df.to_arrow().to_batches()[0]

            yield batch.cast(target_schema)

        else:
            # --- Low-memory disk-streaming path (large tables) ---
            tmp_dir = os.environ.get("MATRYOSHKA_TMP_DIR", "/tmp/matryoshka_tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            fd, final_path = tempfile.mkstemp(suffix=".parquet", dir=tmp_dir)
            os.close(fd)

            tmp_path = final_path + ".writing"

            match pg_table_name:
                case self.feature_selection_table_name:
                    encode_cols = ['sum', 'diag', 'qcr_term_positive', 'qcr_term_negative']

                    # Write the dataframe to Parquet via pyarrow in row slices.
                    # A single Arrow string array is capped at 2 GiB (int32 offsets);
                    # for wide tables one chunk can blow that limit, so we slice the
                    # df into small row groups AND cast string columns to large_string
                    # (int64 offsets) before writing.
                    n_rows = df.height
                    row_chunk = 200

                    def _to_large_string(tbl: pa.Table) -> pa.Table:
                        new_cols = []
                        new_fields = []
                        for i, f in enumerate(tbl.schema):
                            col = tbl.column(i)
                            if pa.types.is_string(f.type):
                                col = col.cast(pa.large_string())
                                f = pa.field(f.name, pa.large_string(), nullable=f.nullable)
                            elif pa.types.is_binary(f.type):
                                col = col.cast(pa.large_binary())
                                f = pa.field(f.name, pa.large_binary(), nullable=f.nullable)
                            new_cols.append(col)
                            new_fields.append(f)
                        return pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))

                    first_slice = _to_large_string(df.slice(0, min(row_chunk, n_rows)).to_arrow())
                    schema = first_slice.schema
                    writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
                    try:
                        writer.write_table(first_slice)
                        del first_slice
                        for start in range(row_chunk, n_rows, row_chunk):
                            sub = _to_large_string(df.slice(start, row_chunk).to_arrow())
                            writer.write_table(sub)
                            del sub
                    finally:
                        writer.close()
                    del df
                    os.replace(tmp_path, final_path)
                    gc.collect()
                    try:
                        pf = pq.ParquetFile(final_path)
                        for batch in pf.iter_batches(batch_size=500):
                            all_cols = batch.schema.names
                            for col in encode_cols:
                                if col in all_cols:
                                    idx = all_cols.index(col)
                                    batch = batch.set_column(
                                        idx,
                                        col,
                                        array_to_utf8_json_array(batch.column(col))
                                    )
                            yield batch.cast(target_schema)
                    finally:
                        os.remove(final_path)

                case self.overlap_table_name:
                    df = df.unique(subset=['key']).select(['key', 'table_index', 'key_col_index', 'row_index'])
                    n_rows = df.height
                    row_chunk = 50_000
                    first_slice = df.slice(0, min(row_chunk, n_rows)).to_arrow()
                    schema = first_slice.schema
                    writer = pq.ParquetWriter(tmp_path, schema, compression="snappy")
                    try:
                        writer.write_table(first_slice)
                        del first_slice
                        for start in range(row_chunk, n_rows, row_chunk):
                            sub = df.slice(start, row_chunk).to_arrow()
                            writer.write_table(sub)
                            del sub
                    finally:
                        writer.close()
                    del df
                    gc.collect()

                    pf = pq.ParquetFile(tmp_path)
                    for batch in pf.iter_batches(batch_size=50_000):
                        yield batch.cast(target_schema)

    def write_to_db(self, feature_selection_index: pl.DataFrame, pg_table_name: str) -> None:
        match pg_table_name:
            case self.feature_selection_table_name:
                target_schema = pa.schema([
                    pa.field('key', pa.string()),
                    pa.field('feature_index', pa.int32()),
                    pa.field('table_index', pa.int32()),
                    pa.field('table_name', pa.string()),
                    pa.field('key_col_index', pa.int32()),
                    pa.field('row_index', pa.int64()),
                    pa.field('count', pa.int32()),
                    pa.field('sum', pa.large_binary()),
                    pa.field('diag', pa.large_binary()),
                    pa.field('cofactors', pa.large_binary()),
                    pa.field('shape', pa.list_(pa.int32())),
                    pa.field('qcr_term_positive', pa.large_binary()),
                    pa.field('qcr_term_negative', pa.large_binary()),
                    pa.field('column_headers', pa.list_(pa.string()))
                ])
            case self.overlap_table_name:
                target_schema = pa.schema([
                    pa.field('key', pa.string()),
                    pa.field('table_index', pa.int32()),
                    pa.field('key_col_index', pa.int32()),
                    pa.field('row_index', pa.int64())
                ])
        # Reuse persistent pool if available (set up by index_lake), otherwise create a transient one.
        transient = False
        conn_pool = getattr(self, '_conn_pool', None)
        if conn_pool is None:
            conn_pool = ConnectionPool(self.conninfo, min_size=8, max_size=32, timeout=1800)
            conn_pool.open()
            transient = True
        try:
            # Project ``target_schema`` (and the input frame) down to the
            # columns the destination Postgres table actually has. Older
            # indexes (CUK, NYC) predate the ``table_name`` /
            # ``column_headers`` schema migration; without this projection the
            # binary COPY produces 14 fields against a 12-field table and the
            # driver raises BadCopyFileFormat.
            with conn_pool.connection() as _intro_conn:
                with _intro_conn.cursor() as _intro_cur:
                    _intro_cur.execute(
                        'SELECT column_name FROM information_schema.columns '
                        'WHERE table_name = %s', (pg_table_name,))
                    _actual_cols = {row[0] for row in _intro_cur.fetchall()}
            _missing = [f.name for f in target_schema if f.name not in _actual_cols]
            if _missing:
                target_schema = pa.schema(
                    [f for f in target_schema if f.name in _actual_cols])
                _drop = [c for c in _missing if c in feature_selection_index.columns]
                if _drop:
                    feature_selection_index = feature_selection_index.drop(_drop)
            for batch in self._preprocess_dataframe(feature_selection_index, target_schema, pg_table_name):
                self._write_batch_to_pg(batch, conn_pool, pg_table_name)
        finally:
            if transient:
                conn_pool.close()


    def _prune_features(
        self,
        df: pl.LazyFrame,
        sample_size: int = 1000,
        null_threshold: float = 0.8,
        high_cardinality_threshold: float = 0.8,
        constant_threshold: float = 0.99,
        boolean_skew_threshold: float = 0.01,
        max_text_avg_len: int = 100,
    ) -> list[str]:
        n_rows = df.select(pl.len()).collect().item()
        if n_rows > sample_size:
            df = df.collect().sample(n=sample_size, seed=42)
        else:
            df = df.collect()

        total = df.height
        if total == 0:
            return []

        width = df.width
        if width == 0:
            return []

        # --- build expressions for stats ---
        exprs = []
        for c in df.columns:
            s = df[c]
            if s.dtype in (pl.Binary, pl.Object):
                continue  # skip early
            exprs.extend([
                pl.col(c).null_count().alias(f"{c}_nulls"),
                pl.col(c).n_unique().alias(f"{c}_uniq"),
                pl.col(c).drop_nulls().value_counts().struct[1].max().alias(f"{c}_top")
            ])
            # text avg len
            if s.dtype == pl.Utf8:
                exprs.append(
                    pl.col(c).drop_nulls().str.len_chars().mean().alias(f"{c}_avglen")
                )
            # boolean counts
            if s.dtype == pl.Boolean:
                exprs.append(
                    pl.col(c).drop_nulls().value_counts().struct[1].min().alias(f"{c}_boolmin")
                )

        try:
            stats = df.select(exprs)
        except ColumnNotFoundError:
            return []

        # --- evaluate heuristics ---
        promising = []
        for c in df.columns:
            s = df[c]
            if s.dtype in (pl.Binary, pl.Object):
                continue

            nulls = stats[f"{c}_nulls"][0] if f"{c}_nulls" in stats.columns else 0
            non_null = total - nulls
            if non_null == 0 or nulls / total > null_threshold:
                continue

            uniq = stats[f"{c}_uniq"][0]
            if uniq / non_null > high_cardinality_threshold:
                continue

            top = stats[f"{c}_top"][0]
            if top / non_null > constant_threshold:
                continue

            if s.dtype == pl.Utf8 and f"{c}_avglen" in stats.columns:
                avglen = stats[f"{c}_avglen"][0]
                if avglen is not None and avglen > max_text_avg_len:
                    continue

            if s.dtype == pl.Boolean and f"{c}_boolmin" in stats.columns:
                boolmin = stats[f"{c}_boolmin"][0]
                if boolmin / non_null < boolean_skew_threshold:
                    continue

            promising.append(c)

        return promising


    def _identify_key_columns(self, table: pl.LazyFrame) -> tuple[list[int]]:
        '''
        Identifies foreign key candidates in the table by selecting columns of string type with no null values at all

        Parameters:
        ----------
        table: `pl.DataFrame`
            Table to identify key columns in
        
        Returns:
        -------
       `tuple[list[int]]`: Key columns, Non-key columns (indices for both lists)
        '''
        DESCRIPTION_LIKE_TOKENS = [
            'desc', 'text', 'comment', 'remark', 'note', 'memo', 'content', 'message', 'feedback', 'log', 'msg'
        ]
        ID_LIKE_TOKENS = [
            'uuid', 'guid', 'key', 'code', 'ref', 'hash', 'token', 'identifier', 'unnamed'
        ]
        invalid_cols_tokens = DESCRIPTION_LIKE_TOKENS + ID_LIKE_TOKENS
        valid_cols = []
        for col in table.collect_schema().names():
            non_null_col = table.select(col).null_count().collect().row(0)[0] < table.select(pl.len()).collect().row(0)[0]
            non_constant_col = table.select(pl.col(col).n_unique()).collect().row(0)[0] > 1
            if non_null_col \
                and non_constant_col \
                and not any(token in col.lower() for token in invalid_cols_tokens) \
                and not (col.endswith('id') or col.startswith('id')):
                    valid_cols.append(col)
        key_columns = [col for col in table.select(pl.all()).collect_schema().names() if col in valid_cols]

        return key_columns


    def _split_columns_by_type(self, table: pl.LazyFrame) -> tuple[pl.LazyFrame, list[str], list[str]]:
        '''
        Splits the columns of a table by their data types

        Parameters:
        ----------
        table: `pl.DataFrame`
            Table to split columns of
        
        Returns:
        -------
        `tuple[list[str], list[str]]`: Numeric columns, non-numeric columns
        '''
        table = table.cast({pl.Boolean: pl.String}).collect().lazy()
        numeric_cols = table.select(pl.all().exclude(pl.String)).collect_schema().names()
        non_numeric_cols = table.select(pl.all().exclude(numeric_cols)).collect_schema().names()

        return table, numeric_cols, non_numeric_cols


    def create_db_table_index(self, conn) -> None:
        self.logger.info(f'Creating in-database index for tables {self.overlap_table_name}, {self.feature_selection_table_name}')
        # When update support is disabled the *_meta / *_num_state / *_cat_state
        # tables don't exist; skip indexing them.
        state_tables = {self.tables_meta_table_name, self.num_state_table_name, self.cat_state_table_name}
        for table_name in self.btree_index_cols:
            if not self.update_support and table_name in state_tables:
                continue
            queries = self.btree_index_cols[table_name]
            for i, q in enumerate(queries):
                query = self.create_index_query.replace('index_name', f'{table_name}_index_{i}').replace('table_name', f'{table_name}').replace('method (columns)', q)
                self.create_index(conn, query)


    # ============================================================================
    # Index update support: state emission, state read+merge+derive, update_table.
    # ============================================================================
    #
    # The offline index encodes only post-aggregation sketches, which are not
    # invertible — once a row's QCR/cofactors are computed, you can't recover the
    # raw values needed to add another row's data. To support `update_table`, the
    # offline path also persists raw per-key state into three side tables:
    #
    #   * <fs>_meta       — one row per indexed table (col types, key columns,
    #                       useful columns, n_rows).
    #   * <fs>_num_state  — per (table_index, key_col_index, key, num_col):
    #                       count, sum, min, max, sorted-values bytes (for median).
    #   * <fs>_cat_state  — per (table_index, key_col_index, key, cat_col,
    #                       category): raw count.
    #
    # On update, we merge the delta into the state, fully re-derive the per-key
    # feature DataFrame, and replace the (table_index, key_col_index) section of
    # the fs index in two non-atomic phases (DELETE, then COPY-insert). This is
    # forced rather than incremental because the QCR sign mask depends on the
    # column mean across all keys, so any new row invalidates every sketch row
    # for the table.
    #
    # v1 limitations (documented; not enforced):
    #   * No null imputation. The offline `_numeric_query` fills numeric nulls
    #     with the *global* column mean before group_by; we don't replicate this.
    #     Tables with many nulls in numeric columns will diverge slightly from
    #     a full rebuild after update.
    #   * Replace-non-atomic. DELETE and INSERT are separate transactions; a
    #     reader hitting the gap sees an empty section. Acceptable for offline
    #     batch updates; revisit if used online.
    #   * No sampling. Tables that originally went through `priority_sampling`
    #     during offline indexing won't preserve that sampling after update;
    #     the update path always derives from the full state.

    # ---------- state record builders ---------------------------------------------

    def _build_num_state_records(
        self,
        table: pl.DataFrame,
        group_col: str,
        group_col_index: int,
        numeric_cols: list[str],
        table_index: int,
    ) -> pl.DataFrame | None:
        '''Aggregate per-(key, numeric_col) raw stats from a Polars DataFrame.'''
        usable = [c for c in numeric_cols if c in table.columns and c != group_col]
        if not usable:
            return None
        # Cast all numeric cols to Float64 so unpivot has a uniform value dtype.
        casted = table.select(
            pl.col(group_col),
            *[pl.col(c).cast(pl.Float64, strict=False).alias(c) for c in usable],
        )
        # Apply the same key normalization that `_numeric_query` applies post-groupby.
        if casted.schema[group_col] == pl.String:
            casted = casted.with_columns(
                pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)
            )
        long = (
            casted.unpivot(index=[group_col], on=usable, variable_name='col_name', value_name='val')
            .drop_nulls(['val', group_col])
        )
        if long.height == 0:
            return None
        agg = long.group_by([group_col, 'col_name']).agg(
            pl.len().cast(pl.Int64).alias('count_'),
            pl.col('val').sum().alias('sum_'),
            pl.col('val').min().alias('min_'),
            pl.col('val').max().alias('max_'),
            pl.col('val').sort().alias('values_list'),
        )
        # Encode the per-(key, col) value stream through the pluggable median
        # backend. With 'exact' (default) this produces the same float64 bytes
        # as the prior inline tobytes() call.
        encode = self._median_backend.encode
        agg = agg.with_columns(
            pl.col('values_list').map_elements(
                lambda lst: encode(lst),
                return_dtype=pl.Binary,
            ).alias('values_'),
        ).drop('values_list')
        return agg.with_columns(
            pl.lit(table_index).cast(pl.Int32).alias('table_index'),
            pl.lit(group_col_index).cast(pl.Int32).alias('key_col_index'),
            pl.col(group_col).cast(pl.String).alias('key'),
        ).select([
            'table_index', 'key_col_index', 'key', 'col_name',
            'count_', 'sum_', 'min_', 'max_', 'values_',
        ])

    def _build_cat_state_records(
        self,
        table: pl.DataFrame,
        group_col: str,
        group_col_index: int,
        cat_cols: list[str],
        table_index: int,
    ) -> pl.DataFrame | None:
        '''Per-(key, cat_col, category) raw counts.'''
        usable = [c for c in cat_cols if c in table.columns and c != group_col]
        if not usable:
            return None
        normalize_key = table.schema[group_col] == pl.String
        chunks = []
        for c in usable:
            sub = table.select([group_col, c]).drop_nulls()
            if sub.height == 0:
                continue
            if normalize_key:
                sub = sub.with_columns(
                    pl.col(group_col).map_elements(lambda s: process_key(s), return_dtype=pl.String)
                )
            chunks.append(
                sub.group_by([group_col, c]).agg(pl.len().cast(pl.Int64).alias('raw_count'))
                .with_columns(
                    pl.lit(table_index).cast(pl.Int32).alias('table_index'),
                    pl.lit(group_col_index).cast(pl.Int32).alias('key_col_index'),
                    pl.lit(c).alias('col_name'),
                    pl.col(group_col).cast(pl.String).alias('key'),
                    pl.col(c).cast(pl.String).alias('category'),
                )
                .select(['table_index', 'key_col_index', 'key', 'col_name', 'category', 'raw_count'])
            )
        return pl.concat(chunks, how='vertical_relaxed') if chunks else None

    # ---------- write paths -------------------------------------------------------

    def _state_pool(self) -> tuple[ConnectionPool, bool]:
        '''Reuse the persistent pool when available, otherwise create a transient one.'''
        pool = getattr(self, '_conn_pool', None)
        if pool is None:
            pool = ConnectionPool(self.conninfo, min_size=2, max_size=8, timeout=1800)
            pool.open()
            return pool, True
        return pool, False

    def _write_state_to_db(self, df: pl.DataFrame | None, pg_table_name: str) -> None:
        if df is None or df.height == 0:
            return
        if pg_table_name == self.num_state_table_name:
            target_schema = pa.schema([
                pa.field('table_index',  pa.int32()),
                pa.field('key_col_index', pa.int32()),
                pa.field('key',          pa.string()),
                pa.field('col_name',     pa.string()),
                pa.field('count_',       pa.int64()),
                pa.field('sum_',         pa.float64()),
                pa.field('min_',         pa.float64()),
                pa.field('max_',         pa.float64()),
                pa.field('values_',      pa.large_binary()),
            ])
        elif pg_table_name == self.cat_state_table_name:
            target_schema = pa.schema([
                pa.field('table_index',  pa.int32()),
                pa.field('key_col_index', pa.int32()),
                pa.field('key',          pa.string()),
                pa.field('col_name',     pa.string()),
                pa.field('category',     pa.string()),
                pa.field('raw_count',    pa.int64()),
            ])
        else:
            raise ValueError(f'_write_state_to_db: unknown table {pg_table_name!r}')

        df = df.select([f.name for f in target_schema])
        batch = df.to_arrow().combine_chunks().to_batches()[0].cast(target_schema)

        pool, transient = self._state_pool()
        try:
            self._write_batch_to_pg(batch, pool, pg_table_name)
        finally:
            if transient:
                pool.close()

    def _write_table_meta(
        self,
        table_index: int,
        table_name: str,
        numeric_cols: list[str],
        cat_cols: list[str],
        key_columns: list[str],
        useful_columns: list[str],
        n_rows: int,
    ) -> None:
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f'INSERT INTO {self.tables_meta_table_name} '
                    '(table_index, table_name, numeric_cols, cat_cols, key_columns, useful_columns, n_rows, last_updated) '
                    'VALUES (%s, %s, %s, %s, %s, %s, %s, NOW()) '
                    'ON CONFLICT (table_index) DO UPDATE SET '
                    'table_name = EXCLUDED.table_name, '
                    'numeric_cols = EXCLUDED.numeric_cols, '
                    'cat_cols = EXCLUDED.cat_cols, '
                    'key_columns = EXCLUDED.key_columns, '
                    'useful_columns = EXCLUDED.useful_columns, '
                    'n_rows = EXCLUDED.n_rows, '
                    'last_updated = NOW();',
                    (
                        int(table_index),
                        table_name,
                        list(numeric_cols),
                        list(cat_cols),
                        list(key_columns),
                        list(useful_columns),
                        int(n_rows),
                    ),
                )
                conn.commit()
        finally:
            if transient:
                pool.close()

    def _emit_state(
        self,
        table: pl.LazyFrame | pl.DataFrame,
        group_col: str,
        group_col_index: int,
        numeric_cols: list[str],
        cat_cols: list[str],
        table_index: int,
    ) -> None:
        df = table.collect() if isinstance(table, pl.LazyFrame) else table
        if group_col not in df.columns:
            return
        df = df.drop_nulls(subset=[group_col])
        if df.height == 0:
            return
        num_recs = self._build_num_state_records(df, group_col, group_col_index, numeric_cols, table_index)
        cat_recs = self._build_cat_state_records(df, group_col, group_col_index, cat_cols,     table_index)
        self._write_state_to_db(num_recs, self.num_state_table_name)
        self._write_state_to_db(cat_recs, self.cat_state_table_name)

    # ---------- read paths --------------------------------------------------------

    def _fetch_table_meta(self, table_name: str) -> dict | None:
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f'SELECT table_index, table_name, numeric_cols, cat_cols, key_columns, useful_columns, n_rows '
                    f'FROM {self.tables_meta_table_name} WHERE table_name = %s',
                    (table_name,),
                )
                row = cur.fetchone()
        finally:
            if transient:
                pool.close()
        if row is None:
            return None
        return {
            'table_index':    row[0],
            'table_name':     row[1],
            'numeric_cols':   list(row[2] or []),
            'cat_cols':       list(row[3] or []),
            'key_columns':    list(row[4] or []),
            'useful_columns': list(row[5] or []),
            'n_rows':         row[6],
        }

    def _read_num_state(self, table_index: int, group_col_index: int) -> dict:
        '''Returns {(key, col): {count, sum, min, max, values}} where values is np.ndarray[float64].'''
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f'SELECT key, col_name, count_, sum_, min_, max_, values_ '
                    f'FROM {self.num_state_table_name} '
                    f'WHERE table_index = %s AND key_col_index = %s',
                    (int(table_index), int(group_col_index)),
                )
                rows = cur.fetchall()
        finally:
            if transient:
                pool.close()
        out = {}
        decode = self._median_backend.decode
        for key, col_name, count_, sum_, min_, max_, values_ in rows:
            out[(key, col_name)] = {
                'count': int(count_),
                'sum':   float(sum_),
                'min':   float(min_),
                'max':   float(max_),
                'values': decode(values_),
            }
        return out

    def _read_cat_state(self, table_index: int, group_col_index: int) -> dict:
        '''Returns {col_name: {key: {category: count}}}.'''
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f'SELECT key, col_name, category, raw_count '
                    f'FROM {self.cat_state_table_name} '
                    f'WHERE table_index = %s AND key_col_index = %s',
                    (int(table_index), int(group_col_index)),
                )
                rows = cur.fetchall()
        finally:
            if transient:
                pool.close()
        out: dict = {}
        for key, col_name, category, raw_count in rows:
            out.setdefault(col_name, {}).setdefault(key, {})[category] = int(raw_count)
        return out

    # ---------- merge + derive ----------------------------------------------------

    def _merge_num_dicts(self, existing: dict, delta_records: pl.DataFrame | None) -> dict:
        if delta_records is None or delta_records.height == 0:
            return existing
        merged = {k: dict(v) for k, v in existing.items()}
        decode = self._median_backend.decode
        merge_states = self._median_backend.merge
        for row in delta_records.iter_rows(named=True):
            kc = (row['key'], row['col_name'])
            new_state = decode(row['values_'])
            if kc in merged:
                m = merged[kc]
                m['count'] += int(row['count_'])
                m['sum']   += float(row['sum_'])
                m['min']    = min(m['min'], float(row['min_']))
                m['max']    = max(m['max'], float(row['max_']))
                m['values'] = merge_states(m['values'], new_state)
            else:
                merged[kc] = {
                    'count': int(row['count_']),
                    'sum':   float(row['sum_']),
                    'min':   float(row['min_']),
                    'max':   float(row['max_']),
                    'values': new_state,
                }
        return merged

    @staticmethod
    def _merge_cat_dicts(existing: dict, delta_records: pl.DataFrame | None) -> dict:
        if delta_records is None or delta_records.height == 0:
            return existing
        merged: dict = {k: {kk: dict(vv) for kk, vv in v.items()} for k, v in existing.items()}
        for row in delta_records.iter_rows(named=True):
            col = row['col_name']
            key = row['key']
            cat = row['category']
            cnt = int(row['raw_count'])
            merged.setdefault(col, {}).setdefault(key, {})
            merged[col][key][cat] = merged[col][key].get(cat, 0) + cnt
        return merged

    def _dict_to_num_state_df(self, num_dict: dict, table_index: int, group_col_index: int) -> pl.DataFrame | None:
        if not num_dict:
            return None
        rows = []
        encode = self._median_backend.encode
        for (key, col), st in num_dict.items():
            rows.append({
                'table_index':   int(table_index),
                'key_col_index': int(group_col_index),
                'key':           key,
                'col_name':      col,
                'count_':        int(st['count']),
                'sum_':          float(st['sum']),
                'min_':          float(st['min']),
                'max_':          float(st['max']),
                'values_':       encode(st['values']),
            })
        return pl.DataFrame(rows, schema={
            'table_index':   pl.Int32,
            'key_col_index': pl.Int32,
            'key':           pl.String,
            'col_name':      pl.String,
            'count_':        pl.Int64,
            'sum_':          pl.Float64,
            'min_':          pl.Float64,
            'max_':          pl.Float64,
            'values_':       pl.Binary,
        })

    def _dict_to_cat_state_df(self, cat_dict: dict, table_index: int, group_col_index: int) -> pl.DataFrame | None:
        if not cat_dict:
            return None
        rows = []
        for col, per_key in cat_dict.items():
            for key, per_cat in per_key.items():
                for cat, cnt in per_cat.items():
                    rows.append({
                        'table_index':   int(table_index),
                        'key_col_index': int(group_col_index),
                        'key':           key,
                        'col_name':      col,
                        'category':      cat,
                        'raw_count':     int(cnt),
                    })
        if not rows:
            return None
        return pl.DataFrame(rows, schema={
            'table_index':   pl.Int32,
            'key_col_index': pl.Int32,
            'key':           pl.String,
            'col_name':      pl.String,
            'category':      pl.String,
            'raw_count':     pl.Int64,
        })

    def _derive_features_from_state(
        self,
        num_dict: dict,
        cat_dict: dict,
        group_col: str,
        numeric_cols: list[str],
        cat_cols: list[str],
    ) -> pl.DataFrame:
        '''Reproduce the schema and values of `_combined_query` (no feature_extraction).'''
        # All keys present in both halves (mirrors the inner-join semantics in `_combined_query`).
        keys_num: dict[str, set[str]] = {}
        for (k, c), _ in num_dict.items():
            keys_num.setdefault(c, set()).add(k)
        keys_cat: dict[str, set[str]] = {c: set(d.keys()) for c, d in cat_dict.items()}

        if numeric_cols:
            common_num = set.intersection(*(keys_num.get(c, set()) for c in numeric_cols)) if numeric_cols else set()
        else:
            common_num = None
        if cat_cols:
            common_cat = set.intersection(*(keys_cat.get(c, set()) for c in cat_cols)) if cat_cols else set()
        else:
            common_cat = None

        if common_num is not None and common_cat is not None:
            keys = sorted(common_num & common_cat)
        elif common_num is not None:
            keys = sorted(common_num)
        elif common_cat is not None:
            keys = sorted(common_cat)
        else:
            return pl.DataFrame()

        # N_c per categorical column = distinct categories across all keys.
        N_c: dict[str, int] = {}
        for c in cat_cols:
            seen: set[str] = set()
            for per_cat in cat_dict.get(c, {}).values():
                seen.update(per_cat.keys())
            N_c[c] = len(seen)

        rows = []
        backend = self._median_backend
        for key in keys:
            row: dict = {group_col: key}
            for c in numeric_cols:
                st = num_dict[(key, c)]
                vs = st['values']
                n = backend.size(vs)
                if n == 0:
                    continue
                row[f'{c}_mean']   = float(st['sum']) / int(st['count'])
                row[f'{c}_median'] = backend.median(vs)
                row[f'{c}_max']    = float(st['max'])
                row[f'{c}_min']    = float(st['min'])
            for c in cat_cols:
                Nc = N_c[c]
                if Nc == 0:
                    continue
                counts = list(cat_dict[c][key].values())
                row[f'{c}_mean']    = sum(counts) / (Nc ** 2)
                row[f'{c}_max']     = max(counts) / Nc
                row[f'{c}_nunique'] = len({*counts})  # mirrors n_unique() of the scaled list
            rows.append(row)
        if not rows:
            return pl.DataFrame()
        return pl.DataFrame(rows)

    def _postprocess_feature_df(self, combined_result: pl.DataFrame, group_col: str) -> pl.DataFrame:
        '''Apply the dedupe-by-hash and constant-column filters from `_combined_query`.'''
        if combined_result.is_empty():
            return combined_result
        feature_cols = [c for c in combined_result.columns if c != group_col]
        if not feature_cols:
            return combined_result
        col_hashes = {c: combined_result[c].hash().sum() for c in feature_cols}
        unique_cols = list({v: k for k, v in col_hashes.items()}.values())
        combined_result = combined_result.select([group_col] + unique_cols)
        constant_check = combined_result.select(
            (pl.all().exclude(group_col) - pl.all().exclude(group_col).mean())
            .sign().var() != 0
        )
        if constant_check.height == 0:
            return pl.DataFrame()
        non_constant_cols = [c for c, v in zip(constant_check.columns, constant_check.row(0)) if v]
        return combined_result.select([group_col] + non_constant_cols)

    # ---------- sketch reuse ------------------------------------------------------

    def _sketch_from_features(
        self,
        result: pl.DataFrame,
        group_col: str,
        group_col_index: int,
        table_index: int,
        table_name: str,
        max_feature_index: int = 0,
    ) -> tuple[list[pl.DataFrame], int]:
        '''Mirror the sketch-building tail of `_index_features` for a derived feature DataFrame.'''
        if result.is_empty() or result.width <= 1 or result.height <= 1:
            return [], max_feature_index
        result = result.with_columns(pl.col(group_col).cast(pl.String).str.slice(0, 335))
        grouped_columns = self._grouped_columns_mapping(result, group_col)
        feature_selection_index: list[pl.DataFrame] = []
        for col in list(grouped_columns.keys()):
            try:
                mapped_cols = grouped_columns[col]
            except KeyError:
                continue
            feature_slice = result.select([group_col] + mapped_cols)
            base_col = None
            for fsc in feature_slice.columns:
                if fsc.count('_') == 1:
                    base_col = fsc
                    break
            if base_col is None:
                continue
            combs = self._combinations_with_element(mapped_cols, base_col)
            features_col_indices = list(range(max_feature_index, len(combs) + max_feature_index))
            max_feature_index = max(features_col_indices) + 1
            key_col = feature_slice.select(group_col).to_series()
            feature_slice = feature_slice.select(mapped_cols).to_numpy()
            feature_selection_index.extend(
                self._inverted_index_from_feature_matrix(
                    feature_slice, key_col, table_index, table_name, group_col_index,
                    combs, features_col_indices, mapped_cols,
                )
            )
            if len(mapped_cols) > 1:
                mapped_cols.remove(base_col)
                for k in mapped_cols:
                    grouped_columns.pop(k, None)
        return feature_selection_index, max_feature_index

    # ---------- public update API -------------------------------------------------

    def update_table(self, table_name: str, new_rows: pl.LazyFrame | pl.DataFrame) -> bool:
        '''Append new rows to an indexed table and refresh its section of the fs index.

        Returns True if the update made changes; False if the table was unknown
        or the delta was empty.'''
        meta = self._fetch_table_meta(table_name)
        if meta is None:
            self.logger.error(f'update_table: {table_name!r} not in {self.tables_meta_table_name}')
            return False
        table_index   = meta['table_index']
        numeric_cols  = meta['numeric_cols']
        cat_cols      = meta['cat_cols']
        key_columns   = meta['key_columns']
        # Note: useful_columns from meta is informational; the actual column set in
        # state is whatever was emitted at offline time. We just pass through cols
        # known to meta and ignore unknown ones from new_rows.
        new_df = new_rows.collect() if isinstance(new_rows, pl.LazyFrame) else new_rows
        if new_df.is_empty():
            return False
        # Match `index_table`'s column-name normalization.
        renames = {orig: process_key(orig) for orig in new_df.columns}
        renames = {o: n for o, n in renames.items() if o != n}
        if renames:
            new_df = new_df.rename(renames)
        # Drop columns not tracked by state (and warn). Coerce numeric cols to Float64.
        known = set(numeric_cols) | set(cat_cols) | set(key_columns)
        unknown = [c for c in new_df.columns if c not in known]
        if unknown:
            self.logger.warning(f'update_table {table_name!r}: dropping columns not in meta: {unknown}')
            new_df = new_df.drop(unknown)

        any_change = False
        rebuilt_total_rows = 0
        for group_col_index, group_col in enumerate(key_columns):
            if group_col not in new_df.columns:
                continue
            sub = new_df.drop_nulls(subset=[group_col])
            if sub.height == 0:
                continue

            # --- 1) compute delta state from the new rows ---
            delta_num = self._build_num_state_records(sub, group_col, group_col_index, numeric_cols, table_index)
            delta_cat = self._build_cat_state_records(sub, group_col, group_col_index, cat_cols,     table_index)
            if delta_num is None and delta_cat is None:
                continue

            # --- 2) read existing state and merge ---
            existing_num = self._read_num_state(table_index, group_col_index)
            existing_cat = self._read_cat_state(table_index, group_col_index)
            merged_num = self._merge_num_dicts(existing_num, delta_num)
            merged_cat = self._merge_cat_dicts(existing_cat, delta_cat)

            # --- 3) re-derive features from merged state ---
            derived = self._derive_features_from_state(merged_num, merged_cat, group_col, numeric_cols, cat_cols)
            derived = self._postprocess_feature_df(derived, group_col)

            sketch_dfs, _ = self._sketch_from_features(
                derived, group_col, group_col_index, table_index, table_name, max_feature_index=0,
            )

            # --- 4) replace the (table_index, key_col_index) section.
            #     Two-phase, non-atomic: DELETE then COPY-INSERT. Concurrent readers
            #     may see an empty section during the gap. Acceptable for batch updates.
            with self._state_pool()[0].connection() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT pg_advisory_xact_lock(%s)', (int(table_index),))
                    cur.execute(
                        f'DELETE FROM {self.feature_selection_table_name} '
                        f'WHERE table_index = %s AND key_col_index = %s',
                        (int(table_index), int(group_col_index)),
                    )
                    cur.execute(
                        f'DELETE FROM {self.num_state_table_name} '
                        f'WHERE table_index = %s AND key_col_index = %s',
                        (int(table_index), int(group_col_index)),
                    )
                    cur.execute(
                        f'DELETE FROM {self.cat_state_table_name} '
                        f'WHERE table_index = %s AND key_col_index = %s',
                        (int(table_index), int(group_col_index)),
                    )
                    conn.commit()

            self._write_state_to_db(self._dict_to_num_state_df(merged_num, table_index, group_col_index), self.num_state_table_name)
            self._write_state_to_db(self._dict_to_cat_state_df(merged_cat, table_index, group_col_index), self.cat_state_table_name)

            if sketch_dfs:
                sketch_df = pl.concat(sketch_dfs, how='vertical_relaxed')
                self.write_to_db(sketch_df, self.feature_selection_table_name)
                rebuilt_total_rows += sketch_df.height

            any_change = True

        if any_change:
            self._write_table_meta(
                table_index=table_index,
                table_name=table_name,
                numeric_cols=numeric_cols,
                cat_cols=cat_cols,
                key_columns=key_columns,
                useful_columns=meta['useful_columns'],
                n_rows=int(meta['n_rows'] or 0) + new_df.height,
            )
            self.logger.info(
                f'update_table {table_name!r} (table_index={table_index}): '
                f'+{new_df.height} rows, rebuilt {rebuilt_total_rows} fs rows'
            )
        return any_change

    def update_lake(self, updates_dir: str, from_tar_archive: bool = False) -> None:
        '''Sequentially apply append-only updates to every table found under `updates_dir`.

        Each subitem must match an already-indexed table by name (sans extension).
        Tables not previously indexed are skipped with a warning.'''
        if os.path.isdir(updates_dir):
            loader = DirectoryLoader()
        elif from_tar_archive:
            loader = TarArchiveLoader()
        else:
            loader = ZipArchiveLoader()

        # Use the persistent pool for the whole run if not already open.
        owns_pool = getattr(self, '_conn_pool', None) is None
        if owns_pool:
            self._conn_pool = ConnectionPool(self.conninfo, min_size=4, max_size=16, timeout=1800)
            self._conn_pool.open()
        try:
            for name, lf in loader.load_tables(updates_dir, self.logger, None):
                table_name = os.path.splitext(name)[0] if os.path.splitext(name)[1] else name
                try:
                    self.update_table(table_name, lf)
                except Exception as e:
                    self.logger.error(f'update_lake: failed updating {table_name!r}: {e!r}')
        finally:
            if owns_pool:
                self._conn_pool.close()
                self._conn_pool = None


    # ============================================================================
    # Union discovery (Matryoshka-native, no BLEND)
    # ============================================================================
    #
    # Implements BLEND's SC seeker + Counter combiner pattern directly on the
    # existing index/state tables:
    #
    #   * `<fs_table>.key`       — distinct cell values from every indexed key column
    #     (already btree-indexed on `key`). Probed for queries against key columns.
    #   * `<fs>_cat_state.category` — distinct categorical cell values from every
    #     categorical column of every indexed table (btree-indexed on
    #     (table_index, key_col_index, col_name)).
    #
    # We probe both with `WHERE col = ANY(%s::text[])` (one binding, no parameter cap),
    # aggregate per (table_index, col), then bipartite-match the per-column overlaps
    # to produce a column mapping {query_col -> peer_col} usable by `update_table`.
    #
    # Numeric columns are not searchable in v1; documented in the paper as future work.

    _UNION_SEEKER_VALUE_CAP = 10_000  # per-column distinct-value cap for IN UNNEST probes

    def _query_value_sets(self, df: pl.DataFrame) -> dict[str, list[str]]:
        '''Distinct string-cast non-null values per column, capped to _UNION_SEEKER_VALUE_CAP.'''
        out: dict[str, list[str]] = {}
        for c in df.columns:
            s = df[c]
            if s.dtype in (pl.Binary, pl.Object):
                continue
            try:
                vals = s.cast(pl.String, strict=False).drop_nulls().unique().to_list()
            except Exception:
                continue
            if not vals:
                continue
            if len(vals) > self._UNION_SEEKER_VALUE_CAP:
                # Deterministic head-cap; values are unique so this is just a slice.
                vals = vals[: self._UNION_SEEKER_VALUE_CAP]
            out[c] = vals
        return out

    def _seek_key_overlaps(
        self,
        value_sets: dict[str, list[str]],
        exclude_table_index: int | None,
        top_k: int,
    ) -> pl.DataFrame:
        '''Per-column SC seeker against `<fs_table>.key` (key-column cells).

        Returns rows: (q_col, table_index, key_col_index, overlap).
        '''
        rows: list[dict] = []
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                for q_col, vals in value_sets.items():
                    if not vals:
                        continue
                    sql = (
                        f'SELECT table_index, key_col_index, COUNT(DISTINCT key) AS overlap '
                        f'FROM {self.feature_selection_table_name} '
                        f'WHERE key = ANY(%s::text[]) '
                    )
                    params: list = [vals]
                    if exclude_table_index is not None:
                        sql += 'AND table_index <> %s '
                        params.append(int(exclude_table_index))
                    sql += (
                        'GROUP BY table_index, key_col_index '
                        'ORDER BY overlap DESC LIMIT %s'
                    )
                    params.append(int(top_k))
                    cur.execute(sql, params)
                    for table_index, key_col_index, overlap in cur.fetchall():
                        rows.append({
                            'q_col':         q_col,
                            'r_kind':        'key',
                            'table_index':   int(table_index),
                            'r_col':         f'__key_col_{int(key_col_index)}__',
                            'key_col_index': int(key_col_index),
                            'overlap':       int(overlap),
                        })
        finally:
            if transient:
                pool.close()
        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
            'q_col': pl.String, 'r_kind': pl.String, 'table_index': pl.Int64,
            'r_col': pl.String, 'key_col_index': pl.Int64, 'overlap': pl.Int64,
        })

    def _seek_cat_overlaps(
        self,
        value_sets: dict[str, list[str]],
        exclude_table_index: int | None,
        top_k: int,
    ) -> pl.DataFrame:
        '''Per-column SC seeker against `<fs>_cat_state.category` (categorical cells).

        Returns rows: (q_col, table_index, col_name, overlap). Aggregates over
        all key_col_index values for the same (table_index, col_name) since cat_state
        replicates raw counts per key column.
        '''
        rows: list[dict] = []
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                for q_col, vals in value_sets.items():
                    if not vals:
                        continue
                    sql = (
                        f'SELECT table_index, col_name, COUNT(DISTINCT category) AS overlap '
                        f'FROM {self.cat_state_table_name} '
                        f'WHERE category = ANY(%s::text[]) '
                    )
                    params: list = [vals]
                    if exclude_table_index is not None:
                        sql += 'AND table_index <> %s '
                        params.append(int(exclude_table_index))
                    sql += (
                        'GROUP BY table_index, col_name '
                        'ORDER BY overlap DESC LIMIT %s'
                    )
                    params.append(int(top_k))
                    cur.execute(sql, params)
                    for table_index, col_name, overlap in cur.fetchall():
                        rows.append({
                            'q_col':       q_col,
                            'r_kind':      'cat',
                            'table_index': int(table_index),
                            'r_col':       str(col_name),
                            'overlap':     int(overlap),
                        })
        finally:
            if transient:
                pool.close()
        return pl.DataFrame(rows) if rows else pl.DataFrame(schema={
            'q_col': pl.String, 'r_kind': pl.String, 'table_index': pl.Int64,
            'r_col': pl.String, 'overlap': pl.Int64,
        })

    def _table_meta_by_index(self, table_indices: list[int]) -> dict[int, dict]:
        '''Bulk fetch meta rows for several table indices. Returns {table_index: meta}.'''
        if not table_indices:
            return {}
        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute(
                    f'SELECT table_index, table_name, numeric_cols, cat_cols, '
                    f'key_columns, useful_columns, n_rows '
                    f'FROM {self.tables_meta_table_name} '
                    f'WHERE table_index = ANY(%s::int[])',
                    (list(map(int, table_indices)),),
                )
                rows = cur.fetchall()
        finally:
            if transient:
                pool.close()
        return {
            int(r[0]): {
                'table_index':    int(r[0]),
                'table_name':     r[1],
                'numeric_cols':   list(r[2] or []),
                'cat_cols':       list(r[3] or []),
                'key_columns':    list(r[4] or []),
                'useful_columns': list(r[5] or []),
                'n_rows':         int(r[6] or 0),
            }
            for r in rows
        }

    def find_union_peer(
        self,
        new_table: pl.LazyFrame | pl.DataFrame,
        exclude_table_index: int | None = None,
        top_k_seeker: int = 200,
        top_k_combiner: int = 10,
        threshold: float = 0.0,
        min_cols: float = 0.5,
    ) -> dict | None:
        '''Find the single best union peer for `new_table` in the index, or None.

        Returns: {'peer_table_index', 'peer_table_name', 'mapping', 'score',
        'matched_cols'} where `mapping` is {q_col -> peer_col_name}. `peer_col_name`
        is a real column name when matched against cat_state, or '__key_col_{i}__'
        when matched against the fs key columns (use the table's `meta.key_columns[i]`
        to resolve).

        Parameters mirror BLEND's SC + Counter combiner. `threshold` filters by
        normalised cell-overlap (sum of matched overlaps / (n_rows_q + n_rows_peer));
        `min_cols` filters by `2 * |mapping| / (|q_cols| + |peer_cols|)`.
        '''
        df = new_table.collect() if isinstance(new_table, pl.LazyFrame) else new_table
        if df.is_empty():
            return None
        # Normalize column names like the rest of the index.
        df = df.rename({c: process_key(c) for c in df.columns})
        value_sets = self._query_value_sets(df)
        if not value_sets:
            return None
        n_rows_q = df.height

        # --- 1) seekers ---
        sk_key = self._seek_key_overlaps(value_sets, exclude_table_index, top_k_seeker)
        sk_cat = self._seek_cat_overlaps(value_sets, exclude_table_index, top_k_seeker)
        all_sk = pl.concat([sk_key, sk_cat], how='diagonal_relaxed') if (sk_key.height or sk_cat.height) else None
        if all_sk is None or all_sk.height == 0:
            return None

        # --- 2) Counter combiner: rank candidate tables by total overlap ---
        candidates = (
            all_sk.group_by('table_index')
            .agg(pl.col('overlap').sum().alias('total_overlap'),
                 pl.col('q_col').n_unique().alias('matched_q_cols'))
            .sort('total_overlap', descending=True)
            .head(top_k_combiner)
        )
        if candidates.height == 0:
            return None

        cand_indices = [int(x) for x in candidates['table_index'].to_list()]
        metas = self._table_meta_by_index(cand_indices)

        # --- 3) per-candidate bipartite matching ---
        results: list[dict] = []
        q_cols_set = set(value_sets.keys())
        for ti in cand_indices:
            meta = metas.get(ti)
            if meta is None:
                continue
            sub = all_sk.filter(pl.col('table_index') == ti)
            # Build edge list: (q_col, r_col_label, weight). Both label spaces
            # (key vs cat) live in `r_col` already; key labels are
            # `__key_col_{i}__`.
            edges = [
                (f'l::{r["q_col"]}', f'r::{r["r_col"]}', float(r['overlap']))
                for r in sub.iter_rows(named=True)
            ]
            if not edges:
                continue
            g = nx.Graph()
            g.add_weighted_edges_from(edges)
            mate = nx.max_weight_matching(g, maxcardinality=False)
            mapping: dict[str, str] = {}
            matched_overlap_sum = 0.0
            for u, v in mate:
                # Make sure left/right labels are in the right order.
                if u.startswith('l::') and v.startswith('r::'):
                    l_lbl, r_lbl = u, v
                elif v.startswith('l::') and u.startswith('r::'):
                    l_lbl, r_lbl = v, u
                else:
                    continue
                q_col = l_lbl[3:]
                r_col = r_lbl[3:]
                # Map placeholder `__key_col_{i}__` to the actual key column name.
                if r_col.startswith('__key_col_') and r_col.endswith('__'):
                    try:
                        idx = int(r_col[len('__key_col_'):-2])
                    except ValueError:
                        continue
                    if idx >= len(meta['key_columns']):
                        continue
                    r_col = meta['key_columns'][idx]
                mapping[q_col] = r_col
                matched_overlap_sum += float(g[u][v].get('weight', 0.0))

            if not mapping:
                continue
            # Filters: minimum column coverage and cell-overlap threshold.
            n_q_cols = len(q_cols_set)
            n_r_cols = max(len(meta['cat_cols']) + len(meta['numeric_cols']), 1)
            col_coverage = (2.0 * len(mapping)) / (n_q_cols + n_r_cols)
            if col_coverage < min_cols:
                continue
            n_rows_r = max(int(meta['n_rows']) or 1, 1)
            cell_overlap_norm = matched_overlap_sum / (n_rows_q + n_rows_r)
            if cell_overlap_norm < threshold:
                continue
            results.append({
                'peer_table_index': ti,
                'peer_table_name':  meta['table_name'],
                'mapping':          mapping,
                'score':            cell_overlap_norm,
                'col_coverage':     col_coverage,
                'matched_cols':     len(mapping),
            })

        if not results:
            return None
        results.sort(key=lambda x: (x['score'], x['col_coverage']), reverse=True)
        return results[0]


    # ============================================================================
    # Scoped rebuild baseline: DELETE table_index's section + index_table over the
    # already-merged (anchor + delta) DataFrame. The baseline forced by the reviewer's
    # comment — "if you know it's the same table, just re-index it from scratch."
    # ============================================================================

    def scoped_rebuild_table(
        self,
        table_name: str,
        merged_table: pl.LazyFrame | pl.DataFrame,
        key_columns: list[str] | None = None,
    ) -> bool:
        '''Delete `table_name`'s entries from fs/overlap/state, then re-index
        `merged_table` under the SAME `table_index`. Output is byte-equal to a
        full-rebuild slice (modulo non-deterministic prune sampling, now seeded).

        Returns True on success, False if the table is unknown.'''
        meta = self._fetch_table_meta(table_name)
        if meta is None:
            self.logger.error(f'scoped_rebuild_table: {table_name!r} not in {self.tables_meta_table_name}')
            return False
        table_index = int(meta['table_index'])
        kc = key_columns if key_columns is not None else (meta.get('key_columns') or None)

        pool, transient = self._state_pool()
        try:
            with pool.connection() as conn, conn.cursor() as cur:
                cur.execute('SELECT pg_advisory_xact_lock(%s)', (table_index,))
                cur.execute(
                    f'DELETE FROM {self.feature_selection_table_name} WHERE table_index = %s',
                    (table_index,),
                )
                cur.execute(
                    f'DELETE FROM {self.overlap_table_name} WHERE table_index = %s',
                    (table_index,),
                )
                cur.execute(
                    f'DELETE FROM {self.num_state_table_name} WHERE table_index = %s',
                    (table_index,),
                )
                cur.execute(
                    f'DELETE FROM {self.cat_state_table_name} WHERE table_index = %s',
                    (table_index,),
                )
                cur.execute(
                    f'DELETE FROM {self.tables_meta_table_name} WHERE table_index = %s',
                    (table_index,),
                )
                conn.commit()
        finally:
            if transient:
                pool.close()

        lf = merged_table.lazy() if isinstance(merged_table, pl.DataFrame) else merged_table
        # `index_table` writes meta + state itself, then emits sketch rows.
        # Calls into the existing single-worker code path.
        index_data = self.index_table(lf, table_index=table_index, table_name=table_name,
                                      key_columns=kc, debug=False)
        if index_data:
            try:
                final_table = pl.concat(index_data, how='vertical_relaxed')
                if final_table.height > 0:
                    self.write_to_db(final_table, self.feature_selection_table_name)
            except Exception as e:
                self.logger.error(f'scoped_rebuild_table {table_name!r}: write_to_db failed: {e!r}')
                return False
        return True
