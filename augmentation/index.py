import gc
import os
import tempfile
import pgpq
import pyarrow as pa
import pyarrow.parquet as pq
import ray
import re
import time
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from augmentation.utils.lakes.archives import TarArchiveLoader, ZipArchiveLoader
from augmentation.utils.lakes.multiprocessing_utils import TableProcessor, RayTableProcessor
import numpy as np
import polars as pl
import polars_hash as plh
from arrow_json import array_to_utf8_json_array
from itertools import combinations
from polars.exceptions import ColumnNotFoundError
from psycopg_pool import ConnectionPool
from ray.experimental.tqdm_ray import tqdm as tqdm_ray
from typing import Callable
from augmentation.utils.common import process_key, semiring_aggregates, profile
from augmentation.utils.database.query_processing import DBHandler
from augmentation.utils.logging.logger_config import setup_logger


class ExhaustiveIndex(DBHandler):
    def __init__(self, data_dir: str = None, feature_selection_table_name: str = None, overlap_table_name: str = None, tunnel: bool = False, batch_size: int = 1, max_workers: int = 1, feature_extraction: bool = False) -> None:
        super().__init__(feature_selection_table_name, overlap_table_name, tunnel)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.feature_extraction = feature_extraction

        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        log_dir = os.path.join(script_dir, 'utils', 'logging', 'logs')
        self.logger = setup_logger(name='exhaustive_index', log_dir=log_dir, log_file=f'exhaustive_index_{timestamp}.log', silent=False)

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
        self.create_tables(conn)
        conn.closeall()

        if from_tar_archive:
            loader = TarArchiveLoader()
        else:
            loader = ZipArchiveLoader()

        checkpoint_name = checkpoint if from_checkpoint else None
        all_tables = list(loader.load_tables(self.data_dir, self.logger, checkpoint_name))
      
        batch_size = self.max_workers if self.batch_size is None else self.batch_size
        batches = [all_tables[i:i + batch_size] for i in range(0, len(all_tables), batch_size)]
        for tables in batches:
            if self.max_workers > 1:
                # materialize tables to DataFrames for parallel processing
                self.logger.info("Materializing tables for parallel processing...")
                table_data = []
                for i, (name, lazy_table) in enumerate(tables):
                    df = lazy_table.collect()
                    table_data.append((name, df, table_index + i))
            else:
                # keep as lazy frames for sequential processing
                table_data = [(name, table, table_index + i) for i, (name, table) in enumerate(tables)]

            if self.max_workers > 1:
                if not ray.is_initialized():
                    ray.init(_temp_dir='/app/data/tmp')
                processors = [RayTableProcessor.remote(self.index_table) for _ in range(self.max_workers)]
                futures = []
                for i, data in enumerate(table_data):
                    processor = processors[i % len(processors)]
                    future = processor.process_table.remote(data)
                    futures.append(future)

                results = []
                index_data = []
                for future in tqdm_ray(futures, desc='Processing tables'):
                    result = ray.get(future)
                    results.append(result[1:])

                    data, _, _, log_message = result
                    for line in log_message.split('\n'):
                        if 'Error' in line:
                            self.logger.error(line)
                        else:
                            self.logger.info(line)
                    index_data.extend(data)
                results = [(idx, success) for idx, success, _ in results]
            else:
                processor = TableProcessor(self.index_table)            
                results = []
                index_data = []
                for data in table_data:
                    result = processor.process_table(data)
                    d, table_index, success, log_message = result
                    for line in log_message.split('\n'):
                        if 'Error' in line:
                            self.logger.error(line)
                        else:
                            self.logger.info(line)
                    results.append((table_index, success))
                    index_data.extend(d)

            if len(index_data) == 0:
                table_index += 1
                continue
            final_table = pl.concat(index_data, how='vertical_relaxed')
            del index_data
            self.write_to_db(final_table, self.feature_selection_table_name)

            successful = sum(1 for _, success in results if success)
            failed = len(results) - successful
            self.logger.info(f'Finished indexing tables: {successful} successful, {failed} failed')
            table_index = max(idx for idx, _ in results) + 1

        self.logger.info('Finished indexing tables in the folder')
        self.logger.handlers.clear()
        if ray.is_initialized():
            ray.shutdown()

        return table_index


    def index_table(self, table: pl.LazyFrame, table_index: int, key_columns: list[str] = None, debug: bool = False) -> None:
        # normalize column names so that further prefixes and suffixes logic works
        new_colnames = [process_key(name) for name in table.collect_schema().keys()]
        duplicate_cols_indices = [i for i, name in enumerate(new_colnames) if new_colnames.count(name) > 1]
        for i in duplicate_cols_indices:
            new_colnames[i] += f'{i}v2'
        table = table.rename({name: new_col for name, new_col in zip(table.collect_schema().keys(), new_colnames)}).collect().lazy()
        if not key_columns:
            useful_columns = self._prune_features(table)
            key_columns = self._identify_key_columns(table)
            if self.max_workers == 1:
                intersection = list(set(useful_columns).intersection(set(key_columns)))
                key_columns = set(key_columns).intersection(set(intersection))
            else:
                intersection = key_columns
            table = table.select(intersection).collect().lazy()
        else:
            useful_columns = self._prune_features(table)
            table = table.select([col for col in table.collect_schema().keys() if col in useful_columns+key_columns]).collect().lazy()

        table, numeric_cols, non_numeric_cols = self._split_columns_by_type(table)

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
                feature_selection_index, new_max_feature_index = self._index_features(table, query_func, group_col, group_col_index, numeric_cols, non_numeric_cols, table_index, max_feature_index, useful_columns, False)
                max_feature_index = new_max_feature_index
                group_col_index += 1
                if not debug:
                    if self.max_workers > 1:
                        if len(feature_selection_index) > 0:
                            index_data.extend(feature_selection_index)
                    else:
                        if len(feature_selection_index) > 0:
                            self.write_to_db(pl.concat(feature_selection_index, how='vertical_relaxed'), self.feature_selection_table_name)
                else:
                    print(feature_selection_index)

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
        max_feature_index: int,
        useful_columns: list[str],
        feature_extraction: bool
    ) -> list[tuple]:
        table = table.drop_nulls(subset=[group_col]).collect()
        table = table[[s.name for s in table if not (s.null_count() == table.height)]]
        numeric_cols = [c for c in numeric_cols if c in table.columns]
        non_numeric_cols = [c for c in non_numeric_cols if c in table.columns]
        if table.width <= 1:
            return [], max_feature_index
        
        table = table.lazy()
        result = query_func(table, group_col, numeric_cols, non_numeric_cols, useful_columns, feature_extraction)
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
            feature_selection_index.extend(self._inverted_index_from_feature_matrix(feature_slice, key_col, table_index, group_col_index, combs, features_col_indices))

            if len(mapped_cols) > 1:
                mapped_cols.remove(base_col)
                for k in mapped_cols:
                    try:
                        del grouped_columns[k]
                    except KeyError:
                        pass

        new_max_feature_index = max_feature_index

        return feature_selection_index, new_max_feature_index


    def _inverted_index_from_feature_matrix(self, feature_slice: np.ndarray, key_col: pl.Series, table_index: int, group_col_index: int, combs: list[list[int]], features_col_indices: list[str]) -> list[tuple]:
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
            tmp_dir = "/app/data/tmp"
            os.makedirs(tmp_dir, exist_ok=True)

            fd, final_path = tempfile.mkstemp(suffix=".parquet", dir=tmp_dir)
            os.close(fd)

            tmp_path = final_path + ".writing"

            match pg_table_name:
                case self.feature_selection_table_name:
                    encode_cols = ['sum', 'diag', 'qcr_term_positive', 'qcr_term_negative']

                    # Write the dataframe to Parquet to avoid huge in-memory Arrow conversion
                    df.write_parquet(tmp_path, compression="snappy")
                    os.replace(tmp_path, final_path)
                    del df
                    gc.collect()
                    try:
                        pf = pq.ParquetFile(final_path)
                        for batch in pf.iter_batches(batch_size=10_000):
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
                    df.write_parquet(tmp_path, compression="snappy")
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
                    pa.field('key_col_index', pa.int32()),
                    pa.field('row_index', pa.int64()),
                    pa.field('count', pa.int32()),
                    pa.field('sum', pa.large_binary()),
                    pa.field('diag', pa.large_binary()),
                    pa.field('cofactors', pa.large_binary()),
                    pa.field('shape', pa.list_(pa.int32())),
                    pa.field('qcr_term_positive', pa.large_binary()),
                    pa.field('qcr_term_negative', pa.large_binary())
                ])
            case self.overlap_table_name:
                target_schema = pa.schema([
                    pa.field('key', pa.string()),
                    pa.field('table_index', pa.int32()),
                    pa.field('key_col_index', pa.int32()),
                    pa.field('row_index', pa.int64())
                ])
        conn_pool = ConnectionPool(self.conninfo, min_size=8, max_size=32, timeout=1800)
        conn_pool.open()
        try:
            for batch in self._preprocess_dataframe(feature_selection_index, target_schema, pg_table_name):
                self._write_batch_to_pg(batch, conn_pool, pg_table_name)
        finally:
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
            df = df.collect().sample(n=sample_size)
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
        for table_name in self.btree_index_cols:
            queries = self.btree_index_cols[table_name]
            for i, q in enumerate(queries):
                query = self.create_index_query.replace('index_name', f'{table_name}_index_{i}').replace('table_name', f'{table_name}').replace('method (columns)', q)
                self.create_index(conn, query)
                self.logger.info(f'Created in-database index for table {table_name}')