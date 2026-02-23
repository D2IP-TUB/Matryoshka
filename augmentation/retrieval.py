import csv
import json
import os
import pickle
import re
import sys
import time
import adbc_driver_postgresql.dbapi as adbc_dbapi
import numpy as np
import polars as pl
import polars.selectors as cs
import polars_hash as plh
import requests
from .utils.database.query_processing import DBHandler
from .utils.exceptions import KeyNotFoundError
from .utils.logging.logger_config import setup_logger
from .Aurum.hnsw_search import HNSWSearcher
from datasketch import MinHash


class JoinDiscovery(DBHandler):
    def __init__(self, feature_selection_table_name: str, overlap_table_name: str, verbose: bool = False, log_file_name: str = None) -> None:
        '''
        Parameters:
        ----------
        token_index_table_name: str
            Name of the table containing the token index
        
        feature_selection_table_name: str
            Name of the table containing the query index
        '''
        super().__init__(feature_selection_table_name, overlap_table_name)

        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        log_dir = os.path.join(script_dir, 'utils', 'logging', 'logs')
        if verbose:
            self.logger = setup_logger(name='join_selection', log_dir=log_dir, log_file=f'{log_file_name}_{timestamp}_retrieval.log')
        else:
            self.logger = setup_logger(name='join_selection', log_dir=log_dir, log_file=f'{log_file_name}_{timestamp}_retrieval.log', silent=True)

        self.overlap_query = 'SELECT oq.key, oq.table_index, oq.key_col_index, oq.row_index, oq.number_of_tokens ' \
                             'FROM (' \
                             "    SELECT ARRAY_AGG(key || '') as key, table_index, key_col_index, ARRAY_AGG(row_index || '') as row_index, COUNT(key) as number_of_tokens " \
                            f'    FROM {feature_selection_table_name} ' \
                             '    WHERE key IN (\'joint_distinct_tokens\' ) ' \
                             '    GROUP BY table_index, key_col_index' \
                             ') as oq ' \
                             'ORDER BY number_of_tokens DESC ' \
                             'LIMIT top_k;'
        
        self.index_col_name = 'index'
    

    def find_joinable_tables(self, query_column: pl.DataFrame, top_k: int, user_table_processed: pl.DataFrame, join_selection: bool = False) -> pl.DataFrame:
        '''
        Main method: finds joinable tables for a given query column.

        Parameters:
        ----------
        query_column: pl.DataFrame
            Query column to find joinable tables for. Currently only supports a `pl.String` type column
        
        top_k: int
            Number of top columns to consider for overlap

        join_selection: bool = False
            If `True`, the output is an array of sketches of the joinable tables which is then passed to feature selection module \n
            If `False`, the output is the augmented table which consists of the joinable tables
        
        Returns:
        -------
        pl.DataFrame: Augmentation table which consists of all joinable tables found
        '''
        query_column_name = query_column.columns[0]
        overlap_query_results, distinct_tokens = self._run_overlap_query(query_column, top_k)
        overlap_query_results, overlap_ratio = self._process_overlap_query_results(overlap_query_results, query_column_name, distinct_tokens)

        if join_selection:
            target_column_name = user_table_processed.columns[0]
            join_selection_query_results = self._run_join_selection_query(overlap_query_results, user_table_processed, query_column_name, target_column_name)    
            join_selection_query_results = join_selection_query_results.sort('key')
            return overlap_query_results, join_selection_query_results, overlap_ratio
        else:
            return overlap_query_results


    def _run_overlap_query(self, query_column: pl.DataFrame, top_k: int) -> list[str]:
        '''
        Runs the overlap query to find the top k columns with the most overlap with the query column.

        Parameters:
        ----------
        query_column: pl.DataFrame
            Query column to find overlap columns for
        
        top_k: int
            Number of top columns to consider for overlap
        
        Returns:
        -------
        list[str]: List of top k columns (`table_column_index` in `feature_selection_table_name` table) with the most overlap with the query column
        '''
        distinct_tokens = np.unique(query_column.select(pl.all().exclude(self.index_col_name).cast(pl.String)).to_numpy().squeeze()).tolist()
        joint_distinct_tokens = '\',\''.join(distinct_tokens)
        overlap_query = self.overlap_query.replace('joint_distinct_tokens', f'{joint_distinct_tokens}').replace('top_k', f'{top_k}')
        overlap_query_results = pl.read_database_uri(overlap_query, self.conninfo)

        return overlap_query_results, distinct_tokens


    def _process_overlap_query_results(self, overlap_query_results: pl.DataFrame, query_column_name: str, distinct_tokens: list[str]) -> pl.DataFrame:
        '''
        Processes the overlap query results to extract the top columns with the most overlap with the query column as well as corresponding rows.

        Parameters:
        ----------
        overlap_query_results: pl.DataFrame
            DataFrame containing the results of the overlap query
        
        Returns:
        -------
        tuple[list[str], list[str]]: List of top columns (`table_column_index` in `feature_selection_table_name` table) with the most overlap with the query column
        '''
        if overlap_query_results.shape[0] > 0:
            overlap_query_results = overlap_query_results.with_columns(pl.col('number_of_tokens') / len(distinct_tokens))
            overlap_ratio = overlap_query_results.select('number_of_tokens').to_numpy()
            if overlap_ratio.shape[0] > 1:
                overlap_ratio = overlap_ratio.squeeze()
            overlap_query_results = overlap_query_results.explode(['key', 'row_index']).cast({'row_index': pl.Int64})
            overlap_query_results = overlap_query_results.cast({'key': pl.String})
        else:
            raise KeyNotFoundError(query_column_name)

        return overlap_query_results, overlap_ratio


    def _generate_joinable_tables_dict(self, token_query_results: pl.DataFrame) -> dict[str, dict[str, int]]:
        '''
        Transforms token query results in a format which allows for faster lookup of the tokens which could be joined with the user table.

        Parameters:
        ----------
        token_query_results: pl.DataFrame
            DataFrame containing the results of the token query
        
        Returns:
        -------
        dict[str, dict[str, int]]: Dictionary containing the mapping of `table_column_index` to the corresponding  {`token`: `table_row_index`} mapping
        '''
        token_query_results = token_query_results.with_columns(
            pl.concat_str(pl.col('table_index'), pl.col('key_col_index'), separator='_').alias('table_column_index'),
            pl.concat_str(pl.col('table_index'), pl.col('row_index'), separator='_').alias('table_row_index')
        )
        token_query_results = token_query_results.unique(subset=['table_column_index', 'table_row_index'])
        joinable_tables_dict = {}
        for name, group in token_query_results.group_by(['table_column_index'], maintain_order=True):
            keys = list(group['key'])
            values = [int(x[x.rfind('_') + 1:]) for x in group['table_row_index']]
            item = dict(zip(keys, values))
            joinable_tables_dict[name[0]] = item

        return joinable_tables_dict
    

    def _generate_rows_map(self, query_column: pl.DataFrame, joinable_tables_dict: dict[str, dict[str, int]]) -> dict[str, int]:
        '''
        Generates a mapping of `table_row_index` to the corresponding `row_index` in the user table.

        Parameters:
        ----------
        query_column: pl.DataFrame
            Query column to find joinable tables for
        
        joinable_tables_dict: dict[str, dict[str, int]]
            Dictionary containing the mapping of `table_column_index` to the corresponding  {`token`: `row_index`} mapping (`row_index` is the part of `table_row_index` after the underscore)

        Returns:
        -------
        dict[str, int]: Mapping of `table_row_index` to the corresponding `row_index` in the user table
        '''
        rows_map = {}
        for table_col_index in joinable_tables_dict:
            join_map = self._generate_join_map(query_column, joinable_tables_dict[table_col_index])
            joinable_rows = np.where(join_map != -1)[0]
            joinable_table_rows = [f'{table_col_index[:table_col_index.rfind("_")]}_{i}' for i in joinable_rows]
            base_table_join_index = join_map[join_map != -1]
            rows_map.update({table_col_index+':'+i: j for i, j in zip(joinable_table_rows, base_table_join_index)})

        return rows_map


    def _generate_join_map(self, query_column: pl.DataFrame, joinable_tables_dict: dict[str, int]) -> np.ndarray[int]:
        '''
        Generates a join map for a given query column and a joinable table. \n
        The join map is a 1D array which size corresponds to the size of the identified foreign key column, where:
            - `-1`: non-matching entry of the foreign key column w.r.t. the query column
            - `i`: query column index (i.e., `row_index`) of the matching entry of the foreign key column w.r.t. the query column
        
        Parameters:
        ----------
        query_column: pl.DataFrame
            Query column to find joinable tables for
        
        joinable_tables_dict: dict[str, int]
            `token` to `row_index` mapping (`row_index` is the part of `table_row_index` after the underscore)
        '''
        vals = joinable_tables_dict.values()
        join_table = np.full(max(vals) + 1, -1)

        q = query_column.to_numpy().squeeze()
        for i in np.arange(len(q)):
            x = q[i]
            index = joinable_tables_dict.get(x, -1)
            if index != -1:
                join_table[index] = i
        
        return join_table


    def _run_join_selection_query(self, token_query_results: pl.DataFrame, user_table_processed: pl.DataFrame, query_column_name: str, target_column_name: str) -> pl.DataFrame:
        token_query_results_arrow = token_query_results.to_arrow()

        conn = adbc_dbapi.connect(self.conninfo)
        try:
            with conn.cursor(adbc_stmt_kwargs={"adbc.postgresql.batch_size_hint_bytes": 512 * 1024 * 1024}) as cur:
                cur.adbc_ingest("temp_valid_combinations", token_query_results_arrow, mode="create", temporary=True)
                join_sql = f"""
                SELECT f.key, f.feature_index, f.table_index, f.key_col_index, f.row_index, f.count, f.sum, f.diag, f.qcr_term_positive, f.qcr_term_negative
                FROM {self.feature_selection_table_name} AS f
                JOIN temp_valid_combinations AS v
                USING (table_index, key_col_index, row_index)
                """
                cur.execute(join_sql)
                join_selection_query_results = cur.fetch_polars()
        finally:
            conn.close()

        join_selection_query_results = join_selection_query_results.with_columns(pl.col(['sum', 'diag', 'qcr_term_positive', 'qcr_term_negative']).cast(pl.Utf8).str.json_decode(dtype=pl.List(pl.Float64)))
        join_selection_query_results = join_selection_query_results.with_columns(
            pl.concat_str(pl.col('table_index'), pl.col('key_col_index'), separator='_').alias('table_column_index'),
            pl.concat_str(pl.col('table_index'), pl.col('row_index'), separator='_').alias('table_row_index')
        )

        return join_selection_query_results
    

    def _prune_features(self, join_selection_query_results: pl.DataFrame, query_col: str, target: str, user_table_processed: pl.DataFrame, task: str, corr_threshold: float = None) ->  pl.DataFrame:
        if corr_threshold is None:
            return join_selection_query_results
        else:
            if task == 'classification':
                join_selection_query_results, status = self._eta_query(join_selection_query_results, user_table_processed, query_col, target, epsilon=corr_threshold)
            elif task == 'regression':
                join_selection_query_results, status = self._correlation_query(user_table_processed, join_selection_query_results, query_col, target, epsilon=corr_threshold)

            join_selection_query_results = (
                join_selection_query_results
                    .with_columns(pl.col('sum').list.len().alias('num_features'))
                    .filter(pl.col('num_features') > 2)
                    .drop('num_features')
            )

            return join_selection_query_results, status
        

    def _eta_query(self, join_selection_query_results: pl.DataFrame, user_table_processed: pl.DataFrame, query_col: str, target: str, epsilon: float):
        n_rows = user_table_processed.height
        joint_counts = user_table_processed.group_by([query_col, target]).len()
        N = 1024
        if N < user_table_processed.height:
            joint_counts = joint_counts.with_columns(
                (
                    (pl.col("len") / n_rows * N)
                    .round()
                    .cast(pl.Int64)
                    .clip(lower_bound=1)
                    .alias("n_sample")
                )
            )
            excess_rows = joint_counts["n_sample"].sum() - N
            if excess_rows > 0:
                joint_counts = joint_counts.with_columns(
                    pl.when(pl.col("n_sample") > 1)
                    .then(pl.col("n_sample") - 1)
                    .otherwise(pl.col("n_sample"))
                    .alias("n_sample")
                )
            user_table_processed_sample = (
                user_table_processed
                .join(joint_counts, on=[query_col, target], how="left")
                .with_columns(pl.int_range(0, pl.len()).over([query_col, target]).alias("_idx"))
                .filter(pl.col("_idx") < pl.col("n_sample"))
                .drop(["len", "n_sample", "_idx"])
            )
        else:
            user_table_processed_sample = user_table_processed
        dfs = []
        for group in join_selection_query_results.sort('key').group_by(['table_index', 'feature_index', 'table_column_index'], maintain_order=True):
            eta_squared = self._compute_eta(group[1], user_table_processed_sample, query_col, target)
            eta_squared = (
                eta_squared
                    .max()
                    .select(
                        pl.concat_list(pl.all())
                        .list.eval(pl.element() > epsilon)
                    )
            )
            features_check = len(set(eta_squared.row(0)[0])) == 2
            if not features_check:
                continue
            _, _, table_column_index = group[0]
            eta_squared = eta_squared.rename({col: f'{table_column_index}_{col.replace("field_", "")}' for col in eta_squared.columns})
            eta_squared = (
                        eta_squared
                            .transpose(include_header=True)
                            .select(
                                pl.nth(0).str.replace(r'_\d$', '', literal=False).alias('table_column_index'),
                                pl.nth(1).cast(pl.List(pl.Int16)).alias('low_correlation_features')
                            )
                            .sql("""
                                SELECT
                                    table_column_index,
                                    ARRAY_TO_STRING(low_correlation_features, ',') AS low_correlation_features
                                FROM self
                                """
                            )
                    )
            dfs.append(eta_squared)
        try:
            eta_results = pl.concat(dfs, how='vertical')
        except ValueError:
            return join_selection_query_results, False

        join_selection_query_results = join_selection_query_results.join(
            eta_results,
            on='table_column_index',
            how='left'
        )
        join_selection_query_results = (
            join_selection_query_results
                .sql("""
                    SELECT key, feature_index, table_index, key_col_index, row_index, count, sum, diag, table_column_index, table_row_index,
                     STRING_TO_ARRAY(low_correlation_features, ',') AS drop_feature FROM self
                    """
                )
                .with_columns(pl.col('drop_feature').cast(pl.List(pl.Int16)))
        )
        join_selection_query_results = join_selection_query_results.drop_nulls()

        return join_selection_query_results, True


    def _compute_eta(self, table: pl.DataFrame, user_table_processed_sample: pl.DataFrame, query_col: str, target: str) -> pl.DataFrame:
        n_cols = table.select(pl.col('sum').list.len()).row(0)[0]
        sub_table_group = user_table_processed_sample.join(
            table.select('key', pl.col('sum').list.to_struct(upper_bound=n_cols).struct.unnest()),
            left_on=query_col,
            right_on='key',
            how='left'
        )
        features_selector = r'^field_\d+$'
        features = [col for col in sub_table_group.columns if re.match(features_selector, col)]
        global_means = sub_table_group.select(
            [pl.col(x).mean().alias(x) for x in features]
        )
        group_stats = (
            sub_table_group
            .group_by(target)
            .agg(
                [pl.count().alias("n")] +
                [pl.col(x).mean().alias(f"{x}_mean") for x in features]
            )
        )
        between = group_stats.select([
            (
                pl.col("n") *
                (pl.col(f"{x}_mean") - global_means[x][0]) ** 2
            ).sum().alias(x)
            for x in features
        ])
        total = sub_table_group.select([
            ((pl.col(x) - global_means[x][0]) ** 2).sum().alias(x)
            for x in features
        ])
        eta_squared = (
            between
            .select([
                (pl.col(x) / total[x][0]).alias(x)
                for x in features
            ])
        )
        return eta_squared


    def _base_pruning_query(self, join_selection_query_results: pl.DataFrame):
        base_query = (
            join_selection_query_results
                .group_by(['table_index', 'feature_index', 'table_column_index'])
                .agg(
                    pl.col('key'),
                    pl.col('sum'),
                    pl.col('sum').map_batches(lambda x: x.to_numpy().mean(axis=0)).alias('features_mean'),
                    pl.col('sum').map_batches(lambda x: x.to_numpy().std(axis=0)).alias('features_std'),
                    pl.col('sum').map_batches(lambda x: np.median(np.vstack(x.to_numpy()), axis=0)).alias('features_median'),
                    pl.col('sum').map_batches(lambda x: np.median(np.abs(np.vstack(x.to_numpy()) - np.median(np.vstack(x.to_numpy()), axis=0)), axis=0)).alias('features_mad'),
                    pl.col('count').sum()
                )
        )
        return base_query


    def _constant_features_query(self, base_query: pl.DataFrame, epsilon: float):
        constant_query = (
            base_query
            .with_columns(
                (
                    (1.4826 * pl.col('features_mad'))
                    /
                    pl.col('features_median').list.eval(
                        pl.when(pl.element() < 1e-12).then(1e-12).otherwise(pl.element())
                    )
                )
                .alias('coef_of_variation')
            )
            .with_columns(
                (pl.col('coef_of_variation').list.eval(pl.element() > epsilon)).alias('low_variance_features')
            )
        )
        return constant_query


    def _correlation_query(self, user_table_processed: pl.DataFrame, join_selection_query_results: pl.DataFrame, query_col: str, target: str, epsilon: float):
        max_cols = join_selection_query_results.select(pl.col('sum').list.len()).max().row(0)[0]
        query_qcr_table = user_table_processed.group_by(query_col, maintain_order=True).agg(pl.col(target).mean()).sort(query_col)
        sub_table = (
            query_qcr_table
                .with_columns(pl.col(query_col).repeat_by(max_cols).list.to_struct(upper_bound=max_cols).struct.unnest())
        )
        dfs = []
        for group in join_selection_query_results.sort('key').group_by(['table_index', 'feature_index', 'table_column_index'], maintain_order=True):
            n_cols = group[1].select(pl.col('sum').list.len()).row(0)[0]
            sub_table_group = (
                sub_table
                    .filter(pl.col(query_col).is_in(group[1].select('key').to_series()))
                    .select(cs.by_index(range(n_cols+2)))  # +2 for query_col and target_col
            )
            query_term_positive = self._build_qcr_term(sub_table_group, True, target, query_col)
            query_term_negative = self._build_qcr_term(sub_table_group, False, target, query_col)

            qcr_candidate_positive = group[1].select(pl.col('qcr_term_positive').list.to_struct(upper_bound=n_cols).struct.unnest())
            qcr_candidate_negative = group[1].select(pl.col('qcr_term_negative').list.to_struct(upper_bound=n_cols).struct.unnest())

            qcr_scores_positive = self._qcr_score(query_term_positive, qcr_candidate_positive, ids=group[0])
            qcr_scores_negative = self._qcr_score(query_term_negative, qcr_candidate_negative, ids=group[0])
            qcr_scores = pl.concat([qcr_scores_positive, qcr_scores_negative], how='vertical')
            qcr_scores = (
                qcr_scores
                    .max()
                    .select(
                        pl.concat_list(pl.all())
                        .list.eval(pl.element() > epsilon)
                    )
            )
            features_check = len(set(qcr_scores.row(0)[0])) == 2
            if not features_check:
                continue
            qcr_scores = (
                        qcr_scores
                            .transpose(include_header=True)
                            .select(
                                pl.nth(0).str.replace(r'_\d$', '', literal=False).alias('table_column_index'),
                                pl.nth(1).cast(pl.List(pl.Int16)).alias('low_correlation_features')
                            )
                            .sql("""
                                SELECT
                                    table_column_index,
                                    ARRAY_TO_STRING(low_correlation_features, ',') AS low_correlation_features
                                FROM self
                                """
                            )
                    )
            dfs.append(qcr_scores)

        try:
            qcr_results = pl.concat(dfs, how='vertical')
        except ValueError:
            return join_selection_query_results, False

        join_selection_query_results = join_selection_query_results.join(
            qcr_results,
            on='table_column_index',
            how='left'
        )
        join_selection_query_results = (
            join_selection_query_results
                .sql("""
                    SELECT key, feature_index, table_index, key_col_index, row_index, count, sum, diag, table_column_index, table_row_index,
                     STRING_TO_ARRAY(low_correlation_features, ',') AS drop_feature FROM self
                    """
                )
                .with_columns(pl.col('drop_feature').cast(pl.List(pl.Int16)))
        )
        join_selection_query_results = join_selection_query_results.drop_nulls()

        return join_selection_query_results, True


    def _build_qcr_term(self, sub_table, is_positive, target: str, query_col: str, max_sha = 2**64 - 1):
        if is_positive:
            qcr_term = (
                sub_table
                    .with_columns(
                    (pl.col(target) - pl.col(target).mean())
                    .sign()
                    .cast(pl.Int16)
                    .cast(pl.String)
                )
            )
        else:
            qcr_term = (
                sub_table
                    .with_columns(
                    (pl.col(target) - pl.col(target).mean())
                    .sign()
                    .mul(-1)
                    .cast(pl.Int16)
                    .cast(pl.String)
                )
            )
        qcr_term = (
            qcr_term.
                select(
                    plh.concat_str(pl.col(query_col), pl.col(target), separator='')
                    .chash.sha3_shake128(length=8).str.to_integer(base=16, dtype=pl.Int128)
                    / max_sha
                )
                .to_series()
        )
        return qcr_term


    def _build_qcr_candidate(self, sub_table: pl.DataFrame, group: pl.DataFrame, num_cols: list[str], n_cols: int, target_col: pl.DataFrame, is_positive: bool, max_sha = 2**64 - 1):
        table = (
            pl.concat(
                [
                    group[1]
                        .select(['key', 'sum'])
                        .with_columns(pl.col('sum').list.to_struct(upper_bound=n_cols).struct.unnest())
                        .drop(['key', 'sum']),
                    target_col
                ],
                how='horizontal'
            )
        )
        if is_positive:
            table = table.with_columns(
                (pl.col(num_cols) - pl.col(num_cols).mean())
                .sign()
                .cast(pl.Int16)
                .cast(pl.String)
            )
        else:
            table = table.with_columns(
                (pl.col(num_cols) - pl.col(num_cols).mean())
                .sign()
                .mul(-1)
                .cast(pl.Int16)
                .cast(pl.String)
            )
        qcr_candidate = pl.DataFrame().select(
            [
                plh.concat_str(sub_table[col], table[col]).chash.sha3_shake128(length=8).str.to_integer(base=16, dtype=pl.Int128)
                / max_sha
                for col in num_cols
            ]
        )
        return qcr_candidate


    def _qcr_score(self, qcr_term: pl.Series, qcr_candidate: pl.DataFrame, ids: tuple[int]):
        _, _, table_column_index = ids
        qcr_scores = (
            qcr_candidate
                .select(
                    [
                        (pl.col(col) == qcr_term).sum()
                        for col in qcr_candidate.columns
                    ]
                )
                /
                qcr_term.len()
        )
        qcr_scores = qcr_scores.rename({col: f'{table_column_index}_{col.replace("field_", "")}' for col in qcr_scores.columns})
        return qcr_scores


class AurumJoinDiscoveryLakeBench:
    def __init__(self, index_file: str, separator: str = '\t', num_perm: int = 128, scale: float = 1.0, random_seed: int = 42) -> None:
        """
        Initialize Aurum-based join discovery using local MinHash index.
        
        Parameters:
        ----------
        index_file: str
            Path to the MinHash embeddings pickle file created by build_hash.py
        hnsw_index_file: str
            Path to precomputed HNSW index file (optional). If provided and exists,
            the index will be loaded. If not provided or doesn't exist, index will
            be built on-the-fly (and saved if path is provided).
        separator: str
            CSV separator used for data lake tables (default: '\t')
        num_perm: int
            Number of MinHash permutations used in index (default: 128)
        scale: float
            Percentage of index to use for scalability experiments (default: 1.0)
        random_seed: int
            Random seed for deterministic table sampling (default: 42)
        """
        self.index_file = f'{index_file.split(".")[0]}_embeddings.pkl'
        self.hnsw_index_file = f'{index_file.split(".")[0]}_index.bin'
        self.separator = separator.encode().decode('unicode_escape')
        self.num_perm = num_perm
        self.scale = scale
        self.random_seed = random_seed
        lake_name = self.index_file.split('/')[-1].split('_embeddings.pkl')[0]
        params_map = {
            'nyc': {
                'K': 20,
                'N': 25,
                'threshold': 0.5,
                'max_join_cols': 3
            },
            'canada_us_uk_open_data': {
                'K': 20,
                'N': 75,
                'threshold': 0.5,
                'max_join_cols': 3
            },
            'gittables': {
                'K': 20,
                'N': 200,
                'threshold': 0.5,
                'max_join_cols': 3
            },
        }
        self.params = params_map[lake_name]
        
        if not os.path.exists(self.index_file):
            raise ValueError(f"Index file does not exist: {self.index_file}")
        if self.hnsw_index_file and not os.path.exists(self.hnsw_index_file):
            print(f"HNSW index file does not exist: {self.hnsw_index_file}. The index will be built on-the-fly.")
            self.hnsw_index_file = None


    def find_joinable_tables(self, query_table_path: str, features: list[str], query_separator: str = ',', output_path: str = None) -> pl.DataFrame:
        """
        Find joinable tables for a query table using MinHash-based similarity search.
        
        Parameters:
        ----------
        query_table_path: str
            Path to the query table CSV file
        query_separator: str
            CSV separator for query table (default: ',')
        K: int
            Number of top candidate tables to return (default: 10)
        N: int
            Number of nearest neighbors per column (default: 10)
        threshold: float
            Similarity threshold for column matching (default: 0.7)
        max_join_cols: int
            Maximum number of matching columns for join mode (default: 3)
        output_path: str
            Optional path to save join paths CSV
            
        Returns:
        -------
        pl.DataFrame: Join paths with schema (from_id, to_id, from_column, to_column, weight)
        """
        query_separator = query_separator.encode().decode('unicode_escape')
        
        if not os.path.exists(query_table_path):
            raise ValueError(f"Query table does not exist: {query_table_path}")
        query_table_splits_path = f'{"/".join(query_table_path.split("/")[:-1])}/splits.json'
        with open(query_table_splits_path, 'r') as f:
            splits_info = json.load(f)
        query_col = splits_info[0]['query_col']
        query_table = pl.read_csv(query_table_path, separator=query_separator, columns=[query_col])
        query_table_path = f'{query_table_path}_aurum_query.csv'
        query_table.write_csv(query_table_path, separator=query_separator)
        
        # Build query embeddings
        query = self._build_query_embeddings(query_table_path, query_separator)
        
        # Initialize searcher with join mode (will load precomputed index if available)
        searcher = HNSWSearcher(self.index_file, self.hnsw_index_file, self.scale, search_mode='join', random_seed=self.random_seed)
        
        # Execute search
        results, num_candidates = searcher.topk(
            'aurum',
            query,
            **self.params
        )
        
        # Read query table header
        with open(query_table_path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=query_separator)
            query_header = next(reader)
        
        # Collect all join path records in a list for efficient batch creation
        join_records = []
        
        for result in results:
            score = result[0]
            column_pairs = result[1]
            candidate_table = result[2]
            
            if len(column_pairs) > 0:
                # Sort column_pairs for deterministic ordering
                sorted_pairs = sorted(column_pairs, key=lambda x: (x[0], x[1], -x[2]))
                for query_col_idx, cand_col_idx, similarity in sorted_pairs:
                    query_col_name = query_header[query_col_idx] if query_col_idx < len(query_header) else f'col_{query_col_idx}'
                    cand_col_name = f'col_{cand_col_idx}'
                    
                    join_records.append({
                        'from_id': query_table_path,
                        'to_id': f'{candidate_table}.csv',
                        'from_column': query_col_name,
                        'to_column': cand_col_name,
                        'weight': similarity
                    })
        
        # Create DataFrame from all records at once
        join_paths = pl.DataFrame(join_records, schema={
            'from_id': pl.String,
            'to_id': pl.String,
            'from_column': pl.String,
            'to_column': pl.String,
            'weight': pl.Float64
        })
        
        # Remove duplicates and sort with all keys for deterministic output
        join_paths = join_paths.filter(pl.col('from_column').is_in(features))
        join_paths = join_paths.unique(maintain_order=True)
        join_paths = join_paths.sort('weight', descending=True).head(50)
        join_paths = join_paths.sort(['from_column', 'to_id', 'to_column', 'weight'], descending=[False, False, False, True])
        
        if output_path:
            join_paths.write_csv(output_path)
        
        return join_paths


    def _build_query_embeddings(self, query_table_path: str, separator: str) -> tuple:
        """
        Build MinHash embeddings for a query table.
        
        Parameters:
        ----------
        query_table_path: str
            Path to query CSV file
        separator: str
            CSV delimiter
            
        Returns:
        -------
        tuple: (table_name, column_embeddings_array)
        """
        data_array = []
        try:
            with open(query_table_path, encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=separator)
                for idx, row in enumerate(csv_reader):
                    if idx == 0:
                        header = row
                    elif row:
                        data_array.append(row)
        except Exception as e:
            raise ValueError(f"Error reading query table: {e}")
        
        if len(data_array) == 0:
            raise ValueError("Query table is empty")
        
        # Build MinHash for each column
        column_embeddings = []
        for col_idx in range(len(data_array[0])):
            m = MinHash(num_perm=self.num_perm)
            unique_vals = list(set([row[col_idx] for row in data_array if col_idx < len(row)]))
            for val in unique_vals:
                m.update(val.encode('utf-8'))
            column_embeddings.append(list(m.hashvalues))
        
        return (query_table_path, np.array(column_embeddings))


class AurumJoinDiscovery:
    def __init__(self, index_dir: str, separator: str = '\t', num_perm: int = 256, scale: float = 1.0, random_seed: int = 42) -> None:
        """
        Initialize LSH Ensemble-based join discovery.

        Parameters:
        ----------
        index_dir: str
            Path to the LSH Ensemble index directory (containing lsh_ensemble.pkl,
            column_map.pkl, and metadata.pkl built by augmentation/LSH/build_index.py)
        separator: str
            CSV separator used for data lake tables (default: '\\t')
        num_perm: int
            Kept for API compatibility; actual value is read from index metadata.
        scale: float
            Kept for API compatibility.
        random_seed: int
            Kept for API compatibility.
        """
        self.index_dir = index_dir.replace('.pkl', '')
        self.separator = separator.encode().decode('unicode_escape')
        self.scale = scale
        self.random_seed = random_seed

        # Load LSH Ensemble index (also sets self._LSHMinHash)
        self._load_index()

        # Lake-specific params
        lake_name = os.path.basename(index_dir).split('_')[0]
        params_map = {
            'nyc': {'top_k': 20},
            'canada': {'top_k': 20},
            'gittables': {'top_k': 20},
        }
        self.params = params_map.get(lake_name, {'top_k': 20})

    def _load_index(self) -> None:
        """
        Load LSH Ensemble index files from disk.

        Temporarily swaps sys.modules to use the local datasketch package
        (augmentation/LSH/datasketch/) for correct unpickling, since the
        index was built with that package rather than the pip-installed one.
        """
        for fname in ('lsh_ensemble.pkl', 'column_map.pkl', 'metadata.pkl'):
            path = os.path.join(self.index_dir, fname)
            if not os.path.exists(path):
                raise ValueError(f"Index file not found: {path}")

        lsh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LSH')

        # Save pip-installed datasketch modules and swap in local ones for unpickling
        saved_modules = {
            key: sys.modules.pop(key)
            for key in list(sys.modules)
            if key == 'datasketch' or key.startswith('datasketch.')
        }
        sys.path.insert(0, lsh_dir)
        try:
            import datasketch as _local_ds
            self._LSHMinHash = _local_ds.MinHash

            with open(os.path.join(self.index_dir, 'lsh_ensemble.pkl'), 'rb') as f:
                self.lsh_ensemble = pickle.load(f)
            with open(os.path.join(self.index_dir, 'column_map.pkl'), 'rb') as f:
                self.column_map = pickle.load(f)
            with open(os.path.join(self.index_dir, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
        finally:
            sys.path.remove(lsh_dir)
            # Remove local datasketch modules and restore pip ones
            for key in list(sys.modules):
                if key == 'datasketch' or key.startswith('datasketch.'):
                    del sys.modules[key]
            sys.modules.update(saved_modules)

        self.num_perm = self.metadata['num_perm']
        self.seed = self.metadata.get('seed', 42)
        self.threshold = self.metadata['threshold']
        # Prefer separator from index metadata (what the lake was built with)
        if 'separator' in self.metadata:
            self.separator = self.metadata['separator']

    def find_joinable_tables(self, query_table_path: str, features: list[str],
                             query_separator: str = ',', output_path: str = None) -> pl.DataFrame:
        """
        Find joinable tables using LSH Ensemble containment search.

        Parameters:
        ----------
        query_table_path: str
            Path to the query table CSV file
        features: list[str]
            List of column names to consider for join discovery
        query_separator: str
            CSV separator for query table (default: ',')
        output_path: str
            Optional path to save join paths CSV

        Returns:
        -------
        pl.DataFrame: Join paths with schema (from_id, to_id, from_column, to_column, weight)
        """
        query_separator = query_separator.encode().decode('unicode_escape')

        if not os.path.exists(query_table_path):
            raise ValueError(f"Query table does not exist: {query_table_path}")

        # Read splits info to get query column
        query_table_splits_path = os.path.join(os.path.dirname(query_table_path), 'splits.json')
        with open(query_table_splits_path, 'r') as f:
            splits_info = json.load(f)
        query_col = splits_info[0]['query_col']

        # Read query table
        query_table = pl.read_csv(query_table_path, separator=query_separator)

        join_records = []

        # Query LSH Ensemble for each feature column
        for col_name in features:
            if col_name not in query_table.columns:
                continue

            # Get unique non-null string values
            values = query_table[col_name].drop_nulls().cast(pl.String).unique().to_list()
            values = [v.strip() for v in values if v and v.strip()]

            if not values:
                continue

            # Build MinHash using the same params as the index
            mh = self._LSHMinHash(num_perm=self.num_perm, seed=self.seed)
            mh.update_batch(values)
            query_size = len(values)
            query_set = set(values)

            # Query LSH Ensemble
            candidates = self.lsh_ensemble.query(mh, query_size)

            if not candidates:
                continue

            # Group candidates by file for efficient IO
            file_candidates = {}
            for key in candidates:
                if key not in self.column_map:
                    continue
                info = self.column_map[key]
                file_candidates.setdefault(info['file_path'], []).append(info['column_name'])

            # Compute exact containment for each candidate
            for file_path, cand_cols in file_candidates.items():
                try:
                    cand_df = pl.read_csv(
                        file_path, separator=self.separator,
                        columns=cand_cols, infer_schema=False
                    )
                except Exception:
                    continue

                for cand_col in cand_cols:
                    try:
                        target_values = set(
                            cand_df[cand_col].drop_nulls()
                            .str.strip_chars().unique().to_list()
                        )
                        containment = len(query_set & target_values) / query_size
                    except Exception:
                        continue

                    if containment > 0:
                        join_records.append({
                            'from_id': query_table_path,
                            'to_id': file_path,
                            'from_column': col_name,
                            'to_column': cand_col,
                            'weight': containment
                        })

        # Build output DataFrame
        join_paths = pl.DataFrame(join_records, schema={
            'from_id': pl.String,
            'to_id': pl.String,
            'from_column': pl.String,
            'to_column': pl.String,
            'weight': pl.Float64
        })

        # Deduplicate and sort
        join_paths = join_paths.unique(maintain_order=True)
        join_paths = join_paths.sort('weight', descending=True).head(self.params['top_k'])
        join_paths = join_paths.sort(
            ['from_column', 'to_id', 'to_column', 'weight'],
            descending=[False, False, False, True]
        )
        join_paths = join_paths.with_columns(pl.col('to_id').str.split('/').list[-1].alias('to_id'))

        if output_path:
            join_paths.write_csv(output_path)

        return join_paths


class AurumJoinDiscoveryOG:
    def __init__(self, api_host: str, api_port: int, es_host: str = "aurum-datadiscovery-elasticsearch-1") -> None:
        self.api_host = api_host
        self.api_port = api_port
        self.base_url = f'http://{self.api_host}:{self.api_port}'
        self.es_host = es_host


    def find_joinable_tables(self, template_path: str, table_path: str, source_name: str, baseline: str, lake: str, output_path: str = None):
        with open(template_path, 'r') as f:
            template = f.read()
        payload = {
            "table_path": table_path,
            "template": template,
            "es_host": self.es_host,
            "lake": lake
        }
        request = requests.post(f'{self.base_url}/update_index', json=payload)
        response = request.json()
        new_id_info = list(response['new_id_info'].keys())
        new_table_ids = list(response['new_table_ids'].values())[0]

        payload = {
            "table_path": source_name,
            "lake": lake
        }
        request = requests.post(f'{self.base_url}/similar_tables', json=payload)
        response = request.json()
        join_paths = self._prepare_join_paths(response, source_name, baseline, output_path)

        return join_paths, new_id_info, new_table_ids, response


    def cleanup(self, new_id_info, new_table_ids, response, lake):
        payload = {
            "id_info": new_id_info,
            "table_ids": new_table_ids,
            "join_paths_dict": response,
            "lake": lake
        }
        request = requests.post(f'{self.base_url}/remove_from_index', json=payload)


    def _prepare_join_paths(self, response: dict, source_name: str, baseline: str, output_path: str):
        match baseline:
            case 'metam':
                join_paths = pl.DataFrame(schema={'tbl1': pl.String, 'col1': pl.String, 'tbl2': pl.String, 'col2': pl.String})
                for pair in response['similar_tables']['edges']:
                    if pair[0]['source_name'] != pair[1]['source_name']:
                        if pair[0]['source_name'] == source_name:
                            tbl1 = source_name#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[0]['field_name']
                            tbl2 = pair[1]['source_name']
                            col2 = pair[1]['field_name']
                        if pair[1]['source_name'] == source_name:
                            tbl1 = pair[1]['field_name']#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[1]['field_name']
                            tbl2 = pair[0]['source_name']
                            col2 = pair[0]['field_name']
                        join_paths = pl.concat([join_paths, pl.DataFrame({'tbl1': [tbl1], 'col1': [col1], 'tbl2': [tbl2], 'col2': [col2]})], how='vertical')
            case 'arda':
                join_paths = pl.DataFrame(schema={'from_id': pl.String, 'to_id': pl.String, 'from_column': pl.String, 'to_column': pl.String, 'weight': pl.Float64})
                for pair in response['similar_tables']['edges']:
                    if pair[0]['source_name'] != pair[1]['source_name']:
                        if pair[0]['source_name'] == source_name:
                            tbl1 = source_name#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[0]['field_name']
                            tbl2 = pair[1]['source_name']
                            col2 = pair[1]['field_name']
                        if pair[1]['source_name'] == source_name:
                            tbl1 = pair[1]['field_name']#.replace('.csv', '_preprocessed.csv')
                            col1 = pair[1]['field_name']
                            tbl2 = pair[0]['source_name']
                            col2 = pair[0]['field_name']
                        weight = np.nan
                        join_paths = pl.concat([join_paths, pl.DataFrame({'from_id': [tbl1], 'to_id': [tbl2], 'from_column': [col1], 'to_column': [col2], 'weight': [weight]})], how='vertical')
        join_paths = join_paths.unique()
        join_paths.write_csv(output_path)

        return join_paths