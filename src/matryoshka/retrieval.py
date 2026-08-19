import os
import re
import time
from typing import Iterable

import adbc_driver_postgresql.dbapi as adbc_dbapi
import numpy as np
import polars as pl
import polars.selectors as cs
import polars_hash as plh

from .db.handler import DBHandler
from .exceptions import KeyNotFoundError
from .utils.logging import default_log_dir, setup_logger


class JoinDiscovery(DBHandler):
    def __init__(self, feature_selection_table_name: str, overlap_table_name: str, verbose: bool = False,
                 log_file_name: str = None, settings=None, exclude_tables: 'Iterable[str] | None' = None,
                 log_dir: str = None) -> None:
        '''
        Parameters:
        ----------
        token_index_table_name: str
            Name of the table containing the token index
        
        feature_selection_table_name: str
            Name of the table containing the query index
        '''
        super().__init__(feature_selection_table_name, overlap_table_name, settings=settings)
        self.feature_selection_table_name = feature_selection_table_name

        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        log_dir = str(log_dir) if log_dir else str(default_log_dir())
        log_file_name = log_file_name or 'retrieval'
        if verbose:
            self.logger = setup_logger(name='join_selection', log_dir=log_dir, log_file=f'{log_file_name}_{timestamp}_retrieval.log')
        else:
            self.logger = setup_logger(name='join_selection', log_dir=log_dir, log_file=f'{log_file_name}_{timestamp}_retrieval.log', silent=True)

        # ``exclusion_clause`` is replaced at query time by either the empty
        # string or ``AND table_index NOT IN (id, id, ...)`` so that callers
        # can suppress specific lake tables from the overlap result. The
        # primary use-case is leave-one-out evaluation where the query table
        # is itself a member of the lake.
        self.overlap_query = 'SELECT oq.key, oq.table_index, oq.key_col_index, oq.row_index, oq.number_of_tokens ' \
                             'FROM (' \
                             "    SELECT ARRAY_AGG(key || '') as key, table_index, key_col_index, ARRAY_AGG(row_index || '') as row_index, COUNT(key) as number_of_tokens " \
                            f'    FROM {feature_selection_table_name} ' \
                             '    WHERE key IN (\'joint_distinct_tokens\' ) exclusion_clause ' \
                             '    GROUP BY table_index, key_col_index' \
                             ') as oq ' \
                             'ORDER BY number_of_tokens DESC ' \
                             'LIMIT top_k;'

        self.index_col_name = 'index'
        # Lake tables excluded from every overlap query of this instance.
        # A query table that is itself a member of the lake must not retrieve
        # itself, or its own target leaks back in as a candidate feature. Set
        # through `exclude_tables=` on the constructor or per query on
        # `find_joinable_tables`; names are resolved against `table_name` in
        # the index and cached.
        self._excluded_table_names: set[str] = set(exclude_tables or ())
        self._excluded_index_cache: dict[str, int | None] = {}


    def find_joinable_tables(self, query_column: pl.DataFrame, top_k: int, user_table_processed: pl.DataFrame,
                             join_selection: bool = False, n_hops: int = 1, joinability_threshold: float = 0.5,
                             table_name=None, exclude_tables: 'Iterable[str] | None' = None) -> pl.DataFrame:
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

        n_hops: int = 1
            If > 1, build a multi-hop join graph (root -> hop-1 -> ... -> hop-n) using
            `augmentation.join_graph.JoinGraph`. Hop-1 children come from the standard
            overlap query; deeper hops are reached via bridge columns of the parent's
            table. Each candidate row's `key` is rewritten back into the root token space.

        joinability_threshold: float = 0.5
            Only used when `n_hops > 1`. Hop>=2 candidates whose overlap ratio with
            their parent is below this threshold are dropped.

        table_name: str | None = None
            Name of the query table as it appears in the lake, if it is a
            member of the lake. Excluded from the overlap result so the query
            table cannot retrieve itself, which would leak its own target back
            in as a candidate feature (leave-one-out semantics).

        exclude_tables: Iterable[str] | None = None
            Additional lake table names to exclude from this query, on top of
            those passed to the constructor.

        Returns:
        -------
        pl.DataFrame: Augmentation table which consists of all joinable tables found
        '''
        names = set(self._excluded_table_names)
        names.update(exclude_tables or ())
        if table_name is not None:
            names.add(table_name)
        exclude_table_indices = self._resolve_excluded_indices(names) or None

        query_column_name = query_column.columns[0]

        if n_hops > 1:
            from .join_graph import JoinGraph
            graph = JoinGraph(
                self,
                feature_selection_table_name=self.feature_selection_table_name,
                top_k=top_k,
                n_hops=n_hops,
                joinability_threshold=joinability_threshold,
            )
            graph.build(query_column)
            js_input = graph.to_join_selection_input()
            if js_input.is_empty():
                raise KeyNotFoundError(query_column_name)
            # Reshape to the same schema that `_process_overlap_query_results` produces.
            # `key` is already rewritten into the root token space.
            overlap_query_results = js_input.select([
                pl.col('key').cast(pl.String),
                pl.col('table_index'),
                pl.col('key_col_index'),
                pl.col('row_index').cast(pl.Int64),
                pl.col('joinability').alias('number_of_tokens'),
            ])
            overlap_ratio = (
                overlap_query_results
                    .select(['table_index', 'key_col_index', 'number_of_tokens'])
                    .unique()
                    .get_column('number_of_tokens')
                    .to_numpy()
            )
            self._graph = graph
        else:
            overlap_query_results, distinct_tokens = self._run_overlap_query(query_column, top_k, exclude_table_indices=exclude_table_indices)
            overlap_query_results, overlap_ratio = self._process_overlap_query_results(overlap_query_results, query_column_name, distinct_tokens)

        if join_selection:
            target_column_name = user_table_processed.columns[0]
            join_selection_query_results = self._run_join_selection_query(overlap_query_results, user_table_processed, query_column_name, target_column_name)
            join_selection_query_results = join_selection_query_results.sort('key')
            return overlap_query_results, join_selection_query_results, overlap_ratio
        else:
            return overlap_query_results


    def _resolve_excluded_indices(self, table_names) -> list[int]:
        '''Map lake table names to their ``table_index`` in the index.

        A name is matched first exactly, then against the basename with and
        without a file extension, so that ``covertype``, ``covertype.csv`` and
        ``/lake/covertype.csv`` all resolve. Names absent from the index
        resolve to nothing and are silently ignored: excluding a table that was
        never indexed is a no-op, not an error.
        '''
        indices: list[int] = []
        unresolved = [n for n in table_names if n not in self._excluded_index_cache]
        if unresolved:
            variants = {}
            for name in unresolved:
                stem = os.path.splitext(os.path.basename(str(name)))[0]
                variants[name] = {str(name), os.path.basename(str(name)), stem}
            wanted = sorted({v for group in variants.values() for v in group})
            placeholders = ','.join(f"'{v}'" for v in wanted)
            query = (
                f'SELECT DISTINCT table_name, table_index '
                f'FROM {self.feature_selection_table_name} '
                f'WHERE table_name IN ({placeholders}) '
                f"   OR regexp_replace(table_name, '\\.[^.]*$', '') IN ({placeholders})"
            )
            try:
                found = pl.read_database_uri(query, self.conninfo)
            except Exception:
                found = pl.DataFrame({'table_name': [], 'table_index': []})
            lookup = {}
            for row_name, row_index in zip(found['table_name'].to_list() if found.height else [],
                                           found['table_index'].to_list() if found.height else []):
                lookup[row_name] = int(row_index)
                lookup[os.path.splitext(row_name)[0]] = int(row_index)
            for name in unresolved:
                match = next((lookup[v] for v in variants[name] if v in lookup), None)
                self._excluded_index_cache[name] = match
        for name in table_names:
            resolved = self._excluded_index_cache.get(name)
            if resolved is not None:
                indices.append(resolved)
        return sorted(set(indices))


    def _run_overlap_query(self, query_column: pl.DataFrame, top_k: int, exclude_table_indices=None) -> list[str]:
        '''
        Runs the overlap query to find the top k columns with the most overlap with the query column.

        Parameters:
        ----------
        query_column: pl.DataFrame
            Query column to find overlap columns for

        top_k: int
            Number of top columns to consider for overlap

        exclude_table_indices: Iterable[int] | None = None
            Optional set of ``table_index`` values to drop from the result.

        Returns:
        -------
        list[str]: List of top k columns (`table_column_index` in `feature_selection_table_name` table) with the most overlap with the query column
        '''
        distinct_tokens = np.unique(query_column.select(pl.all().exclude(self.index_col_name).cast(pl.String)).to_numpy().squeeze()).tolist()
        joint_distinct_tokens = '\',\''.join(distinct_tokens)
        if exclude_table_indices:
            ids = ','.join(str(int(x)) for x in exclude_table_indices)
            exclusion_sql = f'AND table_index NOT IN ({ids})'
        else:
            exclusion_sql = ''
        overlap_query = (
            self.overlap_query
                .replace('joint_distinct_tokens', f'{joint_distinct_tokens}')
                .replace('exclusion_clause', exclusion_sql)
                .replace('top_k', f'{top_k}')
        )
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
                # Use v.key (from the temp table) rather than f.key so that the
                # multi-hop graph's root-token rewrite is preserved end-to-end.
                # For single-hop calls v.key == f.key by construction.
                join_sql = f"""
                SELECT v.key, f.feature_index, f.table_index, f.key_col_index, f.row_index, f.count, f.sum, f.diag, f.qcr_term_positive, f.qcr_term_negative, f.table_name, f.column_headers
                FROM {self.feature_selection_table_name} AS f
                JOIN temp_valid_combinations AS v
                USING (table_index, key_col_index, row_index)
                """
                try:
                    cur.execute(join_sql)
                except Exception:
                    # Rollback the aborted transaction before retrying
                    conn.rollback()
                    cur.adbc_ingest("temp_valid_combinations", token_query_results_arrow, mode="create", temporary=True)
                    join_sql = f"""
                    SELECT v.key, f.feature_index, f.table_index, f.key_col_index, f.row_index, f.count, f.sum, f.diag, f.qcr_term_positive, f.qcr_term_negative
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
        _has_headers = (
            'table_name' in join_selection_query_results.columns
            and 'column_headers' in join_selection_query_results.columns
        )
        _header_cols = 'table_name, column_headers, ' if _has_headers else ''
        join_selection_query_results = (
            join_selection_query_results
                .sql(f"""
                    SELECT key, feature_index, table_index, key_col_index, row_index, count, sum, diag, {_header_cols}table_column_index, table_row_index,
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
        _has_headers = (
            'table_name' in join_selection_query_results.columns
            and 'column_headers' in join_selection_query_results.columns
        )
        _header_cols = 'table_name, column_headers, ' if _has_headers else ''
        join_selection_query_results = (
            join_selection_query_results
                .sql(f"""
                    SELECT key, feature_index, table_index, key_col_index, row_index, count, sum, diag, {_header_cols}table_column_index, table_row_index,
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
