import os
import psycopg
import time
import pandas as pd
import polars as pl
from .qcr_utils.Baseline_DataDiscovery import DataDiscovery_CLS
from polars.exceptions import DuplicateError

from matryoshka.db.settings import resolve_settings


class QcrAugmenter:
    # Connection parameters resolve through the library settings layer
    # (environment, then db_config.yaml), lazily on first access.
    def __init__(self, settings=None) -> None:
        resolved = resolve_settings(settings)
        self.host = resolved.host
        self.dbname = resolved.dbname
        self.user = resolved.user
        self.password = resolved.password
        self.port = resolved.port

    def run(
        self,
        top_k: int,
        query_column_name: str,
        target_column_name: str,
        qcr_table_name: str,
        data_lake_path: str,
        query_table: pl.DataFrame,
        lake_table_sep: str,
        **kwargs
    ) -> pl.DataFrame:
        db_name = 'postgres'
        conn_info = {
            'host': self.host,
            'dbname': self.dbname,
            'user': self.user,
            'password': self.password,
        }
        # Anytime instrumentation (see augmentation/feature_selection/anytime.py).
        # The clock starts at the top of ``run`` so the budget covers both the
        # correlation search and the join materialization; ``budget_seconds=None``
        # / ``trajectory_dir=None`` keep the original behaviour.
        from matryoshka.selection.anytime import BudgetClock, TrajectoryEmitter
        _clock = BudgetClock(kwargs.get('budget_seconds')).start()
        _emitter = (TrajectoryEmitter(kwargs.get('trajectory_dir'), algo='QCR')
                    if kwargs.get('trajectory_dir') else None)
        if _emitter is not None:
            _emitter.emit(0, 0.0, [])

        baseline_system = DataDiscovery_CLS(db_name, qcr_table_name, conn_info)
        connection = psycopg.connect(**conn_info).cursor()
        qcr_df = (
            query_table
                .select([query_column_name, target_column_name])
                .with_columns(pl.col(query_column_name).cast(pl.String).str.to_lowercase())
        )
        qcr_df = qcr_df.to_pandas()
        results, _, fetch_time = baseline_system.qcr_correlation_search(qcr_df, qcr_df.columns.values[0], qcr_df.columns.values[1], top_k, db_con=connection)
        start = time.perf_counter()
        retrieved_results, _ = self._merge_two_columns(results, data_lake_path, lake_table_sep)
        final_df = query_table.with_columns(pl.col(query_column_name).cast(pl.String).str.to_lowercase())
        # Materialize discovered features one join at a time. This is QCR's
        # dominant cost, so the budget is enforced here: we stop before each
        # join once the deadline has passed and return the features materialized
        # so far. ``augplan`` therefore lists exactly the columns present in
        # ``final_df``.
        augplan = []
        for merged_two_col_df in retrieved_results:
            if _clock.expired:
                break
            merged_two_col_df = merged_two_col_df.with_columns(
                pl.col(merged_two_col_df.columns[0]).cast(pl.String).str.to_lowercase()
            )
            merged_two_col_df = merged_two_col_df.rename({merged_two_col_df.columns[0]: query_column_name})
            feature_name = merged_two_col_df.columns[1]
            try:
                final_df = final_df.join(
                    merged_two_col_df,
                    on=query_column_name,
                    how='left'
                )
            except DuplicateError:
                continue
            augplan.append(feature_name)
            if _emitter is not None:
                _emitter.emit(len(augplan), _clock.elapsed_s, list(augplan))
        end = time.perf_counter()
        augmentation_time = end - start
        if _emitter is not None:
            _emitter.close()

        return final_df, fetch_time, augmentation_time, augplan
    

    def _merge_two_columns(self, results, data_lake_path: str, lake_table_sep: str) -> list[pl.DataFrame]:
        import glob as _glob
        retrieved_results = []
        augplan = []
        for tableid,  _ in results:
            try:
                result = tableid.split("_|_")
                stem = result[0].split('/')[-1].replace('.tsv.gz', '').replace('.parquet', '').replace('.csv', '')
                table_path = f"{data_lake_path}/{stem}.csv"
                # Fallback for re-extracted lakes (notably GitTables) where each
                # original table is materialised as one CSV per topic prefix
                # using a ``<prefix>__<original>.csv`` naming convention. The
                # qcr_index still references the original (pre-prefix) name so
                # the literal path above misses every file. Grab the first
                # match by suffix when the direct path does not exist.
                if not os.path.exists(table_path):
                    matches = _glob.glob(f'{data_lake_path}/*__{stem}.csv')
                    if matches:
                        table_path = matches[0]
                result_df = pd.read_csv(table_path, sep=lake_table_sep, encoding='utf-8-sig')
                # Strip BOM character from column names and result values
                result_df.columns = result_df.columns.str.lstrip('\ufeff')
                col1 = result[1].lstrip('\ufeff')
                col2 = result[2].lstrip('\ufeff')
                result_df = result_df.loc[:, [col1, col2]]
                result_df.iloc[:, 0] = result_df.iloc[:, 0].astype(str)
                result_df.iloc[:, 0] = result_df.iloc[:, 0].str.lower()
                cat_col, num_col = result_df.columns[0], result_df.columns[1]
                augplan.append(num_col)
                try:
                    result_df_mean = result_df.groupby(cat_col, as_index=False)[num_col].mean()
                except Exception as e:
                    result_df = result_df.dropna(subset=[num_col])
                    try:
                        result_df_mean = result_df.groupby(cat_col, as_index=False)[num_col].mean()
                    except Exception as e:
                        continue
                retrieved_results.append(pl.from_pandas(result_df_mean))
            except Exception as e:
                continue

        return retrieved_results, augplan