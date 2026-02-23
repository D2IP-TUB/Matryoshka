import psycopg
import time
import yaml
import pandas as pd
import polars as pl
from .qcr_utils.Baseline_DataDiscovery import DataDiscovery_CLS
from polars.exceptions import DuplicateError
with open('augmentation/utils/database/db_config.yaml', 'r') as f:
    db_config = yaml.safe_load(f)['db']


class QcrAugmenter:
    host = db_config['host']
    dbname = db_config['dbname']
    user = db_config['user']
    password = db_config['password']

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
        retrieved_results, augplan = self._merge_two_columns(results, data_lake_path, lake_table_sep)
        final_df = query_table.with_columns(pl.col(query_column_name).cast(pl.String).str.to_lowercase())
        for merged_two_col_df in retrieved_results:
            merged_two_col_df = merged_two_col_df.with_columns(
                pl.col(merged_two_col_df.columns[0]).cast(pl.String).str.to_lowercase()
            )
            merged_two_col_df = merged_two_col_df.rename({merged_two_col_df.columns[0]: query_column_name})
            try:
                final_df = final_df.join(
                    merged_two_col_df,
                    on=query_column_name,
                    how='left'
                )
            except DuplicateError:
                continue
        end = time.perf_counter()
        augmentation_time = end - start

        return final_df, fetch_time, augmentation_time, augplan
    

    def _merge_two_columns(self, results, data_lake_path: str, lake_table_sep: str) -> list[pl.DataFrame]:
        retrieved_results = []
        augplan = []
        for tableid,  _ in results:
            try:
                result = tableid.split("_|_")
                table_path = f"{data_lake_path}/{result[0].split('/')[-1].replace('.tsv.gz', '.csv').replace('.parquet', '.csv')}"
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