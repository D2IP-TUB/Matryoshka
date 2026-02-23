import hashlib
import json
import os
import numpy as np
import polars as pl
from experiments.plots_style import apply_style


class PlotGenerator:

    def _create_end_to_end_df(self, df: pl.DataFrame, config_df: pl.DataFrame):
        end_to_end = (
            df
                .filter(pl.col('message') != 'Join Selection started')
                .select(['config', 'runtime', 'message'])
                .join(config_df, left_on='config', right_on='config_hash', how='left')
                .with_columns(pl.col(['var_threshold', 'corr_threshold']).fill_null(0))
        )
        end_to_end = (
            end_to_end
                .group_by(['query_col_cardinality', 'var_threshold', 'corr_threshold', 'message'])
                .agg(pl.col('runtime').mean())
                .group_by(['query_col_cardinality', 'var_threshold', 'corr_threshold'])
                .agg(pl.col('runtime').sum())
                .sort('query_col_cardinality')
        )
        
        return end_to_end


    def _create_pruning_df(self, df: pl.DataFrame, config_df: pl.DataFrame):
        pruning = (
            df
                .filter(pl.col('message') == 'Pruning')
                .select(['config', 'runtime'])
                .join(config_df, left_on='config', right_on='config_hash', how='left')
                .with_columns(pl.col(['var_threshold', 'corr_threshold']).fill_null(0))
        )
        pruning = pruning.group_by(['query_col_cardinality', 'var_threshold', 'corr_threshold']).agg(pl.col('runtime').mean()).sort('query_col_cardinality')

        return pruning


    def _create_retrieval_df(self, df: pl.DataFrame, config_df: pl.DataFrame):
        retrieval = (
            df
                .filter(pl.col('message') == 'Retrieval')
                .select(['config', 'runtime'])
                .join(config_df, left_on='config', right_on='config_hash', how='left')
                .with_columns(pl.col(['var_threshold', 'corr_threshold']).fill_null(0))
        )
        retrieval = retrieval.group_by(['query_col_cardinality']).agg(pl.col('runtime').mean()).sort('query_col_cardinality')

        return retrieval


    def _create_experiment_dfs(self, experiment_log_dir: str, hashes: list[str]):
        logs = os.listdir(experiment_log_dir)
        log_tuples = []
        for log in logs:
            with open(os.path.join(experiment_log_dir, log), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    log_tuples.append(json.loads(line))
        
        df = pl.DataFrame(log_tuples)
        config_df = pl.DataFrame(hashes, schema=['var_threshold', 'corr_threshold', 'query_col_cardinality', 'config_hash'])
        df = df.join(config_df, left_on='config', right_on='config_hash', how='left')
        df = df.with_columns(pl.col(['config', 'query_col_cardinality']).fill_null(strategy='forward'))
        df = df.filter(pl.col('message') != 'Join Selection started')

        return df, config_df

    
    def _generate_experiment_hash(self, config_table_path: str):
        hashes = []
        df_config = pl.read_csv(config_table_path)
        tuples = df_config.rows()
        for var_threshold, corr_threshold, query_col_cardinality, top_k in tuples:
            run_config = {
                'var_threshold': var_threshold,
                'corr_threshold': corr_threshold,
                'top_k': top_k,
                'query_col_cardinality': query_col_cardinality
            }
            config_str = str(run_config)
            config_hash = hashlib.md5(config_str.encode()).hexdigest()
            hashes.append([var_threshold, corr_threshold, query_col_cardinality, config_hash])
        
        return hashes