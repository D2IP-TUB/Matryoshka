import subprocess
import time
import os
import random
import warnings
import numpy as np
import pandas as pd
import polars as pl
from augmentation.retrieval import AurumJoinDiscovery
from .autofeat_utils.autofeat_pipeline.autofeat import AutoFeat as AutoFeatBase
from .autofeat_utils.autofeat_pipeline.evaluate_join_paths import evaluate_paths
from ...utils.common import process_key

# Set PYTHONHASHSEED for deterministic hashing
os.environ['PYTHONHASHSEED'] = '42'


class AutofeatAugmenter:
    def run(
        self,
        join_paths_df_path: str,
        query_column_name: str,
        data_lake_path: str,
        base_table_sep: str,
        lake_table_sep: str,
        problem_type: str,
        base_node_id: str,
        target_column_name: str,
        features: list[str],
        base_table_label: str = 'base_table',
        save_joins_to_disk: bool = False,
        use_polars: bool = False,
        value_ratio: float = 0.5,
        top_k: int = 15,
        sample_size: int = 3000,
        pearson: bool = False,
        jmi: bool = False,
        no_relevance: bool = False,
        no_redundancy: bool = False,
        algorithm: str = "RF",
        **kwargs
    ):
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            start = time.perf_counter()
            base_table_path = '/'.join(join_paths_df_path.split('/')[:3]) + '/' + base_node_id
            # subprocess.run(['cp', base_table_path, f'{data_lake_path}/{base_node_id}'])
            (
                pl.read_csv(base_table_path, separator=base_table_sep)
                    .select([*features, target_column_name])
                    .write_csv(f'{data_lake_path}/{base_node_id}')
            )
            lake = data_lake_path.split('/')[-2]
            aurum_index_file = f'augmentation/Aurum/graphs/{lake}.pkl'
            aurum = AurumJoinDiscovery(aurum_index_file, separator=lake_table_sep)
            base_path = join_paths_df_path.split('/')[:-1]
            join_paths = aurum.find_joinable_tables(
                query_table_path=f'{"/".join(base_path)}/{base_node_id}',
                output_path=join_paths_df_path,
                features=features
            )
            join_paths_df = pd.read_csv(join_paths_df_path)
            join_paths_df['from_id'] = base_node_id
            autofeat = AutoFeatBase(
                join_paths_df=join_paths_df,
                lake_data_folder=data_lake_path,
                base_table_sep=base_table_sep,
                base_table_label=base_table_label,
                base_table_id=base_node_id,
                target_column=target_column_name,
                save_joins_to_disk=save_joins_to_disk,
                use_polars=use_polars,
                task=problem_type,
                value_ratio=value_ratio,
                top_k=top_k,
                sample_size=sample_size,
                pearson=pearson,
                jmi=jmi,
                no_relevance=no_relevance,
                no_redundancy=no_redundancy
            )
            autofeat.streaming_feature_selection(join_paths_df, data_lake_path, lake_table_sep, queue={base_node_id})
            _, top_k_paths, final_selected_features = evaluate_paths(
                bfs_result=autofeat, problem_type=problem_type, algorithm=algorithm,
                join_paths_df=join_paths_df,
                lake_data_folder=data_lake_path,
                lake_table_sep=lake_table_sep,
                base_table_sep=base_table_sep,
            )

            end = time.perf_counter()
            fetch_time = end - start

            start = time.perf_counter()
            final_selected_features_dict = {}
            left_table = pd.read_csv(f'{data_lake_path}/{base_node_id}', sep=base_table_sep)
            augplan = []
            if len(final_selected_features) > 0:
                for v in final_selected_features:
                    file, col = v.split('.', 1)  # split only at the first '.'
                    # if filenames may contain dots, split from the right instead:
                    # file, col = v.rsplit('.', 1)
                    col = col.split('.', 1)[-1]
                    col = col.split('.', 1)[0]
                    final_selected_features_dict.setdefault(file, []).append(col)
                for table_name, features in final_selected_features_dict.items():
                    if '.csv' not in table_name:
                        table_name += '.csv'
                    try:
                        query_column_name = join_paths_df[join_paths_df["to_id"] == table_name]['from_column'].values[0]
                    except IndexError:
                        continue
                    left_table[query_column_name] = left_table[query_column_name].apply(process_key)
                    join_key = join_paths_df[
                        (join_paths_df["from_column"] == query_column_name) &
                        (join_paths_df["to_id"] == table_name)
                    ]['to_column'].values[0]
                    
                    right_table = pd.read_csv(
                        f'{data_lake_path}/{table_name}',
                        header=0,
                        engine="python",
                        encoding="utf8",
                        sep=lake_table_sep
                    )
                    
                    # Resolve placeholder column name (col_4) to actual column name
                    actual_join_key = join_key
                    if join_key.startswith('col_'):
                        try:
                            col_index = int(join_key.split('_')[1])
                            if col_index < len(right_table.columns):
                                actual_join_key = right_table.columns[col_index]
                        except (ValueError, IndexError):
                            pass
                    
                    right_table = right_table.groupby(actual_join_key).sample(
                        n=1, random_state=42
                    )
                    right_table[actual_join_key] = right_table[actual_join_key].apply(process_key)
                    right_table = right_table[[actual_join_key, *features]]
                    left_table = pd.merge(
                        left_table,
                        right_table,
                        how="left",
                        left_on=query_column_name,
                        right_on=actual_join_key
                    )
                    augplan.extend(features)
            left_table_columns = left_table.columns.tolist()
            left_table_columns_non_unique = [col for col in left_table_columns if left_table_columns.count(col) > 1]
            left_table_columns_non_unique_renamed = [col + '_right' if col in left_table_columns_non_unique else col for col in left_table_columns]
            left_table.columns = left_table_columns_non_unique_renamed
            left_table = pl.from_pandas(left_table)
            end = time.perf_counter()
            augmentation_time = end - start
            subprocess.run(['rm', f'{data_lake_path}/{base_node_id}'])

        return left_table, fetch_time, augmentation_time, augplan