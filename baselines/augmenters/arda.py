import gc
import os
import warnings
import pandas as pd
import polars as pl
from .arda_utils.arda import select_arda_features_budget_join
from baselines.discovery.aurum_join_discovery import AurumJoinDiscovery


class ArdaAugmenter:
    def run(
        self,
        join_paths_df_path: str,
        query_table_path: str,
        features: list[str],
        query_column_name: str,
        data_lake_path: str,
        base_node_id: str,
        target_column_name: str,
        sample_size: int,
        regression: bool,
        lake_table_sep: str,
        **kwargs
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Realestate's natural query key (LONG_NAME) has no usable LSH
            # containment overlap in the NYC lake. ARDA, Kitana, AutoFeat,
            # and CAAFE instead key on the ZIP code extracted from STATE.
            # Matryoshka's forward path keeps the original key.
            if base_node_id == 'realestate.csv':
                query_column_name = 'zip_code'
            _lake_parts = data_lake_path.rstrip('/').split('/')
            lake = _lake_parts[-2] if _lake_parts[-1] == 'extracted' else _lake_parts[-1]
            _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            aurum_index_file = os.path.join(_project_root, 'augmentation', 'Aurum', 'graphs', lake)
            if not os.path.isdir(aurum_index_file):
                aurum_index_file = os.path.join(_project_root, 'augmentation', 'Aurum', 'graphs', f'{lake}.pkl')
            aurum = AurumJoinDiscovery(aurum_index_file, separator=lake_table_sep)
            base_path = join_paths_df_path.split('/')[:-1]
            join_paths = aurum.find_joinable_tables(
                query_table_path=f'{"/".join(base_path)}/{base_node_id}',
                query_col=query_column_name,
                output_path=join_paths_df_path,
                features=[query_column_name]
            )
            del aurum  # Free LSH Ensemble index (~7+ GB for gittables)
            gc.collect()
            query_table = pl.read_csv(query_table_path)
            left_table, fetch_time, augmentation_time, augplan = select_arda_features_budget_join(
                join_paths_df_path=join_paths_df_path,
                query_table=query_table,
                query_column_name=query_column_name,
                data_lake_folder=data_lake_path,
                base_node_id=base_node_id,
                target_column_name=target_column_name,
                sample_size=sample_size,
                regression=regression,
                sep_lake=lake_table_sep,
                budget_seconds=kwargs.get('budget_seconds'),
                trajectory_dir=kwargs.get('trajectory_dir'),
            )
        return left_table, fetch_time, augmentation_time, augplan