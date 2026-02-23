import warnings
import pandas as pd
import polars as pl
from .arda_utils.arda import select_arda_features_budget_join
from augmentation.retrieval import AurumJoinDiscovery


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
            lake = data_lake_path.split('/')[-2]
            aurum_index_file = f'augmentation/Aurum/graphs/{lake}.pkl'
            aurum = AurumJoinDiscovery(aurum_index_file, separator=lake_table_sep)
            base_path = join_paths_df_path.split('/')[:-1]
            join_paths = aurum.find_joinable_tables(
                query_table_path=f'{"/".join(base_path)}/{base_node_id}',
                output_path=join_paths_df_path,
                features=features
            )
            query_table = pl.read_csv(query_table_path)
            query_table = query_table.select([*features, target_column_name])
            left_table, fetch_time, augmentation_time, augplan = select_arda_features_budget_join(
                join_paths_df_path=join_paths_df_path,
                query_table=query_table,
                query_column_name=query_column_name,
                data_lake_folder=data_lake_path,
                base_node_id=base_node_id,
                target_column_name=target_column_name,
                sample_size=sample_size,
                regression=regression,
                sep_lake=lake_table_sep
            )
        return left_table, fetch_time, augmentation_time, augplan