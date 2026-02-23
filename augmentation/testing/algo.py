import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../augmentation/')))

import ray
import shutil
import polars as pl
from augmentation.join_selection import JoinSelection
from augmentation.feature_selection.models import FeatureSelectionModel
from .feature_selection import FeatureSelectionTest


class FeatureSelectionAlgo(JoinSelection):
    def __init__(self, token_index_table_name: str, feature_selection_table_name: str, overlap_table_name: str) -> None:
        super().__init__(token_index_table_name=token_index_table_name,
                         feature_selection_table_name=feature_selection_table_name,
                         overlap_table_name=overlap_table_name)


    def find_best_joins(self, model: FeatureSelectionModel, user_table_processed: pl.DataFrame, query_column_name: str, target_feature_name: str, metric: str, top_k: int, min_overlap_share: float, n_jobs: int = 16) -> pl.DataFrame:
        '''
        Use feature selection algorithm to find the best joins among the joinable columns retrieved from the index.

        Parameters:
        ----------
        model: FeatureSelectionModel
            Feature selection model to use for the feature selection algorithm.

        user_table_processed: pl.DataFrame
            User table to find joinable tables and perform feature selection for. \n
            All rows must be unique and all columns except for the foreign key column must be numeric.

        query_column_name: str
            Name of the query column in the user table

        top_k: int
            Number of columns to select by overlap

        Returns:
        --------
        pl.DataFrame: Augmented table with the best features
        '''
        # check that user_table_processed has only numeric columns
        # raises error and breaks execution otherwise
        user_table_processed = self._user_table_checkup(user_table_processed, query_column_name)
        # target feature should be the last column in the user table for future consistency in sketch operations
        target_feature = user_table_processed.get_column(target_feature_name)
        user_table_processed = user_table_processed.drop(target_feature_name)
        user_table_processed.insert_column(0, target_feature)
        query_column = user_table_processed.select(query_column_name)
        rows_map, joint_overlap_tables, joint_overlap_columns, joint_overlap_rows, duplicate_rows_map = self.find_joinable_tables(query_column, top_k=top_k, join_selection=True)

        join_selection_query_results = self._run_join_selection_query(joint_overlap_tables, joint_overlap_columns, joint_overlap_rows)

        columns_map = self._create_column_row_mapping(rows_map)
        n_foreign_keys_found = len(columns_map)
        self.logger.info(f'Total foreign keys found: {n_foreign_keys_found}/{top_k}')
        columns_map, rows_map, common_rows_subset = self._find_common_joinable_rows(columns_map, rows_map, duplicate_rows_map, min_overlap_share)
        self.logger.info(f'Common rows subset is: {len(common_rows_subset)} rows')
        self.logger.info(f'Remaining foreign keys: {len(columns_map)}/{n_foreign_keys_found}')

        user_table_agg = self._sketch_user_table(user_table_processed, query_column_name, common_rows_subset)

        base_table_sketch, aug_tables_sketches = self._create_fs_sketches(rows_map, join_selection_query_results, user_table_agg)
        top_features = self._feature_selection_algo(query_column_name, target_feature_name, base_table_sketch, aug_tables_sketches, model, metric, n_jobs)

        # joinable_tables_content = self._run_content_query(rows_map=rows_map, top_features=top_features)
        # augmentation_table = self._prepare_augmentation_table(joinable_tables_content, overlap_columns, rows_map, duplicate_rows_map)

        self.logger.handlers.clear()

        return top_features
    

    def _feature_selection_algo(self, query_column_name, target_column_name, base_table_sketch, aug_tables_sketches, model, metric, n_jobs):
        ray.init(num_cpus=n_jobs, ignore_reinit_error=True)
        augmentation_plan = []
        scores = []
        global_score = float('inf')
        N_ITER = len(aug_tables_sketches)
        self.logger.info(f'Total features in the pool: {N_ITER}')
        self.logger.info(f'Running feature selection algorithm with model {model.__name__}')
        actor = FeatureSelectionTest.remote(query_column_name, target_column_name)
        for n_iter in range(N_ITER):
            if n_iter == 0:
                best_feature = None
            futures = [
                actor.run_fs_iteration.remote(
                    best_feature,
                    model,
                    metric,
                    aug_sketch.table_feature_index,
                    base_table_sketch,
                    aug_sketch,
                    n_iter
                ) for aug_sketch in aug_tables_sketches
            ]
            iter_scores = ray.get(futures)
            iter_scores = {k: v for d in iter_scores for k, v in d.items()}
            best_feature = min(iter_scores, key=iter_scores.get)
            best_score = iter_scores[best_feature]
            scores.append(best_score)

            iteration_path = self._compute_abspath(f'logs/iter/iteration_{n_iter}/{best_feature}.csv')
            best_features_path = self._compute_abspath(f'logs/best/iteration_{n_iter}')
            if not os.path.exists(best_features_path):
                os.makedirs(best_features_path)
            shutil.copyfile(iteration_path, f'{best_features_path}/{best_feature}.csv')

            if not global_score or best_score < global_score:
                global_score = best_score
                best_feature_cache = ray.get(actor.retrieve.remote('cache'))[best_feature]
                actor.update_with_best_feature.remote(
                    best_feature_cache['new_feature'],
                    best_feature_cache['joint_cofactor_matrix'],
                    best_feature_cache['joint_feature_target_vector'],
                    best_feature_cache['joint_count'],
                    best_feature_cache['y_t_y'],
                    best_feature_cache['M1'],
                    best_feature_cache['M2'],
                    best_feature_cache['d']
                )
                actor.remove_from_features_pool.remote(best_feature)
                augmentation_plan.append(best_feature)
            else:
                break

        self.logger.info(f'Final score: {global_score}')

        return augmentation_plan
    

    def _compute_abspath(self, filename: str) -> str:
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
        return os.path.abspath(os.path.join(script_dir, filename))