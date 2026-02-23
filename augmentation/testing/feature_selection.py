import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../augmentation/')))

import time
import ray
import numpy as np
import polars as pl
from augmentation.feature_selection.base.model import FeatureSelectionModel
from augmentation.feature_selection.base.statistics import AugSketch, BaseSketch, FeatureSelectionActorNonRemote
from augmentation.utils.logging.logger_config import setup_logger
from .models import TrueSolver
from tabulate import tabulate

@ray.remote
class FeatureSelectionTest(FeatureSelectionActorNonRemote):
    def __init__(self, query_column_name: str, target_column_name: str) -> None:
        super().__init__()
        self.base_table_path = self._compute_abspath('base_table.csv')
        self.query_column_name = query_column_name
        self.target_column_name = target_column_name
        log_dir = self._compute_abspath('testing_logs')
        timestamp = time.asctime(time.localtime()).replace(' ', '_').replace(':', '_')
        self.logger = setup_logger(name='exhaustive_index', log_dir=log_dir, log_file=f'testing_{timestamp}.log', silent=False)
    

    def run_fs_iteration(self, best_feature: str, model: FeatureSelectionModel, metric: str, table_feature_index: str, base_table_sketch: BaseSketch, aug_table_sketch: AugSketch, n_iter: int, best_feature_index: str = None) -> dict[str, float]:
        if n_iter == 0:
            logging_path = self._compute_abspath('logs')
            if not os.path.exists(logging_path):
                os.mkdir(logging_path)
            joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count = self._join_aug_table(table_feature_index, base_table_sketch, aug_table_sketch)
            M1 = M2 = d = None
        else:
            if table_feature_index not in self.cache:
                return {table_feature_index: float('inf')}
            joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count = self._update_from_cache(table_feature_index)
            M1, M2 = self.M1, self.M2
            d = self.d
        
        iteration_path = self._compute_abspath(f'logs/iter/iteration_{n_iter}')
        if not os.path.exists(iteration_path):
            os.makedirs(iteration_path)
        
        true_cofactor, true_feature_target_vector = self._ground_truth_join(best_feature, table_feature_index, n_iter, iteration_path)
        stats_check = self._compare_statistics(joint_cofactor_matrix, true_cofactor, joint_feature_target_vector, true_feature_target_vector)
        if not np.all(stats_check):
            self.logger.error(f'Statistics mismatch for {table_feature_index} at iteration {n_iter}')
            self.logger.info(f'Joint cofactor matrix: {tabulate(joint_cofactor_matrix)}')
            self.logger.info(f'True cofactor matrix: {tabulate(true_cofactor)}')
            self.logger.info(f'Feature target vector: {tabulate(joint_feature_target_vector)}')
            self.logger.info(f'True feature target vector: {tabulate(true_feature_target_vector)}')

        try:
            linreg = model(y_t_y, n_iter, M1, M2, d)
            linreg.fit(joint_cofactor_matrix)
            linreg.predict(joint_feature_target_vector)
            score = linreg.score(joint_cofactor_matrix, joint_feature_target_vector, joint_count, metric=metric, new_features=new_features_count)

            true_linreg = TrueSolver(model.__name__, y_t_y, n_iter)
            true_linreg.fit(true_cofactor)
            true_linreg.predict(true_feature_target_vector)

            coefs = linreg.coef_
            true_coefs = true_linreg.coef_
            check_coefs = np.allclose(coefs, true_coefs, rtol=1e-03)

        except np.linalg.LinAlgError as e:
            score = np.array([float('inf')])

        self.cache[table_feature_index]['M1'] = linreg.M1
        # self.logger.info(f'M1 for {table_feature_index} at iteration {n_iter}: {tabulate(linreg.M1)}')
        self.cache[table_feature_index]['M2'] = linreg.M2
        self.cache[table_feature_index]['d'] = linreg.d

        return {table_feature_index: score.item()}


    def _ground_truth_join(self, best_feature: str, table_feature_index: str, n_iter: int, iteration_path: str) -> tuple[np.ndarray, np.ndarray]:
        if n_iter == 0:
            base_table = pl.read_csv(self.base_table_path)
        else:
            prev_iter = n_iter - 1
            prev_iter_str = f'best/iteration_{prev_iter}'
            prev_iter_path = self._compute_abspath(f'logs/{prev_iter_str}')
            base_table_path = os.path.join(prev_iter_path, f'{best_feature}.csv')
            base_table = pl.read_csv(base_table_path)

        aug_table = pl.read_csv(self._compute_abspath(f'processed_aug_tables/{table_feature_index}.csv'))
        joined_table = base_table.join(aug_table, on='id', how='left')
        joined_table_path = os.path.join(iteration_path, f'{table_feature_index}.csv')
        joined_table.write_csv(joined_table_path)

        y = joined_table.select(self.target_column_name).to_numpy()
        X = joined_table.select(pl.exclude(self.target_column_name, self.query_column_name)).to_numpy()
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        cofactor = X.T @ X
        feature_target_vector = X.T @ y

        return cofactor, feature_target_vector
    

    def _compare_statistics(self, cofactor, true_cofactor, feature_target_vector, true_feature_target_vector):
        cofactors_comparison = np.allclose(cofactor, true_cofactor, rtol=1e-05, atol=1e-08)
        feature_target_vector_comparison = np.allclose(feature_target_vector, true_feature_target_vector, rtol=1e-05, atol=1e-08)

        check = np.array([feature_target_vector_comparison, cofactors_comparison])

        return check


    def _compute_abspath(self, filename: str) -> str:
        script_path = os.path.realpath(__file__)
        script_dir = os.path.dirname(script_path)
        return os.path.abspath(os.path.join(script_dir, filename))