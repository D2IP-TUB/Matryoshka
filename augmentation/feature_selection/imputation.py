import ray
import numpy as np
import polars as pl
from augmentation.utils.common import compute_outer_products
from .base.model import FeatureSelectionModel
from .base.sketch import AugSketch, BaseSketch
from .base.statistics import FeatureSelectionActor
from .base.user_table import BaseTable


@ray.remote
class MultivariateImputer(FeatureSelectionActor):
    def __init__(self, logger, base_table: BaseTable):
        super().__init__()
        self.logger = logger
        self.base_table = base_table
        self.query_column_name = base_table.query_column_name
        self.user_table_processed = base_table.table.sort(self.query_column_name)
        self.target_column_name = base_table.target_column_name
        self.nan_mask = base_table.nan_mask


    def run_fs_iteration(self, model: FeatureSelectionModel, table_feature_index: str, base_table_sketch: BaseSketch, aug_table_sketch: AugSketch, n_trials: int):
        joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count = self._join_aug_table(table_feature_index, base_table_sketch, aug_table_sketch)
        M1 = M2 = d = None
        linreg = model(y_t_y, 0, M1, M2, d)
        linreg.fit(joint_cofactor_matrix)
        collinear_indices = self._collinearity_check(linreg.M1, linreg.M2, new_features_count)
        if len(collinear_indices) == new_features_count:
            pass
        else:
            if len(collinear_indices) > 0:
                pos_indices = [i if i >= 0 else joint_cofactor_matrix.shape[1] + i for i in collinear_indices]
                all_indices = np.arange(joint_cofactor_matrix.shape[1])
                complement = np.setdiff1d(all_indices, pos_indices)
                joint_cofactor_matrix = joint_cofactor_matrix[np.ix_(complement, complement)]
                joint_feature_target_vector = joint_feature_target_vector[complement]

        X = self.user_table_processed.drop([self.query_column_name, self.target_column_name]).to_numpy()
        query_col = self.user_table_processed.select(self.query_column_name).to_numpy()
        features_names = [col for col in self.user_table_processed.columns if col not in [self.query_column_name, self.target_column_name]]
        X_imp = self._em_imputation(X, query_col, self.nan_mask, table_feature_index, base_table_sketch, aug_table_sketch, n_trials)

        imp_table = pl.concat(
            [
                self.user_table_processed.select([self.query_column_name, self.target_column_name]),
                pl.from_numpy(data=X_imp, schema=features_names)
            ],
            how='horizontal'
        )

        augmentation_plan = [] # for consistency of output with the other methods; external features are not needed here

        return augmentation_plan, imp_table


    def _em_imputation(self, X: np.ndarray, query_col: np.ndarray, nan_mask: np.ndarray, table_feature_index: str, base_table_sketch: BaseSketch, aug_table_sketch: AugSketch, n_trials: int, tol: float = 1e-6):
        nan_rows = np.any(~nan_mask, axis=1)
        X_imp = X.copy()
        for _ in range(n_trials):
            joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count = self._join_aug_table(table_feature_index, base_table_sketch, aug_table_sketch)
            cov = joint_cofactor_matrix / (joint_count - 1)
            X_prev = X_imp.copy()
            
            unique_patterns = {}
            for i in np.where(nan_rows)[0]:
                pattern = tuple(nan_mask[i])
                if pattern not in unique_patterns:
                    unique_patterns[pattern] = []
                unique_patterns[pattern].append(i)

            for pattern, row_indices in unique_patterns.items():
                missing_idx = np.array([not p for p in pattern])
                observed_idx = np.array(pattern)

                sigma_oo = cov[np.ix_(observed_idx, observed_idx)]
                sigma_mo = cov[np.ix_(missing_idx, observed_idx)]

                row_indices = np.array(row_indices)
                X_obs = X_imp[row_indices][:, observed_idx]

                alpha = np.linalg.pinv(sigma_oo) @ X_obs.T
                cond_means = sigma_mo @ alpha
                X_imp[np.ix_(row_indices, missing_idx)] = cond_means.T

            diff = X_imp - X_prev
            diff_missing_only = diff[~nan_mask]
            X_imp_missing_only = X_imp[~nan_mask]

            rel_change = np.linalg.norm(diff_missing_only) / (np.linalg.norm(X_imp_missing_only) + 1e-12)
            if rel_change < tol:
                break

            base_table_sketch = self._calibrate_base_sketch(base_table_sketch, X_imp, query_col)

        return X_imp


    def _calibrate_base_sketch(self, base_table_sketch: BaseSketch, X_imp: np.ndarray, query_col: np.ndarray):
        X_imp = np.hstack([query_col, X_imp])
        X_imp_group = np.split(X_imp[:, 1:], np.unique(X_imp[:, 0], return_index=True)[1][1:])
        new_sum = np.array(np.vstack([arr.sum(axis=0) for arr in X_imp_group]), dtype=np.float64)
        base_table_sketch.features_sum = new_sum
        base_table_sketch.features_cofactors = compute_outer_products(new_sum)

        return base_table_sketch

    
    def _join_aug_sketches(self, aug_tables_sketches: list[AugSketch]):
        joint_sum = np.concatenate([aug_table_sketch.sum_ for aug_table_sketch in aug_tables_sketches], axis=1)
        joint_count = aug_tables_sketches[0].count
        outer_prod = compute_outer_products(joint_sum)
        joint_diag = np.vstack([np.diag(outer_prod[i]) for i in range(outer_prod.shape[0])])

        triu = np.triu(outer_prod, k=1)
        tril = np.tril(outer_prod, k=-1)
        cofactors = triu+tril
        batch, n, _ = cofactors.shape
        diag_mask = ~np.eye(n, dtype=bool)
        diag_mask = np.broadcast_to(diag_mask, (batch, n, n))
        cofactors = cofactors[diag_mask].reshape(batch, n, -1)
        cofactors = cofactors.transpose(0, 2, 1)
        cofactors = cofactors.squeeze()
        if np.isnan(cofactors).all():
            joint_cofactors = np.expand_dims(cofactors, 1)
        else:
            if len(cofactors.shape) == 2:
                joint_cofactors = np.expand_dims(cofactors, 2)
            else:
                joint_cofactors = cofactors

        feature_index = 'joint_features'
        joint_sketch = AugSketch(feature_index, joint_count, joint_sum, joint_diag, joint_cofactors)

        return joint_sketch


    def _collinearity_check(self, M1, M2, new_features_count, tol=1e-6):
        triangular = self._find_upper_triangular([M1, M2])
        triangular_diag = np.diag(triangular)
        max_coef = np.max(np.abs(triangular_diag[:-1]))
        collinear_indices = []
        for i in range(1, new_features_count + 1):
            new_feature_coef = triangular_diag[-i]
            if np.isnan(new_feature_coef) or np.isinf(new_feature_coef):
                collinear_indices.append(-i)
            elif( np.abs(new_feature_coef) / max_coef) < tol:
                collinear_indices.append(-i)
            else:
                pass
        return collinear_indices


    def _find_upper_triangular(self, matrices: list[np.ndarray]) -> np.ndarray:
        triangular = np.zeros_like(matrices[0])
        for matrix in matrices:
            if matrix is None:
                continue
            if np.all(np.triu(matrix, k=1) == 0) or np.all(np.tril(matrix, k=-1) == 0):
                triangular += np.triu(matrix)
        return triangular