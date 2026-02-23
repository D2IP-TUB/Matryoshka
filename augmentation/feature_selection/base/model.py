import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import t
from ...utils.exceptions import LinAlgError


class FeatureSelectionModel(ABC):
    @abstractmethod
    def __init__(
        self,
        y_t_y,
        n_iter,
        r=None,
        c=None
    ):
        super().__init__()
        self.y_t_y = y_t_y
        self.n_iter = n_iter
        self.r, self.c = r, c


    @staticmethod
    def _cit_pvalue(r: float, k: int, n: int) -> float:
        """Two-sided p-value for partial correlation r (df = n-k-2)."""
        p_vals = []
        for r_i in r:
            df = max(n - k - 2, 1)
            r_cl = max(min(r_i, 1.0), -1.0)
            t_stat = r_cl * np.sqrt(df / max(1e-16, 1 - r_cl * r_cl))
            p_val = 2 * t.sf(abs(t_stat), df)
            p_vals.append(p_val)
        return np.array([np.min(p_vals)])


    @abstractmethod
    def fit(self, joint_cofactor):
        pass


    @abstractmethod
    def predict(self, cofactor_least_squares, return_coef: bool = False):
        pass

    
    def score(
        self,
        metric: str,
        joint_cofactor: np.ndarray = None,
        feature_target: np.ndarray = None,
        joint_count: int = None,
        sum_y: np.ndarray | None = None,
        new_features: int = None,
        **kwargs
    ) -> float:
        match metric:
            case 'mse':
                singularity, cond_number = self._is_near_singular(joint_cofactor)
                if singularity:
                    raise LinAlgError(cond_number)
                metric_score = -( ( self.y_t_y - np.multiply(2, (self.coef_.T @ feature_target)) + (self.coef_.T @ joint_cofactor @ self.coef_) ) / joint_count )

            case 'rmse':
                singularity, cond_number = self._is_near_singular(joint_cofactor)
                if singularity:
                    raise LinAlgError(cond_number)
                metric_score = -np.sqrt( ( ( self.y_t_y - np.multiply(2, (self.coef_.T @ feature_target)) + (self.coef_.T @ joint_cofactor @ self.coef_) ) / joint_count ) )

            case 'r2':
                singularity, cond_number = self._is_near_singular(joint_cofactor)
                if singularity:
                    raise LinAlgError(cond_number)
                rss = self.y_t_y - self.coef_.T @ feature_target
                tss = self.y_t_y - ((sum_y / joint_count) ** 2) / joint_count
                metric_score = 1 - (rss / tss)

            case 'adj_r2':
                singularity, cond_number = self._is_near_singular(joint_cofactor)
                if singularity:
                    raise LinAlgError(cond_number)
                rss = self.y_t_y - self.coef_.T @ feature_target
                tss = self.y_t_y - ((sum_y / joint_count) ** 2) / joint_count
                n = joint_count
                p = joint_cofactor.shape[0]
                metric_score = 1 - ( (1 - (1 - (rss / tss))) * (n - 1) / max(1, (n - p - 1)) )

            case 'cit':
                # --------------  CIT branch -------------------------------------
                # We approximate the partial-corr r of the *last* column that was
                # just appended to `cofactor` (and is therefore tested now).
                #
                #   r ≈  (b_j) / √(G_jj · yᵀy)
                #
                #   * b_j   : last element of Aᵀy (= feature_target_vector[-1])
                #   * G_jj  : last diagonal of AᵀA (= cofactor[-1, -1])
                #   * k     : number of already-chosen columns (assume = cofactor.shape[0]-1)
                # ----------------------------------------------------------------
                base_features = joint_cofactor.shape[1] - new_features
                candidate_cofactor = joint_cofactor[:, base_features:]
                candidate_solution = self.predict(candidate_cofactor, True)
                candidate_cofactor_norm = joint_cofactor[-new_features:, -new_features:]
                candidate_residuals = np.max([np.trace(candidate_cofactor_norm - candidate_solution.T @ candidate_cofactor), 1e16])
                
                solution = self.coef_.reshape(-1)
                feature_target_vector = feature_target_vector.reshape(-1)
                cofactor_residuals = self.y_t_y - solution.T @ feature_target_vector

                feature_target_candidate = feature_target_vector[-new_features:].reshape(-1)
                cross_product = feature_target_candidate - candidate_solution.T @ feature_target_vector - solution.T @ candidate_cofactor + candidate_solution.T @ joint_cofactor @ feature_target_vector
 
                r = cross_product / np.sqrt(candidate_residuals * cofactor_residuals)
                k = joint_cofactor.shape[0] - new_features
                metric_score = self._cit_pvalue(r, k, joint_count)
            
            case 'average_mahalanobis':
                metric_score = np.array([[np.sum(np.abs(self.coef_))]]) # so that the output is consistent with other metrics
    
            case _:
                raise ValueError(f"Unknown metric '{metric}'")

        return metric_score
    

    def _is_near_singular(self, cofactor) -> tuple[bool, float]:
        try:
            cond_number = np.linalg.cond(cofactor)
        except np.linalg.LinAlgError as e:
            cond_number = np.inf
        eps = np.finfo(cofactor.dtype).eps
        threshold = (1 / eps)
        return cond_number > threshold, cond_number


    def _is_skipped(self, joint_count:int) -> bool:
        return joint_count is None