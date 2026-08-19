from abc import ABC, abstractmethod

import numpy as np
from scipy.stats import t

from ...exceptions import LinAlgError


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

            case 'gcv_rmse':
                # Generalised cross-validation RMSE. Sketch-only LOO surrogate
                # (Craven and Wahba 1979) that replaces row-wise leverages with
                # their trace average tr(H)/n = p/n. The penalty 1/(1 - p/n)
                # blows up as the model approaches saturation, which actively
                # discourages over-selection.
                singularity, cond_number = self._is_near_singular(joint_cofactor)
                if singularity:
                    raise LinAlgError(cond_number)
                n = joint_count
                p = joint_cofactor.shape[0]
                if p >= n:
                    # Pathological case. More parameters than samples means
                    # GCV is undefined. Treat as a singular fit so the caller
                    # drops this candidate.
                    raise LinAlgError(float('inf'))
                rss = self.y_t_y - np.multiply(2, (self.coef_.T @ feature_target)) + (self.coef_.T @ joint_cofactor @ self.coef_)
                gcv_denom = 1.0 - p / n
                metric_score = -np.sqrt(rss / n) / gcv_denom

            case 'conditional_gcv':
                # Incremental conditional OLS proxy (regression). Score the
                # candidate block -- the last ``new_features`` columns of the
                # joint Gram -- given the current set (the preceding columns),
                # ranked by GCV. Uses only second-moment statistics; the OLS
                # ``coef_`` is not required. Returning ``-gcv_new`` is order-
                # equivalent to ranking by ``gcv_improvement`` (``gcv_current``
                # is constant across candidates at a given iteration) and keeps
                # the "higher is better" + relative-improvement-stop convention.
                from ..models import conditional_regression_score
                n = joint_count
                P = joint_cofactor.shape[0]
                q = int(new_features) if new_features else P
                q = max(1, min(q, P))
                p0 = P - q
                res = conditional_regression_score(
                    current_xtx=joint_cofactor[:p0, :p0] if p0 > 0 else None,
                    current_xty=feature_target[:p0].reshape(-1) if p0 > 0 else None,
                    candidate_ztz=joint_cofactor[p0:, p0:],
                    candidate_zty=feature_target[p0:].reshape(-1),
                    yty=float(self.y_t_y),
                    n=n,
                    cross_ztx=joint_cofactor[p0:, :p0] if p0 > 0 else None,
                    lambda_x=kwargs.get('lambda_x', 1e-3),
                    lambda_z=kwargs.get('lambda_z', 1e-3),
                )
                metric_score = np.array([[-res['gcv_new']]])

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
                # EXPERIMENTAL, unvalidated. Not reachable through
                # `DiscoveryConfig`: 'cit' appears in neither
                # `REGRESSION_METRICS` nor `CLASSIFICATION_METRICS`, and it has
                # no test coverage. The branch referenced an undefined name
                # (`feature_target_vector`) and so raised NameError on every
                # call; that name is the `score` parameter `feature_target`,
                # corrected here. The numerics remain unverified.
                # --------------  CIT branch -------------------------------------
                # We approximate the partial-corr r of the *last* column that was
                # just appended to `cofactor` (and is therefore tested now).
                #
                #   r ≈  (b_j) / √(G_jj · yᵀy)
                #
                #   * b_j   : last element of Aᵀy (= feature_target[-1])
                #   * G_jj  : last diagonal of AᵀA (= cofactor[-1, -1])
                #   * k     : number of already-chosen columns (assume = cofactor.shape[0]-1)
                # ----------------------------------------------------------------
                base_features = joint_cofactor.shape[1] - new_features
                candidate_cofactor = joint_cofactor[:, base_features:]
                candidate_solution = self.predict(candidate_cofactor, True)
                candidate_cofactor_norm = joint_cofactor[-new_features:, -new_features:]
                candidate_residuals = np.max([np.trace(candidate_cofactor_norm - candidate_solution.T @ candidate_cofactor), 1e16])

                solution = self.coef_.reshape(-1)
                feature_target = feature_target.reshape(-1)
                cofactor_residuals = self.y_t_y - solution.T @ feature_target

                feature_target_candidate = feature_target[-new_features:].reshape(-1)
                cross_product = feature_target_candidate - candidate_solution.T @ feature_target - solution.T @ candidate_cofactor + candidate_solution.T @ joint_cofactor @ feature_target

                r = cross_product / np.sqrt(candidate_residuals * cofactor_residuals)
                k = joint_cofactor.shape[0] - new_features
                metric_score = self._cit_pvalue(r, k, joint_count)

            case 'average_mahalanobis' | 'average_bhattacharyya' | 'robust_moment' | 'conditional_mahalanobis':
                # `coef_` already holds the pairwise class-separability scores
                # (squared Mahalanobis / Bhattacharyya distances, or the
                # robust moment-based separability S for binary classification).
                # Aggregating by sum-of-abs is identical for all three metrics;
                # the choice only affects which kernel filled `coef_` upstream.
                metric_score = np.array([[np.sum(np.abs(self.coef_))]]) # so that the output is consistent with other metrics

            case _:
                raise ValueError(f"Unknown metric '{metric}'")

        return metric_score


    def _is_near_singular(self, cofactor) -> tuple[bool, float]:
        try:
            cond_number = np.linalg.cond(cofactor)
        except np.linalg.LinAlgError:
            cond_number = np.inf
        eps = np.finfo(cofactor.dtype).eps
        threshold = (1 / eps)
        return cond_number > threshold, cond_number


    def _is_skipped(self, joint_count:int) -> bool:
        return joint_count is None
