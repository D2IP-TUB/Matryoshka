import numpy as np
from scipy.linalg import cholesky, solve_triangular
from skopt import gp_minimize
from skopt.space import Real
from .base.model import FeatureSelectionModel
from ..utils.exceptions import LinAlgError
from skglm import Lasso as LinRegL1
from skglm import SparseLogisticRegression as LogRegL1


class RegressionQR(FeatureSelectionModel):
    def __init__(self, y_t_y, n_iter, M1=None, M2=None, d=None) -> None:
        skipped_feature = self._is_skipped(y_t_y)
        if skipped_feature:
            raise LinAlgError('Feature is skipped.')
        self.y_t_y = y_t_y
        self.n_iter = n_iter
        self.M1 = M1
        self.M2 = M2
        self.d = d


    def fit(self, joint_cofactor: np.ndarray, **kwargs) -> None:
        '''
        Implemented for the sake of sklearn-like API development.
        '''
        self.M1, self.M2 = np.linalg.qr(joint_cofactor)


    def predict(self, feature_target, **kwargs):
        solution = np.linalg.solve(self.M2, self.M1.T @ feature_target)
        self.coef_ = solution

    
    def fit_predict(self, joint_cofactor, feature_target, return_coef: bool = False, **kwargs) -> np.ndarray:
        self.fit(joint_cofactor)
        self.predict(feature_target)

        if return_coef:
            return self.coef_


class IncrementalRegressionFGS(FeatureSelectionModel):
    """
    OLS solver via Factorized Gram-Schmidt (F-GS) decomposition with
    incremental updates for forward feature selection.

    Decomposes the Gram matrix  G_k = R_k^T R_k  where R_k is upper
    triangular, and maintains  C_k = R_k^{-1}.  When a new candidate
    feature is appended, the factorization is updated in O(k^2) using
    only the cross-covariance vector  g_k = X_k^T x_{k+1}  and the
    self-covariance  gamma = x_{k+1}^T x_{k+1}, both extracted from
    the augmented Gram block.

    Attributes
    ----------
    M1 : C = R^{-1}  (coefficient matrix, upper triangular)
    M2 : R            (Cholesky-like factor, upper triangular)
    d  : C^T @ (X^T y), cached transformed RHS for inter-iteration reuse
    """
    def __init__(self, y_t_y, n_iter, M1=None, M2=None, d=None) -> None:
        skipped_feature = self._is_skipped(y_t_y)
        if skipped_feature:
            raise LinAlgError('Feature is skipped.')
        self.y_t_y = y_t_y
        self.n_iter = n_iter
        self.M1 = M1          # C = R^{-1} (cached)
        self.M2 = M2          # R          (cached)
        self.d = d             # C^T @ s    (cached transformed RHS)


    def fit(self, joint_cofactor: np.ndarray, **kwargs) -> None:
        if self.M1 is None or self.M2 is None:
            # Cold start – Cholesky factorisation of the Gram matrix
            L = cholesky(joint_cofactor, lower=True)
            self.M2 = L.T                                                  # R (upper triangular)
            self.M1 = solve_triangular(self.M2, np.eye(self.M2.shape[0]))  # C = R^{-1}
        else:
            self._incremental_fgs_update(joint_cofactor)


    def _incremental_fgs_update(self, joint_cofactor: np.ndarray) -> None:
        """
        Incremental F-GS update of the Gram matrix factorisation.

        Given cached  R_k, C_k  and the augmented Gram matrix  G_{k+1},
        extracts the cross-covariance  g_k = G_{k+1}[:k, k]  and self-
        covariance  gamma = G_{k+1}[k, k],  then computes:

            alpha_k = C_k^T  g_k            (projection coefficients)
            rho     = sqrt(gamma - ||alpha||^2)   (residual norm)

        and assembles the block-extended factors:

            R_{k+1} = [ R_k   alpha ]     C_{k+1} = [ C_k   -C_k alpha / rho ]
                      [  0     rho  ]                [  0        1 / rho      ]
        """
        C_k = self.M1
        R_k = self.M2
        k = R_k.shape[0]

        # Extract cross-covariance and self-covariance from augmented Gram
        g_k = joint_cofactor[:k, k]      # X_k^T x_{k+1}
        gamma = joint_cofactor[k, k]     # x_{k+1}^T x_{k+1}

        # Projection coefficients
        alpha = C_k.T @ g_k

        # Residual norm (variance unexplained by current feature set)
        rho_sq = gamma - alpha @ alpha
        if rho_sq <= 0:
            raise LinAlgError('Feature is linearly dependent.')
        rho = np.sqrt(rho_sq)

        # Build R_{k+1}
        R_new = np.zeros((k + 1, k + 1))
        R_new[:k, :k] = R_k
        R_new[:k, k] = alpha
        R_new[k, k] = rho

        # Build C_{k+1} = R_{k+1}^{-1}
        C_alpha = C_k @ alpha
        C_new = np.zeros((k + 1, k + 1))
        C_new[:k, :k] = C_k
        C_new[:k, k] = -C_alpha / rho
        C_new[k, k] = 1.0 / rho

        self.M2 = R_new
        self.M1 = C_new


    def predict(self, feature_target, **kwargs):
        # d = C^T @ s  where s = X^T y (feature_target)
        self.d = self.M1.T @ feature_target
        # Solve  R beta = d  via back-substitution
        solution = solve_triangular(self.M2, self.d)
        self.coef_ = solution


    def fit_predict(self, joint_cofactor, feature_target, return_coef: bool = False, **kwargs) -> np.ndarray:
        self.fit(joint_cofactor)
        self.predict(feature_target)

        if return_coef:
            return self.coef_


class RegressionCholesky(FeatureSelectionModel):
    def __init__(self, y_t_y, n_iter, M1=None, M2=None, d=None) -> None:
        skipped_feature = self._is_skipped(y_t_y)
        if skipped_feature:
            raise LinAlgError('Feature is skipped.')
        self.y_t_y = y_t_y
        self.n_iter = n_iter
        self.M1 = M1
        self.M2 = M2
        self.d = d


    def fit(self, joint_cofactor: np.ndarray, **kwargs) -> None:
        '''
        Implemented for the sake of sklearn-like API development.
        '''
        self.M1 = cholesky(joint_cofactor, lower=True)


    def predict(self, feature_target, **kwargs):
        solution = solve_triangular(self.M1, feature_target, lower=True)
        self.coef_ = solution

    
    def fit_predict(self, joint_cofactor, feature_target, return_coef: bool = False, **kwargs) -> np.ndarray:
        self.fit(joint_cofactor)
        self.predict(feature_target)
        
        if return_coef:
            return self.coef_


class ClassificationCholesky(FeatureSelectionModel):
    def __init__(self, y_t_y, n_iter, M1=None, M2=None, d=None) -> None:
        skipped_feature = self._is_skipped(y_t_y)
        if skipped_feature:
            raise LinAlgError('Feature is skipped.')
        self.n_iter = n_iter
        self.M1 = M1
        self.M2 = M2
        self.d = d


    def fit(self, joint_cofactor: np.ndarray, **kwargs) -> None:
        '''
        Implemented for the sake of sklearn-like API development.
        '''
        try:
            self.M1 = cholesky(joint_cofactor, lower=True)
        except np.linalg.LinAlgError as e:
            self.M1, self.M2 = np.linalg.qr(joint_cofactor)


    def predict(self, feature_target, return_coef: bool = False):
        pass


    def fit_predict(self, class_covariances, class_means, return_coef: bool = False, **kwargs) -> np.ndarray:
        n_labels = class_means.shape[0]
        i_indices, j_indices = np.triu_indices(n_labels, k=1)
        mean_diffs = class_means[i_indices] - class_means[j_indices]
        # divide by 2 because of pairwise distance computations
        cov_pooled = (class_covariances[i_indices] + class_covariances[j_indices]) / 2.0

        Ls = np.linalg.cholesky(cov_pooled)
        z = np.linalg.solve(Ls, mean_diffs[..., None])

        # squared Mahalanobis distances (n_pairs,)
        d = np.sum(z.squeeze(-1)**2, axis=1)
        self.coef_ = d

        if return_coef:
            return d


class Lasso(FeatureSelectionModel):
    def __init__(self, y_t_y, n_iter=None, M1=None, M2=None, d=None) -> None:
        self.y_t_y = y_t_y
        self.n_iter = n_iter
        self.M1 = M1
        self.M2 = M2
        self.d = d


    def fit_predict(self, cofactor, feature_target, alphas=(1e-4, 100.0), n_trials=100, return_coef=False) -> np.ndarray:
        scale = np.sqrt(np.mean(cofactor ** 2))
        cofactor_scaled = cofactor / scale
        feature_target_scaled = feature_target / scale
        alphas_scaled = (alphas[0] / scale, alphas[1] / scale)
        search_space = [Real(*alphas_scaled, prior='log-uniform', name='lambda')]

        result = gp_minimize(
            lambda lmb: self._lasso_objective(lmb, cofactor_scaled, feature_target_scaled),
            search_space,
            n_calls=n_trials,
            random_state=42
        )

        best_lambda = result.x[0]
        best_beta = self._minimize_lasso(cofactor, feature_target, best_lambda)
        
        if return_coef:
            return best_beta
        else:
            self.coef_ = best_beta
            self.best_params_ = best_lambda


    def _minimize_lasso(self, cofactor, feature_target, lam, tol=1e-6, max_iter=1000) -> np.ndarray:
        p = feature_target.shape[0]
        beta = np.zeros(p)
        diag_cofactor = np.diag(cofactor)

        for _ in range(max_iter):
            beta_old = beta.copy()

            for j in range(p):
                r_j = feature_target[j] - (cofactor[j, :] @ beta - cofactor[j, j] * beta[j])
                raw_update = r_j / diag_cofactor[j]

                if j == 0:
                    beta[j] = raw_update
                else:
                    beta[j] = self._soft_thresholding(raw_update, lam / diag_cofactor[j])

            if np.linalg.norm(beta - beta_old, ord=np.inf) < tol:
                break

        return beta


    def _lasso_objective(self, lambda_, cofactor, feature_target):
        lam = lambda_[0]
        beta = self._minimize_lasso(cofactor, feature_target, lam)
        loss = 0.5 * beta @ (cofactor @ beta) - beta @ feature_target + lam * np.sum(np.abs(beta[1:]))  # no penalty on intercept
        loss = loss.item()

        return loss


    def _soft_thresholding(self, z, gamma):
        return np.sign(z) * max(abs(z) - gamma, 0)
    
    
    def fit(self, joint_cofactor: np.ndarray) -> None:
        pass


    def predict(self, feature_target: np.ndarray, return_coef: bool = False) -> None:
        pass