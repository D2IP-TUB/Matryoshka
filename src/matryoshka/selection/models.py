import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar

from ..exceptions import LinAlgError
from .base.model import FeatureSelectionModel


def conditional_fisher_score(
    candidate_class_means,
    candidate_class_covs,
    class_counts,
    current_class_means=None,
    current_class_covs=None,
    cross_class_covs=None,
    lambda_x: float = 1e-3,
    lambda_z: float = 1e-3,
):
    r"""Incremental conditional multiclass Fisher (Mahalanobis) feature score.

    Scores how much a candidate feature block ``Z`` (q features) improves
    geometric class separability *beyond* an existing feature set ``X``
    (p features), using only class-level summary statistics — class means,
    class covariances, and class counts. No access to raw rows is required.

    The criterion is the conditional generalization of the multiclass Fisher
    ratio ``trace(S_W^{-1} S_B)``: it scores ``Z`` after projecting out the
    part of its class-mean separation already explained by ``X`` (the partial
    Mahalanobis contribution). A candidate that merely duplicates information
    already in ``X`` therefore receives a low incremental score even if its
    standalone (raw) score is high.

    Parameters
    ----------
    candidate_class_means : (K, q) array
        Per-class means of the candidate features ``Z``.
    candidate_class_covs : (K, q, q) array
        Per-class covariance matrices of ``Z``.
    class_counts : (K,) array
        Per-class sample counts ``n_k``.
    current_class_means : (K, p) array or None
        Per-class means of the current features ``X``. ``None``/empty -> p=0
        (standalone scoring).
    current_class_covs : (K, p, p) array or None
        Per-class covariance matrices of ``X``.
    cross_class_covs : (K, q, p) array or None
        Per-class covariance blocks between ``Z`` and ``X``. If ``None`` the
        pooled cross-covariance ``S_ZX`` is assumed zero, which makes the
        incremental score an approximation (redundancy between ``X`` and ``Z``
        cannot be estimated).
    lambda_x, lambda_z : float
        Ridge regularization for the current and candidate-conditional
        covariances, respectively.

    Returns
    -------
    dict with keys ``incremental_score``, ``raw_score``, ``S_Z_given_X``,
    ``S_B_Z_given_X`` and ``diagnostics``.
    """
    cand_means = np.asarray(candidate_class_means, dtype=float)
    if cand_means.ndim == 1:
        cand_means = cand_means.reshape(cand_means.shape[0], 1)
    cand_covs = np.asarray(candidate_class_covs, dtype=float)
    counts = np.asarray(class_counts, dtype=float).reshape(-1)

    K, q = cand_means.shape
    total = counts.sum()
    pi = counts / total if total > 0 else np.full(K, 1.0 / K)

    Iq = np.eye(q)

    def _pooled(covs):                       # sum_k pi_k Sigma_k
        return np.einsum('k,kij->ij', pi, covs)

    def _between(d):                         # sum_k pi_k d_k d_k^T
        return np.einsum('k,ki,kj->ij', pi, d, d)

    # --- candidate (Z) global mean, centered shifts, pooled covariance ---
    mu_Z = np.einsum('k,kq->q', pi, cand_means)
    d_Z = cand_means - mu_Z                  # (K, q)
    S_ZZ = _pooled(cand_covs)                # (q, q)
    S_B_Z = _between(d_Z)                     # (q, q)

    # raw standalone Fisher score: how separable Z is by itself.
    raw_score = float(np.trace(np.linalg.solve(S_ZZ + lambda_z * Iq, S_B_Z)))

    # --- number of current features p ---
    cur_means = None if current_class_means is None else np.asarray(current_class_means, dtype=float)
    if cur_means is not None and cur_means.ndim == 1:
        cur_means = cur_means.reshape(cur_means.shape[0], 1)
    p = int(cur_means.shape[1]) if (cur_means is not None and cur_means.size > 0) else 0
    cross_provided = cross_class_covs is not None and p > 0
    cond_xx = None

    if p == 0:
        # No current features: the incremental score is the standalone score.
        incremental_score = raw_score
        S_Z_given_X = S_ZZ
        S_B_Z_given_X = S_B_Z
    else:
        cur_covs = np.asarray(current_class_covs, dtype=float)
        Ip = np.eye(p)
        mu_X = np.einsum('k,kp->p', pi, cur_means)
        d_X = cur_means - mu_X               # (K, p)
        S_XX = _pooled(cur_covs)             # (p, p)
        if cross_provided:
            S_ZX = _pooled(np.asarray(cross_class_covs, dtype=float))   # (q, p)
        else:
            S_ZX = np.zeros((q, p))
        S_XZ = S_ZX.T                        # (p, q)

        A = S_XX + lambda_x * Ip
        cond_xx = float(np.linalg.cond(A))
        # Part of each class-mean shift in Z not explained by X:
        # d_{Z|X}_k = d_Z_k - S_ZX A^{-1} d_X_k   (solve, never invert).
        d_Z_given_X = d_Z - (S_ZX @ np.linalg.solve(A, d_X.T)).T        # (K, q)
        # Conditional within-class covariance: S_ZZ - S_ZX A^{-1} S_XZ.
        S_Z_given_X = S_ZZ - S_ZX @ np.linalg.solve(A, S_XZ)            # (q, q)
        S_Z_given_X = 0.5 * (S_Z_given_X + S_Z_given_X.T)              # symmetrize
        S_B_Z_given_X = _between(d_Z_given_X)
        incremental_score = float(np.trace(
            np.linalg.solve(S_Z_given_X + lambda_z * Iq, S_B_Z_given_X)))

    cond_zgx = float(np.linalg.cond(S_Z_given_X + lambda_z * Iq))
    # A genuinely negative score (beyond numerical tolerance) is unexpected for
    # a PSD between-class scatter against a regularized SPD covariance.
    negative_warning = (incremental_score < -1e-10) or (raw_score < -1e-10)

    # Clip tiny negatives from round-off to exactly zero.
    if abs(raw_score) < 1e-10:
        raw_score = 0.0
    if abs(incremental_score) < 1e-10:
        incremental_score = 0.0

    diagnostics = {
        'K': int(K),
        'p': int(p),
        'q': int(q),
        'class_priors': pi.tolist(),
        'lambda_x': float(lambda_x),
        'lambda_z': float(lambda_z),
        'cross_covariance_provided': bool(cross_provided),
        'cond_S_XX_reg': cond_xx,
        'cond_S_Z_given_X_reg': cond_zgx,
        'negative_score_warning': bool(negative_warning),
    }
    return {
        'incremental_score': incremental_score,
        'raw_score': raw_score,
        'S_Z_given_X': S_Z_given_X,
        'S_B_Z_given_X': S_B_Z_given_X,
        'diagnostics': diagnostics,
    }


def conditional_regression_score(
    current_xtx,
    current_xty,
    candidate_ztz,
    candidate_zty,
    yty,
    n,
    cross_ztx=None,
    p_current=None,
    q_candidate=None,
    lambda_x: float = 1e-3,
    lambda_z: float = 1e-3,
):
    r"""Incremental conditional OLS-proxy feature score for regression.

    Scores how much a candidate feature block ``Z`` (q features) reduces the
    residual sum of squares of a ridge-OLS proxy *beyond* an existing feature
    set ``X`` (p features), using only second-moment summary statistics — Gram
    matrices and cross-products. No access to raw rows ``X``/``y`` is required.
    The downstream model may be non-linear, so this is a proxy for useful
    predictive signal rather than the downstream loss itself.

    The criterion is the conditional incremental RSS reduction of ``Z`` given
    ``X``. Writing ``A = X^T X + lambda_x I``, the part of ``Z`` orthogonal to
    ``X`` has conditional cross-product ``c = Z^T y - Z^T X A^{-1} X^T y`` and
    conditional Gram ``G = Z^T Z - Z^T X A^{-1} X^T Z`` (a regularized Schur
    complement). The RSS removed by adding ``Z`` is
    ``delta_rss = c^T (G + lambda_z I)^{-1} c``. A candidate that merely
    duplicates information already in ``X`` therefore contributes little even
    if its standalone reduction is large.

    Selection uses GCV, a validation-error surrogate that penalizes model
    complexity: ``gcv = (rss / n) / (1 - k / n)^2`` with ``k`` the parameter
    count. ``gcv_improvement = gcv_current - gcv_new > 0`` indicates the RSS
    reduction justifies the extra ``q`` features. Raw training RSS is not used
    directly because it monotonically over-selects; BIC is reported only as a
    diagnostic since it is too conservative when OLS is a proxy for a non-linear
    downstream model.

    Parameters
    ----------
    current_xtx : (p, p) array or None
        ``X^T X`` for the current set. ``None``/empty -> ``p = 0`` (standalone).
    current_xty : (p,) array or None
        ``X^T y`` for the current set.
    candidate_ztz : (q, q) array
        ``Z^T Z`` for the candidate block.
    candidate_zty : (q,) array
        ``Z^T y`` for the candidate block.
    yty : float
        ``y^T y``.
    n : int
        Number of samples.
    cross_ztx : (q, p) array or None
        ``Z^T X``. If ``None`` the cross-covariance is assumed zero and the
        result is flagged ``approximate`` in diagnostics (redundancy between
        ``X`` and ``Z`` cannot be estimated).
    p_current, q_candidate : int or None
        Optional, validated against the matrix shapes; the shapes are
        authoritative and any mismatch is recorded in diagnostics.
    lambda_x, lambda_z : float
        Ridge regularization for ``A = X^T X + lambda_x I`` and for the
        conditional candidate Gram ``G + lambda_z I``.

    Returns
    -------
    dict with keys ``delta_rss``, ``rss_current``, ``rss_new``, ``gcv_current``,
    ``gcv_new``, ``gcv_improvement``, ``aic_*``, ``bic_*`` and ``diagnostics``.
    GCV is the default ranking criterion; AIC and BIC are diagnostics only.
    """
    ZtZ = np.asarray(candidate_ztz, dtype=float)
    if ZtZ.ndim == 0:
        ZtZ = ZtZ.reshape(1, 1)
    Zty = np.asarray(candidate_zty, dtype=float).reshape(-1)
    q = int(ZtZ.shape[0])
    yty = float(yty)
    n = int(n)
    eps = 1e-12

    if current_xtx is None or np.asarray(current_xtx).size == 0:
        p = 0
        XtX = np.zeros((0, 0))
        Xty = np.zeros(0)
    else:
        XtX = np.asarray(current_xtx, dtype=float)
        p = int(XtX.shape[0])
        Xty = np.asarray(current_xty, dtype=float).reshape(-1)

    shape_mismatch = (
        (p_current is not None and int(p_current) != p)
        or (q_candidate is not None and int(q_candidate) != q)
    )

    cross_provided = (cross_ztx is not None) and (p > 0)
    Iq = np.eye(q)
    cond_xx = None

    # Conditional candidate cross-product c and Gram G given X (solve, no inverse)
    if p == 0:
        G = ZtZ
        c = Zty
        rss_current = yty
    else:
        A = XtX + lambda_x * np.eye(p)
        cond_xx = float(np.linalg.cond(A))
        if cross_provided:
            ZtX = np.asarray(cross_ztx, dtype=float).reshape(q, p)
        else:
            ZtX = np.zeros((q, p))
        A_inv_Xty = np.linalg.solve(A, Xty)                 # A^{-1} X^T y
        c = Zty - ZtX @ A_inv_Xty
        G = ZtZ - ZtX @ np.linalg.solve(A, ZtX.T)
        G = 0.5 * (G + G.T)                                 # symmetrize
        rss_current = yty - float(Xty @ A_inv_Xty)

    G_reg = G + lambda_z * Iq
    cond_zgx = float(np.linalg.cond(G_reg))
    delta_rss = float(c @ np.linalg.solve(G_reg, c))

    # Tiny round-off negatives -> 0; genuinely negative values are kept + warned.
    negative_delta_warning = delta_rss < -1e-10
    if abs(delta_rss) < 1e-10:
        delta_rss = 0.0

    rss_new = rss_current - delta_rss
    rss_new_nonpos_warning = rss_new <= 0
    if rss_new <= 0:                                        # numerical guard only
        rss_new = eps
    rss_current_pos = rss_current if rss_current > eps else eps

    def _gcv(rss, k):
        denom = 1.0 - (k / n)
        return float('inf') if denom <= 0 else (rss / n) / (denom * denom)

    saturation_warning = (p + q) >= n
    gcv_current = _gcv(rss_current_pos, p)
    gcv_new = _gcv(rss_new, p + q)
    gcv_improvement = gcv_current - gcv_new

    log_cur = np.log(rss_current_pos / n)
    log_new = np.log(rss_new / n)
    log_n = np.log(n)
    aic_current = n * log_cur + 2 * p
    aic_new = n * log_new + 2 * (p + q)
    aic_improvement = aic_current - aic_new
    bic_current = n * log_cur + p * log_n
    bic_new = n * log_new + (p + q) * log_n
    bic_improvement = bic_current - bic_new

    warns = []
    if negative_delta_warning:
        warns.append('delta_rss substantially negative')
    if rss_new_nonpos_warning:
        warns.append('rss_new <= 0 (clipped to eps)')
    if saturation_warning:
        warns.append('p_current + q_candidate >= n')
    if shape_mismatch:
        warns.append('p_current/q_candidate disagree with matrix shapes')

    diagnostics = {
        'n': n,
        'p_current': p,
        'q_candidate': q,
        'lambda_x': float(lambda_x),
        'lambda_z': float(lambda_z),
        'cross_ztx_provided': bool(cross_provided),
        'approximate': bool(p > 0 and not cross_provided),
        'cond_A_reg': cond_xx,
        'cond_G_Z_given_X_reg': cond_zgx,
        'delta_rss': float(delta_rss),
        'rss_current': float(rss_current),
        'rss_new': float(rss_new),
        'gcv_current': float(gcv_current),
        'gcv_new': float(gcv_new),
        'gcv_improvement': float(gcv_improvement),
        'aic_improvement': float(aic_improvement),
        'bic_improvement': float(bic_improvement),
        'negative_delta_rss_warning': bool(negative_delta_warning),
        'rss_new_nonpositive_warning': bool(rss_new_nonpos_warning),
        'saturation_warning': bool(saturation_warning),
        'shape_mismatch_warning': bool(shape_mismatch),
        'warnings': warns,
    }
    return {
        'delta_rss': float(delta_rss),
        'rss_current': float(rss_current),
        'rss_new': float(rss_new),
        'gcv_current': float(gcv_current),
        'gcv_new': float(gcv_new),
        'gcv_improvement': float(gcv_improvement),
        'aic_current': float(aic_current),
        'aic_new': float(aic_new),
        'aic_improvement': float(aic_improvement),
        'bic_current': float(bic_current),
        'bic_new': float(bic_new),
        'bic_improvement': float(bic_improvement),
        'diagnostics': diagnostics,
    }


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
    # Metric → pairwise distance routine. The metric scoring branch in
    # `FeatureSelectionModel.score` and the distance kernel here are the
    # two halves of the same choice; keeping the mapping in one place keeps
    # them in sync.
    _METRIC_TO_DISTANCE = {
        'average_mahalanobis': 'mahalanobis',
        'average_bhattacharyya': 'bhattacharyya',
        'robust_moment': 'robust_moment',
        'conditional_mahalanobis': 'conditional_mahalanobis',
    }
    # Ridge regularization for the incremental conditional Fisher metric.
    _COND_LAMBDA_X = 1e-3
    _COND_LAMBDA_Z = 1e-3

    def __init__(self, y_t_y, n_iter, M1=None, M2=None, d=None, metric: str = 'average_mahalanobis') -> None:
        skipped_feature = self._is_skipped(y_t_y)
        if skipped_feature:
            raise LinAlgError('Feature is skipped.')
        self.n_iter = n_iter
        self.M1 = M1
        self.M2 = M2
        self.d = d
        if metric not in self._METRIC_TO_DISTANCE:
            raise ValueError(
                f"Unknown metric '{metric}' for ClassificationCholesky. "
                f"Expected one of {list(self._METRIC_TO_DISTANCE)}."
            )
        self.metric = metric
        self.distance = self._METRIC_TO_DISTANCE[metric]


    def fit(self, joint_cofactor: np.ndarray, **kwargs) -> None:
        '''
        Implemented for the sake of sklearn-like API development.
        '''
        try:
            self.M1 = cholesky(joint_cofactor, lower=True)
        except np.linalg.LinAlgError:
            self.M1, self.M2 = np.linalg.qr(joint_cofactor)


    def predict(self, feature_target, return_coef: bool = False):
        pass


    def fit_predict(self, class_covariances, class_means, return_coef: bool = False, **kwargs) -> np.ndarray:
        class_counts = kwargs.get('class_counts', None)

        if self.distance == 'conditional_mahalanobis':
            # Incremental conditional Fisher: score the candidate block (the
            # last ``new_features_count`` columns of the joint) given the
            # current feature set (the preceding columns).
            d = self._conditional_mahalanobis(
                class_covariances, class_means, class_counts,
                kwargs.get('new_features_count'),
            )
            self.coef_ = d
            return d if return_coef else None

        n_labels = class_means.shape[0]
        i_indices, j_indices = np.triu_indices(n_labels, k=1)
        mean_diffs = class_means[i_indices] - class_means[j_indices]

        if self.distance == 'mahalanobis':
            d = self._mahalanobis(class_covariances, mean_diffs, class_counts)
        elif self.distance == 'robust_moment':
            d = self._robust_moment(class_covariances, class_means, class_counts=class_counts)
        else:
            d = self._bhattacharyya(class_covariances, mean_diffs, i_indices, j_indices)

        self.coef_ = d

        if return_coef:
            return d


    def _mahalanobis(self, class_covariances, mean_diffs, class_counts) -> np.ndarray:
        # Globally-pooled within-class covariance (standard LDA)
        if class_counts is not None:
            weights = class_counts - 1                                      # (n_k - 1)
            cov_pooled = np.einsum('i,ijk->jk', weights, class_covariances) / weights.sum()
        else:
            cov_pooled = np.mean(class_covariances, axis=0)

        # Ridge-regularize the pooled within-class covariance so the Cholesky
        # factorization is well-defined even when features are collinear or a
        # class is rank deficient (common on dense, synthetic lakes such as the
        # AutoFeat benchmark, where the un-regularized factorization raises
        # LinAlgError for every candidate and forward selection then picks no
        # features). The plain factorization is attempted first, so well-
        # conditioned matrices are unchanged; jitter is only added on failure,
        # scaled to the covariance magnitude and escalated until the matrix is
        # positive definite.
        p = cov_pooled.shape[0]
        eye_p = np.eye(p)
        tr = float(np.trace(cov_pooled))
        base = (tr / p) if (p > 0 and tr > 0) else 1.0
        jitter = 0.0
        L = None
        for _ in range(9):
            try:
                L = np.linalg.cholesky(cov_pooled + jitter * eye_p)
                break
            except np.linalg.LinAlgError:
                jitter = 1e-10 * base if jitter == 0.0 else jitter * 10.0
        if L is None:
            raise LinAlgError('Pooled within-class covariance is not positive definite.')
        # solve  L z_j = mean_diff_j  for every pair j
        z = np.linalg.solve(L, mean_diffs.T)                               # (d, n_pairs)

        # squared Mahalanobis distances (n_pairs,)
        return np.sum(z ** 2, axis=0)


    def _conditional_mahalanobis(self, class_covariances, class_means,
                                 class_counts, new_features_count) -> np.ndarray:
        """Incremental conditional Fisher score for the candidate block.

        ``class_covariances`` (K, P, P) and ``class_means`` (K, P) are the
        per-class statistics of the *joint* feature vector ``[X, Z]``, where the
        candidate block ``Z`` is the trailing ``new_features_count`` columns and
        the current set ``X`` is everything before it. Returns a single-element
        array holding the running cumulative score ``D^2(X) + dD^2(Z|X)`` so the
        forward selection's relative-improvement stopping rule (improvement =
        marginal / cumulative) continues to apply: the score difference between
        consecutive iterations is exactly the candidate's non-redundant
        contribution ``dD^2(Z|X)``."""
        class_covariances = np.asarray(class_covariances, dtype=float)
        class_means = np.asarray(class_means, dtype=float)
        P = class_means.shape[1]
        q = int(new_features_count) if new_features_count else P
        q = max(1, min(q, P))
        p = P - q

        z_means = class_means[:, p:]
        z_covs = class_covariances[:, p:, p:]
        if p > 0:
            x_means = class_means[:, :p]
            x_covs = class_covariances[:, :p, :p]
            cross = class_covariances[:, p:, :p]                 # (K, q, p)
        else:
            x_means = x_covs = cross = None

        res = conditional_fisher_score(
            z_means, z_covs, class_counts,
            current_class_means=x_means, current_class_covs=x_covs,
            cross_class_covs=cross,
            lambda_x=self._COND_LAMBDA_X, lambda_z=self._COND_LAMBDA_Z,
        )
        # D^2(X): standalone Fisher of the current set (constant across
        # candidates at a given iteration); 0 when there is no current set.
        if p > 0:
            d2_x = conditional_fisher_score(
                x_means, x_covs, class_counts, lambda_z=self._COND_LAMBDA_X,
            )['raw_score']
        else:
            d2_x = 0.0
        return np.array([[d2_x + res['incremental_score']]])


    def _robust_moment(
        self,
        class_covariances: np.ndarray,
        class_means: np.ndarray,
        class_counts: np.ndarray | None = None,
        lam: float = 0.1,
        delta: float = 1e-12,
        degenerate_tol: float = 1e-12,
        aggregation: str = 'weighted',
    ) -> np.ndarray:
        r"""
        Multiclass moment-based conservative accuracy lower bound.

        For each class pair (a,b), computes a binary Cantelli-style error bound,
        minimizes it over threshold t in [m_a, m_b], converts to pair score
        (1 - risk), and aggregates pairwise risks into one multiclass score.

        Pair weights follow normalized pi_a * pi_b when ``aggregation='weighted'``.
        Optionally, ``aggregation='worst'`` returns the worst-pair score.
        """
        n_classes = class_means.shape[0]
        if n_classes < 2:
            return np.array([0.0])
        if aggregation not in {'weighted', 'worst'}:
            raise ValueError("aggregation must be one of {'weighted', 'worst'}")

        # Normalize priors from counts when available.
        if class_counts is not None and len(class_counts) == n_classes and np.sum(class_counts) > 0:
            priors = np.asarray(class_counts, dtype=float)
            priors = priors / np.sum(priors)
        else:
            priors = np.full(n_classes, 1.0 / n_classes, dtype=float)

        p = class_means.shape[1]
        eye_p = np.eye(p)

        pair_scores: list[float] = []
        pair_risks: list[float] = []
        pair_weights: list[float] = []
        pair_details: list[dict] = []

        for a in range(n_classes):
            for b in range(a + 1, n_classes):
                sigma_a = class_covariances[a]
                sigma_b = class_covariances[b]
                mu_a = class_means[a]
                mu_b = class_means[b]

                pi_a = float(priors[a])
                pi_b = float(priors[b])
                pi_pair_sum = pi_a + pi_b
                if pi_pair_sum <= 0:
                    pair_prior_a = 0.5
                    pair_prior_b = 0.5
                else:
                    pair_prior_a = pi_a / pi_pair_sum
                    pair_prior_b = pi_b / pi_pair_sum

                tr_sum = float(np.trace(sigma_a) + np.trace(sigma_b))
                eps = 1e-6 * tr_sum / (2 * p) if (p > 0 and tr_sum > 0) else 1e-6
                sigma_a_reg = (1.0 - lam) * sigma_a + lam * np.diag(np.diag(sigma_a)) + eps * eye_p
                sigma_b_reg = (1.0 - lam) * sigma_b + lam * np.diag(np.diag(sigma_b)) + eps * eye_p

                delta_ab = mu_b - mu_a
                sigmaR_ab = sigma_a_reg + sigma_b_reg
                try:
                    L = np.linalg.cholesky(sigmaR_ab)
                    z = solve_triangular(L, delta_ab, lower=True)
                    w_ab = solve_triangular(L.T, z, lower=False)
                except np.linalg.LinAlgError:
                    w_ab = np.linalg.lstsq(sigmaR_ab, delta_ab, rcond=None)[0]

                m_a = float(w_ab @ mu_a)
                m_b = float(w_ab @ mu_b)
                if m_a > m_b:
                    w_ab = -w_ab
                    m_a, m_b = m_b, m_a
                gap_ab = m_b - m_a

                s_a_sq = max(float(w_ab @ sigma_a_reg @ w_ab), delta)
                s_b_sq = max(float(w_ab @ sigma_b_reg @ w_ab), delta)

                if gap_ab <= degenerate_tol:
                    pair_score = max(pair_prior_a, pair_prior_b)
                    pair_risk = 1.0 - pair_score
                    t_star = float(0.5 * (m_a + m_b))
                else:
                    def pair_risk_bound(t: float) -> float:
                        err_a = s_a_sq / (s_a_sq + (t - m_a) ** 2)
                        err_b = s_b_sq / (s_b_sq + (m_b - t) ** 2)
                        return pair_prior_a * err_a + pair_prior_b * err_b

                    try:
                        opt = minimize_scalar(pair_risk_bound, bounds=(m_a, m_b), method='bounded')
                        if opt.success:
                            t_star = float(opt.x)
                            pair_risk = float(opt.fun)
                        else:
                            raise RuntimeError('bounded minimization failed')
                    except Exception:
                        grid = np.linspace(m_a, m_b, num=257)
                        risks = np.array([pair_risk_bound(float(t)) for t in grid], dtype=float)
                        best_i = int(np.argmin(risks))
                        t_star = float(grid[best_i])
                        pair_risk = float(risks[best_i])

                    pair_risk = float(np.clip(pair_risk, 0.0, 1.0))
                    pair_score = 1.0 - pair_risk

                pair_weight = float(priors[a] * priors[b])
                pair_scores.append(float(np.clip(pair_score, 0.0, 1.0)))
                pair_risks.append(float(np.clip(pair_risk, 0.0, 1.0)))
                pair_weights.append(pair_weight)
                pair_details.append(
                    {
                        'class_a': int(a),
                        'class_b': int(b),
                        'pair_prior_a': float(pair_prior_a),
                        'pair_prior_b': float(pair_prior_b),
                        'weight_raw': float(pair_weight),
                        'score': float(np.clip(pair_score, 0.0, 1.0)),
                        'risk_bound': float(np.clip(pair_risk, 0.0, 1.0)),
                        'threshold': float(t_star),
                        'gap': float(gap_ab),
                        'proj_var_a': float(s_a_sq),
                        'proj_var_b': float(s_b_sq),
                        'projection_direction': w_ab.tolist(),
                    }
                )

        if len(pair_scores) == 0:
            return np.array([0.0])

        w = np.asarray(pair_weights, dtype=float)
        if np.sum(w) <= 0:
            w = np.full_like(w, 1.0 / len(w), dtype=float)
        else:
            w = w / np.sum(w)

        pair_scores_np = np.asarray(pair_scores, dtype=float)
        pair_risks_np = np.asarray(pair_risks, dtype=float)

        weighted_avg_pair_score = float(np.sum(w * pair_scores_np))
        overall_risk_weighted = float(np.sum(w * pair_risks_np))
        best_pair_score = float(np.max(pair_scores_np))
        worst_pair_score = float(np.min(pair_scores_np))

        if aggregation == 'worst':
            overall_score = worst_pair_score
            overall_risk = 1.0 - overall_score
        else:
            overall_score = 1.0 - overall_risk_weighted
            overall_risk = overall_risk_weighted

        # Optional diagnostics for analysis/reporting.
        self.robust_moment_details_ = {
            'overall_score': float(np.clip(overall_score, 0.0, 1.0)),
            'overall_risk_bound': float(np.clip(overall_risk, 0.0, 1.0)),
            'weighted_average_pair_score': float(np.clip(weighted_avg_pair_score, 0.0, 1.0)),
            'worst_pair_score': float(np.clip(worst_pair_score, 0.0, 1.0)),
            'best_pair_score': float(np.clip(best_pair_score, 0.0, 1.0)),
            'pair_weights': w.tolist(),
            'pair_scores': pair_scores_np.tolist(),
            'pair_risk_bounds': pair_risks_np.tolist(),
            'pairs': pair_details,
        }

        return np.array([float(np.clip(overall_score, 0.0, 1.0))])


    def _bhattacharyya(
        self,
        class_covariances: np.ndarray,
        mean_diffs: np.ndarray,
        i_indices: np.ndarray,
        j_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Pairwise Bhattacharyya distance between class-conditional Gaussians:

            D_B(p, q) = 1/8 (mu_p - mu_q)^T Sigma^{-1} (mu_p - mu_q)
                      + 1/2 log( det(Sigma) / sqrt(det(Sigma_p) det(Sigma_q)) )

        with  Sigma = (Sigma_p + Sigma_q) / 2.  Per-class log-determinants
        are computed once and reused; the quadratic form is evaluated via
        a per-pair Cholesky solve to avoid explicit inversion.
        """
        # Per-class log-determinants (stable, sign-aware)
        signs, log_dets = np.linalg.slogdet(class_covariances)
        if np.any(signs <= 0):
            raise LinAlgError('Class covariance is not positive definite.')

        cov_p = class_covariances[i_indices]                                # (n_pairs, d, d)
        cov_q = class_covariances[j_indices]
        cov_avg = 0.5 * (cov_p + cov_q)

        L = np.linalg.cholesky(cov_avg)                                     # (n_pairs, d, d)
        # Solve  L z = mean_diff  for every pair  (z: (n_pairs, d))
        z = np.linalg.solve(L, mean_diffs[..., None])[..., 0]
        quad = np.sum(z ** 2, axis=1)                                       # (n_pairs,)

        # log det(cov_avg) via Cholesky diagonal
        log_det_avg = 2.0 * np.sum(np.log(np.diagonal(L, axis1=1, axis2=2)), axis=1)
        log_det_pq = 0.5 * (log_dets[i_indices] + log_dets[j_indices])

        return 0.125 * quad + 0.5 * (log_det_avg - log_det_pq)


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
        # scikit-optimize is an optional dependency of the L1 proxy only.
        from skopt import gp_minimize
        from skopt.space import Real

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
