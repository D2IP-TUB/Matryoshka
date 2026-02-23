import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../augmentation/')))

import numpy as np
from augmentation.feature_selection.base.model import FeatureSelectionModel
from scipy.linalg import lu, solve_triangular, pinv


class TrueSolver(FeatureSelectionModel):
    def __init__(self, model, y_t_y, n_iter, M1=None, M2=None):
        self.model = model
        self.y_t_y = y_t_y
        self.n_iter = n_iter
        self.M1 = M1
        self.M2 = M2
        self.coef_ = None
        self.lam = 1e-10


    def fit(self, X):
        match self.model:
            case 'FactorizedLeastSquares':
                Q, R = np.linalg.qr(X)
                self.M1 = Q
                self.M2 = R
            case 'FactorizedLeastSquaresIncremental':
                Q, R = np.linalg.qr(X)
                self.M1 = Q
                self.M2 = R
            case 'CholeskyIncremental':
                L = np.linalg.cholesky(X @ X.T)
                self.M1 = L
            case 'LUIncremental':
                _, L, U = lu(X, permute_l=True)
                self.M1 = L
                self.M2 = U
            case 'ScipyPseudoInverse':
                cofactor_matrix = X.T @ X
                identity_matrix = np.eye(cofactor_matrix.shape[1])
                self.M1 = cofactor_matrix + self.lam * identity_matrix
            case _:
                raise ValueError('Unknown model type')


    def predict(self, y):
        match self.model:
            case 'FactorizedLeastSquares':
                self.coef_ = np.linalg.solve(self.M2, self.M1.T @ y)
            case 'FactorizedLeastSquaresIncremental':
                self.coef_ = np.linalg.solve(self.M2, self.M1.T @ y)
            case 'CholeskyIncremental':
                z = solve_triangular(self.M1, y, lower=True)
                self.coef_ = solve_triangular(self.M1.T, z, lower=False)
            case 'LUIncremental':
                z = solve_triangular(self.M1, y, lower=True)
                self.coef_ = solve_triangular(self.M2, z, lower=False)
            case 'ScipyPseudoInverse':
                self.coef_ = pinv(self.M1) @ y
            case _:
                raise ValueError('Unknown model type')