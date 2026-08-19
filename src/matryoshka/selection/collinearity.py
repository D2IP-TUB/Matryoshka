from collections import namedtuple

import numpy as np

from ..exceptions import EmptyAugmentation
from .base.model import FeatureSelectionModel
from .models import *


class CollinearityAnalysis:
    def __init__(self):
        pass


    def eliminate(self, model: FeatureSelectionModel, task: str, joint_tuples, **kwargs):
        match task:
            case 'regression':
                objects = joint_tuples._asdict()
                joint_cofactor_matrix = objects['joint_cofactor_matrix']
                joint_feature_target_vector = objects['joint_feature_target_vector']
                aug_feature_indices = objects['aug_feature_indices']
                new_features_count = objects['new_features_count']
                y_t_y = objects['y_t_y']
                joint_count = objects['joint_count']
                sum_y = objects['sum_y']
                model_instance = model(y_t_y, 0)
                checks = objects.get('aug_table_sketches', None), objects.get('base_table_sketch', None)
                if checks[0] is None or checks[1] is None:
                    objects['aug_table_sketches'], objects['base_table_sketch'] = None, None

                model_instance.fit(joint_cofactor_matrix)
                M1, M2 = model_instance.M1, model_instance.M2
                collinear_indices = self._collinearity_check(M1, M2, new_features_count)
                try:
                    if len(collinear_indices) == new_features_count:
                        raise EmptyAugmentation()
                except EmptyAugmentation as e:
                    raise RuntimeError(str(e))
                pos_indices = [i if i >= 0 else joint_cofactor_matrix.shape[1] + i for i in collinear_indices]
                all_indices = np.arange(joint_cofactor_matrix.shape[1])
                complement = np.setdiff1d(all_indices, pos_indices)

                joint_cofactor_matrix = joint_cofactor_matrix[np.ix_(complement, complement)]
                joint_feature_target_vector = joint_feature_target_vector[complement]
                aug_feature_indices = {i: aug_feature_indices[i] for i in aug_feature_indices if i not in pos_indices}
                new_features_count = len(aug_feature_indices)

                Result = namedtuple('JointTuple', ['joint_cofactor_matrix', 'joint_feature_target_vector', 'joint_count', 'y_t_y', 'new_features_count', 'aug_feature_indices', 'sum_y', 'aug_table_sketches', 'base_table_sketch'])

                return Result(joint_cofactor_matrix, joint_feature_target_vector, joint_count, y_t_y, new_features_count, aug_feature_indices, sum_y, objects['aug_table_sketches'], objects['base_table_sketch'])

            case 'classification':
                joint_cofactor_matrix = (
                    np.stack(
                        [
                            joint_tuples.class_dict[label]['joint_cofactor_matrix'] / joint_tuples.class_dict[label]['joint_count']
                            for label in joint_tuples.class_dict
                        ],
                        axis=2
                    )
                    .sum(axis=2)
                )
                joint_count = joint_tuples.class_dict[0]['joint_count']
                new_features_count = joint_tuples.new_features_count
                aug_feature_indices = joint_tuples.aug_feature_indices
                aug_table_sketches_per_class = joint_tuples.aug_table_sketches_per_class
                model_instance = model(joint_count, 0)

                model_instance.fit(joint_cofactor_matrix)
                M1, M2 = model_instance.M1, model_instance.M2
                collinear_indices = self._collinearity_check(M1, M2, new_features_count)
                pos_indices = [i if i >= 0 else joint_cofactor_matrix.shape[1] + i for i in collinear_indices]
                all_indices = np.arange(joint_cofactor_matrix.shape[1])
                complement = np.setdiff1d(all_indices, pos_indices)

                for label in joint_tuples.class_dict:
                    joint_cofactor_matrix = joint_tuples.class_dict[label]['joint_cofactor_matrix']
                    joint_tuples.class_dict[label]['joint_cofactor_matrix'] = joint_cofactor_matrix[np.ix_(complement, complement)]
                    features_mean = joint_tuples.class_dict[label]['features_mean']
                    joint_tuples.class_dict[label]['features_mean'] = features_mean[complement]

                aug_feature_indices = {i: aug_feature_indices[i] for i in aug_feature_indices if i not in pos_indices}
                new_features_count = len(aug_feature_indices)

                Result = namedtuple('JointTuple', ['class_dict', 'joint_count', 'new_features_count', 'aug_feature_indices', 'aug_table_sketches_per_class'])

                return Result(joint_tuples.class_dict, joint_count, new_features_count, aug_feature_indices, aug_table_sketches_per_class)


    def _collinearity_check(self, M1, M2, new_features_count, tol=1e-4):
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
