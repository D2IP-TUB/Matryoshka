import ray
import numpy as np
from augmentation.feature_selection.sketch_processing import SketchProcessor
from collections import namedtuple
from kneed import KneeLocator
from ..models import *
from ..base.model import FeatureSelectionModel
from ...utils.exceptions import LinAlgError


@ray.remote(num_cpus=0)
class IncrementalSelection:
    def __init__(self, metric: str):
        self.metric = metric


    def run(self, model: FeatureSelectionModel, task: str, joint_tuples, **kwargs):
        sketch_proc = SketchProcessor.remote()
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
                aug_table_sketches = objects['aug_table_sketches']
                model_instance = model(y_t_y, 0)

                n_features = joint_cofactor_matrix.shape[1]
                n_init_features = n_features - new_features_count
                init_features_idx = np.arange(n_init_features)
                new_features_idx = np.setdiff1d(np.arange(n_features), init_features_idx)
                aug_feature_indices = {k: v for k, v in zip(new_features_idx, aug_feature_indices.values())}

                coefs = model_instance.fit_predict(joint_cofactor_matrix, joint_feature_target_vector, return_coef=True)
                keys_to_keep = self._contribution_score(coefs, aug_feature_indices, new_features_count)
                keys_to_keep_all = np.concatenate([init_features_idx, keys_to_keep])

                joint_cofactor_matrix = joint_cofactor_matrix[np.ix_(keys_to_keep_all, keys_to_keep_all)]
                joint_feature_target_vector = joint_feature_target_vector[keys_to_keep_all]
                score = self._evaluate_features(
                    **{
                        'model': model_instance,
                        'task': task,
                        'joint_cofactor': joint_cofactor_matrix,
                        'feature_target': joint_feature_target_vector,
                        'joint_count': joint_count,
                        'sum_y': sum_y
                    }
                )
                augmentation_plan = [aug_feature_indices[i] for i in keys_to_keep]

                splits = ray.get(
                    [
                        sketch_proc._prepare_split_inputs.remote(aug_sketch)
                        for aug_sketch in aug_table_sketches
                    ]
                )
                split_sums, split_diags, split_feature_indices = zip(*splits)

                aug_table_sketches = []
                for l in zip(split_sums, split_diags, split_feature_indices):
                    aug_table_sketches.extend(
                        ray.get(
                            [
                                sketch_proc._split_aug_table_sketch.remote(sums, diags, feature_indices)
                                for sums, diags, feature_indices in zip(*l) if feature_indices in augmentation_plan
                            ]
                        )
                    )
                aug_sums = np.hstack([aug_sketch.sum_ for aug_sketch in aug_table_sketches])

                Result = namedtuple('JointTuple', ['score', 'augmentation_plan', 'aug_sums'])

            case 'classification':
                class_dict = joint_tuples.class_dict
                new_features_count = joint_tuples.new_features_count
                aug_feature_indices = joint_tuples.aug_feature_indices
                aug_table_sketches_per_class = joint_tuples.aug_table_sketches_per_class
                class_covariances = []
                class_means = []
                for label, _ in class_dict.items():
                    class_covariances.append(class_dict[label]['joint_cofactor_matrix'] / class_dict[label]['joint_count'])
                    class_means.append(class_dict[label]['features_mean'])
                class_covariances = np.transpose(
                    np.stack(class_covariances, axis=2),
                    (2, 0, 1)
                )
                class_means = np.transpose(
                    np.stack(class_means, axis=1),
                    (1, 0)
                )
                score = self._evaluate_features(
                    **{
                        'model': model_instance,
                        'task': task,
                        'class_covariances': class_covariances,
                        'class_means': class_means
                    }
                )
                augmentation_plan = list(aug_feature_indices.values())
                
                for l in aug_table_sketches_per_class:
                    dic = {}
                    for sketch in aug_table_sketches_per_class[l]:
                        dic[sketch.table_feature_index] = sketch
                    aug_table_sketches_per_class[l] = dic
                table_index = '_'.join(augmentation_plan[0].split('_')[:-1]+['0'])
                feature_indices = [int(s.split('_')[-1]) for s in augmentation_plan]
                aug_sums = []
                for l in aug_table_sketches_per_class:
                    aug_sum = aug_table_sketches_per_class[l][table_index].sum_[:, feature_indices]
                    aug_sums.append(aug_sum)
                aug_sums = np.vstack(aug_sums)

                Result = namedtuple('JointTuple', ['score', 'augmentation_plan', 'aug_sums'])

        return Result(score, augmentation_plan, aug_sums)


    def _contribution_score(self, coefs, aug_feature_indices, new_features_count):
        new_features_first_idx = coefs.shape[0] - new_features_count
        abs_coefs = np.abs(coefs[new_features_first_idx:])
        importance_sum = abs_coefs / abs_coefs.sum()
        importance_dict = {i: c.item() for i, c in zip(list(aug_feature_indices.keys()), importance_sum)}
        importance_dict = dict(sorted(importance_dict.items(), key=lambda item: item[1], reverse=True))
        elbow_idx = self._find_elbow(importance_dict)
        keys_to_keep = list(importance_dict.keys())[:elbow_idx+1]

        return keys_to_keep


    def _find_elbow(self, importance_dict: dict[int, float]) -> int:
        x = np.array(list(importance_dict.keys()))
        y = np.array(list(importance_dict.values()))
        kneedle = KneeLocator(x, y, curve='convex', direction='decreasing', interp_method='interp1d', S=1)
        elbow_idx = kneedle.knee
        if elbow_idx is None:
            elbow_idx = len(importance_dict) - 1

        return elbow_idx
    

    def _evaluate_features(self, model: FeatureSelectionModel, task: str, **kwargs):
        try:
            match task:
                case 'regression':
                    model.fit(**kwargs)
                    model.predict(**kwargs)
                    score = model.score(self.metric, **kwargs).item()
                case 'classification':
                    model.fit_predict(return_coef=False, **kwargs)
                    score = model.score(self.metric, **kwargs).item()
        except LinAlgError:
            score = None
        return score