from collections import namedtuple
from copy import deepcopy
import ray
from abc import ABC, abstractmethod

from ..models import *
from ..base.model import FeatureSelectionModel
from ..sketch_processing import SketchProcessor
from ...utils.exceptions import LinAlgError


class RemoteMixin:
    def call_as_task(self, method_name, *args, **kwargs):
        method = getattr(self, method_name)

        # create a Ray task from the method
        @ray.remote
        def task_wrapper():
            return method(*args, **kwargs)
        
        return task_wrapper.remote()

    def call_multiple_tasks(self, method_calls):
        futures = []
        for method_name, args, kwargs in method_calls:
            future = self.call_as_task(method_name, *args, **kwargs)
            futures.append(future)

        return futures


class GreedyAlgo(ABC):
    @abstractmethod
    def run(self, model: str, task: str, joint_tuples, tol: float):
        pass

    def _first_iteration(self, model: FeatureSelectionModel, metric: str, **kwargs):
        model.fit_predict(**kwargs)
        return model.score(metric, **kwargs).item()

    def _evaluate_feature(self, model: FeatureSelectionModel, feature_idx: int, metric: str, **kwargs) -> dict[int, float]:
        try:
            model.fit_predict(**kwargs)
            score = model.score(metric, **kwargs)
        except (LinAlgError, np.linalg.LinAlgError):
            score = np.array([[float('-inf')]])
            sketch_proc = kwargs.get('sketch_proc', None)
            if sketch_proc is None:
                sketch_proc = kwargs['sketch_proc_per_class']
                for label in sketch_proc:
                    sketch_proc[label].remove_from_features_pool.remote(feature_idx)
            else:
                sketch_proc.remove_from_features_pool.remote(feature_idx)

        return {feature_idx: score}

    def _stopping_criteria(self, current_score: float, previous_score: float | None, tol: float, model_size: int, task: str = 'selection') -> bool:
        if current_score is None:
            return False
        denom = max(abs(previous_score), 1e-12)
        improvement = (current_score - previous_score) / denom

        # Decaying tolerance
        tol_t = tol / (model_size + 1) ** 0.5
        if task == 'selection':
            stopping = improvement < tol_t
        elif task == 'elimination':
            stopping = improvement > tol_t
        return stopping


@ray.remote
class ForwardSelection(GreedyAlgo, RemoteMixin):
    def __init__(self, metric: str):
        self.metric = metric


    def run(self, model: FeatureSelectionModel, task: str, joint_tuples, tol: float, **kwargs):
        sketch_proc = SketchProcessor.remote(conninfo=None, feature_selection_table_name=None)
        match task:
            case 'regression':
                base_table_sketch = joint_tuples.base_table_sketch
                aug_table_sketches = joint_tuples.aug_table_sketches
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
                                for sums, diags, feature_indices in zip(*l)
                            ]
                        )
                    )
                
                # do not use the output, just populate the cache in `sketch_proc` object
                _ = ray.get(
                    [
                        sketch_proc._join_aug_table.remote(
                            aug_sketch.table_feature_index,
                            base_table_sketch,
                            aug_sketch,
                            task
                        ) for aug_sketch in aug_table_sketches
                    ]
                )
                n_iter = 0
                remaining_features = ray.get(sketch_proc.retrieve.remote('cache')).keys()
                task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc': sketch_proc})
                futures = self.call_multiple_tasks(task_calls)
                results = ray.get(futures)
                feature_scores = {}
                for result_dict in results:
                    feature_scores.update(result_dict)
                best_feature = max(feature_scores, key=feature_scores.get)
                base_score = feature_scores[best_feature].item()
                if base_score == float('-inf'):
                    augmentation_plan = []
                    aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
                    Result = namedtuple('JointTuple', 'aug_feature_indices')
                    return Result(aug_feature_indices)
                augmentation_plan = [best_feature]

                best_feature_cache = ray.get(sketch_proc.retrieve.remote('cache'))
                best_feature_cache = best_feature_cache[best_feature]
                sketch_proc.update_with_best_feature.remote(
                    best_feature_cache['new_feature'],
                    best_feature_cache['joint_cofactor_matrix'],
                    best_feature_cache['joint_feature_target_vector'],
                    best_feature_cache['joint_count'],
                    best_feature_cache['y_t_y'],
                    best_feature_cache['joint_sum_vec']
                )
                sketch_proc.remove_from_features_pool.remote(best_feature)

                new_features_count = len(ray.get(sketch_proc.retrieve.remote('cache')))
                early_stopping = self._stopping_criteria(None, base_score, tol, new_features_count)
                while (not early_stopping) and (new_features_count > 0):
                    n_iter += 1
                    remaining_features = ray.get(sketch_proc.retrieve.remote('cache')).keys()
                    _ = ray.get(
                        [
                            sketch_proc._update_from_cache.remote(
                                table_feature_index, task
                            ) for table_feature_index in remaining_features
                        ]
                    )

                    task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc': sketch_proc})
                    futures = self.call_multiple_tasks(task_calls)
                    results = ray.get(futures)
                    feature_scores = {}
                    for result_dict in results:
                        feature_scores.update(result_dict)
                    best_feature = max(feature_scores, key=feature_scores.get)
                    iter_score = feature_scores[best_feature].item()
                    early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count)

                    if not early_stopping:
                        augmentation_plan.append(best_feature)
                        best_feature_cache = ray.get(sketch_proc.retrieve.remote('cache'))[best_feature]
                        sketch_proc.update_with_best_feature.remote(
                            best_feature_cache['new_feature'],
                            best_feature_cache['joint_cofactor_matrix'],
                            best_feature_cache['joint_feature_target_vector'],
                            best_feature_cache['joint_count'],
                            best_feature_cache['y_t_y'],
                            best_feature_cache['joint_sum_vec']
                        )
                        sketch_proc.remove_from_features_pool.remote(best_feature)

                        new_features_count = len(ray.get(sketch_proc.retrieve.remote('cache')))
                        base_score = iter_score

            case 'classification':
                base_table_sketches_per_class = joint_tuples.base_table_sketches_per_class
                aug_table_sketches_per_class = joint_tuples.aug_table_sketches_per_class
                aug_table_sketches_per_class = dict(sorted(aug_table_sketches_per_class.items(), key=lambda item: item[0]))
                aug_table_sketches_per_class_split = {}
                for label, aug_table_sketches in aug_table_sketches_per_class.items():
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
                                    for sums, diags, feature_indices in zip(*l)
                                ]
                            )
                        )
                    aug_table_sketches_per_class_split[label] = aug_table_sketches
                sketch_proc_per_class = {}
                for label in base_table_sketches_per_class:
                    sketch_proc_per_class[label] = SketchProcessor.remote(conninfo=None, feature_selection_table_name=None)
                    base_table_sketch = base_table_sketches_per_class[label]
                    aug_table_sketches = aug_table_sketches_per_class_split[label]
                    
                    # do not use the output, just populate the cache in `sketch_proc` object
                    _ = ray.get(
                        [
                            sketch_proc_per_class[label]._join_aug_table.remote(
                                aug_sketch.table_feature_index,
                                base_table_sketch,
                                aug_sketch,
                                task,
                                label
                            ) for aug_sketch in aug_table_sketches
                        ]
                    )
                n_iter = 0
                remaining_features = ray.get(sketch_proc_per_class[label].retrieve.remote('cache')).keys()
                task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc_per_class': sketch_proc_per_class})
                futures = self.call_multiple_tasks(task_calls)
                results = ray.get(futures)
                feature_scores = {}
                for result_dict in results:
                    feature_scores.update(result_dict)
                best_feature = max(feature_scores, key=feature_scores.get)
                base_score = feature_scores[best_feature].item()
                if base_score == float('-inf'):
                    augmentation_plan = []
                    aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
                    Result = namedtuple('JointTuple', 'aug_feature_indices')
                    return Result(aug_feature_indices)
                augmentation_plan = [best_feature]
                
                for label in sketch_proc_per_class:
                    best_feature_cache = ray.get(sketch_proc_per_class[label].retrieve.remote('cache'))[best_feature]
                    sketch_proc_per_class[label].update_with_best_feature.remote(
                        best_feature_cache['new_feature'],
                        best_feature_cache['joint_cofactor_matrix'],
                        best_feature_cache['joint_feature_target_vector'],
                        best_feature_cache['joint_count'],
                        best_feature_cache['y_t_y'],
                        best_feature_cache['joint_sum_vec']
                    )
                    sketch_proc_per_class[label].remove_from_features_pool.remote(best_feature)

                new_features_count = len(ray.get(sketch_proc_per_class[label].retrieve.remote('cache')))
                early_stopping = self._stopping_criteria(None, base_score, tol, new_features_count)
                while (not early_stopping) and (new_features_count > 0):
                    n_iter += 1
                    remaining_features = ray.get(sketch_proc_per_class[label].retrieve.remote('cache')).keys()
                    for label in sketch_proc_per_class:
                        _ = ray.get(
                            [
                                sketch_proc_per_class[label]._update_from_cache.remote(
                                    table_feature_index, task
                                ) for table_feature_index in remaining_features
                            ]
                        )

                    task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc_per_class': sketch_proc_per_class})
                    futures = self.call_multiple_tasks(task_calls)
                    results = ray.get(futures)
                    feature_scores = {}
                    for result_dict in results:
                        feature_scores.update(result_dict)
                    best_feature = max(feature_scores, key=feature_scores.get)
                    iter_score = feature_scores[best_feature].item()
                    early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count)

                    if not early_stopping:
                        augmentation_plan.append(best_feature)
                        for label in sketch_proc_per_class:
                            best_feature_cache = ray.get(sketch_proc_per_class[label].retrieve.remote('cache'))[best_feature]
                            sketch_proc_per_class[label].update_with_best_feature.remote(
                                best_feature_cache['new_feature'],
                                best_feature_cache['joint_cofactor_matrix'],
                                best_feature_cache['joint_feature_target_vector'],
                                best_feature_cache['joint_count'],
                                best_feature_cache['y_t_y'],
                                best_feature_cache['joint_sum_vec']
                            )
                            sketch_proc_per_class[label].remove_from_features_pool.remote(best_feature)
                        new_features_count = len(ray.get(sketch_proc_per_class[label].retrieve.remote('cache')))
                        base_score = iter_score

        Result = namedtuple('JointTuple', 'aug_feature_indices')
        aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
    
        return Result(aug_feature_indices)


    def _collect_tasks(self, task: str, model: FeatureSelectionModel, remaining_features: list[str], n_iter: int, **kwargs):
        task_calls = []
        match task:
            case 'regression':
                sketch_proc = kwargs['sketch_proc']
                statistics = ray.get(sketch_proc.retrieve.remote('fs_output'))
                for idx in remaining_features:
                    dic = statistics[idx]
                    if dic['joint_count'] is None:
                        sketch_proc.remove_from_features_pool.remote(idx)
                        continue
                    task_params = {
                        'joint_cofactor': dic['joint_cofactor_matrix'],
                        'feature_target': dic['joint_feature_target_vector'],
                        'joint_count': dic['joint_count'],
                        'y_t_y': dic['y_t_y'],
                        'sum_y': dic['base_target_sum_vec'],
                        'sketch_proc': sketch_proc
                    }
                    task_calls.append((
                        '_evaluate_feature',
                        (model(dic['y_t_y'], n_iter), idx, self.metric),
                        task_params
                    ))
            case 'classification':
                sketch_proc_per_class = kwargs['sketch_proc_per_class']
                statistics_per_class = {
                    label: ray.get(sketch_proc_per_class[label].retrieve.remote('fs_output'))
                    for label in sketch_proc_per_class
                }
                for idx in remaining_features:
                    dic_per_class = {
                        label: statistics_per_class[label][idx]
                        for label in statistics_per_class
                    }

                    if any([dic_per_class[label]['joint_count'] is None for label in dic_per_class]):
                        for label in dic_per_class:
                            sketch_proc_per_class[label].remove_from_features_pool.remote(idx)
                        continue

                    task_params = {
                        'class_covariances': np.transpose(
                            np.stack(
                                [
                                    dic_per_class[label]['joint_cofactor_matrix'] / dic_per_class[label]['joint_count']
                                    for label in dic_per_class
                                ],
                                axis=2
                            ),
                            (2, 0, 1)
                        ),
                        'class_means': np.stack(
                            [dic_per_class[label]['features_mean'] for label in dic_per_class],
                            axis=0
                        ),
                        'sketch_proc_per_class': sketch_proc_per_class
                    }
                    task_calls.append((
                        '_evaluate_feature',
                        (model(0, 0), idx, self.metric),
                        task_params
                    ))

        return task_calls


@ray.remote
class BackwardElimination(GreedyAlgo, RemoteMixin):
    def __init__(self, metric: str):
        self.metric = metric


    def run(self, model: FeatureSelectionModel, task: str, joint_tuples, tol: float, **kwargs):
        tol = -tol # because we are looking for decrease in score
        match task:
            case 'regression':
                objects = joint_tuples._asdict()
                joint_cofactor_matrix = objects['joint_cofactor_matrix']
                joint_feature_target_vector = objects['joint_feature_target_vector']
                joint_count = objects['joint_count']
                sum_y = objects['sum_y']
                aug_feature_indices = objects['aug_feature_indices']
                new_features_count = objects['new_features_count']
                model_instance = model(objects['y_t_y'], 0)

                n_features = joint_cofactor_matrix.shape[1]
                n_init_features = n_features - new_features_count
                init_features_idx = np.arange(n_init_features)
                new_features_idx = np.setdiff1d(np.arange(n_features), init_features_idx)
                all_features_idx = np.arange(n_features)
                aug_feature_indices = {k: v for k, v in zip(new_features_idx, aug_feature_indices.values())}

                params = {
                    'joint_cofactor': joint_cofactor_matrix,
                    'feature_target': joint_feature_target_vector,
                    'joint_count': joint_count,
                    'all_features_idx': all_features_idx,
                    'sum_y': sum_y
                }
                base_score = self._first_iteration(model_instance, self.metric, **params)

            case 'classification':
                class_dict = joint_tuples.class_dict
                new_features_count = joint_tuples.new_features_count
                aug_feature_indices = joint_tuples.aug_feature_indices
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

                n_features = class_means.shape[1]
                n_init_features = n_features - new_features_count
                init_features_idx = np.arange(n_init_features)
                new_features_idx = np.setdiff1d(np.arange(n_features), init_features_idx)
                all_features_idx = np.arange(n_features)
                aug_feature_indices = {k: v for k, v in zip(new_features_idx, aug_feature_indices.values())}

                model_instance = model(joint_tuples.joint_count, 0)
                params = {
                    'class_covariances': np.transpose(
                        np.stack(
                            [class_covariances[i] for i in range(len(class_covariances))],
                            axis=2
                        ),
                        (2, 0, 1)
                    ),
                    'class_means': class_means,
                    'all_features_idx': all_features_idx
                }
                base_score = self._first_iteration(model_instance, self.metric, **params)

        iter_score = None
        early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count, task='elimination')
        augmentation_plan = list(aug_feature_indices.values())
        while (not early_stopping) and (new_features_count > 0):
            task_calls = self._collect_tasks(task, model_instance, new_features_idx, **params)
            futures = self.call_multiple_tasks(task_calls)
            results = ray.get(futures)
            feature_scores = {}
            for result_dict in results:
                feature_scores.update(result_dict)
            best_feature = max(feature_scores, key=feature_scores.get)
            iter_score = feature_scores[best_feature].item()
            early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count, task='elimination')

            if not early_stopping:
                augmentation_plan.remove(aug_feature_indices[best_feature])
                new_features_idx = np.setdiff1d(new_features_idx, [best_feature])
                all_features_idx = np.setdiff1d(all_features_idx, [best_feature])
                params['all_features_idx'] = all_features_idx
                new_features_count -= 1
                base_score = iter_score
        aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
        Result = namedtuple('JointTuple', 'aug_feature_indices')

        return Result(aug_feature_indices)
    

    def _collect_tasks(self, task: str, model: FeatureSelectionModel, new_features_idx: np.ndarray[int], **kwargs):
        task_calls = []
        for idx in new_features_idx:
            indices = np.setdiff1d(kwargs['all_features_idx'], idx)
            match task:
                case 'regression':
                    joint_cofactor = kwargs['joint_cofactor'][np.ix_(indices, indices)]
                    feature_target_vector = kwargs['feature_target'][indices]
                    task_params = deepcopy(kwargs)
                    task_params['joint_cofactor'] = joint_cofactor
                    task_params['feature_target'] = feature_target_vector
                case 'classification':
                    task_params = {
                        'class_covariances': np.transpose(
                            np.stack(
                                [kwargs['class_covariances'][i][np.ix_(indices, indices)] 
                                    for i in range(len(kwargs['class_covariances']))],
                                axis=2
                            ),
                            (2, 0, 1)
                        ),
                        'class_means': kwargs['class_means'][:, indices]
                    }
            task_calls.append((
                '_evaluate_feature',
                (model, idx, self.metric),
                task_params
            ))

        return task_calls