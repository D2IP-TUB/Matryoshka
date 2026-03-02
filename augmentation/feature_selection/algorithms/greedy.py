from collections import namedtuple
from copy import deepcopy
import ray
from abc import ABC, abstractmethod

from ..models import *
from ..base.model import FeatureSelectionModel
from ..sketch_processing import SketchProcessor
from ...utils.exceptions import LinAlgError


def _evaluate_feature_fn(model: FeatureSelectionModel, feature_idx: int, metric: str, **kwargs) -> dict[int, float]:
    '''Standalone evaluate function — avoids capturing actor state in Ray tasks.'''
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


@ray.remote
def _evaluate_batch(calls):
    '''Evaluate a batch of (model, feature_idx, metric, kwargs) tuples as a single Ray task.'''
    results = []
    for model, feature_idx, metric, kwargs in calls:
        results.append(_evaluate_feature_fn(model, feature_idx, metric, **kwargs))
    return results


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
        return _evaluate_feature_fn(model, feature_idx, metric, **kwargs)

    def _dispatch_evaluate(self, task_calls, n_jobs: int):
        '''Dispatch feature evaluation calls as batched Ray tasks.
        task_calls: list of ('_evaluate_feature', (model, idx, metric), kwargs)
        Returns flat list of result dicts.'''
        if len(task_calls) == 0:
            return []
        # Convert method-call tuples to data tuples for _evaluate_batch
        eval_items = [(args[0], args[1], args[2], kwargs) for _, args, kwargs in task_calls]
        n_batches = max(1, min(n_jobs, len(eval_items)))
        batch_size = (len(eval_items) + n_batches - 1) // n_batches
        # Put shared data in object store once to avoid repeated serialization
        batches = []
        for i in range(0, len(eval_items), batch_size):
            batch_ref = ray.put(eval_items[i:i + batch_size])
            batches.append(_evaluate_batch.remote(batch_ref))
        batch_results = ray.get(batches)
        results = []
        for batch in batch_results:
            results.extend(batch)
        return results

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


@ray.remote(num_cpus=0)
class ForwardSelection(GreedyAlgo, RemoteMixin):
    def __init__(self, metric: str):
        self.metric = metric


    def run(self, model: FeatureSelectionModel, task: str, joint_tuples, tol: float, n_jobs: int = 1, **kwargs):
        import time as _time
        _t_cache_total = 0.0
        _t_eval_total = 0.0
        _t_overhead_total = 0.0
        _n_iters = 0
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
                results = self._dispatch_evaluate(task_calls, n_jobs)
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

                new_features_count = ray.get(sketch_proc.accept_best_feature.remote(best_feature))
                early_stopping = self._stopping_criteria(None, base_score, tol, new_features_count)
                while (not early_stopping) and (new_features_count > 0):
                    n_iter += 1
                    _n_iters += 1
                    _t0 = _time.perf_counter()
                    remaining_features, statistics = ray.get(sketch_proc._update_cache_and_get_stats.remote(task, n_jobs))
                    _t1 = _time.perf_counter()
                    _t_cache_total += _t1 - _t0

                    task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc': sketch_proc, 'statistics': statistics})
                    _t2 = _time.perf_counter()
                    results = self._dispatch_evaluate(task_calls, n_jobs)
                    _t3 = _time.perf_counter()
                    _t_eval_total += _t3 - _t2
                    feature_scores = {}
                    for result_dict in results:
                        feature_scores.update(result_dict)
                    best_feature = max(feature_scores, key=feature_scores.get)
                    iter_score = feature_scores[best_feature].item()
                    early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count)

                    if not early_stopping:
                        _t4 = _time.perf_counter()
                        augmentation_plan.append(best_feature)
                        new_features_count = ray.get(sketch_proc.accept_best_feature.remote(best_feature))
                        base_score = iter_score
                        _t5 = _time.perf_counter()
                        _t_overhead_total += _t5 - _t4

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
                results = self._dispatch_evaluate(task_calls, n_jobs)
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
                
                accept_futures = [
                    sketch_proc_per_class[label].accept_best_feature.remote(best_feature)
                    for label in sketch_proc_per_class
                ]
                counts = ray.get(accept_futures)
                new_features_count = counts[-1]
                early_stopping = self._stopping_criteria(None, base_score, tol, new_features_count)
                while (not early_stopping) and (new_features_count > 0):
                    n_iter += 1
                    # Update all class sketch processors in parallel, each returns (remaining, fs_output)
                    update_futures = [
                        sketch_proc_per_class[label]._update_cache_and_get_stats.remote(task, n_jobs)
                        for label in sketch_proc_per_class
                    ]
                    update_results = ray.get(update_futures)
                    remaining_features = update_results[0][0]
                    statistics_per_class = {
                        label: result[1]
                        for label, result in zip(sketch_proc_per_class.keys(), update_results)
                    }

                    task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc_per_class': sketch_proc_per_class, 'statistics_per_class': statistics_per_class})
                    results = self._dispatch_evaluate(task_calls, n_jobs)
                    feature_scores = {}
                    for result_dict in results:
                        feature_scores.update(result_dict)
                    best_feature = max(feature_scores, key=feature_scores.get)
                    iter_score = feature_scores[best_feature].item()
                    early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count)

                    if not early_stopping:
                        augmentation_plan.append(best_feature)
                        accept_futures = [
                            sketch_proc_per_class[label].accept_best_feature.remote(best_feature)
                            for label in sketch_proc_per_class
                        ]
                        counts = ray.get(accept_futures)
                        new_features_count = counts[-1]
                        base_score = iter_score

        if _n_iters > 0:
            print(f'[ForwardSelection] iters={_n_iters}, cache_update={_t_cache_total:.2f}s, eval={_t_eval_total:.2f}s, overhead={_t_overhead_total:.2f}s, n_jobs={n_jobs}')
        Result = namedtuple('JointTuple', 'aug_feature_indices')
        aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
    
        return Result(aug_feature_indices)


    def _collect_tasks(self, task: str, model: FeatureSelectionModel, remaining_features: list[str], n_iter: int, **kwargs):
        task_calls = []
        match task:
            case 'regression':
                sketch_proc = kwargs['sketch_proc']
                statistics = kwargs.get('statistics') or ray.get(sketch_proc.retrieve.remote('fs_output'))
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
                statistics_per_class = kwargs.get('statistics_per_class') or {
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
                        'class_counts': np.array([dic_per_class[label]['joint_count'] for label in dic_per_class]),
                        'sketch_proc_per_class': sketch_proc_per_class
                    }
                    task_calls.append((
                        '_evaluate_feature',
                        (model(0, 0), idx, self.metric),
                        task_params
                    ))

        return task_calls


@ray.remote(num_cpus=0)
class BackwardElimination(GreedyAlgo, RemoteMixin):
    def __init__(self, metric: str):
        self.metric = metric


    def run(self, model: FeatureSelectionModel, task: str, joint_tuples, tol: float, n_jobs: int = 1, **kwargs):
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
                class_counts = np.array([class_dict[label]['joint_count'] for label in class_dict])
                params = {
                    'class_covariances': np.transpose(
                        np.stack(
                            [class_covariances[i] for i in range(len(class_covariances))],
                            axis=2
                        ),
                        (2, 0, 1)
                    ),
                    'class_means': class_means,
                    'class_counts': class_counts,
                    'all_features_idx': all_features_idx
                }
                base_score = self._first_iteration(model_instance, self.metric, **params)

        iter_score = None
        early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count, task='elimination')
        augmentation_plan = list(aug_feature_indices.values())
        while (not early_stopping) and (new_features_count > 0):
            task_calls = self._collect_tasks(task, model_instance, new_features_idx, **params)
            results = self._dispatch_evaluate(task_calls, n_jobs)
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
                        'class_means': kwargs['class_means'][:, indices],
                        'class_counts': kwargs['class_counts']
                    }
            task_calls.append((
                '_evaluate_feature',
                (model, idx, self.metric),
                task_params
            ))

        return task_calls