import csv
import json
import time
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import deepcopy
from pathlib import Path

import ray

from ...exceptions import LinAlgError
from ...utils.logging import default_log_dir
from ..base.model import FeatureSelectionModel
from ..models import *
from ..sketch_processing import (
    SketchProcessor,
    _build_group_aug_sketch,
    expand_with_polynomials,
    expand_with_quantile_bins,
)

# Metrics whose score is computed purely from the joint Gram summary statistics
# (see ``FeatureSelectionModel.score``) and therefore do not use the model's OLS
# fit (``coef_``). For these the (possibly singular) factorization in
# ``fit_predict`` is skipped, so the ridge-regularized conditional-GCV criterion
# can still rank a candidate whose joint Gram is singular instead of dropping it
# at the fit step. The classification conditional metric already factors-free in
# ``ClassificationCholesky.fit_predict``, so only the regression metric is listed.
_FIT_FREE_METRICS = {'conditional_gcv'}


# Suffix vocabulary used by the indexer when materializing per-key
# aggregations of a numeric column (see matryoshka.index._index_features).
# Two candidates whose raw header names differ only by one of these suffixes
# are considered redundant for selection purposes.
_AGGREGATION_SUFFIXES = ('min', 'max', 'median', 'mean', 'nunique')


def _strip_aggregation_suffix(header: str | None) -> str | None:
    '''Return the source-column identifier for ``header`` with any trailing
    ``_<aggregation>`` token stripped. Polynomial power suffixes (``^k``) are
    preserved so that ``x`` and ``x^2`` of the same source column remain
    distinct candidates.'''
    if header is None:
        return None
    name, _, power = header.partition('^')
    parts = name.rsplit('_', 1)
    if len(parts) == 2 and parts[1] in _AGGREGATION_SUFFIXES:
        name = parts[0]
    return f'{name}^{power}' if power else name


def _candidate_sketches_from_splits(splits, polynomial_features: dict | None,
                                    dummify_features: dict | None = None,
                                    bin_edges_registry: dict[str, np.ndarray] | None = None):
    '''Materialize candidate AugSketch objects from the output of
    `SketchProcessor._prepare_split_inputs`.

    Each polynomial power is emitted as an *independent* single-column
    candidate sketch with its own cache key. With ``degree=k`` and ``N`` base
    features, forward selection therefore evaluates ``k * N`` candidates per
    round (base feature `x` plus `x^2, ..., x^k`).

    Args:
        splits: list of (split_sums, split_diags, split_feature_indices,
            split_column_names) tuples, one per original aug_table_sketch;
            inner lists are per-column. ``split_column_names`` carries the
            raw DB header name for each column (used to derive ``base_column``).
        polynomial_features: optional dict with keys
            `enabled: bool`, `degree: int >= 2`, `interactions: bool`.
            When None or enabled=False, behaves like the legacy per-column split.
        dummify_features: optional dict with keys `enabled: bool` and
            `n_bins: int >= 2`. When enabled, each per-key aggregated column
            is dummified into ``n_bins`` quantile-based indicator groups in
            addition to the legacy single-column emission. Mutually
            exclusive with polynomial expansion on the same column.
        bin_edges_registry: optional dict the caller can pass to recover the
            per-column quantile edges used at sketch time. Keys are the
            sketch cache names emitted for the dummified group; values are
            the interior quantile thresholds. Materialization re-applies the
            same edges to reproduce the indicator values on the augmented
            frame.

    Returns:
        (sketches, group_members, feature_to_base):
          * sketches: list of single-column AugSketch \u2014 each is one candidate.
          * group_members: dict mapping each sketch's cache key to a one-element
            list ``[cache_key]``.
          * feature_to_base: dict mapping each sketch's cache key to its
            qualified ``base_column`` identifier (or ``None`` when unknown).
    '''
    pf = polynomial_features or {}
    enabled = bool(pf.get('enabled', False))
    degree = int(pf.get('degree', 2))
    interactions = bool(pf.get('interactions', False))
    if enabled and interactions:
        raise NotImplementedError(
            'polynomial_features.interactions=True is not yet supported; '
            'only powers (x, x^2, ..., x^degree) are currently handled.'
        )

    df = dummify_features or {}
    df_enabled = bool(df.get('enabled', False))
    n_bins = int(df.get('n_bins', 4))

    sketches = []
    group_members: dict[str, list[str]] = {}
    feature_to_base: dict[str, str | None] = {}
    for split in splits:
        # Backwards-compatible unpack: older callers may pass a 3-tuple.
        if len(split) == 4:
            split_sums, split_diags, split_feature_indices, split_column_names = split
        else:
            split_sums, split_diags, split_feature_indices = split
            split_column_names = [None] * len(split_feature_indices)
        for sums, diags, name, header in zip(
            split_sums, split_diags, split_feature_indices, split_column_names
        ):
            # Qualify the base-column identifier with the source table /
            # key-column prefix carried by ``name`` (``<ti>_<kci>_<i>``) so
            # candidates from different tables never collide.
            stripped = _strip_aggregation_suffix(header)
            if stripped is not None:
                name_parts = name.split('_')
                prefix = '_'.join(name_parts[:-1]) if len(name_parts) >= 2 else ''
                base_col = f'{prefix}_{stripped}' if prefix else stripped
            else:
                base_col = None
            if enabled and degree >= 2:
                # A base column and its powers form ONE atomic group candidate
                # `[x, x^2, ..., x^degree]`: forward selection evaluates them
                # jointly (the group's per-row cofactors give the joint Gram
                # block) and, when the group is selected, emits the base
                # together with every power. A power is never selected or
                # materialized without its base.
                group_sketch = expand_with_polynomials(
                    sums[:, 0:1], name, degree, table_feature_index=name,
                    base_column=base_col,
                )
                sketches.append(group_sketch)
                group_members[name] = [name] + [f'{name}^{k}' for k in range(2, degree + 1)]
                feature_to_base[name] = base_col
            else:
                # Legacy single-column candidate under its original cache key.
                base_sketch = _build_group_aug_sketch(
                    sums, name, column_names=[name], base_column=base_col,
                )
                sketches.append(base_sketch)
                group_members[name] = [name]
                feature_to_base[name] = base_col

            # Dummification group emitted in ADDITION to the per-column
            # candidate above. The group key is suffixed with `=dummies` so
            # it never collides with the original cache key and forward
            # treats it as an independent candidate.
            if df_enabled:
                dummy_name = f'{name}=dummies'
                dummy_sketch, edges = expand_with_quantile_bins(
                    sums[:, 0:1], name, n_bins, table_feature_index=dummy_name,
                    base_column=base_col,
                )
                if dummy_sketch is not None:
                    sketches.append(dummy_sketch)
                    # The materialized member columns are the per-bin
                    # indicators emitted by ``expand_with_quantile_bins``.
                    if dummy_sketch.aug_feature_indices is not None:
                        members = [dummy_sketch.aug_feature_indices[i]
                                   for i in sorted(dummy_sketch.aug_feature_indices)]
                    else:
                        members = [dummy_name]
                    group_members[dummy_name] = members
                    feature_to_base[dummy_name] = base_col
                    if bin_edges_registry is not None:
                        bin_edges_registry[dummy_name] = edges
                        for member in members:
                            bin_edges_registry[member] = edges
    return sketches, group_members, feature_to_base


def _drop_aggregation_siblings(
    accepted_feature: str,
    feature_to_base: dict[str, str | None],
    sketch_procs,
) -> list[str]:
    '''After ``accepted_feature`` is committed, remove from every sketch
    processor's pool any other candidate sharing the same ``base_column``.

    Args:
        accepted_feature: cache key just passed to ``accept_best_feature``.
        feature_to_base: cache-key \u2192 qualified base column map; mutated in
            place to drop both the accepted feature and any siblings so they
            cannot be reconsidered (or re-removed) on later iterations.
        sketch_procs: a single SketchProcessor handle or an iterable of them
            (one per class for classification).

    Returns:
        The list of sibling cache keys that were dropped.
    '''
    base = feature_to_base.pop(accepted_feature, None)
    if base is None:
        return []
    siblings = [k for k, v in feature_to_base.items() if v == base]
    if not siblings:
        return []
    procs = sketch_procs if isinstance(sketch_procs, (list, tuple, dict)) else [sketch_procs]
    if isinstance(procs, dict):
        procs = list(procs.values())
    for sib in siblings:
        for proc in procs:
            proc.remove_from_features_pool.remote(sib)
        feature_to_base.pop(sib, None)
    return siblings


def _evaluate_feature_fn(model: FeatureSelectionModel, feature_idx: int, metric: str, **kwargs) -> dict[int, float]:
    '''Standalone evaluate function — avoids capturing actor state in Ray tasks.'''
    try:
        if metric not in _FIT_FREE_METRICS:
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


def _evaluate_base_only_score(base_table_sketch, model_cls, metric: str) -> tuple[float, int, int]:
    '''Fit the proxy on the base table sketch alone (no augmentation) and
    return its score under ``metric``.

    Provides the no-augmentation reference point used at iter 0 of forward
    selection. Aggregates the per-key sketch state, applies the same
    centred + scaled Gram standardisation that the joint path uses
    (``_standardize_gram_inputs_pure`` in sketch_processing.py), then fits
    the proxy on the standardised matrix. Without standardisation the raw
    base Gram is routinely ill-conditioned by tens of orders of magnitude,
    matching what the joint path would observe absent standardisation.

    Returns
    -------
    score : float
        Scalar score under ``metric``. The convention is "higher is better"
        (RMSE / GCV-RMSE are returned negated, matching ``score()``).
    n : int
        Total joined rows in the base table (= sum of per-key counts).
    p_base : int
        Number of base-side features (excluding the intercept).
    '''
    from ..sketch_processing import _standardize_gram_inputs_pure

    bc = base_table_sketch.count                # (K, 1)
    bfs = base_table_sketch.features_sum        # (K, p_base)
    bts = base_table_sketch.target_sum          # (K, 1)
    # Use the true per-key Gram blocks (sum_{i in k} v_i v_i^T) populated
    # alongside the legacy s_k s_k^T sketch. The legacy `features_cofactors`
    # / `target_cofactors` slots cannot be used here because their per-key
    # outer product weighs each group by n_k^2 instead of accumulating
    # within-group second-order mass.
    bfg = base_table_sketch.features_gram        # (K, p_base, p_base)
    btgr = base_table_sketch.target_gram_row     # (K, p_base+1, 1)
    if bfg is None or btgr is None:
        raise ValueError(
            'base_table_sketch is missing features_gram/target_gram_row; '
            'rebuild it via SketchProcessor._create_base_table_sketch on a '
            'user_table_agg that carries the `cofactor_upper` column.'
        )

    n = int(bc.sum())
    p_base = int(bfs.shape[1])

    # Aggregate per-key sketch state into a single (p_base, p_base) Gram
    # and a (p_base,) X'y vector. The intercept is absorbed by the
    # centred standardisation that follows.
    features_sum = bfs.sum(axis=0).reshape(-1).astype(np.float64)
    target_sum = float(bts.sum())
    cofactor_matrix = bfg.sum(axis=0).astype(np.float64)
    feature_target_vector = btgr[:, 1:, 0].sum(axis=0).reshape(-1).astype(np.float64)

    # Centred y'y for the standardised regression target.
    y_t_y_raw = float(btgr[:, 0, 0].sum())
    y_t_y_centered = y_t_y_raw - (target_sum * target_sum) / n

    G_std, s_std = _standardize_gram_inputs_pure(
        cofactor_matrix=cofactor_matrix,
        feature_target_vector=feature_target_vector,
        features_sum=features_sum,
        target_sum=target_sum,
        joint_count=n,
    )

    model = model_cls(y_t_y_centered, 0)
    if metric not in _FIT_FREE_METRICS:
        model.fit_predict(joint_cofactor=G_std, feature_target=s_std)
    score = model.score(
        metric,
        joint_cofactor=G_std,
        feature_target=s_std,
        joint_count=n,
        sum_y=np.array([[target_sum]]),
        new_features=0,
    )
    return float(np.asarray(score).reshape(-1)[0]), n, p_base


def _score_item(v) -> float:
    if isinstance(v, np.ndarray):
        return float(v.item())
    return float(v)


def _record_candidate_history(
    rows: list[dict],
    iteration: int,
    feature_scores: dict,
    selected_feature: str,
    previous_score: float | None,
    accepted: bool,
) -> float:
    sorted_items = sorted(
        ((str(k), _score_item(v)) for k, v in feature_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    top_score = float(sorted_items[0][1]) if sorted_items else float('-inf')
    selected_score = float(dict(sorted_items)[str(selected_feature)])
    abs_improvement = None if previous_score is None else (selected_score - previous_score)
    if previous_score is None:
        rel_improvement = None
    else:
        rel_improvement = abs_improvement / max(abs(previous_score), 1e-12)

    for rank, (candidate, score) in enumerate(sorted_items, start=1):
        is_selected = candidate == str(selected_feature)
        rows.append(
            {
                'iteration': int(iteration),
                'candidate_feature': candidate,
                'score': float(score),
                'rank': int(rank),
                'is_selected': bool(is_selected),
                'is_argmax_score': bool(np.isclose(score, top_score, rtol=0.0, atol=1e-12)),
                'accepted': bool(accepted) if is_selected else False,
                'selected_feature': str(selected_feature),
                'selected_score': float(selected_score),
                'previous_score': None if previous_score is None else float(previous_score),
                'improvement': None if abs_improvement is None else float(abs_improvement),
                'relative_improvement': None if rel_improvement is None else float(rel_improvement),
                'score_gap_to_best': float(top_score - score),
            }
        )

    return float(selected_score)


def _write_selection_history_artifacts(
    task: str,
    metric: str,
    candidate_rows: list[dict],
    augmentation_plan: list[str],
    final_score: float,
) -> dict:
    # Written under the library's log directory ($MATRYOSHKA_LOG_DIR, else
    # ./.matryoshka/logs) rather than a bare ./logs, so a run does not
    # scatter artefacts through the caller's working directory.
    out_dir = default_log_dir() / 'selection_history'
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = f'{task}_{metric}_{int(time.time())}_{int(time.time_ns() % 10_000_000)}'
    candidates_path = out_dir / f'{run_id}_candidates.csv'
    selected_path = out_dir / f'{run_id}_selected.csv'
    summary_path = out_dir / f'{run_id}_summary.json'

    fieldnames = [
        'iteration',
        'candidate_feature',
        'score',
        'rank',
        'is_selected',
        'is_argmax_score',
        'accepted',
        'selected_feature',
        'selected_score',
        'previous_score',
        'improvement',
        'relative_improvement',
        'score_gap_to_best',
    ]

    with candidates_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            writer.writerow(row)

    selected_rows = [row for row in candidate_rows if row['is_selected']]
    with selected_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow(row)

    summary = {
        'run_id': run_id,
        'task': task,
        'metric': metric,
        'iterations': int(len(selected_rows)),
        'selected_features_count': int(len(augmentation_plan)),
        'selected_features': [str(x) for x in augmentation_plan],
        'final_score': None if final_score is None else float(final_score),
        'candidates_path': str(candidates_path),
        'selected_path': str(selected_path),
        'generated_at_epoch_sec': int(time.time()),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


@ray.remote
def _evaluate_batch(calls):
    '''Evaluate a batch of (model, feature_idx, metric, kwargs) tuples as a single Ray task.
    Returns (results, compute_s) where compute_s is wall time spent on actual computation.'''
    import time as _time
    t0 = _time.perf_counter()
    results = []
    for model, feature_idx, metric, kwargs in calls:
        results.append(_evaluate_feature_fn(model, feature_idx, metric, **kwargs))
    return results, _time.perf_counter() - t0


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
        if metric not in _FIT_FREE_METRICS:
            model.fit_predict(**kwargs)
        return model.score(metric, **kwargs).item()

    def _evaluate_feature(self, model: FeatureSelectionModel, feature_idx: int, metric: str, **kwargs) -> dict[int, float]:
        return _evaluate_feature_fn(model, feature_idx, metric, **kwargs)

    def _dispatch_evaluate(self, task_calls, n_jobs: int):
        '''Dispatch feature evaluation calls as batched Ray tasks.
        task_calls: list of ('_evaluate_feature', (model, idx, metric), kwargs)
        Returns (flat list of result dicts, timing_dict).
        timing_dict keys: put_s (object-store serialization), dispatch_s (task dispatch),
        get_s (wall time waiting for ray.get), compute_s (actual computation inside tasks).'''
        import time as _time
        _ZERO_TIMING = {'put_s': 0.0, 'dispatch_s': 0.0, 'get_s': 0.0, 'compute_s': 0.0}
        if len(task_calls) == 0:
            return [], _ZERO_TIMING
        # Convert method-call tuples to data tuples for _evaluate_batch
        eval_items = [(args[0], args[1], args[2], kwargs) for _, args, kwargs in task_calls]
        n_batches = max(1, min(n_jobs, len(eval_items)))
        batch_size = (len(eval_items) + n_batches - 1) // n_batches
        # Put each batch into the object store, then dispatch
        batches = []
        t_put_total = 0.0
        t_dispatch_total = 0.0
        for i in range(0, len(eval_items), batch_size):
            t0 = _time.perf_counter()
            batch_ref = ray.put(eval_items[i:i + batch_size])
            t1 = _time.perf_counter()
            batches.append(_evaluate_batch.remote(batch_ref))
            t2 = _time.perf_counter()
            t_put_total += t1 - t0
            t_dispatch_total += t2 - t1
        t_get_start = _time.perf_counter()
        batch_results_raw = ray.get(batches)
        t_get_end = _time.perf_counter()
        results = []
        compute_times = []
        for batch_results, compute_s in batch_results_raw:
            results.extend(batch_results)
            compute_times.append(compute_s)
        timing = {
            'put_s': t_put_total,
            'dispatch_s': t_dispatch_total,
            'get_s': t_get_end - t_get_start,
            'compute_s': max(compute_times) if compute_times else 0.0,
        }
        return results, timing

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

        from ..anytime import BudgetClock, TrajectoryEmitter
        _t_cache_total = 0.0
        _t_cache_put_total = 0.0
        _t_cache_compute_total = 0.0
        _t_eval_total = 0.0
        _t_eval_put_total = 0.0
        _t_eval_compute_total = 0.0
        _t_overhead_total = 0.0
        _n_iters = 0
        track_selection_history = kwargs.get('track_selection_history', True)
        candidate_history_rows: list[dict] = []
        polynomial_features = kwargs.get('polynomial_features')
        dummify_features = kwargs.get('dummify_features')
        # Quantile edges per dummified candidate. Populated by
        # ``_candidate_sketches_from_splits`` when ``dummify_features`` is
        # enabled and consumed by the augmentation materialization step so
        # the same bin edges are re-applied to the joined column.
        _dummify_bin_edges: dict[str, np.ndarray] = {}

        # Anytime instrumentation. ``budget_seconds=None`` disables the deadline,
        # ``trajectory_dir=None`` disables emission, both keep the
        # backwards-compatible behaviour. See augmentation/feature_selection/anytime.py.
        _budget_seconds = kwargs.get('budget_seconds')
        _trajectory_dir = kwargs.get('trajectory_dir')
        _clock = BudgetClock(_budget_seconds).start()
        _emitter = (TrajectoryEmitter(_trajectory_dir, algo='ForwardSelection')
                    if _trajectory_dir else None)
        # Holds the base-only proxy score so we can include it on the
        # iter-0 trajectory row's `extra` payload (set below in the
        # regression branch before any aug feature is selected).
        _base_only_score: float | None = None

        def _emit_progress(iter_idx, plan, gm):
            if _emitter is None:
                return
            # Emit the cumulative selected plan (one entry per augmentation-plan
            # feature), so ``n_features`` aligns with the augplan ordering used
            # by the anytime evaluator. ``gm`` (group members) is kept only for
            # signature compatibility; the group-expanded view is not emitted.
            extra = None
            if iter_idx == 0 and _base_only_score is not None:
                extra = {'base_only_score': _base_only_score}
            _emitter.emit(iter_idx, _clock.elapsed_s, list(plan), extra=extra)
        sketch_proc = SketchProcessor.remote(conninfo=None, feature_selection_table_name=None)
        match task:
            case 'regression':
                base_table_sketch = joint_tuples.base_table_sketch
                aug_table_sketches = joint_tuples.aug_table_sketches

                # Base-only baseline score. No augmentation features in the
                # proxy yet. Acts as the "no-augmentation" reference point at
                # iter 0 of the trajectory. Failures here are recoverable —
                # we log them and proceed without the baseline.
                try:
                    _base_only_score, _n_base, _p_base = _evaluate_base_only_score(
                        base_table_sketch, model, self.metric,
                    )
                    print(
                        f'[ForwardSelection] base-only baseline ({self.metric}): '
                        f'{_base_only_score:.6f}  n={_n_base}  p_base={_p_base}'
                    )
                except (LinAlgError, np.linalg.LinAlgError, ValueError) as _e:
                    print(f'[ForwardSelection] base-only baseline skipped: {_e!r}')
                    _base_only_score = None
                # Initial empty-plan trajectory row, annotated with the
                # base-only baseline score if it could be computed.
                if _emitter is not None:
                    _extra0 = ({'base_only_score': _base_only_score}
                               if _base_only_score is not None else None)
                    _emitter.emit(0, _clock.elapsed_s, [], extra=_extra0)
                splits = ray.get(
                    [
                        sketch_proc._prepare_split_inputs.remote(aug_sketch)
                        for aug_sketch in aug_table_sketches
                    ]
                )
                aug_table_sketches, group_members, feature_to_base = _candidate_sketches_from_splits(
                    splits, polynomial_features,
                    dummify_features=dummify_features,
                    bin_edges_registry=_dummify_bin_edges,
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
                results, _eval_t = self._dispatch_evaluate(task_calls, n_jobs)
                _t_eval_put_total += _eval_t['put_s']
                _t_eval_compute_total += _eval_t['compute_s']
                feature_scores = {}
                for result_dict in results:
                    feature_scores.update(result_dict)
                best_feature = max(feature_scores, key=feature_scores.get)
                base_score = feature_scores[best_feature].item()
                # `-inf` indicates a numerical breakdown (no candidate could be
                # fit). Bail out immediately with an empty plan.
                bail_out = base_score == float('-inf')
                if track_selection_history:
                    _record_candidate_history(
                        candidate_history_rows,
                        iteration=n_iter,
                        feature_scores=feature_scores,
                        selected_feature=best_feature,
                        previous_score=_base_only_score,
                        accepted=(not bail_out),
                    )
                if bail_out:
                    augmentation_plan = []
                    aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
                    Result = namedtuple('JointTuple', ['aug_feature_indices', 'timing_breakdown'])
                    return Result(aug_feature_indices, {})
                augmentation_plan = [best_feature]
                _emit_progress(n_iter, augmentation_plan, group_members)

                new_features_count = ray.get(sketch_proc.accept_best_feature.remote(best_feature))
                _drop_aggregation_siblings(best_feature, feature_to_base, sketch_proc)
                early_stopping = self._stopping_criteria(None, base_score, tol, new_features_count)
                while (not early_stopping) and (new_features_count > 0):
                    if _clock.expired:
                        break
                    n_iter += 1
                    _n_iters += 1
                    _t0 = _time.perf_counter()
                    remaining_features, statistics, _cache_t = ray.get(sketch_proc._update_cache_and_get_stats.remote(task, n_jobs))
                    _t1 = _time.perf_counter()
                    _t_cache_total += _t1 - _t0
                    _t_cache_put_total += _cache_t['put_s']
                    _t_cache_compute_total += _cache_t['compute_s']

                    task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc': sketch_proc, 'statistics': statistics})
                    _t2 = _time.perf_counter()
                    results, _eval_t = self._dispatch_evaluate(task_calls, n_jobs)
                    _t3 = _time.perf_counter()
                    _t_eval_total += _t3 - _t2
                    _t_eval_put_total += _eval_t['put_s']
                    _t_eval_compute_total += _eval_t['compute_s']
                    feature_scores = {}
                    for result_dict in results:
                        feature_scores.update(result_dict)
                    best_feature = max(feature_scores, key=feature_scores.get)
                    iter_score = feature_scores[best_feature].item()
                    early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count)
                    if track_selection_history:
                        _record_candidate_history(
                            candidate_history_rows,
                            iteration=n_iter,
                            feature_scores=feature_scores,
                            selected_feature=best_feature,
                            previous_score=base_score,
                            accepted=(not early_stopping),
                        )

                    if not early_stopping:
                        _t4 = _time.perf_counter()
                        augmentation_plan.append(best_feature)
                        new_features_count = ray.get(sketch_proc.accept_best_feature.remote(best_feature))
                        _drop_aggregation_siblings(best_feature, feature_to_base, sketch_proc)
                        base_score = iter_score
                        _t5 = _time.perf_counter()
                        _t_overhead_total += _t5 - _t4
                        _emit_progress(n_iter, augmentation_plan, group_members)

            case 'classification':
                # Initial empty-plan trajectory row. Classification has no
                # base-only baseline computation yet (TODO if a per-class
                # equivalent of GCV is added) so the row carries no extra.
                if _emitter is not None:
                    _emitter.emit(0, _clock.elapsed_s, [])

                base_table_sketches_per_class = joint_tuples.base_table_sketches_per_class
                aug_table_sketches_per_class = joint_tuples.aug_table_sketches_per_class
                aug_table_sketches_per_class = dict(sorted(aug_table_sketches_per_class.items(), key=lambda item: item[0]))
                # Label dtypes can drift between the two dicts (e.g. int vs float)
                # depending on how each was assembled in `ranking.py`. Re-key the
                # base-sketch dict against the aug-sketch labels in sorted order so
                # subsequent lookups are dtype-agnostic.
                base_table_sketches_per_class = dict(
                    zip(
                        aug_table_sketches_per_class.keys(),
                        [base_table_sketches_per_class[k] for k in sorted(base_table_sketches_per_class.keys())],
                    )
                )
                aug_table_sketches_per_class_split = {}
                group_members: dict[str, list[str]] = {}
                feature_to_base: dict[str, str | None] = {}
                for label, aug_table_sketches in aug_table_sketches_per_class.items():
                    splits = ray.get(
                        [
                            sketch_proc._prepare_split_inputs.remote(aug_sketch)
                            for aug_sketch in aug_table_sketches
                        ]
                    )
                    sketches, gm, ftb = _candidate_sketches_from_splits(
                        splits, polynomial_features,
                        dummify_features=None,
                        bin_edges_registry=None,
                    )
                    aug_table_sketches_per_class_split[label] = sketches
                    # Groups are identical across classes (same synthetic names).
                    group_members.update(gm)
                    feature_to_base.update(ftb)
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
                if not sketch_proc_per_class:
                    augmentation_plan = []
                    aug_feature_indices = {}
                    Result = namedtuple('JointTuple', ['aug_feature_indices', 'timing_breakdown'])
                    return Result(aug_feature_indices, {})
                n_iter = 0
                _any_label = next(iter(sketch_proc_per_class))
                remaining_features = ray.get(sketch_proc_per_class[_any_label].retrieve.remote('cache')).keys()
                task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc_per_class': sketch_proc_per_class})
                results, _eval_t = self._dispatch_evaluate(task_calls, n_jobs)
                _t_eval_put_total += _eval_t['put_s']
                _t_eval_compute_total += _eval_t['compute_s']
                feature_scores = {}
                for result_dict in results:
                    feature_scores.update(result_dict)
                best_feature = max(feature_scores, key=feature_scores.get)
                base_score = feature_scores[best_feature].item()
                if track_selection_history:
                    _record_candidate_history(
                        candidate_history_rows,
                        iteration=n_iter,
                        feature_scores=feature_scores,
                        selected_feature=best_feature,
                        previous_score=None,
                        accepted=True,
                    )
                if base_score == float('-inf'):
                    augmentation_plan = []
                    aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
                    Result = namedtuple('JointTuple', ['aug_feature_indices', 'timing_breakdown'])
                    return Result(aug_feature_indices, {})
                augmentation_plan = [best_feature]
                _emit_progress(n_iter, augmentation_plan, group_members)

                accept_futures = [
                    sketch_proc_per_class[label].accept_best_feature.remote(best_feature)
                    for label in sketch_proc_per_class
                ]
                counts = ray.get(accept_futures)
                new_features_count = counts[-1]
                _drop_aggregation_siblings(best_feature, feature_to_base, sketch_proc_per_class)
                early_stopping = self._stopping_criteria(None, base_score, tol, new_features_count)
                while (not early_stopping) and (new_features_count > 0):
                    if _clock.expired:
                        break
                    n_iter += 1
                    _n_iters += 1
                    # Update all class sketch processors in parallel, each returns (remaining, fs_output, timing)
                    _t0 = _time.perf_counter()
                    update_futures = [
                        sketch_proc_per_class[label]._update_cache_and_get_stats.remote(task, n_jobs)
                        for label in sketch_proc_per_class
                    ]
                    update_results = ray.get(update_futures)
                    _t1 = _time.perf_counter()
                    _t_cache_total += _t1 - _t0
                    remaining_features = update_results[0][0]
                    statistics_per_class = {
                        label: result[1]
                        for label, result in zip(sketch_proc_per_class.keys(), update_results)
                    }
                    _cache_timings = [result[2] for result in update_results]
                    _t_cache_put_total += sum(t['put_s'] for t in _cache_timings)
                    _t_cache_compute_total += max((t['compute_s'] for t in _cache_timings), default=0.0)

                    task_calls = self._collect_tasks(task, model, remaining_features, n_iter, **{'sketch_proc_per_class': sketch_proc_per_class, 'statistics_per_class': statistics_per_class})
                    _t2 = _time.perf_counter()
                    results, _eval_t = self._dispatch_evaluate(task_calls, n_jobs)
                    _t3 = _time.perf_counter()
                    _t_eval_total += _t3 - _t2
                    _t_eval_put_total += _eval_t['put_s']
                    _t_eval_compute_total += _eval_t['compute_s']
                    feature_scores = {}
                    for result_dict in results:
                        feature_scores.update(result_dict)
                    best_feature = max(feature_scores, key=feature_scores.get)
                    iter_score = feature_scores[best_feature].item()
                    early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count)
                    if track_selection_history:
                        _record_candidate_history(
                            candidate_history_rows,
                            iteration=n_iter,
                            feature_scores=feature_scores,
                            selected_feature=best_feature,
                            previous_score=base_score,
                            accepted=(not early_stopping),
                        )

                    if not early_stopping:
                        augmentation_plan.append(best_feature)
                        accept_futures = [
                            sketch_proc_per_class[label].accept_best_feature.remote(best_feature)
                            for label in sketch_proc_per_class
                        ]
                        counts = ray.get(accept_futures)
                        new_features_count = counts[-1]
                        _drop_aggregation_siblings(best_feature, feature_to_base, sketch_proc_per_class)
                        base_score = iter_score
                        _emit_progress(n_iter, augmentation_plan, group_members)

        _t_computation = _t_cache_compute_total + _t_eval_compute_total
        _t_ray_overhead = (_t_cache_total - _t_cache_compute_total) + (_t_eval_total - _t_eval_compute_total)
        _t_total = _t_cache_total + _t_eval_total + _t_overhead_total
        if _n_iters > 0:
            print(
                f'[ForwardSelection] iters={_n_iters} n_jobs={n_jobs}\n'
                f'  cache_update : wall={_t_cache_total:.3f}s  put={_t_cache_put_total:.3f}s  compute={_t_cache_compute_total:.3f}s  ray_overhead={_t_cache_total - _t_cache_compute_total:.3f}s\n'
                f'  eval         : wall={_t_eval_total:.3f}s   put={_t_eval_put_total:.3f}s   compute={_t_eval_compute_total:.3f}s   ray_overhead={_t_eval_total - _t_eval_compute_total:.3f}s\n'
                f'  accept_best  : wall={_t_overhead_total:.3f}s\n'
                f'  TOTAL: wall={_t_total:.3f}s  computation={_t_computation:.3f}s  ray_overhead={_t_ray_overhead:.3f}s'
                f'  ({100 * _t_ray_overhead / (_t_total + 1e-12):.1f}% overhead)'
            )
        timing_breakdown = {
            'n_iters': _n_iters,
            'computation_s': round(_t_computation, 4),
            'ray_overhead_s': round(_t_ray_overhead, 4),
        }
        Result = namedtuple('JointTuple', ['aug_feature_indices', 'timing_breakdown'])
        # Expand each selected group (cache key) into its member column names.
        flat_members = []
        for group_key in augmentation_plan:
            members = group_members.get(group_key, [group_key])
            flat_members.extend(members)
        aug_feature_indices = {k: v for k, v in enumerate(flat_members)}

        if track_selection_history:
            _write_selection_history_artifacts(
                task=task,
                metric=self.metric,
                candidate_rows=candidate_history_rows,
                augmentation_plan=augmentation_plan,
                final_score=base_score if len(augmentation_plan) > 0 else None,
            )

        return Result(aug_feature_indices, timing_breakdown)


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
                        # Candidate block size (trailing n_Z columns), used by
                        # the 'conditional_gcv' metric to split the joint Gram
                        # into current (X) and candidate (Z) blocks.
                        'new_features': dic.get('new_features_count'),
                        'sketch_proc': sketch_proc
                    }
                    task_calls.append((
                        '_evaluate_feature',
                        (model(dic['y_t_y'], n_iter), idx, self.metric),
                        task_params
                    ))
            case 'classification':
                sketch_proc_per_class = kwargs['sketch_proc_per_class']
                # Only ClassificationCholesky binds the metric to a distance kernel;
                # other classification-eligible models (e.g. RegressionQR) ignore it.
                model_kwargs = {'metric': self.metric} if model.__name__ == 'ClassificationCholesky' else {}
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
                        # Candidate block size (last n_Z columns of the joint),
                        # needed by the conditional-Mahalanobis proxy to split
                        # the joint into current (X) and candidate (Z) blocks.
                        'new_features_count': next(iter(dic_per_class.values()))['new_features_count'],
                        'sketch_proc_per_class': sketch_proc_per_class
                    }
                    task_calls.append((
                        '_evaluate_feature',
                        (model(0, 0, **model_kwargs), idx, self.metric),
                        task_params
                    ))

        return task_calls


@ray.remote(num_cpus=0)
class BackwardElimination(GreedyAlgo, RemoteMixin):
    def __init__(self, metric: str):
        self.metric = metric


    @staticmethod
    def _infer_feature_groups(aug_feature_indices: dict[int, str]) -> list[tuple[int, ...]]:
        '''Each selected column (including polynomial powers `<base>^k`) is
        considered its own atomic candidate during backward elimination,
        mirroring the per-power forward-selection semantics.
        '''
        return [(idx,) for idx in aug_feature_indices.keys()]


    def run(self, model: FeatureSelectionModel, task: str, joint_tuples, tol: float, n_jobs: int = 1, **kwargs):
        import time as _time

        from ..anytime import BudgetClock, TrajectoryEmitter
        _n_iters = 0
        _t_eval_total = 0.0
        _t_eval_compute_total = 0.0
        tol = -tol # because we are looking for decrease in score

        # Anytime instrumentation. Trajectory for backward elimination records
        # the SHRINKING feature set (iter 0 is the full pool; each subsequent
        # row drops one group).
        _budget_seconds = kwargs.get('budget_seconds')
        _trajectory_dir = kwargs.get('trajectory_dir')
        _clock = BudgetClock(_budget_seconds).start()
        _emitter = (TrajectoryEmitter(_trajectory_dir, algo='BackwardElimination')
                    if _trajectory_dir else None)
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

                model_instance = model(
                    joint_tuples.joint_count,
                    0,
                    **({'metric': self.metric} if model.__name__ == 'ClassificationCholesky' else {})
                )
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
        if _emitter is not None:
            _emitter.emit(0, _clock.elapsed_s, list(augmentation_plan))
        # Infer groups (columns sharing a base name before `^`). When no
        # polynomial/interaction naming is present, each group is a singleton
        # → behavior matches the legacy per-column elimination.
        active_groups = self._infer_feature_groups(aug_feature_indices)
        while (not early_stopping) and (new_features_count > 0) and len(active_groups) > 0:
            if _clock.expired:
                break
            task_calls = self._collect_tasks(task, model_instance, active_groups, **params)
            _t0 = _time.perf_counter()
            results, _eval_t = self._dispatch_evaluate(task_calls, n_jobs)
            _t1 = _time.perf_counter()
            _t_eval_total += _t1 - _t0
            _t_eval_compute_total += _eval_t['compute_s']
            _n_iters += 1
            feature_scores = {}
            for result_dict in results:
                feature_scores.update(result_dict)
            best_group_key = max(feature_scores, key=feature_scores.get)
            iter_score = feature_scores[best_group_key].item()
            early_stopping = self._stopping_criteria(iter_score, base_score, tol, new_features_count, task='elimination')

            if not early_stopping:
                # `best_group_key` is the first column index of the chosen group
                # (set by `_collect_tasks`); look up the full group tuple.
                best_group = next(g for g in active_groups if g[0] == best_group_key)
                for col_idx in best_group:
                    augmentation_plan.remove(aug_feature_indices[col_idx])
                new_features_idx = np.setdiff1d(new_features_idx, list(best_group))
                all_features_idx = np.setdiff1d(all_features_idx, list(best_group))
                params['all_features_idx'] = all_features_idx
                new_features_count -= len(best_group)
                active_groups = [g for g in active_groups if g != best_group]
                base_score = iter_score
                if _emitter is not None:
                    _emitter.emit(_n_iters, _clock.elapsed_s, list(augmentation_plan))
        aug_feature_indices = {k: v for k, v in zip(range(len(augmentation_plan)), augmentation_plan)}
        timing_breakdown = {
            'n_iters': _n_iters,
            'computation_s': round(_t_eval_compute_total, 4),
            'ray_overhead_s': round(_t_eval_total - _t_eval_compute_total, 4),
        }
        Result = namedtuple('JointTuple', ['aug_feature_indices', 'timing_breakdown'])

        return Result(aug_feature_indices, timing_breakdown)


    def _collect_tasks(self, task: str, model: FeatureSelectionModel, active_groups: list[tuple[int, ...]], **kwargs):
        task_calls = []
        for group in active_groups:
            group_list = list(group)
            indices = np.setdiff1d(kwargs['all_features_idx'], group_list)
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
            # Task key is the first column index of the group — lets the caller
            # recover the full group tuple after picking the best score.
            task_calls.append((
                '_evaluate_feature',
                (model, group[0], self.metric),
                task_params
            ))

        return task_calls
