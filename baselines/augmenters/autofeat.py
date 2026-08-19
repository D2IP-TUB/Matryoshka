import subprocess
import time
import os
import random
import warnings
import numpy as np
import pandas as pd
import polars as pl
from baselines.discovery.aurum_join_discovery import AurumJoinDiscovery
from .autofeat_utils.autofeat_pipeline.autofeat import AutoFeat as AutoFeatBase
from .autofeat_utils.autofeat_pipeline.evaluate_join_paths import evaluate_paths
from .autofeat_utils.autofeat_pipeline.neo4j_transactions import clear_df_cache
from matryoshka.selection.anytime import BudgetClock, TrajectoryEmitter
from matryoshka.utils.common import process_key

# Set PYTHONHASHSEED for deterministic hashing
os.environ['PYTHONHASHSEED'] = '42'


class AutofeatAugmenter:
    def run(
        self,
        join_paths_df_path: str,
        query_column_name: str,
        data_lake_path: str,
        base_table_sep: str,
        lake_table_sep: str,
        problem_type: str,
        base_node_id: str,
        target_column_name: str,
        features: list[str],
        base_table_label: str = 'base_table',
        save_joins_to_disk: bool = False,
        use_polars: bool = False,
        value_ratio: float = 0.5,
        top_k: int = 15,
        sample_size: int = 3000,
        pearson: bool = False,
        jmi: bool = False,
        no_relevance: bool = False,
        no_redundancy: bool = False,
        algorithm: str = "RF",
        **kwargs
    ):
        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)
        os.environ['PYTHONHASHSEED'] = '42'
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Realestate's natural query key (LONG_NAME) has no usable LSH
            # containment overlap in the NYC lake. ARDA, Kitana, AutoFeat,
            # and CAAFE instead key on the ZIP code extracted from STATE.
            # Matryoshka's forward path keeps the original key.
            if base_node_id == 'realestate.csv':
                query_column_name = 'zip_code'
            start = time.perf_counter()
            base_table_path = '/'.join(join_paths_df_path.split('/')[:3]) + '/' + base_node_id
            # subprocess.run(['cp', base_table_path, f'{data_lake_path}/{base_node_id}'])
            (
                pl.read_csv(base_table_path, separator=base_table_sep)
                .write_csv(f'{data_lake_path}/{base_node_id}')
            )
            _lake_parts = data_lake_path.rstrip('/').split('/')
            lake = _lake_parts[-2] if _lake_parts[-1] == 'extracted' else _lake_parts[-1]
            _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            aurum_index_file = os.path.join(_project_root, 'augmentation', 'Aurum', 'graphs', lake)
            if not os.path.isdir(aurum_index_file):
                aurum_index_file = os.path.join(_project_root, 'augmentation', 'Aurum', 'graphs', f'{lake}.pkl')
            aurum = AurumJoinDiscovery(aurum_index_file, separator=lake_table_sep)
            base_path = join_paths_df_path.split('/')[:-1]
            join_paths = aurum.find_joinable_tables(
                query_table_path=f'{"/".join(base_path)}/{base_node_id}',
                query_col=query_column_name,
                output_path=join_paths_df_path,
                features=[query_column_name]
            )
            join_paths_df = pd.read_csv(join_paths_df_path)
            join_paths_df['from_id'] = base_node_id
            autofeat = AutoFeatBase(
                join_paths_df=join_paths_df,
                lake_data_folder=data_lake_path,
                base_table_sep=base_table_sep,
                base_table_label=base_table_label,
                base_table_id=base_node_id,
                target_column=target_column_name,
                save_joins_to_disk=save_joins_to_disk,
                use_polars=use_polars,
                task=problem_type,
                value_ratio=value_ratio,
                top_k=top_k,
                sample_size=sample_size,
                pearson=pearson,
                jmi=jmi,
                no_relevance=no_relevance,
                no_redundancy=no_redundancy
            )
            # Anytime instrumentation. The BFS emits per-iteration trajectory
            # rows recording the cumulative pool of candidate features the
            # graph walk has retained so far. ``evaluate_paths`` is run once
            # after BFS terminates (cooperatively or because of an expired
            # budget); we append one final trajectory row carrying the
            # natural top-k selection so the right edge of the anytime curve
            # matches the un-budgeted endpoint.
            _budget_seconds = kwargs.get('budget_seconds')
            _trajectory_dir = kwargs.get('trajectory_dir')
            _clock = BudgetClock(_budget_seconds).start()
            _emitter = (TrajectoryEmitter(_trajectory_dir, algo='AutoFeat')
                        if _trajectory_dir else None)

            # Three-tier wall-clock cap. ``evaluate_paths`` is uncapped
            # upstream and previously ran 62+ minutes on fire after BFS had
            # already used 9. We layer three independent mechanisms so the
            # budget is honored even when one is suppressed by third-party
            # libraries (autogluon installs its own signal handlers, sklearnex
            # holds the GIL through long C calls).
            #   1. Cooperative: ``evaluate_paths`` polls ``budget_clock.expired``
            #      at the top of each per-path iteration and breaks early; any
            #      iterations already completed contribute their selected
            #      features to the augplan.
            #   2. Watchdog thread: a background ``threading.Timer`` flips
            #      ``budget_clock.expired`` at the deadline regardless of
            #      whether SIGALRM was overwritten. The cooperative loop then
            #      observes the flag at the next iteration boundary.
            #   3. SIGALRM at deadline * 1.1: redundant hard cap that raises
            #      ``_AutofeatBudgetExceeded`` if the main thread is still
            #      inside Python bytecode. We restore partial state from disk.
            import signal as _signal
            import tempfile as _tempfile
            import pickle as _pickle
            import threading as _threading
            class _AutofeatBudgetExceeded(Exception):
                pass
            def _on_timeout(_signum, _frame):
                raise _AutofeatBudgetExceeded(
                    f'AutoFeat hard budget exceeded ({_budget_seconds}s)')
            _prev_handler = None
            _watchdog = None
            _partial_state_path = None
            if _budget_seconds is not None:
                _prev_handler = _signal.signal(_signal.SIGALRM, _on_timeout)
                _signal.alarm(int(_budget_seconds * 1.1))
                # Watchdog flips ``_clock.expired`` exactly at the deadline,
                # which is what ``evaluate_paths`` polls. Daemon thread so it
                # never blocks process shutdown.
                def _on_deadline():
                    # ``BudgetClock.expired`` is a derived property; force it
                    # to evaluate True by collapsing the deadline. The slot
                    # itself is writable even though the property is not.
                    try:
                        _clock.seconds = 0
                    except Exception:
                        pass
                _watchdog = _threading.Timer(float(_budget_seconds), _on_deadline)
                _watchdog.daemon = True
                _watchdog.start()
                _partial_state_path = _tempfile.mktemp(
                    suffix='.pkl', prefix=f'autofeat_partial_{base_node_id}_'
                )

            def _bfs_fallback():
                _cumulative, _seen = [], set()
                for _feats in autofeat.partial_join_selected_features.values():
                    for _f in _feats:
                        if _f not in _seen:
                            _seen.add(_f)
                            _cumulative.append(_f)
                return _cumulative[:top_k], []

            def _recover_partial_state():
                # Read the last completed iteration's features from disk; fall
                # back to the BFS-discovered pool if nothing was persisted yet.
                if _partial_state_path and os.path.exists(_partial_state_path):
                    try:
                        with open(_partial_state_path, 'rb') as _f:
                            _state = _pickle.load(_f)
                        return (list(_state.get('selected_features') or []),
                                _state.get('top_k_path_list') or [])
                    except Exception:
                        pass
                return _bfs_fallback()

            try:
                autofeat.streaming_feature_selection(
                    join_paths_df, data_lake_path, lake_table_sep,
                    queue={base_node_id},
                    budget_clock=_clock, trajectory_emitter=_emitter,
                )
                # Soft check: if BFS itself exhausted the budget, skip
                # evaluate_paths entirely. The hard SIGALRM below catches the
                # case where evaluate_paths runs but goes long.
                if _budget_seconds is not None and _clock.expired:
                    final_selected_features, top_k_paths = _bfs_fallback()
                else:
                    _, top_k_paths, final_selected_features = evaluate_paths(
                        bfs_result=autofeat, problem_type=problem_type, algorithm=algorithm,
                        join_paths_df=join_paths_df,
                        lake_data_folder=data_lake_path,
                        lake_table_sep=lake_table_sep,
                        base_table_sep=base_table_sep,
                        budget_clock=_clock,
                        partial_state_path=_partial_state_path,
                    )
                    # If the cooperative break fired (budget_clock.expired
                    # observed at iter top) the loop returned empty selected
                    # features for the un-evaluated tail; merge with whatever
                    # the watchdog persisted from completed iterations.
                    if _budget_seconds is not None and _clock.expired and not final_selected_features:
                        final_selected_features, top_k_paths = _recover_partial_state()
            except _AutofeatBudgetExceeded:
                final_selected_features, top_k_paths = _recover_partial_state()
            finally:
                if _budget_seconds is not None:
                    _signal.alarm(0)
                    if _prev_handler is not None:
                        _signal.signal(_signal.SIGALRM, _prev_handler)
                    if _watchdog is not None:
                        _watchdog.cancel()
                    if _partial_state_path and os.path.exists(_partial_state_path):
                        try:
                            os.remove(_partial_state_path)
                        except Exception:
                            pass
            if _emitter is not None:
                # Final point: natural endpoint after the post-BFS top-k filter.
                _emitter.emit(
                    iter_idx=10_000,  # arbitrary high marker for the "final" row
                    t_elapsed_s=_clock.elapsed_s,
                    features=list(final_selected_features),
                )

            end = time.perf_counter()
            fetch_time = end - start

            start = time.perf_counter()
            final_selected_features_dict = {}
            left_table = pd.read_csv(base_table_path, sep=base_table_sep)
            augplan = []
            if len(final_selected_features) > 0:
                for v in final_selected_features:
                    file, col = v.split('.', 1)  # split only at the first '.'
                    # if filenames may contain dots, split from the right instead:
                    # file, col = v.rsplit('.', 1)
                    col = col.split('.', 1)[-1]
                    col = col.split('.', 1)[0]
                    final_selected_features_dict.setdefault(file, []).append(col)
                for table_name, features in final_selected_features_dict.items():
                    if '.csv' not in table_name:
                        table_name += '.csv'
                    try:
                        query_column_name = join_paths_df[join_paths_df["to_id"] == table_name]['from_column'].values[0]
                    except IndexError:
                        continue
                    left_table[query_column_name] = left_table[query_column_name].apply(process_key)
                    join_key = join_paths_df[
                        (join_paths_df["from_column"] == query_column_name) &
                        (join_paths_df["to_id"] == table_name)
                    ]['to_column'].values[0]
                    
                    right_table = pd.read_csv(
                        f'{data_lake_path}/{table_name}',
                        header=0,
                        engine="c",
                        encoding="utf8",
                        on_bad_lines='skip',
                        sep=lake_table_sep
                    )
                    # Drop duplicate column names (keep first occurrence)
                    right_table = right_table.loc[:, ~right_table.columns.duplicated()]
                    
                    right_table = right_table.groupby(join_key).sample(
                        n=1, random_state=42
                    )
                    right_table[join_key] = right_table[join_key].apply(process_key)
                    # Exclude join key from features to avoid duplicate columns
                    features = [f for f in features if f != join_key]
                    right_table = right_table[[join_key, *features]]
                    left_table = pd.merge(
                        left_table,
                        right_table,
                        how="left",
                        left_on=query_column_name,
                        right_on=join_key,
                        suffixes=('', '_right')
                    )
                    # Drop duplicate columns introduced by the merge
                    left_table = left_table.loc[:, ~left_table.columns.duplicated()]
                    augplan.extend(features)
            left_table_columns = left_table.columns.tolist()
            left_table_columns_non_unique = [col for col in left_table_columns if left_table_columns.count(col) > 1]
            left_table_columns_non_unique_renamed = [col + '_right' if col in left_table_columns_non_unique else col for col in left_table_columns]
            left_table.columns = left_table_columns_non_unique_renamed
            left_table.reset_index(drop=True, inplace=True)
            left_table = pl.from_pandas(left_table)
            end = time.perf_counter()
            augmentation_time = end - start
            subprocess.run(['rm', f'{data_lake_path}/{base_node_id}'])
            clear_df_cache()

        return left_table, fetch_time, augmentation_time, augplan