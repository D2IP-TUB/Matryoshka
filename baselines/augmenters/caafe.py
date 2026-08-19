"""CAAFE baseline.

Discovers joinable tables with Aurum, sample-and-joins each candidate onto the
buyer table to keep the working frame tractable, then runs CAAFE
(https://github.com/noahho/CAAFE) for LLM-driven feature engineering /
selection on the joined frame.

Notes:
- CAAFE generates Python code via an OpenAI-compatible LLM and executes it
    in-process under CAAFE's whitelist. Set ``OPENAI_API_KEY`` (and optionally
    ``OPENAI_BASE_URL``) before running.
- For regression tasks, the target is discretized into quantile bins only for
    CAAFE's internal classification loop; generated feature code is then applied
    to the original table with the original continuous target unchanged.
"""
import json
import os
import random
import re
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier

from baselines.discovery.aurum_join_discovery import AurumJoinDiscovery
from baselines._compat import PreProcessor
from .neo4j_join_discovery import (
    discover_neo4j_join_paths,
    neo4j_chain_data_lake,
)
from matryoshka.selection.anytime import BudgetClock, TrajectoryEmitter
from matryoshka.utils.common import process_key


# Soft column-budget multiplier: stop adding seller tables once
# ``augmented_df.shape[1] > top_k * COL_BUDGET_MULT``.
COL_BUDGET_MULT = 5


class CaafeAugmenter:
    def run(
        self,
        join_paths_df_path: str,
        base_node_id: str,
        query_column_name: str,
        query_table_path: str,
        features: list[str],
        target_column_name: str,
        data_lake_path: str,
        problem_type: str = 'binary',
        buyer_sep: str = ',',
        lake_table_sep: str = ',',
        sample_size: int = 3000,
        per_seller_sample: int = 2000,
        top_k: int = 15,
        iterations: int = 10,
        llm_model: str = 'gpt-4o-mini',
        n_splits: int = 10,
        n_repeats: int = 2,
        dataset_description: Optional[str] = None,
        splits_path: Optional[str] = None,
        multihop: bool = False,
        multihop_depth: int = 2,
        **kwargs,
    ):
        random.seed(42)
        np.random.seed(42)

        # Anytime instrumentation. CAAFE's outer iteration loop lives inside
        # ``caafe_clf.fit_pandas`` which has no per-iteration callback in the
        # upstream API. We budget the run at the wrapper level and emit two
        # trajectory points: empty plan at start, final plan after fit. The
        # net effect on the multi-budget curve is informative on its own.
        # When the budget is below CAAFE's first-LLM-call latency, the curve
        # stays at zero quality, which is exactly the contrast the experiment
        # wants to surface.
        _budget_seconds = kwargs.get('budget_seconds')
        _trajectory_dir = kwargs.get('trajectory_dir')
        _clock = BudgetClock(_budget_seconds).start()
        _emitter = (TrajectoryEmitter(_trajectory_dir, algo='CAAFE')
                    if _trajectory_dir else None)
        if _emitter is not None:
            _emitter.emit(0, 0.0, [])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Realestate's natural query key (LONG_NAME) has no usable LSH
            # containment overlap in the NYC lake. ARDA, Kitana, AutoFeat,
            # and CAAFE instead key on the ZIP code extracted from STATE.
            # Matryoshka's forward path keeps the original key.
            if base_node_id == 'realestate.csv':
                query_column_name = 'zip_code'
            start = time.perf_counter()

            # ----------------------------------------------------------------
            # Join discovery: Aurum (default) or Neo4j multi-hop.
            # ----------------------------------------------------------------
            if multihop:
                discover_neo4j_join_paths(
                    base_node_id=base_node_id,
                    data_lake_path=data_lake_path,
                    lake_table_sep=lake_table_sep,
                    output_path=join_paths_df_path,
                    max_depth=multihop_depth,
                )
                data_lake_path = neo4j_chain_data_lake(
                    join_paths_df_path, data_lake_path
                )
            else:
                _lake_parts = data_lake_path.rstrip('/').split('/')
                lake = _lake_parts[-2] if _lake_parts[-1] == 'extracted' else _lake_parts[-1]
                _project_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), '..', '..', '..')
                )
                aurum_index_file = os.path.join(
                    _project_root, 'augmentation', 'Aurum', 'graphs', lake
                )
                if not os.path.isdir(aurum_index_file):
                    aurum_index_file = os.path.join(
                        _project_root, 'augmentation', 'Aurum', 'graphs', f'{lake}.pkl'
                    )
                aurum = AurumJoinDiscovery(aurum_index_file, separator=lake_table_sep)
                base_path = join_paths_df_path.split('/')[:-1]
                aurum.find_joinable_tables(
                    query_table_path=f'{"/".join(base_path)}/{base_node_id}',
                    query_col=query_column_name,
                    output_path=join_paths_df_path,
                    features=[query_column_name],
                )
            join_paths_df = pd.read_csv(join_paths_df_path)

            # ----------------------------------------------------------------
            # Buyer-side preprocessing (mirrors kitana).
            # ----------------------------------------------------------------
            if splits_path is None:
                splits_path = (
                    f'experiments/base_tables/{base_node_id.split(".")[0]}/splits.json'
                )
            preprocessor = PreProcessor(query_table_path, splits_path, 0)
            buyer_df, _, _, _ = preprocessor.run()
            buyer_df = buyer_df.to_pandas()
            if buyer_df[target_column_name].dtype == 'object':
                unique_categories = buyer_df[target_column_name].unique()
                category_mapping = {c: i for i, c in enumerate(unique_categories)}
                buyer_df[target_column_name] = buyer_df[target_column_name].map(
                    category_mapping
                )
            if sample_size and len(buyer_df) > sample_size:
                buyer_df = buyer_df.sample(sample_size, random_state=42).reset_index(
                    drop=True
                )

            end = time.perf_counter()
            fetch_time = end - start

            # ----------------------------------------------------------------
            # Per-seller sample-and-join.
            # ----------------------------------------------------------------
            start = time.perf_counter()
            augmented_df = buyer_df.copy()
            base_cols = list(augmented_df.columns)   # query-table (retained base) columns
            join_keys = sorted(join_paths_df['from_column'].unique().tolist())
            for jk in join_keys:
                if jk in augmented_df.columns:
                    augmented_df[jk] = augmented_df[jk].apply(process_key)

            items = join_paths_df[['from_column', 'to_column']].drop_duplicates()
            schema_mapping = dict(zip(items['to_column'], items['from_column']))
            col_budget = max(int(top_k * COL_BUDGET_MULT), augmented_df.shape[1] + 1)

            lake_tables = sorted(join_paths_df['to_id'].unique().tolist())
            joined_feature_cols = []   # lake feature columns actually merged in
            for table in lake_tables:
                if augmented_df.shape[1] > col_budget:
                    break
                seller_path = f"{data_lake_path}/{table}"
                try:
                    header = pd.read_csv(
                        seller_path, sep=lake_table_sep, nrows=0
                    ).columns.tolist()
                except Exception:
                    continue
                seller_join_col = next(
                    (c for c in header if c in schema_mapping), None
                )
                if seller_join_col is None:
                    continue
                buyer_join_col = schema_mapping[seller_join_col]
                if buyer_join_col not in augmented_df.columns:
                    continue
                seller_feats = [c for c in header if c != seller_join_col]
                if not seller_feats:
                    continue
                try:
                    seller_df = pd.read_csv(
                        seller_path,
                        sep=lake_table_sep,
                        usecols=[seller_join_col, *seller_feats],
                    )
                except Exception:
                    continue
                seller_df[seller_join_col] = seller_df[seller_join_col].apply(
                    process_key
                )
                # Reduce to 1 row per join key to enforce M:1 joins.
                try:
                    seller_df = seller_df.groupby(seller_join_col).sample(
                        n=1, random_state=42
                    )
                except (ValueError, KeyError):
                    seller_df = seller_df.drop_duplicates(subset=[seller_join_col])
                if len(seller_df) > per_seller_sample:
                    seller_df = seller_df.sample(
                        per_seller_sample, random_state=42
                    )
                # Prefix non-key columns with table name to avoid clashes.
                stem = table.replace('.csv', '')
                rename_map = {
                    c: f'{stem}__{c}' for c in seller_df.columns if c != seller_join_col
                }
                seller_df = seller_df.rename(columns=rename_map)
                try:
                    augmented_df = augmented_df.merge(
                        seller_df,
                        how='left',
                        left_on=buyer_join_col,
                        right_on=seller_join_col,
                        suffixes=('', f'__{stem}'),
                    )
                except Exception:
                    continue
                # Only drop the seller key when buyer/seller key names differ;
                # otherwise the merge already collapsed them into one column.
                if seller_join_col != buyer_join_col:
                    try:
                        augmented_df.drop(
                            columns=[seller_join_col], inplace=True
                        )
                    except KeyError:
                        pass
                # Record the lake feature columns this join contributed (exclude
                # synthetic Key_* columns, which are join ids, not features).
                joined_feature_cols.extend(
                    v for v in rename_map.values()
                    if v in augmented_df.columns
                    and not str(v).rsplit('__', 1)[-1].startswith('Key_')
                )

            # Drop the FK column itself: CAAFE should not feature-engineer on
            # the join key (it has no semantics for the downstream task), and
            # leaving it in inflates the prompt without value. The buyer-side
            # query column is preserved verbatim so the §7.2 post-processing
            # pipeline can re-join CAAFE's selected features with the shared
            # preprocessed base on the original query key.
            for jk in join_keys:
                if (jk in augmented_df.columns
                        and jk != target_column_name
                        and jk != query_column_name):
                    augmented_df = augmented_df.drop(columns=[jk])

            end = time.perf_counter()
            join_time = end - start

            is_classification = problem_type in ('binary', 'classification', 'multiclass')
            caafe_task_type = 'classification' if is_classification else 'regression'
            continuous_target_for_eval = None

            # ----------------------------------------------------------------
            # Build dataset description.
            # ----------------------------------------------------------------
            if dataset_description is None:
                dataset_description = self._build_description(
                    base_node_id=base_node_id,
                    target_column_name=target_column_name,
                    augmented_df=augmented_df,
                    splits_path=splits_path,
                    task_type=caafe_task_type,
                    lake_feature_cols=joined_feature_cols,
                )

            # ----------------------------------------------------------------
            # CAAFE feature engineering loop.
            # ----------------------------------------------------------------
            start = time.perf_counter()
            from caafe import CAAFEClassifier
            from caafe.run_llm_code import run_llm_code
            import caafe.sklearn_wrapper as _csw
            import openai as _openai

            # Shim ``openai.ChatCompletion.create`` (removed in openai>=1.0)
            # using the modern client so legacy CAAFE keeps working.
            _cc = getattr(_openai, 'ChatCompletion', None)
            _is_removed_proxy = (
                _cc is None
                or type(_cc).__name__ == 'APIRemovedInV1Proxy'
                or 'lib._old_api' in type(_cc).__module__
            )
            if _is_removed_proxy and not getattr(_openai, '_fdd_chatcompletion_patched', False):
                # Prefer the native Gemini ``generateContent`` endpoint when
                # ``GEMINI_API_KEY`` is set: the ``/v1beta/openai/`` compat
                # endpoint advertises a much lower (broken) free-tier quota
                # than the native one. Fall back to the OpenAI SDK otherwise.
                import httpx as _httpx
                _gemini_key = os.environ.get('GEMINI_API_KEY')
                _openai_client = None
                if not _gemini_key:
                    _openai_client = _openai.OpenAI()

                def _call_gemini_native(model, messages, **kw):
                    # FDD_LLM_MODEL overrides the configured model at request
                    # time (e.g. when the configured paper-era model is not
                    # accessible to the active API key). experiments.csv keeps
                    # recording the original provenance.
                    model = os.environ.get('FDD_LLM_MODEL') or model
                    # Translate OpenAI-style ``messages`` into Gemini's
                    # ``contents`` (system messages → ``systemInstruction``).
                    sys_parts, contents = [], []
                    for m in messages:
                        role = m.get('role', 'user')
                        text = m.get('content', '') or ''
                        if role == 'system':
                            sys_parts.append({'text': text})
                            continue
                        contents.append({
                            'role': 'model' if role == 'assistant' else 'user',
                            'parts': [{'text': text}],
                        })
                    payload = {'contents': contents}
                    if sys_parts:
                        payload['systemInstruction'] = {'parts': sys_parts}
                    gen_cfg = {}
                    if 'temperature' in kw:
                        gen_cfg['temperature'] = kw['temperature']
                    # gemini-2.5-* spend output tokens on hidden reasoning. With
                    # CAAFE's small request (max_tokens=500) the reasoning eats
                    # the whole budget and the code block is truncated to a few
                    # comment lines (no executable code -> 0 improvement). Disable
                    # thinking for this structured code-gen task and keep a
                    # generous output budget so the full block is emitted.
                    out_tokens = int(kw.get('max_tokens', 1024) or 1024)
                    gen_cfg['maxOutputTokens'] = max(out_tokens, 2048)
                    if '2.5' in str(model):
                        gen_cfg['thinkingConfig'] = {'thinkingBudget': 0}
                    if 'stop' in kw and kw['stop']:
                        gen_cfg['stopSequences'] = (
                            kw['stop'] if isinstance(kw['stop'], list) else [kw['stop']]
                        )
                    if gen_cfg:
                        payload['generationConfig'] = gen_cfg

                    url = (
                        f'https://generativelanguage.googleapis.com/v1beta/'
                        f'models/{model}:generateContent?key={_gemini_key}'
                    )
                    r = _httpx.post(url, json=payload, timeout=120)
                    r.raise_for_status()
                    data = r.json()
                    text = ''
                    for cand in data.get('candidates', []):
                        for p in cand.get('content', {}).get('parts', []) or []:
                            if 'text' in p:
                                text += p['text']
                        break
                    return text

                class _ChatCompletionShim:
                    _last_call_ts = 0.0

                    @staticmethod
                    def create(**kw):
                        min_interval = float(os.environ.get('FDD_LLM_MIN_INTERVAL', '4.5'))
                        elapsed = time.time() - _ChatCompletionShim._last_call_ts
                        if elapsed < min_interval:
                            time.sleep(min_interval - elapsed)

                        max_attempts = int(os.environ.get('FDD_LLM_MAX_RETRIES', '6'))
                        delay = float(os.environ.get('FDD_LLM_INITIAL_BACKOFF', '5'))
                        last_err = None
                        content = None
                        for attempt in range(max_attempts):
                            try:
                                if _gemini_key:
                                    content = _call_gemini_native(**kw)
                                else:
                                    resp = _openai_client.chat.completions.create(**kw)
                                    content = resp.choices[0].message.content
                                break
                            except Exception as e:
                                msg = str(e)
                                last_err = e
                                is_rate = (
                                    getattr(e, 'status_code', None) == 429
                                    or '429' in msg
                                    or 'RESOURCE_EXHAUSTED' in msg
                                    or 'rate limit' in msg.lower()
                                )
                                if not is_rate or attempt == max_attempts - 1:
                                    break
                                m = re.search(r'retryDelay["\']?\s*:\s*["\']?(\d+)', msg)
                                wait = int(m.group(1)) if m else delay
                                wait += random.uniform(0, 1.0)
                                print(
                                    f"[caafe] Gemini 429; retry {attempt + 1}/{max_attempts} "
                                    f"in {wait:.1f}s",
                                    flush=True,
                                )
                                time.sleep(wait)
                                delay = min(delay * 2, 60)

                        _ChatCompletionShim._last_call_ts = time.time()

                        # Upstream CAAFE infinite-loops on exceptions (it
                        # ``continue``s without incrementing the counter).
                        # Return an empty choice so the loop advances.
                        if content is None:
                            print(
                                f"[caafe] LLM call failed after {max_attempts} attempts: "
                                f"{last_err!r}; skipping iteration.",
                                flush=True,
                            )
                            content = ''

                        # Gemini often emits the stop-sequence token ``end``
                        # as a bare trailing line (`` ``` `` + newline + ``end``
                        # rather than ``\`\`\`end`` on one line), which CAAFE's
                        # cleanup leaves as a standalone ``end`` statement that
                        # causes NameError during exec.  Strip it here before
                        # returning to CAAFE.
                        if content:
                            _lines = content.rstrip().split('\n')
                            while _lines and _lines[-1].strip() in (
                                'end', '```end', '```',
                            ):
                                _lines.pop()
                            content = '\n'.join(_lines)

                        return {
                            'choices': [
                                {
                                    'message': {'role': 'assistant', 'content': content},
                                    'finish_reason': 'stop',
                                    'index': 0,
                                }
                            ],
                        }

                _openai.ChatCompletion = _ChatCompletionShim
                _openai._fdd_chatcompletion_patched = True

            # Force CAAFE to use plain ``print`` instead of IPython's
            # ``display(Markdown(...))``, which leaks ``<IPython.core.display
            # .Markdown object>`` lines outside a notebook.
            if not getattr(_csw, '_fdd_print_patched', False):
                _orig_generate_features = _csw.generate_features

                def _generate_features_print(*args, **kw):
                    kw['display_method'] = 'print'
                    return _orig_generate_features(*args, **kw)

                _csw.generate_features = _generate_features_print
                _csw._fdd_print_patched = True

            # CAAFE's ``evaluate_dataset`` calls ``tabpfn.scripts.tabular_metrics``
            # which does not exist in tabpfn >= 7.x, and invokes ``y.long()``
            # (a PyTorch method) even for plain sklearn estimators. Replace it
            # with a task-aware sklearn evaluator.
            import caafe.caafe_evaluate as _caafe_eval
            import caafe.caafe as _caafe_core
            if not getattr(_caafe_eval, '_fdd_evaluate_patched', False):
                from sklearn.metrics import (
                    accuracy_score,
                    roc_auc_score,
                    mean_absolute_error,
                    mean_squared_error,
                )

                def _sklearn_evaluate_dataset(
                    df_train, df_test, prompt_id, name, method,
                    metric_used, target_name, max_time=300, seed=0,
                ):
                    import copy
                    from caafe.data import get_X_y
                    from caafe.preprocessing import (
                        make_datasets_numeric, make_dataset_numeric,
                    )
                    from sklearn.base import BaseEstimator
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.linear_model import LogisticRegression

                    df_train, df_test = copy.deepcopy(df_train), copy.deepcopy(df_test)
                    df_train, _, mappings = make_datasets_numeric(
                        df_train, None, target_name, return_mappings=True
                    )
                    df_test = make_dataset_numeric(df_test, mappings=mappings)

                    test_x, test_y = get_X_y(df_test, target_name=target_name)
                    x, y = get_X_y(df_train, target_name=target_name)

                    def _to_np(a):
                        if hasattr(a, 'numpy'):
                            return a.numpy()
                        import numpy as _np
                        return _np.asarray(a)

                    task_is_classification = bool(
                        getattr(_caafe_eval, '_fdd_task_is_classification', True)
                    )

                    if task_is_classification:
                        x, y = _to_np(x), _to_np(y).astype(int)
                        test_x, test_y = _to_np(test_x), _to_np(test_y).astype(int)

                        if isinstance(method, BaseEstimator):
                            clf = method
                        else:
                            clf = LogisticRegression(max_iter=200, random_state=seed)

                        clf.fit(x, y)
                        ys = clf.predict_proba(test_x)

                        y_pred = ys.argmax(axis=1)
                        acc = float(accuracy_score(test_y, y_pred))

                        n_classes = ys.shape[1]
                        try:
                            if n_classes == 2:
                                roc = float(roc_auc_score(test_y, ys[:, 1]))
                            else:
                                roc = float(roc_auc_score(
                                    test_y, ys, multi_class='ovr', average='macro'
                                ))
                        except Exception:
                            roc = acc

                        method_str = method if isinstance(method, str) else 'sklearn'
                        return {
                            'acc': acc,
                            'roc': roc,
                            'prompt': prompt_id,
                            'seed': seed,
                            'name': name,
                            'size': len(df_train),
                            'method': method_str,
                            'max_time': max_time,
                            'feats': x.shape[-1],
                        }

                    # Regression path: score feature code by minimizing loss.
                    x = _to_np(x).astype(float)
                    test_x = _to_np(test_x).astype(float)

                    target_series = getattr(_caafe_eval, '_fdd_continuous_target_for_eval', None)
                    if target_series is not None:
                        y = target_series.reindex(df_train.index).to_numpy(dtype=float)
                        test_y = target_series.reindex(df_test.index).to_numpy(dtype=float)
                    else:
                        y = _to_np(y).astype(float)
                        test_y = _to_np(test_y).astype(float)

                    train_valid = np.isfinite(y)
                    test_valid = np.isfinite(test_y)
                    x, y = x[train_valid], y[train_valid]
                    test_x, test_y = test_x[test_valid], test_y[test_valid]

                    if len(y) < 5 or len(test_y) < 2:
                        return {
                            'acc': -1e12,
                            'roc': -1e12,
                            'prompt': prompt_id,
                            'seed': seed,
                            'name': name,
                            'size': len(df_train),
                            'method': 'regression_invalid',
                            'max_time': max_time,
                            'feats': x.shape[-1] if x.ndim == 2 else 0,
                            'mse': float('inf'),
                            'mae': float('inf'),
                        }

                    reg = RandomForestRegressor(
                        n_estimators=120,
                        max_depth=6,
                        random_state=seed,
                        n_jobs=-1,
                    )
                    reg.fit(x, y)
                    pred = reg.predict(test_x)

                    mse = float(mean_squared_error(test_y, pred))
                    mae = float(mean_absolute_error(test_y, pred))

                    return {
                        'acc': -mse,
                        'roc': -mae,
                        'prompt': prompt_id,
                        'seed': seed,
                        'name': name,
                        'size': len(df_train),
                        'method': 'regression_rf',
                        'max_time': max_time,
                        'feats': x.shape[-1],
                        'mse': mse,
                        'mae': mae,
                    }

                _caafe_eval.evaluate_dataset = _sklearn_evaluate_dataset
                _caafe_core.evaluate_dataset = _sklearn_evaluate_dataset
                _caafe_eval._fdd_evaluate_patched = True

            # Update task context on every run (important in mixed clf/reg sessions).
            _caafe_eval._fdd_task_is_classification = bool(is_classification)
            _caafe_eval._fdd_continuous_target_for_eval = continuous_target_for_eval

            # Coerce all non-target columns to numeric so the RandomForest base
            # classifier inside CAAFE's CV loop never sees object dtypes.
            caafe_df = augmented_df.copy()
            for c in caafe_df.columns:
                if c == target_column_name:
                    continue
                if caafe_df[c].dtype == 'object':
                    caafe_df[c] = pd.to_numeric(caafe_df[c], errors='coerce')
                caafe_df[c] = caafe_df[c].fillna(caafe_df[c].mean())
            caafe_df = caafe_df.dropna(subset=[target_column_name])

            # CAAFE natively supports classification only. For regression tasks,
            # bin the continuous target into quantiles for CAAFE fitting while
            # keeping the original target in the returned dataframe.
            if not is_classification:
                y = pd.to_numeric(caafe_df[target_column_name], errors='coerce')
                valid_mask = y.notna()
                caafe_df = caafe_df.loc[valid_mask].copy()
                y = y.loc[valid_mask]

                if len(caafe_df) < 10 or float(y.nunique()) < 2:
                    warnings.warn(
                        'CAAFE regression fallback skipped: insufficient target variance.'
                    )
                    augmentation_time = join_time
                    base_only = augmented_df[[c for c in base_cols if c in augmented_df.columns]]
                    return pl.from_pandas(base_only), fetch_time, augmentation_time, []

                n_bins = int(kwargs.get('regression_bins', 5))
                n_bins = max(2, min(n_bins, int(y.nunique())))
                try:
                    y_binned = pd.qcut(y, q=n_bins, labels=False, duplicates='drop')
                except ValueError:
                    # Fallback when quantile cuts are unstable (e.g., many ties).
                    y_binned = (y >= y.median()).astype(int)

                y_binned = pd.Series(y_binned, index=caafe_df.index).astype(float)
                valid_bins = y_binned.notna()
                caafe_df = caafe_df.loc[valid_bins].copy()
                y_binned = y_binned.loc[valid_bins].astype(int)
                continuous_target_for_eval = y.loc[valid_bins].astype(float)
                _caafe_eval._fdd_continuous_target_for_eval = continuous_target_for_eval

                if int(y_binned.nunique()) < 2:
                    warnings.warn(
                        'CAAFE regression fallback skipped: binned target collapsed to one class.'
                    )
                    augmentation_time = join_time
                    base_only = augmented_df[[c for c in base_cols if c in augmented_df.columns]]
                    return pl.from_pandas(base_only), fetch_time, augmentation_time, []

                caafe_df[target_column_name] = y_binned

            # CAAFE's author-default base classifier is TabPFN, but tabpfn>=7
            # requires a one-time license/token download that fails in this
            # non-interactive environment. CAAFE itself recommends a plain
            # RandomForest as the alternative, so use it with scikit-learn
            # DEFAULTS (notably max_depth=None) rather than an artificially
            # shallow model, to keep the evaluator principled and able to
            # register genuine gains from engineered/selected features.
            base_clf = RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            )
            caafe_clf = CAAFEClassifier(
                base_classifier=base_clf,
                llm_model=llm_model,
                iterations=iterations,
                optimization_metric='accuracy',
                n_splits=n_splits,
                n_repeats=n_repeats,
            )
            _pre_llm_cols = set(augmented_df.columns)
            llm_new = []
            code = ''
            try:
                caafe_clf.fit_pandas(
                    caafe_df,
                    dataset_description=dataset_description,
                    target_column_name=target_column_name,
                )
                code = caafe_clf.code or ''
                if code.strip():
                    augmented_df = run_llm_code(code, augmented_df.copy())
                    llm_new = [c for c in augmented_df.columns
                               if c not in _pre_llm_cols and c != target_column_name]
            except Exception as e:
                warnings.warn(f'CAAFE feature engineering failed: {e!r}')

            # CAAFE's augmentation = the lake features it actually picked for
            # engineering, i.e. those referenced in its kept LLM code, plus any
            # columns that code engineered. Word-boundary matching avoids prefix
            # collisions (e.g. ``D13`` matching ``D130``). If CAAFE kept no
            # engineering, it picked nothing -> empty plan -> scored as base.
            picked_lake = [
                c for c in joined_feature_cols
                if c in augmented_df.columns
                and re.search(r'\b' + re.escape(c) + r'\b', code)
            ]
            augplan = picked_lake + [c for c in llm_new if c not in picked_lake]

            # Restrict the returned frame to base + picked. The anytime evaluator
            # keeps base + augplan[:k] by dropping only augplan[k:], so any joined
            # lake column left outside the plan would leak into the score.
            keep_cols = [c for c in base_cols if c in augmented_df.columns]
            keep_cols += [c for c in augplan
                          if c in augmented_df.columns and c not in keep_cols]
            augmented_df = augmented_df[keep_cols]

            end = time.perf_counter()
            augmentation_time = join_time + (end - start)

        if _emitter is not None:
            _emitter.emit(1, _clock.elapsed_s, augplan)

        return pl.from_pandas(augmented_df), fetch_time, augmentation_time, augplan

    # ------------------------------------------------------------------
    @staticmethod
    def _build_description(
        base_node_id: str,
        target_column_name: str,
        augmented_df: pd.DataFrame,
        splits_path: str,
        task_type: Optional[str] = None,
        lake_feature_cols: Optional[list] = None,
    ) -> str:
        """Build a textual dataset description for the CAAFE prompt.

        Starts from ``description.txt`` next to ``splits.json`` (or a synthesized
        base description) and then **foregrounds the joined lake features**. The
        curated description covers only the base table, so without this the LLM
        engineers from base columns and never touches the auxiliary columns that
        actually carry the signal. The synthetic foreign-key sentence is dropped
        because the ``Key_*`` columns are removed from the frame before CAAFE
        runs, so it only distracts the model into trying to drop them.
        """
        lake_set = set(lake_feature_cols or [])

        # 1. Base description: prefer a curated description.txt, else synthesize.
        base_desc = None
        desc_path = os.path.join(os.path.dirname(splits_path), 'description.txt')
        if os.path.isfile(desc_path):
            with open(desc_path, 'r') as f:
                text = f.read().strip()
            if text:
                base_desc = text

        if base_desc is None:
            target_type = 'classification'
            try:
                with open(splits_path, 'r') as f:
                    splits = json.load(f)
                target_type = splits[0].get('target_type', 'classification')
            except (FileNotFoundError, KeyError, IndexError, json.JSONDecodeError):
                pass
            inferred_task = task_type or (
                'regression' if target_type in ('continuous', 'regression')
                else 'classification'
            )
            dataset_name = base_node_id.split('.')[0]
            base_cols = [c for c in augmented_df.columns
                         if c != target_column_name and c not in lake_set]
            col_summary = ', '.join(
                f"{c} ({augmented_df[c].dtype})" for c in base_cols[:40]
            )
            if len(base_cols) > 40:
                col_summary += f", ... ({len(base_cols) - 40} more)"
            base_desc = (
                f"Dataset '{dataset_name}'. Task: {inferred_task}. "
                f"Predict the column '{target_column_name}' from the base table. "
                f"Base columns: {col_summary}."
            )

        # 2. Drop the synthetic foreign-key sentence (Key_* already removed).
        base_desc = re.sub(
            r"\s*The column '[^']*' is a synthetic foreign key[^.]*\.", '', base_desc
        ).strip()

        # 3. Foreground the joined lake features so the LLM engineers from them.
        lake_cols = [c for c in (lake_feature_cols or []) if c in augmented_df.columns]
        if lake_cols:
            shown = ', '.join(lake_cols[:30])
            if len(lake_cols) > 30:
                shown += f", ... (+{len(lake_cols) - 30} more)"
            base_desc += (
                f" In addition to the base columns, this dataframe has been augmented "
                f"with {len(lake_cols)} columns joined from auxiliary tables discovered "
                f"in a data lake; their names are prefixed with the source-table id "
                f"(e.g. '{lake_cols[0]}'). These joined columns are the main candidate "
                f"predictors for '{target_column_name}': some are informative and some "
                f"are noise, and the underlying feature names are anonymized. Build "
                f"useful combinations, ratios, and aggregations of these joined columns, "
                f"and drop joined columns that do not improve the classifier. Joined "
                f"columns: {shown}."
            )

        return base_desc
