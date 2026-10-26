"""The evaluation protocol established in the paper.

This module reproduces how the paper's experiments (Section 7) prepare query
tables, prepare augmented tables for training, guard against target leakage
and score the downstream model. It is the reference for comparing against the
published numbers; :func:`matryoshka.preprocessing.prepare_query_table` remains
the lightweight default for using Matryoshka on its own.

The protocol has four steps.

1. :func:`prepare_query_table_paper` turns a raw query table into the layout
   discovery expects, under one of the paper's feature setups: rows with a
   missing join key or target are dropped, categorical features are imputed
   with the mode and numeric features with an iterative Bayesian Ridge
   imputer, AutoGluon's ``AutoMLPipelineFeatureGenerator`` encodes the
   features, and classification targets are ordinal-encoded.
2. :func:`prepare_augmented_table` readies the augmented table for training:
   values beyond ``1e17`` are masked as missing, numeric columns are imputed
   with an iterative imputer around a histogram gradient boosting regressor,
   and remaining rows with missing values are dropped.
3. :func:`detect_target_leakage` removes numeric columns that leak the target:
   any column with absolute Pearson correlation above ``0.99``, and any set of
   columns of which the target is a near-exact linear combination (OLS
   ``R^2 > 0.999``).
4. :func:`score_random_forest` tunes ``max_depth`` and ``min_samples_leaf`` of
   a random forest with successive-halving grid search, then reports 5-fold
   cross-validated RMSE and ``R^2`` for regression, accuracy and weighted F1
   for multiclass and accuracy, F1 and ROC AUC for binary classification.

:func:`evaluate_paper_protocol` chains steps 2 to 4.

The feature setups of :func:`prepare_query_table_paper` are those of the
paper: ``join_key`` keeps only the join key, ``categorical`` the categorical
features, ``binned`` all features with numeric ones quantile-binned into four
buckets and dummy-coded, and ``full`` all features as they are.

AutoGluon and pandas are optional dependencies, installed with the ``paper``
extra (``pip install -e ".[paper]"``). The paper's setup uses AutoGluon
1.4.0, scikit-learn 1.7.2 and the random forests of scikit-learn-intelex
2025.10 (``intelex=True``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import polars as pl

from .utils.common import process_key

if TYPE_CHECKING:
    import pandas as pd

PAPER_SETUPS = ('join_key', 'categorical', 'binned', 'full')

# Columns whose magnitude exceeds this bound are missing-category sentinels
# (int64 minimum) or overflowed aggregates, not values.
_MAX_ABS_VALUE = 1e17

_NUMERIC_DTYPES = (
    pl.Float32, pl.Float64,
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
)

_RF_PARAM_GRID = {'max_depth': [5, 10, 20], 'min_samples_leaf': [10, 20, 50]}


def _require_paper_extra():
    try:
        import pandas  # noqa: F401
        from autogluon.features.generators import AutoMLPipelineFeatureGenerator
    except ImportError as e:
        raise ImportError(
            'The paper evaluation protocol needs AutoGluon and pandas; '
            'install them with `pip install -e ".[paper]"`'
        ) from e

    return AutoMLPipelineFeatureGenerator


def _feature_generator():
    AutoMLPipelineFeatureGenerator = _require_paper_extra()

    return AutoMLPipelineFeatureGenerator(
        enable_text_special_features=False,
        enable_text_ngram_features=False,
        verbosity=0,
    )


def _bayesian_ridge_imputer():
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.linear_model import BayesianRidge

    return IterativeImputer(estimator=BayesianRidge(), sample_posterior=False, random_state=42, n_nearest_features=None)


def _gradient_boosting_imputer():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    return IterativeImputer(
        estimator=HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.1, random_state=42),
        sample_posterior=False,
        random_state=42,
        n_nearest_features=10,
        max_iter=5,
        tol=1e-2,
    ).set_output(transform='pandas')


def prepare_query_table_paper(
    table: pl.DataFrame,
    key: str,
    target: str,
    *,
    categorical_features: Iterable[str],
    numeric_features: Iterable[str] = (),
    task: str = 'classification',
    setup: str = 'categorical',
    n_bins: int = 4,
) -> pl.DataFrame:
    """Prepare a query table as the paper's experiments do.

    Parameters
    ----------
    table
        Raw query table.
    key
        Join column. It is kept as the first column, cast to ``String`` and
        normalised with :func:`matryoshka.utils.common.process_key`; if it is
        also listed as a categorical feature, its encoding is a feature too.
    target
        Prediction target, kept as the last column. Ordinal-encoded (classes
        in sorted order) for classification.
    categorical_features, numeric_features
        Feature columns of the query table, by type. The ``Features`` column
        of the paper's dataset table counts both.
    task
        ``'classification'`` or ``'regression'``.
    setup
        One of :data:`PAPER_SETUPS`.
    n_bins
        Quantile bins per numeric feature in the ``binned`` setup.

    Returns
    -------
    pl.DataFrame
        Key column first, numeric feature columns named ``cat__<feature>``
        (AutoGluon may derive several from one feature, e.g.
        ``cat__<date>.year``), and the target last.

    Notes
    -----
    ``join_key`` keeps the columns whose name contains the key or the target
    name, so a feature whose name contains the target name
    (``sidewalk_condition`` for the target ``condition`` of Street Trees)
    stays in that setup.
    """
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    if setup not in PAPER_SETUPS:
        raise ValueError(f'setup must be one of {PAPER_SETUPS}, got {setup!r}')
    if task not in ('classification', 'regression'):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    categorical_cols = list(dict.fromkeys(categorical_features))
    numeric_cols = [c for c in dict.fromkeys(numeric_features) if c not in categorical_cols]
    skip_numeric = setup == 'categorical'
    features = categorical_cols if skip_numeric else categorical_cols + numeric_cols
    for name in [key, target] + features:
        if name not in table.columns:
            raise KeyError(f'column {name!r} is not in the query table')
    if not features:
        raise ValueError('the paper protocol needs at least one feature column')

    df = table.drop_nulls(subset=[key, target])
    for name in (key, target):
        if df.schema[name] in (pl.Float32, pl.Float64):
            df = df.filter(pl.col(name).is_not_nan())

    if task == 'classification':
        encoder = OrdinalEncoder()
        y = encoder.fit_transform(df.select(target).to_pandas()).ravel()
        y = pl.Series(name=target, values=y)
    else:
        y = df.get_column(target)

    X = df.select(features).to_pandas().replace({None: np.nan})
    transformers = [('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent'))]), categorical_cols)]
    if not skip_numeric and numeric_cols:
        transformers.append(('num', Pipeline([('imputer', _bayesian_ridge_imputer())]), numeric_cols))
    imputed = ColumnTransformer(transformers).set_output(transform='pandas').fit_transform(X)
    imputed.columns = [c.replace('num__', '', 1).replace('cat__', '', 1) for c in imputed.columns]
    imputed = imputed[features]

    generator = _feature_generator()
    generator.fit(imputed)
    encoded = generator.transform(imputed)
    feature_names = [f'cat__{name}' for name in generator.feature_metadata.get_features()]
    encoded = pd.DataFrame(np.asarray(encoded.values, dtype=float), columns=feature_names)
    encoded = _bayesian_ridge_imputer().set_output(transform='pandas').fit_transform(encoded)

    out = pl.DataFrame(encoded.to_numpy(), schema=feature_names, orient='row')
    out.insert_column(out.width, y)
    out.insert_column(0, df.get_column(key))
    out = out.with_columns(pl.col(key).cast(pl.String).map_elements(process_key, return_dtype=pl.String))

    if setup == 'join_key':
        out = out.select([c for c in out.columns if key in c or target in c])
    elif setup == 'binned':
        middle = out.columns[1:-1]
        binned = [
            pl.col(c).qcut(n_bins, labels=[str(i) for i in range(n_bins)], allow_duplicates=True).alias(c)
            if out.schema[c] in _NUMERIC_DTYPES else pl.col(c)
            for c in middle
        ]
        out = out.select([pl.col(key)] + binned + [pl.col(target)])
        out = pl.concat([
            out.select(key),
            out.select(middle).to_dummies(drop_first=True).cast(pl.Float64),
            out.select(target),
        ], how='horizontal')

    oversized = [c for c in out.columns[1:-1] if out.get_column(c).max() > _MAX_ABS_VALUE]

    return out.drop(oversized)


def prepare_augmented_table(
    augmented: 'pl.DataFrame | pd.DataFrame',
    key: str,
    target: str,
    *,
    mask_sentinels: bool = True,
) -> 'pd.DataFrame':
    """Ready an augmented table for training, as the paper's evaluation does.

    The join key is dropped. Numeric columns are imputed with an iterative
    imputer around a histogram gradient boosting regressor; if the table
    still has non-numeric columns, they are imputed with the mode and encoded
    with AutoGluon's feature generator first. Rows with remaining missing
    values are dropped.

    ``mask_sentinels`` masks numeric values beyond ``1e17`` as missing before
    imputation.
    """
    _require_paper_extra()
    import pandas as pd
    from sklearn.impute import SimpleImputer

    X = augmented if isinstance(augmented, pl.DataFrame) else pl.from_pandas(augmented)
    if target not in X.columns:
        raise KeyError(f'column {target!r} is not in the augmented table')
    others = [c for c in X.columns if c not in (key, target)]
    numeric_cols = [c for c in others if X.schema[c] in _NUMERIC_DTYPES]
    categorical_cols = [c for c in others if c not in numeric_cols]

    X = X.to_pandas().drop(columns=[key, 'Unnamed: 0'], errors='ignore').replace([np.inf, -np.inf], np.nan)
    if mask_sentinels and numeric_cols:
        X[numeric_cols] = X[numeric_cols].mask(X[numeric_cols].abs() > _MAX_ABS_VALUE)
    if categorical_cols:
        X[categorical_cols] = SimpleImputer(strategy='most_frequent').set_output(transform='pandas').fit_transform(X[categorical_cols])
        if numeric_cols:
            X[numeric_cols] = _gradient_boosting_imputer().fit_transform(X[numeric_cols])
        X = _feature_generator().fit_transform(X)
        # The feature generator can introduce missing values, e.g. for
        # high-cardinality text treated as a category.
        for col in X.columns[X.isna().any()]:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].median())
            else:
                mode = X[col].mode(dropna=True)
                fill = mode.iloc[0] if len(mode) else 'missing'
                if hasattr(X[col], 'cat') and fill not in X[col].cat.categories:
                    X[col] = X[col].cat.add_categories([fill])
                X[col] = X[col].fillna(fill)
    else:
        X = _gradient_boosting_imputer().fit_transform(X)

    return X.dropna()


def detect_target_leakage(
    X: 'pd.DataFrame',
    target: str,
    task: str,
    *,
    corr_threshold: float = 0.99,
    r2_threshold: float = 0.999,
) -> list[str]:
    """Return the numeric columns of ``X`` that leak ``target``.

    Two criteria are evaluated in a loop over the remaining columns, dropping
    columns until neither holds:

    1. a single column whose absolute Pearson correlation with the target
       exceeds ``corr_threshold``: the most correlated one is dropped;
    2. the OLS fit of the target on all remaining columns reaches
       ``R^2 > r2_threshold``: every column whose standardised coefficient is
       above 1% of the largest is dropped. This catches additive identities
       such as ``Global_Sales = NA_Sales + EU_Sales + JP_Sales + Other_Sales``,
       whose components are only moderately correlated with the sum one by
       one.

    Classification targets are compared by their integer class codes.
    """
    import pandas as pd

    if target not in X.columns:
        return []
    try:
        y = pd.Categorical(X[target]).codes.astype(float) if task == 'classification' else X[target].astype(float)
    except (TypeError, ValueError):
        return []
    y = np.asarray(y, dtype=float)
    if not np.isfinite(y).any() or float(np.nanstd(y)) == 0.0:
        return []
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0.0:
        return []

    col_data: dict[str, np.ndarray] = {}
    for c in X.columns:
        if c == target or not pd.api.types.is_numeric_dtype(X[c]):
            continue
        col = X[c].astype(float).to_numpy()
        if not np.isfinite(col).any() or float(np.nanstd(col)) == 0.0:
            continue
        col_data[c] = np.where(np.isnan(col), np.nanmean(col), col)

    leaky: list[str] = []
    remaining = list(col_data)
    while remaining:
        corrs = []
        for c in remaining:
            with np.errstate(invalid='ignore', divide='ignore'):
                r = float(np.corrcoef(col_data[c], y)[0, 1])
            corrs.append(abs(r) if np.isfinite(r) else 0.0)
        top = int(np.argmax(corrs))
        if corrs[top] > corr_threshold:
            leaky.append(remaining.pop(top))
            continue
        Xn = np.column_stack([col_data[c] for c in remaining])
        design = np.c_[np.ones(len(Xn)), Xn]
        try:
            coefs, *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            break
        r2 = 1.0 - float(np.sum((y - design @ coefs) ** 2)) / ss_tot
        if r2 <= r2_threshold:
            break
        impact = np.abs(coefs[1:]) * Xn.std(axis=0)
        if not np.isfinite(impact).any() or impact.max() <= 0:
            break
        participating = [i for i in range(len(remaining)) if impact[i] > 0.01 * float(impact.max())]
        if not participating:
            break
        dropped = [remaining.pop(i) for i in sorted(participating, reverse=True)]
        leaky.extend(reversed(dropped))

    return leaky


def score_random_forest(
    X: 'pd.DataFrame',
    target: str,
    task: str,
    *,
    cv_folds: int = 5,
    n_jobs: int = 8,
    intelex: bool = False,
) -> dict[str, float]:
    """Score ``X`` with the paper's random forest pipeline.

    ``max_depth`` and ``min_samples_leaf`` are chosen by successive-halving
    grid search (3-fold, factor 3; plain grid search on tables too small for
    halving) over a 100-tree forest. A 50-tree forest with those parameters
    and ``max_features='log2'`` is then cross-validated over ``cv_folds``
    shuffled folds, stratified for classification.

    Returns ``{'r2', 'rmse'}`` for regression, ``{'accuracy', 'f1_weighted'}``
    for multiclass and ``{'accuracy', 'f1', 'roc_auc'}`` for binary
    classification, averaged over the folds (RMSE is the root of the mean
    squared error averaged over the folds).

    ``intelex`` uses the random forests of scikit-learn-intelex, as the
    paper's setup does; they are not numerically identical to
    scikit-learn's.
    """
    if intelex:
        try:
            from sklearnex.ensemble import RandomForestClassifier, RandomForestRegressor
        except ImportError as e:
            raise ImportError('intelex=True needs scikit-learn-intelex; `pip install scikit-learn-intelex`') from e
    else:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, KFold, StratifiedKFold

    if task not in ('classification', 'regression'):
        raise ValueError(f"task must be 'classification' or 'regression', got {task!r}")
    regression = task == 'regression'
    features = X.drop(columns=[target])
    y = X[target]
    binary = not regression and y.nunique() == 2

    def splitter(n_splits):
        cls = KFold if regression else StratifiedKFold
        return cls(n_splits=n_splits, shuffle=True, random_state=42)

    search_folds = 3
    forest = (RandomForestRegressor if regression else RandomForestClassifier)(n_estimators=100, random_state=42, n_jobs=1)
    scoring = 'neg_mean_squared_error' if regression else 'accuracy'
    if len(features) >= max(search_folds * len(_RF_PARAM_GRID['min_samples_leaf']), search_folds * 2):
        search = HalvingGridSearchCV(forest, _RF_PARAM_GRID, cv=splitter(search_folds), scoring=scoring,
                                     n_jobs=n_jobs, factor=3, random_state=42)
    else:
        search = GridSearchCV(forest, _RF_PARAM_GRID, cv=splitter(min(search_folds, max(2, len(features)))),
                              scoring=scoring, n_jobs=n_jobs)
    search.fit(features, y)
    best_params = search.best_params_

    shuffled = X.sample(frac=1, random_state=42).reset_index(drop=True)
    features, y = shuffled.drop(columns=[target]), shuffled[target]
    folds: dict[str, list[float]] = {}
    for train_idx, val_idx in splitter(cv_folds).split(features, y):
        model = (RandomForestRegressor if regression else RandomForestClassifier)(
            n_estimators=50, random_state=42, n_jobs=n_jobs, max_features='log2', **best_params)
        model.fit(features.iloc[train_idx], y.iloc[train_idx])
        X_val, y_val = features.iloc[val_idx], y.iloc[val_idx]
        y_pred = model.predict(X_val)
        if regression:
            scores = {'r2': r2_score(y_val, y_pred), 'mse': mean_squared_error(y_val, y_pred)}
        elif binary:
            try:
                auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            except ValueError:
                auc = roc_auc_score(y_val, y_pred)
            scores = {'accuracy': accuracy_score(y_val, y_pred), 'f1': f1_score(y_val, y_pred), 'roc_auc': auc}
        else:
            scores = {'accuracy': accuracy_score(y_val, y_pred), 'f1_weighted': f1_score(y_val, y_pred, average='weighted')}
        for metric, value in scores.items():
            folds.setdefault(metric, []).append(float(value))

    result = {metric: float(np.mean(values)) for metric, values in folds.items()}
    if regression:
        result['rmse'] = float(np.sqrt(result.pop('mse')))

    return result


def evaluate_paper_protocol(
    augmented: 'pl.DataFrame | pd.DataFrame',
    key: str,
    target: str,
    task: str,
    *,
    leakage_filter: bool = True,
    mask_sentinels: bool = True,
    cv_folds: int = 5,
    n_jobs: int = 8,
    intelex: bool = False,
) -> dict:
    """Prepare, leakage-filter and score an augmented table as the paper does.

    Score the unaugmented query table the same way for the base model the
    paper's improvements are relative to: RMSE improvement is
    ``(base - augmented) / base`` and F1 improvement
    ``(augmented - base) / |base|``.

    Returns the metrics of :func:`score_random_forest`, plus
    ``'leaky_columns'``, the columns removed by :func:`detect_target_leakage`.
    """
    X = prepare_augmented_table(augmented, key, target, mask_sentinels=mask_sentinels)
    leaky = detect_target_leakage(X, target, task) if leakage_filter else []
    X = X.drop(columns=leaky)
    metrics = score_random_forest(X, target, task, cv_folds=cv_folds, n_jobs=n_jobs, intelex=intelex)

    return {**metrics, 'leaky_columns': leaky}
