import json
import numpy as np
import pandas as pd
import polars as pl
from augmentation.utils.common import process_key
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from polars.exceptions import InvalidOperationError
from sklearn.experimental import enable_iterative_imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from typing import Literal


class AGFeatureWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.fg_kwargs = kwargs
        self.fg_ = None

    def fit(self, X, y=None):
        self.fg_ = AutoMLPipelineFeatureGenerator(
            enable_text_special_features=False,
            enable_text_ngram_features=False,
            verbosity=0,
            **self.fg_kwargs
        )
        self.fg_.fit(X)
        return self

    def transform(self, X):
        X_out = self.fg_.transform(X)
        return X_out.values
    
    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "fg_"):
            raise RuntimeError(
                "AGFeatureWrapper must be fitted before calling get_feature_names_out()"
            )

        # AutoGluon stores feature metadata here
        feature_metadata = self.fg_.feature_metadata

        # All generated features (raw + derived)
        feature_names = feature_metadata.get_features()

        return np.asarray(feature_names, dtype=object)


class PandasSimpleImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.imputer = SimpleImputer(**kwargs)

    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.index_ = X.index
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_imp = self.imputer.transform(X)
        return pd.DataFrame(X_imp, columns=self.columns_, index=X.index)


class PreProcessor:
    def __init__(self, base_table_path: str, base_table_splits_path: str, split_index: int = 0):
        self.base_table_path = base_table_path
        self.base_table_splits_path = base_table_splits_path
        self.split_index = split_index


    def run(self, augmentation_plan: list[str] = None, binning: bool = True, skip_num_features: bool = False) -> tuple[pl.DataFrame, str, str, np.ndarray]:
        with open(self.base_table_splits_path, 'r') as f:
            base_table_splits = json.load(f)[self.split_index]
        df = pl.read_csv(self.base_table_path, ignore_errors=True)
        features = base_table_splits['features']
        for f in features:
            if f in df.columns:
                continue
            elif f'{f}_x' in df.columns:
                features[features.index(f)] = f'{f}_x'
            elif f'{f}_y' in df.columns:
                features[features.index(f)] = f'{f}_y'
        df_columns = set(df.columns)
        if augmentation_plan is not None:
            for i in range(len(augmentation_plan)):
                f = augmentation_plan[i]
                if f in df_columns:
                    continue
                elif f'{f}_x' in df_columns:
                    augmentation_plan[i] = f'{f}_x'
                elif f'{f}_y' in df_columns:
                    augmentation_plan[i] = f'{f}_y'
            features = features + augmentation_plan

        features = list(set(features))
        query_col = base_table_splits['query_col']
        if query_col not in df.columns:
            if f'{query_col}_x' in df.columns:
                query_col = f'{query_col}_x'
            elif f'{query_col}_y' in df.columns:
                query_col = f'{query_col}_y'
        target = base_table_splits['target']
        if target not in df.columns:
            if f'{target}_x' in df.columns:
                target = f'{target}_x'
            elif f'{target}_y' in df.columns:
                target = f'{target}_y'
        target_type = base_table_splits['target_type']
        numeric_cols = base_table_splits['numeric_features']
        categorical_cols = base_table_splits['categorical_features']
        nan_mask = df.select([~pl.col(c).is_null().alias(c) for c in features]).to_numpy()
        
        transforms = []
        steps = []
        steps.append(('target', AGFeatureWrapper()))
        categorical_transformer = Pipeline(steps=steps)
        transforms.append(('cat', categorical_transformer, features))

        preprocessor = ColumnTransformer(
            transformers=transforms,
            remainder='drop'
        )

        df = df.drop_nulls(subset=[query_col, target])
        try:
            df = df.drop_nans(subset=[query_col, target])
        except InvalidOperationError:
            pass

        if target_type != 'continuous':
            target_transforms = [
                ('', Pipeline(steps=[('ord', OrdinalEncoder())]), [target])
            ]
            y = df.select(target)
            target_preprocessor = ColumnTransformer(transformers=target_transforms)
            y = target_preprocessor.fit_transform(y).ravel()
            y = pl.Series(name=target, values=y)
        else:
            y = df.select(target).to_series()

        if skip_num_features:
            features = categorical_cols
        X = df.select(features)
        X = X.to_pandas().replace({None: np.nan})
        # X = imputer.fit_transform(X)
        # X = self._convert_nan_values(X, numeric_features, categorical_features, missing_value_categorical)
        imputer = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=False,
                random_state=42,
                n_nearest_features=None
            )
        if not skip_num_features:
            numeric_pipeline = Pipeline([
                ("imputer", imputer)
            ])
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ])
            col_transform = ColumnTransformer([
                ("cat", categorical_pipeline, categorical_cols),
                ("num", numeric_pipeline, numeric_cols)
            ]).set_output(transform="pandas")
        else:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent"))
            ])
            col_transform = ColumnTransformer([
                ("cat", categorical_pipeline, categorical_cols)
            ]).set_output(transform="pandas")
            transforms = []
            steps = []
            steps.append(('target', AGFeatureWrapper()))
            categorical_transformer = Pipeline(steps=steps)
            transforms.append(('cat', categorical_transformer, categorical_cols))
            preprocessor = ColumnTransformer(
                transformers=transforms,
                remainder='drop'
            )
        X = col_transform.fit_transform(X)
        X.columns = [col.replace('num__', '').replace('cat__', '') for col in X.columns]
        X = preprocessor.fit_transform(X, y)
        imputer = IterativeImputer(
                estimator=BayesianRidge(),
                sample_posterior=False,
                random_state=42,
                n_nearest_features=None
            ).set_output(transform="pandas")
        X = imputer.fit_transform(X)

        feature_names = preprocessor.get_feature_names_out()
        X = pl.DataFrame(X, schema=feature_names.tolist())
        X.insert_column(X.shape[1], y)
        X.insert_column(0, df.get_column(query_col))
        assert X.null_count().to_numpy().sum() == 0, 'X contains null values'

        X = X.with_columns(pl.col(query_col).cast(pl.String).map_elements(lambda x: process_key(x)).alias(query_col))
        if binning:
            cols_to_bin = X.columns[1:-1]
            n_bins = 4
            binned_exprs = []
            for col in cols_to_bin:
                if X[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64]:
                    binned_exprs.append(
                        pl.col(col).qcut(n_bins, labels=[str(i) for i in range(n_bins)], allow_duplicates=True).alias(col)
                    )
                else:
                    binned_exprs.append(pl.col(col))
            X = X.select(
                [pl.col(X.columns[0])] + binned_exprs + [pl.col(X.columns[-1])]
            )
            X = pl.concat([
                X.select(pl.col(X.columns[0])),
                X.select(pl.col(cols_to_bin)).to_dummies(drop_first=True),
                X.select(pl.col(X.columns[-1]))
            ], how='horizontal')
        # X = X.with_columns(pl.exclude(query_col, target).fill_nan(pl.exclude(query_col, target).filter(pl.exclude(query_col, target).is_not_nan()).mean()))

        return X, query_col, target, nan_mask

    
    def run_only_query_col(self) -> tuple[pl.DataFrame, str, str, np.ndarray]:
        with open(self.base_table_splits_path, 'r') as f:
            base_table_splits = json.load(f)[self.split_index]
        df = pl.read_csv(self.base_table_path, ignore_errors=True)

        query_col = base_table_splits['query_col']
        if query_col not in df.columns:
            if f'{query_col}_x' in df.columns:
                query_col = f'{query_col}_x'
            elif f'{query_col}_y' in df.columns:
                query_col = f'{query_col}_y'

        target = base_table_splits['target']
        if target not in df.columns:
            if f'{target}_x' in df.columns:
                target = f'{target}_x'
            elif f'{target}_y' in df.columns:
                target = f'{target}_y'
        target_type = base_table_splits['target_type']

        features = [query_col]
        nan_mask = df.select([~pl.col(c).is_null().alias(c) for c in features]).to_numpy()

        df = df.drop_nulls(subset=[query_col, target])
        try:
            df = df.drop_nans(subset=[query_col, target])
        except InvalidOperationError:
            pass

        # Encode target
        if target_type != 'continuous':
            target_transforms = [
                ('', Pipeline(steps=[('ord', OrdinalEncoder())]), [target])
            ]
            y = df.select(target)
            target_preprocessor = ColumnTransformer(transformers=target_transforms)
            y = target_preprocessor.fit_transform(y).ravel()
            y = pl.Series(name=target, values=y)
        else:
            y = df.select(target).to_series()

        # Encode query_col with AutoGluon
        X_raw = df.select([query_col]).to_pandas().replace({None: np.nan})
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent"))
        ])
        col_transform = ColumnTransformer([
            ("cat", categorical_pipeline, [query_col])
        ]).set_output(transform="pandas")
        X_imputed = col_transform.fit_transform(X_raw)
        X_imputed.columns = [col.replace('cat__', '') for col in X_imputed.columns]

        preprocessor = ColumnTransformer(
            transformers=[('cat', Pipeline(steps=[('target', AGFeatureWrapper())]), [query_col])],
            remainder='drop'
        )
        X_encoded = preprocessor.fit_transform(X_imputed, y)
        imputer = IterativeImputer(
            estimator=BayesianRidge(),
            sample_posterior=False,
            random_state=42,
            n_nearest_features=None
        ).set_output(transform="pandas")
        X_encoded = imputer.fit_transform(X_encoded)

        feature_names = preprocessor.get_feature_names_out()
        X = pl.DataFrame(X_encoded, schema=feature_names.tolist())
        X.insert_column(X.shape[1], y)
        X.insert_column(0, df.get_column(query_col))
        assert X.null_count().to_numpy().sum() == 0, 'X contains null values'

        X = X.with_columns(
            pl.col(query_col).cast(pl.String).map_elements(lambda x: process_key(x)).alias(query_col)
        )

        return X, query_col, target, nan_mask


    def _convert_nan_values(self, df: pl.DataFrame, numeric_features: list, categorical_features: list, missing_value_categorical: str):
        df = df.select(
            [
                pl.col(categorical_features)
                    .cast(pl.String)
                    .fill_null(value=missing_value_categorical),
                pl.col(numeric_features)
                    .cast(pl.Float64)
                    .fill_null(value=np.nan)
            ]
        )
        return df
