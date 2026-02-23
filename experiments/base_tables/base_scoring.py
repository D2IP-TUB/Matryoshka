"""
AutoGluon Model Training Script

This script trains regression or classification models using AutoGluon,
which automatically tries multiple models and ensembles them.
"""

import argparse
import json
import os
os.environ['PYTHONPATH'] = '.'
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, r2_score, roc_auc_score
from experiments.base_tables.base_table_preprocessing import PreProcessor

from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV as GridSearchCV
import pickle


class SimpleTrainer:
    """Wrapper for training simple Random Forest models with basic hyperparameter tuning."""
    
    def __init__(
        self,
        problem_type: Literal["regression", "binary", "multiclass"],
        target_column: str,
        output_dir: str = "./simple_models",
        dry_run: bool = False,
        **kwargs
    ):
        """
        Initialize Simple Random Forest trainer.
        
        Args:
            problem_type: Type of problem - "regression", "binary", or "multiclass"
            target_column: Name of the target column in the dataset
            output_dir: Directory to save trained models
            dry_run: If True, skip actual training
        """
        self.problem_type = problem_type
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.dry_run = dry_run
        self.model = None
        self.best_params = None
        
    def train(
        self,
        train_data: pd.DataFrame,
        cv_folds: int = 3,
        verbosity: int = 2,
        **kwargs
    ):
        """
        Train the Random Forest model with grid search for hyperparameter tuning.
        
        Args:
            train_data: Training dataset with features and target
            cv_folds: Number of cross-validation folds for hyperparameter tuning
            verbosity: Verbosity level (0-3)
            
        Returns:
            Trained model (GridSearchCV object)
        """
        print(f"Starting Simple Random Forest training for {self.problem_type}...")
        print(f"Target column: {self.target_column}")
        print(f"Training samples: {len(train_data)}")
        
        if self.dry_run:
            print("[DRY RUN] Skipping actual model training...")
            print(f"  - Would use Random Forest {'Regressor' if self.problem_type == 'regression' else 'Classifier'}")
            print(f"  - Would save to: {self.output_dir}")
            print(f"  - CV folds: {cv_folds}")
            return None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate features and target
        X = train_data.drop(columns=[self.target_column])
        y = train_data[self.target_column]
        
        # Initialize base model and CV strategy
        if self.problem_type == "regression":
            base_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=64)
            scoring = 'neg_mean_squared_error'
            cv_strategy = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:  # binary or multiclass
            base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=64)
            scoring = 'accuracy'
            cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Define hyperparameter grid
        param_grid = {
            'max_depth': [5, 10, 20],
            'min_samples_leaf': [10, 20, 50]
        }
        
        # Perform grid search
        if verbosity > 0:
            print(f"Performing grid search with {cv_folds}-fold cross-validation...")
            print(f"Parameter grid: {param_grid}")
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv_strategy,
            scoring=scoring,
            n_jobs=64,
            verbose=verbosity,
            return_train_score=True,
            factor=3
        )
        
        grid_search.fit(X, y)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        if verbosity > 0:
            print(f"\nBest parameters: {self.best_params}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Save the model
        model_path = self.output_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save best parameters
        params_path = self.output_dir / "best_params.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        print(f"\nTraining complete! Model saved to: {self.output_dir}")
        return self.model
    
    def evaluate(self, test_data: pd.DataFrame, cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate the model using k-fold cross-validation and return average metrics.
        
        Args:
            test_data: Test dataset with features and target
            cv_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary of average evaluation metrics from cross-validation
        """
        if self.dry_run:
            print(f"[DRY RUN] Skipping model evaluation on {len(test_data)} test samples...")
            return {"dry_run": True}
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"\nEvaluating with {cv_folds}-fold cross-validation on {len(test_data)} samples...")
        
        # Shuffle the data
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X = test_data.drop(columns=[self.target_column])
        y = test_data[self.target_column]
        
        # Initialize cross-validation (stratified for classification)
        if self.problem_type == "regression":
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Perform cross-validation
        fold_scores = {}
        
        if self.problem_type == "regression":
            scoring = ['r2', 'neg_mean_squared_error']
        elif self.problem_type == "binary":
            scoring = ['accuracy', 'f1', 'roc_auc']
        else:  # multiclass
            scoring = ['accuracy', 'f1_weighted']
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Retrain model on this fold's training data with best params
            if self.problem_type == "regression":
                fold_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=64, max_features='log2', **(self.best_params or {}))
            else:
                fold_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=64, max_features='log2', **(self.best_params or {}))
            fold_model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = fold_model.predict(X_val)
            
            # Calculate metrics for this fold
            if self.problem_type == "regression":
                if 'r2' not in fold_scores:
                    fold_scores['r2'] = []
                    fold_scores['neg_mean_squared_error'] = []
                fold_scores['r2'].append(r2_score(y_val, y_pred))
                fold_scores['neg_mean_squared_error'].append(-mean_squared_error(y_val, y_pred))
            elif self.problem_type == "binary":
                if 'accuracy' not in fold_scores:
                    fold_scores['accuracy'] = []
                    fold_scores['f1'] = []
                    fold_scores['roc_auc'] = []
                fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_scores['f1'].append(f1_score(y_val, y_pred))
                # For ROC AUC, use predict_proba
                try:
                    y_pred_proba = fold_model.predict_proba(X_val)[:, 1]
                    fold_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
                except:
                    fold_scores['roc_auc'].append(roc_auc_score(y_val, y_pred))
            else:  # multiclass
                if 'accuracy' not in fold_scores:
                    fold_scores['accuracy'] = []
                    fold_scores['f1_weighted'] = []
                fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_scores['f1_weighted'].append(f1_score(y_val, y_pred, average='weighted'))
        
        # Calculate average scores
        avg_performance = {}
        for metric, scores in fold_scores.items():
            avg_score = np.mean(scores)
            # Convert negative MSE back to positive RMSE for better interpretability
            if metric == 'neg_mean_squared_error':
                avg_performance['rmse'] = np.sqrt(-avg_score)
            else:
                avg_performance[metric] = avg_score
        
        print("\nAverage Cross-Validation Performance:")
        for metric, value in avg_performance.items():
            print(f"  {metric}: {value:.4f}")
        
        return avg_performance
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores from the Random Forest.
        
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Decision trees have feature_importances_ attribute
        importance_scores = self.model.feature_importances_
        feature_names = self.model.feature_names_in_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame with features (without target column)
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.model.predict(data)
        return predictions
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            save_path: Optional path to save the plot
        """
        importance = self.get_feature_importance()
        
        # Get top N
        importance_top = importance.head(top_n).sort_values('importance', ascending=True)
        
        # Create plot
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(importance_top['feature'], importance_top['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model directory
        """
        model_file = Path(model_path) / "model.pkl"
        params_file = Path(model_path) / "best_params.json"
        
        with open(model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        if params_file.exists():
            with open(params_file, 'r') as f:
                self.best_params = json.load(f)
        
        print(f"Model loaded from: {model_path}")
        if self.best_params:
            print(f"Best parameters: {self.best_params}")


class AutoGluonTrainer:
    """Wrapper for training regression and classification models with AutoGluon."""
    
    def __init__(
        self,
        problem_type: Literal["regression", "binary", "multiclass"],
        target_column: str,
        eval_metric: Optional[str] = None,
        time_limit: int = 600,
        presets: str = "best_quality",
        output_dir: str = "./autogluon_models",
        dry_run: bool = False
    ):
        """
        Initialize AutoGluon trainer.
        
        Args:
            problem_type: Type of problem - "regression", "binary", or "multiclass"
            target_column: Name of the target column in the dataset
            eval_metric: Metric to optimize (e.g., 'rmse', 'r2' for regression,
                        'accuracy', 'f1', 'roc_auc' for classification)
            time_limit: Time limit in seconds for training
            presets: Quality preset - "best_quality", "high_quality", "good_quality", 
                    "medium_quality", or "optimize_for_deployment"
            output_dir: Directory to save trained models
        """
        self.problem_type = problem_type
        self.target_column = target_column
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.presets = presets
        self.output_dir = Path(output_dir)
        self.predictor = None
        self.dry_run = dry_run
        
    def train(
        self,
        train_data: pd.DataFrame,
        validation_data: Optional[pd.DataFrame] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        verbosity: int = 2
    ) -> TabularPredictor:
        """
        Train the AutoGluon model.
        
        Args:
            train_data: Training dataset with features and target
            validation_data: Optional validation dataset
            hyperparameters: Optional custom hyperparameters for models
            verbosity: Verbosity level (0-4)
            
        Returns:
            Trained TabularPredictor
        """
        print(f"Starting AutoGluon training for {self.problem_type}...")
        print(f"Target column: {self.target_column}")
        print(f"Training samples: {len(train_data)}")
        
        if self.dry_run:
            print("[DRY RUN] Skipping actual model training...")
            print(f"  - Would use presets: {self.presets}")
            print(f"  - Would save to: {self.output_dir}")
            print(f"  - Time limit: {self.time_limit} seconds")
            if validation_data is not None:
                print(f"  - Validation samples: {len(validation_data)}")
            return None
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize predictor
        self.predictor = TabularPredictor(
            label=self.target_column,
            problem_type=self.problem_type,
            eval_metric=self.eval_metric,
            path=str(self.output_dir),
            verbosity=verbosity
        )
        
        # Train the model
        self.predictor.fit(
            train_data=train_data,
            tuning_data=validation_data,
            time_limit=self.time_limit,
            presets=self.presets,
            hyperparameters=hyperparameters,
            keep_only_best=True,  # Only keep the best model
            dynamic_stacking=False
        )
        
        print(f"\nTraining complete! Best model saved to: {self.output_dir}")
        return self.predictor
    
    def evaluate(self, test_data: pd.DataFrame, cv_folds: int = 5) -> Dict[str, float]:
        """
        Evaluate the model using k-fold cross-validation and return average metrics.
        
        Args:
            test_data: Test dataset with features and target
            cv_folds: Number of folds for cross-validation
            
        Returns:
            Dictionary of average evaluation metrics from cross-validation
        """
        if self.dry_run:
            print(f"[DRY RUN] Skipping model evaluation on {len(test_data)} test samples...")
            return {"dry_run": True}
        
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        print(f"\nEvaluating with {cv_folds}-fold cross-validation on {len(test_data)} samples...")
        
        # Shuffle the data
        test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        X = test_data.drop(columns=[self.target_column])
        y = test_data[self.target_column]
        
        # Initialize cross-validation (stratified for classification)
        if self.problem_type == "regression":
            kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Define scoring metrics based on problem type
        if self.problem_type == "regression":
            scoring = ['r2', 'neg_mean_squared_error']
        elif self.problem_type == "binary":
            scoring = ['accuracy', 'f1', 'roc_auc']
        else:  # multiclass
            scoring = ['accuracy', 'f1_weighted']
        
        # Perform cross-validation using AutoGluon predictor
        fold_scores = {metric: [] for metric in scoring}
        
        for train_idx, val_idx in kfold.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Create validation data with target column
            val_data = X_val.copy()
            val_data[self.target_column] = y_val
            
            # Get predictions
            y_pred = self.predictor.predict(X_val)
            
            # Calculate metrics for this fold
            if self.problem_type == "regression":
                fold_scores['r2'].append(r2_score(y_val, y_pred))
                fold_scores['neg_mean_squared_error'].append(-mean_squared_error(y_val, y_pred))
            elif self.problem_type == "binary":
                fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_scores['f1'].append(f1_score(y_val, y_pred))
                # For ROC AUC, use predict_proba if available
                try:
                    y_pred_proba = self.predictor.predict_proba(X_val)[:, 1]
                    fold_scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
                except:
                    fold_scores['roc_auc'].append(roc_auc_score(y_val, y_pred))
            else:  # multiclass
                fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_scores['f1_weighted'].append(f1_score(y_val, y_pred, average='weighted'))
        
        # Calculate average scores
        avg_performance = {}
        for metric, scores in fold_scores.items():
            avg_score = np.mean(scores)
            # Convert negative MSE back to positive RMSE for better interpretability
            if metric == 'neg_mean_squared_error':
                avg_performance['rmse'] = np.sqrt(-avg_score)
            else:
                avg_performance[metric] = avg_score
        
        print("\nAverage Cross-Validation Performance:")
        for metric, value in avg_performance.items():
            print(f"  {metric}: {value:.4f}")
        
        return avg_performance
    
    def get_leaderboard(self, test_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Get leaderboard of all trained models.
        
        Args:
            test_data: Optional test data to evaluate models
            
        Returns:
            DataFrame with model performance
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        leaderboard = self.predictor.leaderboard(test_data, silent=True)
        return leaderboard
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        importance = self.predictor.feature_importance()
        return importance
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            data: DataFrame with features (without target column)
            
        Returns:
            Array of predictions
        """
        if self.predictor is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        predictions = self.predictor.predict(data)
        return predictions
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            save_path: Optional path to save the plot
        """
        importance = self.get_feature_importance()
        
        # Sort by importance and get top N
        importance_sorted = importance.sort_values(by='importance', ascending=True).tail(top_n)
        
        # Create plot
        plt.figure(figsize=(10, max(6, top_n * 0.3)))
        plt.barh(importance_sorted.index, importance_sorted['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to: {save_path}")
        
        plt.show()
    
    def load_model(self, model_path: str):
        """
        Load a previously trained model.
        
        Args:
            model_path: Path to the saved model directory
        """
        self.predictor = TabularPredictor.load(model_path)
        print(f"Model loaded from: {model_path}")


def train_from_csv(
    csv_path: str,
    target_column: str,
    problem_type: Literal["regression", "binary", "multiclass"],
    test_size: float = 0.2,
    time_limit: int = 600,
    output_dir: str = "./autogluon_models",
    dry_run: bool = False
):
    """
    Train AutoGluon model from a CSV file.
    
    Args:
        csv_path: Path to CSV file
        target_column: Name of the target column
        problem_type: Type of problem
        test_size: Fraction of data to use for testing
        time_limit: Time limit in seconds for training
        output_dir: Directory to save models
    """
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Initialize trainer
    trainer = AutoGluonTrainer(
        problem_type=problem_type,
        target_column=target_column,
        time_limit=time_limit,
        presets="best_quality",
        output_dir=output_dir,
        dry_run=dry_run
    )
    
    if dry_run:
        print("\n[DRY RUN] Skipping model training and evaluation...")
        print(f"Problem type: {problem_type}")
        print(f"Target column: {target_column}")
        return trainer
    
    # Train
    trainer.train(df)
    
    # Evaluate
    trainer.evaluate(df)
    
    # Show results
    print("\nModel Leaderboard:")
    print(trainer.get_leaderboard(df))
    
    print("\nFeature Importance:")
    print(trainer.get_feature_importance())
    
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run base table scoring with AutoGluon or Simple Decision Tree',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python experiments/base_tables/base_scoring.py                      # Run with AutoGluon (default)
  python experiments/base_tables/base_scoring.py --trainer simple     # Run with Simple Decision Tree
  python experiments/base_tables/base_scoring.py --dry-run            # Dry run without training
        '''
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Execute everything except training and results calculation'
    )
    parser.add_argument(
        '--trainer',
        choices=['autogluon', 'simple'],
        default='autogluon',
        help='Trainer to use: autogluon (default) or simple (Decision Tree)'
    )
    args = parser.parse_args()
    
    data_dirs = os.listdir('experiments/base_tables/')
    data_dirs = [
        d for d in data_dirs if os.path.isdir(os.path.join('experiments/base_tables/', d)) \
        and d != '__pycache__'
    ]
    all_results = pd.DataFrame()
    
    if args.dry_run:
        print(f"[DRY RUN MODE] Will process data but skip training and scoring using {args.trainer} trainer...\n")
    
    for data_dir in data_dirs:
        base_table_path = f'experiments/base_tables/{data_dir}/{data_dir}.csv'
        base_table_splits_path = f'experiments/base_tables/{data_dir}/splits.json'
        preprocessor = PreProcessor(base_table_path, base_table_splits_path, 0)
        X, query_col, target, nan_mask = preprocessor.run()
        X.write_csv(f'experiments/base_tables/{data_dir}/{data_dir}_preprocessed.csv')
        X = X.drop(query_col).to_pandas()
        with open(base_table_splits_path, 'r') as f:
            splits = json.load(f)
        problem_type = splits[0]['target_type']
        if problem_type == 'continuous':
            problem_type = 'regression'

        if args.dry_run:
            print(f"[DRY RUN] Would create {args.trainer.title()}Trainer for {data_dir}:")
            print(f"  - Table name: {data_dir}")
            print(f"  - Problem type: {problem_type}")
            print(f"  - Target column: {target}")
            print(f"  - Query column: {query_col}")
            print(f"  - Training data shape: {X.shape}")
            if args.trainer == 'autogluon':
                print(f"  - Output directory: experiments/base_tables/{data_dir}/autogluon_model")
                print(f"  - Time limit: 1800 seconds")
                print(f"  - Presets: good_quality")
            else:  # simple
                print(f"  - Output directory: experiments/base_tables/{data_dir}/simple_model")
                print(f"  - CV folds: 5")
            print()
            continue
        
        if args.trainer == 'autogluon':
            model_path = f'experiments/base_tables/{data_dir}/autogluon_model'
            
            trainer = AutoGluonTrainer(
                problem_type=problem_type,
                target_column=target,
                time_limit=1800,
                presets="good_quality",
                output_dir=model_path
            )
        else:  # simple
            model_path = f'experiments/base_tables/{data_dir}/simple_model'
            
            trainer = SimpleTrainer(
                problem_type=problem_type,
                target_column=target,
                output_dir=model_path
            )
        X = X.dropna(subset=[target])
        trainer.train(X)

        performance = trainer.evaluate(X)
        df = pd.DataFrame([performance], index=[data_dir])
        all_results = pd.concat([all_results, df], ignore_index=False)

    if not args.dry_run:
        output_file = f'experiments/base_tables/base_{args.trainer}_scores.csv'
        all_results.to_csv(output_file, index=True)
        print(f"\nResults saved to {output_file}")
    else:
        print(f"[DRY RUN] Skipped saving results to base_{args.trainer}_scores.csv")