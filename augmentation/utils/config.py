from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


@dataclass
class DiscoveryConfig:
    """
    A flexible configuration class for machine learning tasks with automatic validation.
    
    Args:
        task: One of "regression", "classification", "imputation", 
              "regression explanation", "classification explanation"
        ranking: Ranking method (auto-validated based on task)
        strategy: Strategy method (auto-validated based on task and ranking)
        model: Model type (auto-validated based on task and strategy)
        params: Additional parameters (auto-validated based on strategy)
        baseline: If True, only params are allowed (no task/ranking/strategy/model)
    """
    
    task: Optional[str] = None
    ranking: Optional[str] = None
    strategy: Optional[str] = None
    model: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    baseline: bool = False
    
    # Configuration rules (avoids hard-coding in methods)
    _task_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "imputation": {
            "allowed_ranking": ["passthrough"],
            "required_ranking": "passthrough",
            "allowed_model": ["WeightedSum"],
            "allowed_strategy": ["IterativeImputer"],
            "required_params": {"tol": float, "n_iterations": int}
        },
        "regression explanation": {
            "allowed_ranking": ["correlation"],
            "required_ranking": "correlation",
            "allowed_model": ["Cholesky", "QR"],
            "allowed_strategy": [],
            "allowed_params": []
        },
        "classification explanation": {
            "allowed_ranking": ["eta"],
            "required_ranking": "eta",
            "allowed_model": ["Cholesky", "QR"],
            "allowed_strategy": [],
            "allowed_params": []
        },
        "regression": {
            "allowed_ranking": ["passthrough", "joinability", "correlation"],
            "ranking_strategy_rules": {
                "passthrough": {"allowed_strategy": [None, "LassoFeatureSelector", "CofactorLASSO", "IncrementalSelection", "ForwardSelection"]},
                "joinability": {"allowed_strategy": ["IncrementalSelection"]},
                "correlation": {"allowed_strategy": [None, "BackwardElimination", "LassoFeatureSelector", "CofactorLASSO"]},
                "eta": {"impossible": True}
            }
        },
        "classification": {
            "allowed_ranking": ["passthrough", "joinability", "eta"],
            "ranking_strategy_rules": {
                "passthrough": {"allowed_strategy": [None, "LassoFeatureSelector", "IncrementalSelection", "ForwardSelection"]},
                "joinability": {"allowed_strategy": ["IncrementalSelection"]},
                "eta": {"allowed_strategy": [None, "BackwardElimination", "LassoFeatureSelector", "CofactorLASSO"]},
                "correlation": {"impossible": True}
            }
        }
    }, init=False, repr=False)
    
    _strategy_rules: Dict[str, Dict] = field(default_factory=lambda: {
        "LassoFeatureSelector": {
            "model_rules": {
                "regression": ["LinRegL1"],
                "classification": ["LogRegL1"]
            },
            "required_params": {
                "metric": {"type": str, "allowed_values": ["mse", "r2", "f1", "f1_weighted"]},
                "alphas": {"type": list, "validator": lambda x: all(isinstance(a, (int, float)) for a in x) and len(x) > 0},
                "cv": {"type": int, "validator": lambda x: x > 1}
            }
        },
        "CofactorLASSO": {
            "allowed_model": ["RegressionCofactorLASSO"],
            "required_params": {"alphas": list}
        },
        "IncrementalSelection": {
            "model_rules": {
                "regression": ["RegressionQR", "RegressionCholesky"],
                "classification": ["ClassificationCholesky"]
            },
            "required_params": {
                "metric": {"type": str, "allowed_values": ["mse", "r2", "adj_r2", "average_mahalanobis"]},
                "table_batch_size": {"type": int, "validator": lambda x: x > 0},
                "tol": {"type": float, "validator": lambda x: x > 0}
            }
        },
        "ForwardSelection": {
            "model_rules": {
                "regression": ["RegressionQR", "RegressionCholesky"],
                "classification": ["ClassificationCholesky", "RegressionQR"]
            },
            "required_params": {
                "metric": {"type": str, "allowed_values": ["mse", "r2", "adj_r2", "average_mahalanobis"]},
                "tol": {"type": float, "validator": lambda x: x > 0}
            }
        },
        "BackwardElimination": {
            "model_rules": {
                "regression": ["RegressionQR", "RegressionCholesky"],
                "classification": ["ClassificationCholesky", "RegressionQR"]
            },
            "required_params": {
                "metric": {"type": str, "allowed_values": ["mse", "r2", "adj_r2", "average_mahalanobis"]},
                "tol": {"type": float, "validator": lambda x: x > 0}
            }
        }
    }, init=False, repr=False)
    
    def __post_init__(self):
        """Validate the configuration after initialization."""
        if self.baseline:
            self._validate_baseline_config()
        else:
            self._validate_config()
    
    def _validate_baseline_config(self):
        """Validate baseline configuration (only params allowed)."""
        if any([self.task, self.ranking, self.model]):
            raise ConfigValidationError(
                "When baseline=True, task, ranking, and model must all be None"
            )
        # Params can be anything for baseline mode
    
    def _validate_config(self):
        """Comprehensive validation of the entire configuration."""
        if not self.task:
            raise ConfigValidationError("task is required when baseline=False")
        self._validate_task()
        self._validate_ranking()
        self._validate_strategy()
        self._validate_model()
        self._validate_params()
    
    def _validate_task(self):
        """Validate the task parameter."""
        allowed_tasks = list(self._task_rules.keys())
        if self.task not in allowed_tasks:
            raise ConfigValidationError(
                f"Invalid task '{self.task}'. Allowed tasks: {allowed_tasks}"
            )
    
    def _validate_ranking(self):
        """Validate ranking based on task constraints."""
        task_rule = self._task_rules[self.task]
        
        # Check if ranking is required but not provided
        if "required_ranking" in task_rule and not self.ranking:
            self.ranking = task_rule["required_ranking"]
        
        # Check if ranking is allowed
        if "allowed_ranking" in task_rule:
            if self.ranking and self.ranking not in task_rule["allowed_ranking"]:
                raise ConfigValidationError(
                    f"Invalid ranking '{self.ranking}' for task '{self.task}'. "
                    f"Allowed: {task_rule['allowed_ranking']}"
                )
        
        # Check for impossible combinations
        if (self.task in ["regression", "classification"] and 
            "ranking_strategy_rules" in task_rule and 
            self.ranking in task_rule["ranking_strategy_rules"]):
            
            ranking_rule = task_rule["ranking_strategy_rules"][self.ranking]
            if ranking_rule.get("impossible"):
                raise ConfigValidationError(
                    f"Ranking '{self.ranking}' is not allowed for task '{self.task}'"
                )
    
    def _validate_strategy(self):
        """Validate strategy based on task and ranking constraints."""
        task_rule = self._task_rules[self.task]
        
        # Tasks that don't allow strategy
        if "allowed_strategy" in task_rule and not task_rule["allowed_strategy"]:
            if self.strategy:
                raise ConfigValidationError(
                    f"Task '{self.task}' does not allow strategy parameter"
                )
            return
        
        # For regression/classification tasks
        if self.task in ["regression", "classification"] and self.ranking:
            ranking_rules = task_rule["ranking_strategy_rules"]
            if self.ranking in ranking_rules:
                allowed_strategies = ranking_rules[self.ranking].get("allowed_strategy", [])
                if self.strategy and self.strategy not in allowed_strategies:
                    raise ConfigValidationError(
                        f"Invalid strategy '{self.strategy}' for task '{self.task}' "
                        f"with ranking '{self.ranking}'. Allowed: {allowed_strategies}"
                    )
    
    def _validate_model(self):
        """Validate model based on task and strategy constraints."""
        task_rule = self._task_rules[self.task]
        
        # Tasks with specific model requirements
        if "allowed_model" in task_rule:
            if not task_rule["allowed_model"] and self.model:
                raise ConfigValidationError(
                    f"Task '{self.task}' does not allow model parameter"
                )
            elif (task_rule["allowed_model"] and self.model and 
                  self.model not in task_rule["allowed_model"]):
                raise ConfigValidationError(
                    f"Invalid model '{self.model}' for task '{self.task}'. "
                    f"Allowed: {task_rule['allowed_model']}"
                )
        
        # Strategy-based model validation
        if self.strategy and self.strategy in self._strategy_rules:
            strategy_rule = self._strategy_rules[self.strategy]
            
            # Strategy with fixed model options
            if "allowed_model" in strategy_rule:
                if self.model and self.model not in strategy_rule["allowed_model"]:
                    raise ConfigValidationError(
                        f"Invalid model '{self.model}' for strategy '{self.strategy}'. "
                        f"Allowed: {strategy_rule['allowed_model']}"
                    )
            
            # Strategy with task-dependent models
            elif "model_rules" in strategy_rule:
                task_base = self.task.split()[0]  # Handle "regression explanation" -> "regression"
                if task_base in strategy_rule["model_rules"]:
                    allowed_models = strategy_rule["model_rules"][task_base]
                    if self.model and self.model not in allowed_models:
                        raise ConfigValidationError(
                            f"Invalid model '{self.model}' for strategy '{self.strategy}' "
                            f"and task '{self.task}'. Allowed: {allowed_models}"
                        )
    
    def _validate_params(self):
        """Validate params based on task and strategy constraints."""
        task_rule = self._task_rules[self.task]
        
        # Task-specific param requirements
        if "required_params" in task_rule:
            required_params = task_rule["required_params"]
            if not self.params:
                raise ConfigValidationError(
                    f"Task '{self.task}' requires params: {list(required_params.keys())}"
                )
            
            for param_name, param_type in required_params.items():
                if param_name not in self.params:
                    raise ConfigValidationError(
                        f"Missing required parameter '{param_name}' for task '{self.task}'"
                    )
                if not isinstance(self.params[param_name], param_type):
                    raise ConfigValidationError(
                        f"Parameter '{param_name}' must be of type {param_type.__name__}"
                    )
        
        # Strategy-specific param requirements
        if self.strategy and self.strategy in self._strategy_rules:
            strategy_rule = self._strategy_rules[self.strategy]
            
            # Required parameters
            if "required_params" in strategy_rule:
                required_params = strategy_rule["required_params"]
                if not self.params:
                    raise ConfigValidationError(
                        f"Strategy '{self.strategy}' requires params: {list(required_params.keys())}"
                    )
                
                for param_name, param_type in required_params.items():
                    if param_name not in self.params:
                        raise ConfigValidationError(
                            f"Missing required parameter '{param_name}' for strategy '{self.strategy}'"
                        )
                    if not isinstance(self.params[param_name], param_type['type']):
                        raise ConfigValidationError(
                            f"Parameter '{param_name}' must be of type {param_type.__name__}"
                        )
            
            # Optional parameters validation
            if "optional_params" in strategy_rule and self.params:
                optional_params = strategy_rule["optional_params"]
                for param_name, param_value in self.params.items():
                    if param_name in optional_params:
                        param_spec = optional_params[param_name]
                        
                        # Type validation
                        expected_type = param_spec["type"]
                        if not isinstance(param_value, expected_type):
                            raise ConfigValidationError(
                                f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
                                f"got {type(param_value).__name__}"
                            )
                        
                        # Allowed values validation
                        if "allowed_values" in param_spec:
                            allowed_values = param_spec["allowed_values"]
                            if param_value not in allowed_values:
                                raise ConfigValidationError(
                                    f"Parameter '{param_name}' must be one of {allowed_values}, "
                                    f"got '{param_value}'"
                                )
                        
                        # Custom validator
                        if "validator" in param_spec:
                            validator = param_spec["validator"]
                            if not validator(param_value):
                                raise ConfigValidationError(
                                    f"Parameter '{param_name}' with value '{param_value}' "
                                    f"failed validation for strategy '{self.strategy}'"
                                )
            
            # Check for strategies that don't allow parameters
            elif ("allowed_params" in strategy_rule and 
                  not strategy_rule["allowed_params"] and 
                  self.params):
                raise ConfigValidationError(
                    f"Strategy '{self.strategy}' does not allow parameters"
                )
    
    def get_valid_options(self, field: str) -> List[str]:
        """
        Get valid options for a specific field given the current configuration.
        
        Args:
            field: The field to get options for ('ranking', 'strategy', 'model')
            
        Returns:
            List of valid options for the specified field
        """
        if field == "ranking":
            return self._get_valid_rankings()
        elif field == "strategy":
            return self._get_valid_strategies()
        elif field == "model":
            return self._get_valid_models()
        else:
            raise ValueError(f"Unknown field '{field}'")
    
    def _get_valid_rankings(self) -> List[str]:
        """Get valid ranking options for the current task."""
        task_rule = self._task_rules.get(self.task, {})
        if "allowed_ranking" in task_rule:
            return task_rule["allowed_ranking"]
        return []
    
    def _get_valid_strategies(self) -> List[str]:
        """Get valid strategy options for the current task and ranking."""
        if not self.ranking:
            return []
        
        task_rule = self._task_rules.get(self.task, {})
        if "ranking_strategy_rules" in task_rule and self.ranking in task_rule["ranking_strategy_rules"]:
            ranking_rule = task_rule["ranking_strategy_rules"][self.ranking]
            return ranking_rule.get("allowed_strategy", [])
        return []
    
    def _get_valid_models(self) -> List[str]:
        """Get valid model options for the current task and strategy."""
        models = []
        
        # Task-based models
        task_rule = self._task_rules.get(self.task, {})
        if "allowed_model" in task_rule:
            models.extend(task_rule["allowed_model"])
        
        # Strategy-based models
        if self.strategy and self.strategy in self._strategy_rules:
            strategy_rule = self._strategy_rules[self.strategy]
            if "allowed_model" in strategy_rule:
                models.extend(strategy_rule["allowed_model"])
            elif "model_rules" in strategy_rule:
                task_base = self.task.split()[0]
                if task_base in strategy_rule["model_rules"]:
                    models.extend(strategy_rule["model_rules"][task_base])
        
        return list(set(models))  # Remove duplicates
    
    def get_valid_param_options(self, param_name: str) -> List[str]:
        """
        Get valid options for a specific parameter given the current strategy.
        
        Args:
            param_name: The parameter name to get options for
            
        Returns:
            List of valid options for the specified parameter
        """
        if not self.strategy or self.strategy not in self._strategy_rules:
            return []
        
        strategy_rule = self._strategy_rules[self.strategy]
        if ("optional_params" in strategy_rule and 
            param_name in strategy_rule["optional_params"]):
            param_spec = strategy_rule["optional_params"][param_name]
            return param_spec.get("allowed_values", [])
        
        return []
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        if self.baseline:
            return f"DiscoveryConfig(baseline=True, params={self.params})"
        return (f"DiscoveryConfig(task='{self.task}', ranking='{self.ranking}', "
                f"strategy='{self.strategy}', model='{self.model}', params={self.params})")


# Example usage and testing
if __name__ == "__main__":
    # Valid configurations
    try:
        # Imputation task
        config1 = DiscoveryConfig(
            task="imputation",
            params={"tol": 0.01, "n_iterations": 100}
        )
        print("✓", config1)
        
        # Regression with correlation ranking
        config2 = DiscoveryConfig(
            task="regression",
            ranking="correlation",
            strategy="LASSO",
            model="RegressionLASSO",
            params={"alphas": [0.1, 1.0, 10.0]}
        )
        print("✓", config2)
        
        # Classification explanation
        config3 = DiscoveryConfig(
            task="classification explanation",
            model="QR"
        )
        print("✓", config3)
        
        # Get valid options
        config4 = DiscoveryConfig(task="regression", ranking="passthrough")
        print(f"Valid strategies: {config4.get_valid_options('strategy')}")
        
        # Regression with ranking but no strategy
        config5 = DiscoveryConfig(
            task="regression",
            ranking="passthrough",
            strategy=None
        )
        print("✓", config5)
        
        # IncrementalSelection with parameters
        config6 = DiscoveryConfig(
            task="regression",
            ranking="passthrough", 
            strategy="IncrementalSelection",
            model="RegressionQR",
            params={"metric": "mse", "table_batch_size": 32}
        )
        print("✓", config6)
        
        # Baseline configuration
        config7 = DiscoveryConfig(baseline=True, params={"custom_param": "value"})
        print("✓", config7)
        
        # Get valid parameter options
        config8 = DiscoveryConfig(task="regression", ranking="passthrough", strategy="ForwardSelection")
        print(f"Valid metric options: {config8.get_valid_param_options('metric')}")
        
    except ConfigValidationError as e:
        print("✗", e)
    
    # Invalid configurations (will raise errors)
    print("\nTesting invalid configurations:")
    
    test_cases = [
        {"task": "invalid_task"},
        {"task": "imputation", "strategy": "LASSO"},
        {"task": "regression", "ranking": "eta"},
        {"task": "classification", "ranking": "correlation"},
        {"task": "regression", "ranking": "correlation", "strategy": "IncrementalSelection"},
        {"baseline": True, "task": "regression"},  # baseline with other params
        {"strategy": "IncrementalSelection", "params": {"metric": "invalid_metric"}},  # invalid metric
        {"strategy": "IncrementalSelection", "params": {"table_batch_size": -1}},  # invalid batch size
    ]
    
    for case in test_cases:
        try:
            config = DiscoveryConfig(**case)
            print("✗ Should have failed:", case)
        except ConfigValidationError as e:
            print("✓ Correctly caught error:", str(e)[:80] + "...")