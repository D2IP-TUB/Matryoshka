import inspect
import sys
import os
import ast
from dataclasses import dataclass, field
from typing import Dict, Any, Set, ClassVar, Optional
from importlib import import_module

# Import DiscoveryConfig for validation
from augmentation.utils.config import DiscoveryConfig, ConfigValidationError


@dataclass
class ExperimentExecutor:
    """Experiment executor with intelligent parameter filtering based on config structure and algorithm validation."""
    
    init_args: Dict[str, Any] = field(default_factory=dict)
    run_args: Dict[str, Any] = field(default_factory=dict)
    discovery_config_args: Dict[str, Any] = field(default_factory=dict)
    unknown_args: Dict[str, Any] = field(default_factory=dict)
    _strict_validation: bool = field(default=True, init=False)
    
    # Always go to init_args
    INIT_PARAM_KEYS: ClassVar[Set[str]] = {
        'feature_selection_table_name',
        'overlap_table_name'
    }
    
    # Algorithm-specific run_args (only for arda/qcr/kitana)
    ALGORITHM_SPECIFIC_RUN_KEYS: ClassVar[Set[str]] = {
        'lake_table_sep',
        'data_lake_path', 
        'qcr_table_name'
    }
    
    # Algorithm name to module mapping
    ALGORITHM_MODULES: ClassVar[Dict[str, str]] = {
        'arda': 'augmentation.feature_selection.algorithms.arda',
        'qcr': 'augmentation.feature_selection.algorithms.qcr', 
        'kitana': 'augmentation.feature_selection.algorithms.kitana'
    }
    
    @classmethod
    def from_params(
        cls, 
        params: Dict[str, Any], 
        algorithm_name: str,
        strict_validation: bool = True
    ) -> 'ExperimentExecutor':
        """
        Create instance by filtering parameters based on config structure and algorithm requirements.
        
        Args:
            params: Dictionary containing all parameters from config
            algorithm_name: Name of the algorithm (from experiments.csv)
            strict_validation: If True, validate run_args against algorithm's run method signature
            
        Returns:
            ExperimentExecutor instance with filtered and validated parameters
        """
        init_args = {}
        run_args = {}
        discovery_config_args = {}
        unknown_args = {}
        
        # Get valid run method parameters for this algorithm
        valid_run_params = cls._get_algorithm_run_params(algorithm_name) if strict_validation else set()
        
        # For non-baseline algorithms, get valid discovery config parameters
        valid_discovery_params = cls._get_discovery_config_params() if not cls._is_baseline_algorithm(algorithm_name) else set()
        
        for key, value in params.items():
            if key in cls.INIT_PARAM_KEYS:
                # Always goes to init_args
                init_args[key] = value
            elif key == 'baseline':
                # baseline parameter is handled specially - never goes to discovery_config_args
                # For baseline algorithms, it can go to run_args if needed
                if cls._is_baseline_algorithm(algorithm_name):
                    run_args[key] = value
                # For non-baseline algorithms, we ignore it (DiscoveryConfig infers baseline=False)
            elif key == 'params' and cls._is_baseline_algorithm(algorithm_name):
                # For baseline algorithms (arda/qcr/kitana), merge params dict into run_args
                # Handle string representations of dictionaries
                parsed_params = cls._parse_params_value(value)
                if isinstance(parsed_params, dict):
                    for param_key, param_value in parsed_params.items():
                        run_args[param_key] = param_value
                else:
                    run_args[key] = parsed_params
            elif key in ['query_column_name', 'target_column_name']:
                # These always go to run_args for both baseline and non-baseline
                run_args[key] = value
            elif not cls._is_baseline_algorithm(algorithm_name) and key in valid_discovery_params:
                # For non-baseline algorithms, most parameters go to discovery_config_args
                # Handle string representations of dictionaries for params
                if key == 'params':
                    discovery_config_args[key] = cls._parse_params_value(value)
                else:
                    discovery_config_args[key] = value
            elif cls._is_config_run_param(key):
                # Everything under regression/classification sections
                # Handle string representations of dictionaries for params
                if key == 'params':
                    parsed_value = cls._parse_params_value(value)
                    if cls._is_baseline_algorithm(algorithm_name):
                        run_args[key] = parsed_value
                    else:
                        discovery_config_args[key] = parsed_value
                elif key == 'baseline':
                    # Baseline parameter should not go to discovery_config_args
                    # It's only used for algorithm classification
                    pass  # Skip baseline parameter - it's handled implicitly
                else:
                    if cls._is_baseline_algorithm(algorithm_name):
                        run_args[key] = value
                    else:
                        discovery_config_args[key] = value
            elif key in cls.ALGORITHM_SPECIFIC_RUN_KEYS:
                # Algorithm-specific parameters
                if cls._algorithm_supports_param(algorithm_name, key):
                    if cls._is_baseline_algorithm(algorithm_name):
                        run_args[key] = value
                    else:
                        discovery_config_args[key] = value
                else:
                    unknown_args[key] = value
            else:
                # Check if it's a valid run parameter for this algorithm
                if not strict_validation or not valid_run_params or key in valid_run_params:
                    # If we can't validate or it's valid, determine where to put it
                    # Common experiment parameters that should go to run_args
                    common_run_params = {
                        'join_paths_df_path', 'base_node_id', 'query_table', 'query_table_path', 
                        'buyer_sep', 'seller_sep'
                    }
                    if cls._is_baseline_algorithm(algorithm_name):
                        # For baseline algorithms, put in run_args
                        if key in common_run_params or not valid_run_params:
                            run_args[key] = value
                        else:
                            run_args[key] = value
                    else:
                        # For non-baseline algorithms, put in discovery_config_args
                        if key in valid_discovery_params or not valid_discovery_params:
                            discovery_config_args[key] = value
                        else:
                            unknown_args[key] = value
                else:
                    unknown_args[key] = value
        
        # For baseline algorithms, merge run_args into discovery_config_args by updating 'params' dictionary
        if cls._is_baseline_algorithm(algorithm_name) and run_args:
            # Initialize params dict in discovery_config_args if it doesn't exist
            if 'params' not in discovery_config_args:
                discovery_config_args['params'] = {}
            elif not isinstance(discovery_config_args['params'], dict):
                discovery_config_args['params'] = {}
            
            # Merge run_args into discovery_config_args['params']
            discovery_config_args['params'].update(run_args)
            
            # Clear run_args for baseline algorithms since they're now in discovery_config_args['params']
            run_args = {}

        instance = cls(
            init_args=init_args,
            run_args=run_args,
            discovery_config_args=discovery_config_args,
            unknown_args=unknown_args
        )
        
        # Store strict_validation flag for later use
        instance._strict_validation = strict_validation
        
        return instance
    
    @classmethod
    def _parse_params_value(cls, value: Any) -> Any:
        """
        Parse params value, handling string representations of dictionaries.
        
        Args:
            value: The value to parse (could be dict, string, or other)
            
        Returns:
            Parsed value (dict if it was a string representation of a dict, otherwise original value)
        """
        if isinstance(value, str):
            # Try to evaluate string as a dictionary literal
            try:
                # Use ast.literal_eval for safe evaluation of literals
                parsed_value = ast.literal_eval(value)
                return parsed_value
            except (ValueError, SyntaxError):
                # If parsing fails, return the original string
                return value
        else:
            # If it's not a string, return as-is
            return value
    
    @classmethod
    def validate_discovery_config(cls, discovery_config_args: Dict[str, Any]) -> Optional[str]:
        """
        Validate discovery_config_args using DiscoveryConfig class.
        
        Args:
            discovery_config_args: Dictionary containing discovery configuration parameters
            
        Returns:
            None if valid, error message string if invalid
        """
        if not discovery_config_args:
            return None
            
        try:
            # Extract DiscoveryConfig parameters from discovery_config_args
            config_params = {}
            
            # Map discovery_config_args to DiscoveryConfig parameters
            if 'task' in discovery_config_args:
                config_params['task'] = discovery_config_args['task']
            if 'ranking' in discovery_config_args:
                config_params['ranking'] = discovery_config_args['ranking']
            if 'strategy' in discovery_config_args:
                config_params['strategy'] = discovery_config_args['strategy']
            if 'model' in discovery_config_args:
                config_params['model'] = discovery_config_args['model']
            if 'params' in discovery_config_args:
                config_params['params'] = discovery_config_args['params']
            # Never include 'baseline' - DiscoveryConfig infers baseline=False for non-baseline algorithms
            
            # Create and validate DiscoveryConfig
            discovery_config = DiscoveryConfig(**config_params)
            return None  # Valid configuration
            
        except ConfigValidationError as e:
            return f"Discovery config validation failed: {str(e)}"
        except Exception as e:
            return f"Unexpected error during discovery config validation: {str(e)}"
    
    @classmethod
    def _is_config_run_param(cls, key: str) -> bool:
        """Check if parameter is from regression/classification config sections."""
        # These are typical config parameters that appear under regression/classification
        config_run_params = {
            'task', 'ranking', 'model', 'strategy', 'params',  # Removed 'baseline' 
            'tol', 'metric', 'alphas', 'cv', 'regression', 'sample_size',
            'top_k', 'n_iter'
        }
        return key in config_run_params
    
    @classmethod
    def _is_baseline_algorithm(cls, algorithm_name: str) -> bool:
        """Check if algorithm is a baseline algorithm that needs params dict expansion."""
        baseline_algorithms = {'arda', 'qcr', 'kitana'}
        return algorithm_name.lower() in baseline_algorithms
    
    @classmethod
    def _get_discovery_config_params(cls) -> Set[str]:
        """Get valid parameters for find_best_joins method in non-baseline algorithms."""
        # Based on find_best_joins method signature in JoinSelection class
        discovery_params = {
            'top_k', 'corr_threshold', 'n_jobs', 'debug',
            # Config parameters that go into DiscoveryConfig (excluding 'baseline')
            'task', 'ranking', 'model', 'strategy', 'params',
            'tol', 'metric', 'alphas', 'cv', 'regression', 'sample_size',
            'n_iter',
            # Algorithm-specific parameters for non-baselines (excluding qcr_table_name)
            'lake_table_sep', 'data_lake_path'
        }
        return discovery_params
    
    @classmethod
    def _algorithm_supports_param(cls, algorithm_name: str, param_key: str) -> bool:
        """Check if algorithm supports specific parameter."""
        # Algorithm-specific parameter support mapping
        algorithm_param_support = {
            'arda': {'data_lake_path', 'lake_table_sep'},
            'qcr': {'data_lake_path', 'lake_table_sep', 'qcr_table_name'},  # qcr_table_name only for QCR
            'kitana': {'data_lake_path', 'lake_table_sep'}
        }
        
        supported_params = algorithm_param_support.get(algorithm_name, set())
        return param_key in supported_params
    
    @classmethod
    def _get_algorithm_run_params(cls, algorithm_name: str) -> Set[str]:
        """
        Get valid run method parameters for the specified algorithm by inspecting its run method.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Set of valid parameter names for the algorithm's run method
        """
        try:
            # Add the project root to Python path for imports
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.join(current_dir, '..', '..')
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            module_path = cls.ALGORITHM_MODULES.get(algorithm_name)
            if not module_path:
                return set()
            
            # Import the algorithm module
            module = import_module(module_path)
            
            # Get the class name (assume it's AlgorithmNameAugmenter)
            class_name = f"{algorithm_name.title()}Augmenter"
            algorithm_class = getattr(module, class_name, None)
            
            if algorithm_class is None:
                return set()
            
            # Get the run method signature
            run_method = getattr(algorithm_class, 'run', None)
            if run_method is None:
                return set()
            
            # Extract parameter names from signature
            signature = inspect.signature(run_method)
            # Exclude 'self' and include **kwargs parameters
            valid_params = {
                param.name for param in signature.parameters.values() 
                if param.name != 'self'
            }
            
            return valid_params
            
        except Exception as e:
            # Silently fail and return empty set - we'll use fallback logic
            return set()
    
    def get_combined_params(self) -> Dict[str, Any]:
        """Get all parameters combined into one dictionary."""
        return {**self.init_args, **self.run_args, **self.discovery_config_args, **self.unknown_args}
    
    def has_unknown_params(self) -> bool:
        """Check if there are any unknown parameters."""
        return bool(self.unknown_args)
    
    def has_discovery_config_args(self) -> bool:
        """Check if there are any discovery config arguments."""
        return bool(self.discovery_config_args)
    
    def validate_for_algorithm(self, algorithm_name: str) -> Dict[str, Any]:
        """
        Validate current run_args against algorithm requirements and discovery_config_args.
        
        Args:
            algorithm_name: Name of the algorithm to validate against
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'algorithm_validation': {},
            'discovery_config_validation': {},
            'overall_valid': True,
            'messages': []
        }
        
        # Validate algorithm parameters only if strict validation was enabled
        if hasattr(self, '_strict_validation') and self._strict_validation:
            valid_params = self._get_algorithm_run_params(algorithm_name)
            
            if not valid_params:
                validation_result['algorithm_validation'] = {
                    'valid': True,  # If we can't validate, assume it's valid
                    'reason': f'Could not determine valid parameters for {algorithm_name} - assuming valid',
                    'missing_params': [],
                    'extra_params': []
                }
            else:
                current_params = set(self.run_args.keys())
                missing_params = valid_params - current_params - {'kwargs'}  # Exclude kwargs
                extra_params = current_params - valid_params
                
                algorithm_valid = len(extra_params) == 0
                validation_result['algorithm_validation'] = {
                    'valid': algorithm_valid,
                    'missing_params': list(missing_params),
                    'extra_params': list(extra_params),
                    'reason': f'Extra parameters: {extra_params}' if extra_params else 'Valid'
                }
                
                if not algorithm_valid:
                    validation_result['overall_valid'] = False
                    validation_result['messages'].append(f"Algorithm validation failed: {validation_result['algorithm_validation']['reason']}")
        else:
            # Non-strict validation - assume algorithm parameters are valid
            validation_result['algorithm_validation'] = {
                'valid': True,
                'reason': 'Skipped due to non-strict validation mode',
                'missing_params': [],
                'extra_params': []
            }
        
        # Validate discovery config for non-baseline algorithms
        if not self._is_baseline_algorithm(algorithm_name) and self.discovery_config_args:
            discovery_error = self.validate_discovery_config(self.discovery_config_args)
            if discovery_error:
                validation_result['discovery_config_validation'] = {
                    'valid': False,
                    'error': discovery_error
                }
                validation_result['overall_valid'] = False
                validation_result['messages'].append(f"Discovery config validation failed: {discovery_error}")
            else:
                validation_result['discovery_config_validation'] = {
                    'valid': True,
                    'error': None
                }
                validation_result['messages'].append("Discovery config validation passed")
        elif self._is_baseline_algorithm(algorithm_name):
            validation_result['discovery_config_validation'] = {
                'valid': True,
                'error': None,
                'note': 'Baseline algorithm - discovery config validation skipped'
            }
        else:
            validation_result['discovery_config_validation'] = {
                'valid': True,
                'error': None,
                'note': 'No discovery config args to validate'
            }
        
        return validation_result
    
    def get_discovery_config(self) -> Optional[DiscoveryConfig]:
        """
        Create a DiscoveryConfig instance from discovery_config_args.
        
        Returns:
            DiscoveryConfig instance if discovery_config_args is not empty and valid, None otherwise
        """
        if not self.discovery_config_args:
            return None
            
        try:
            config_params = {}
            
            # Map discovery_config_args to DiscoveryConfig parameters
            for key in ['task', 'ranking', 'strategy', 'model', 'params']:
                if key in self.discovery_config_args:
                    config_params[key] = self.discovery_config_args[key]
            # Never include 'baseline' - DiscoveryConfig infers baseline=False for non-baseline algorithms
            
            return DiscoveryConfig(**config_params)
        except Exception:
            return None