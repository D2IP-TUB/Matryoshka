from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from .config import DiscoveryConfig


class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """Represents a single step in the execution plan."""
    name: str
    step_type: str  # e.g., "query", "ranking", "strategy", "model"
    function: Callable
    params: Dict[str, Any]
    dependencies: List[str] = None  # Step names this depends on
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class ExecutionContext:
    """Holds the state and results during plan execution."""

    def __init__(self):
        self.results: Dict[str, Any] = {}
        self.intermediate_data: Dict[str, Any] = {}
        self.config: Optional[DiscoveryConfig] = None

    def set_result(self, step_name: str, result: Any):
        """Store the result of a step."""
        self.results[step_name] = result

    def get_result(self, step_name: str) -> Any:
        """Get the result of a completed step."""
        return self.results.get(step_name)

    def set_data(self, key: str, value: Any):
        """Store intermediate data."""
        self.intermediate_data[key] = value

    def get_data(self, key: str) -> Any:
        """Get intermediate data."""
        return self.intermediate_data.get(key)


class PlanBuilder:
    """Builds execution plans based on DiscoveryConfig."""

    def __init__(self, function_registry: Dict[str, Callable]):
        """
        Initialize with a registry of available functions.
        
        Args:
            function_registry: Dict mapping function names to callable functions
        """
        self.task_rules = DiscoveryConfig(baseline=True)._task_rules
        self.functions = function_registry
        self._default_strategies = {
            'regression': 'LassoFeatureSelector',
            'classification': 'LassoFeatureSelector'
        }
        self._default_models = {
            ('regression', 'LassoFeatureSelector'): 'LinRegL1',
            ('classification', 'LassoFeatureSelector'): 'LogRegL1',
            ('regression', 'IncrementalSelection'): 'RegressionQR',
            ('regression', 'ForwardSelection'): 'RegressionQR',
            ('regression', 'BackwardElimination'): 'RegressionQR',
            ('classification', 'IncrementalSelection'): 'ClassificationCholesky',
            ('classification', 'ForwardSelection'): 'ClassificationCholesky',
            ('classification', 'BackwardElimination'): 'ClassificationCholesky',
            ('regression', 'None'): 'RegressionQR',
            ('classification', 'None'): 'ClassificationCholesky'
        }

    def build_plan(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Build an execution plan based on the configuration."""
        if config.baseline:
            return self._build_baseline_plan(config)

        plan = []

        # Step 1: Always start with overlap and token queries
        plan.extend(self._add_initial_queries())

        if config.strategy == 'IncrementalSelection':
            # incremental selection requires a loop entry step to run steps 2-4 iteratively
            plan.extend(self._add_loop_entry_step(config))

        # Step 2: Add ranking step
        # LassoFeatureSelector strategy does not need ranking
        # Ranking step also implies sketches computation which are not used in this strategy
        if config.strategy != 'LassoFeatureSelector':
            plan.extend(self._add_ranking_step(config))

        # Step 3: Handle different task types
        last_step_name = None

        if config.task in ['regression explanation', 'classification explanation']:
            # Explanation tasks: ranking -> collinearity (no model step needed)
            plan.extend(self._add_collinearity_step(config))
            last_step_name = 'collinearity_analysis'
        elif config.task == 'imputation':
            # Imputation tasks: ranking -> collinearity -> strategy_with_model
            plan.extend(self._add_collinearity_step(config))
            plan.extend(self._add_imputation_strategy_step(config))
            last_step_name = 'strategy_with_model'
        elif config.task in ['regression', 'classification']:
            # Check if we need collinearity analysis
            if self._needs_collinearity_analysis(config):
                plan.extend(self._add_collinearity_step(config))

            # Check if we need strategy execution
            if self._needs_strategy(config):
                plan.extend(self._add_strategy_step(config))
                last_step_name = 'strategy_with_model'
            # If no strategy is needed, we might still need a direct model step
            elif self._needs_direct_model(config):
                plan.extend(self._add_direct_model_step(config))
                last_step_name = 'direct_model'
            else:
                last_step_name = 'collinearity_analysis'

        if config.strategy == 'IncrementalSelection':
            # Add loop exit step to conclude the iterative strategy
            plan.extend(self._add_loop_exit_step(config))
            last_step_name = 'loop_exit'

        # Step 4: Always end with augmentation (non-baseline only)
        plan.extend(self._add_augmentation_step(config, last_step_name))

        return plan

    def _needs_collinearity_analysis(self, config: DiscoveryConfig) -> bool:
        """Determine if a collinearity analysis step is needed."""
        # Collinearity analysis is needed for regression and classification tasks
        if config.ranking == 'passthrough' or config.strategy == 'LassoFeatureSelector':
            return False
        return True

    def _needs_strategy(self, config: DiscoveryConfig) -> bool:
        """Determine if a strategy step is needed."""
        # Only need strategy if it's explicitly specified
        if config.strategy:
            return True

        # If no strategy is specified, we don't need a strategy step
        return False

    def _needs_direct_model(self, config: DiscoveryConfig) -> bool:
        """Determine if a direct model step is needed (without strategy)."""
        # If we already determined we need strategy, don't need direct model
        if self._needs_strategy(config):
            return False

        # If a model is explicitly specified, we might need a direct model step
        if config.model:
            return True

        # For now, assume no direct model step is needed if no strategy
        return False

    def _get_task_rule(self, task: str) -> Dict:
        """Get the rule dictionary for a specific task."""
        return self.task_rules.get(task, {})

    def _build_baseline_plan(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Build a plan for baseline configuration."""
        return [
            PlanStep(
                name='baseline_execution',
                step_type='baseline',
                function=self.functions.get('run_baseline', self._default_baseline),
                params=config.params or {}
            )
        ]

    def _add_initial_queries(self) -> List[PlanStep]:
        """Add the common initial steps."""
        return [
            PlanStep(
                name='find_joinable_tables',
                step_type='query',
                function=self.functions.get('run_find_joinable_tables', self._default_find_joinable_tables),
                params={}
            )
        ]

    def _add_ranking_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add ranking step based on config."""
        ranking_method = config.ranking or 'passthrough'  # Default to passthrough

        return [
            PlanStep(
                name='ranking',
                step_type='ranking',
                function=self.functions.get('run_ranking', self._default_ranking),
                params={'method': ranking_method},
                dependencies=['find_joinable_tables']
            )
        ]

    def _add_loop_entry_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add a loop entry step for iterative strategies."""
        return [
            PlanStep(
                name='loop_entry',
                step_type='incremental_selection_loop',
                function=self.functions.get('run_loop_entry', self._default_loop_entry),
                params={
                    'strategy': config.strategy,
                    'task': config.task,
                    'method': config.ranking,
                    'model': config.model,
                    **config.params

                },
                dependencies=['find_joinable_tables']
            )
        ]

    def _add_loop_exit_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add a loop exit step to conclude iterative strategies."""
        return [
            PlanStep(
                name='loop_exit',
                step_type='incremental_selection_loop_exit',
                function=self.functions.get('run_loop_exit', self._default_loop_exit),
                params={'strategy': config.strategy},
                dependencies=['strategy_with_model']
            )
        ]

    def _add_collinearity_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add collinearity analysis step. This step uses the model to analyze feature collinearity."""
        # Determine the model to use for collinearity analysis
        model = self._get_collinearity_model(config)

        collinearity_params = {
            'model': model,
            'task': config.task
        }

        # Add any relevant config parameters
        if config.params:
            # Filter params that are relevant to collinearity analysis
            collinearity_params.update({
                k: v for k, v in config.params.items()
                if k in ['correlation_threshold', 'vif_threshold', 'condition_number_threshold']
            })

        return [
            PlanStep(
                name='collinearity_analysis',
                step_type='collinearity',
                function=self.functions.get('run_collinearity_analysis', self._default_collinearity),
                params=collinearity_params,
                dependencies=['ranking']
            )
        ]

    def _add_strategy_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add strategy step based on config. The model is integrated into the strategy."""
        # Strategy should only be called if explicitly specified
        if not config.strategy:
            raise ValueError("Strategy step should only be added when strategy is explicitly specified")

        strategy = config.strategy
        model = config.model or self._get_model_for_strategy(config, strategy)

        # Prepare strategy parameters
        strategy_params = config.params.copy() if config.params else {}
        strategy_params['strategy'] = strategy
        strategy_params['model'] = model  # Include model in strategy params
        if config.ranking in ['joinability', 'eta', 'correlation']:
            dependencies = ['collinearity_analysis']
        elif config.ranking is None:
            dependencies = ['find_joinable_tables']
        else:
            dependencies = ['ranking']

        return [
            PlanStep(
                name='strategy_with_model',
                step_type='strategy',
                function=self.functions.get('run_strategy_with_model', self._default_strategy),
                params=strategy_params,
                dependencies=dependencies
            )
        ]

    def _add_explanation_model_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add model step for explanation tasks (no strategy needed)."""
        model = config.model or 'QR'  # Default model for explanation tasks

        return [
            PlanStep(
                name='explanation_model',
                step_type='model',
                function=self.functions.get('run_model}', self._default_model),
                params={'model': model, 'task': config.task},
                dependencies=['collinearity_analysis']  # Now depends on collinearity
            )
        ]

    def _add_direct_model_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add direct model step when no strategy is used."""
        model = config.model or f'{config.task.title()}QR'  # Fallback model

        return [
            PlanStep(
                name='direct_model',
                step_type='model',
                function=self.functions.get('run_direct_model', self._default_model),
                params={'model': model, 'task': config.task},
                dependencies=['collinearity_analysis']  # Now depends on collinearity
            )
        ]

    def _get_default_strategy(self, config: DiscoveryConfig) -> str:
        """Get default strategy for a task."""
        task_base = config.task.split()[0]  # Handle "regression explanation" -> "regression"
        return self._default_strategies.get(task_base, 'LASSO')

    def _get_model_for_strategy(self, config: DiscoveryConfig, strategy: str) -> str:
        """Get the appropriate model for a given task and strategy."""
        task_base = config.task.split()[0]
        model_key = (task_base, strategy)
        return self._default_models.get(model_key, f'{task_base.title()}QR')

    def _get_collinearity_model(self, config: DiscoveryConfig) -> str:
        """Get the appropriate model for collinearity analysis."""
        # For collinearity analysis, we typically want to use a basic model
        # that can compute correlation matrices and condition numbers

        if config.model:
            # If a specific model is configured, use it
            return config.model

        task_base = config.task.split()[0]  # Handle "regression explanation" -> "regression"

        # For collinearity analysis, QR decomposition is often preferred
        # as it provides good numerical stability for correlation analysis
        if task_base == 'regression':
            return 'RegressionQR'
        elif task_base == 'classification':
            return 'ClassificationCholesky'
        elif task_base == 'imputation':
            # For imputation, we might use a simpler correlation-based approach
            # or a specialized imputation collinearity model
            return 'ImputationQR'
        else:
            # Fallback for other tasks
            return 'QR'

    def _add_imputation_strategy_step(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Add strategy step specifically for imputation tasks."""
        # For imputation, we always use a fixed strategy and model
        strategy = 'imputation'  # Fixed strategy for imputation tasks
        model = 'imputation'     # Fixed model for imputation tasks

        # Prepare strategy parameters - use config params for imputation-specific settings
        strategy_params = config.params.copy() if config.params else {}
        strategy_params['strategy'] = strategy
        strategy_params['model'] = model

        return [
            PlanStep(
                name='strategy_with_model',
                step_type='strategy',
                function=self.functions.get('run_imputation', self._default_imputation_strategy),
                params=strategy_params,
                dependencies=['collinearity_analysis']
            )
        ]

    def _add_augmentation_step(self, config: DiscoveryConfig, last_step_name: str) -> List[PlanStep]:
        """Add the final augmentation step that concludes any non-baseline plan."""
        augmentation_params = {
            'ranking': config.ranking,
            'strategy': config.strategy,
            'model': config.model
        }

        # Add any relevant config parameters
        if config.params:
            augmentation_params['config_params'] = config.params

        # Determine the dependency based on the last step
        dependencies = [last_step_name] if last_step_name else []

        return [
            PlanStep(
                name='augmentation',
                step_type='augmentation',
                function=self.functions.get('run_augmentation', self._default_augmentation),
                params=augmentation_params,
                dependencies=dependencies
            )
        ]

    def _default_find_joinable_tables(self, context: ExecutionContext, **kwargs):
        return {'overlap_data': 'mock_overlap_result'}

    def _default_loop_entry(self, context: ExecutionContext, strategy: str, **kwargs):
        return {'loop_entry': f'Initialized loop for strategy {strategy}'}

    def _default_loop_exit(self, context: ExecutionContext, strategy: str, **kwargs):
        return {'loop_exit': f'Concluded loop for strategy {strategy}'}

    def _default_ranking(self, context: ExecutionContext, method: str, **kwargs):
        return {'ranked_features': ['feature1', 'feature2', 'feature3']}

    def _default_collinearity(self, context: ExecutionContext, model: str, task: str, **kwargs):
        """Default collinearity analysis implementation."""
        # Simulate collinearity analysis results
        result = {
            'model_used': model,
            'task': task,
            'correlation_matrix': 'mock_correlation_matrix',
            'vif_scores': {'feature1': 1.2, 'feature2': 2.1, 'feature3': 1.5},
            'condition_number': 12.5,
            'collinear_features': ['feature2'],  # Features identified as potentially collinear
            'recommended_features': ['feature1', 'feature3']  # Features recommended after analysis
        }

        # Include any threshold parameters that were passed
        for param in ['correlation_threshold', 'vif_threshold', 'condition_number_threshold']:
            if param in kwargs:
                result[param] = kwargs[param]

        return result

    def _default_strategy(self, context: ExecutionContext, strategy: str, model: str = None, **kwargs):
        """Default strategy implementation that uses the model internally."""
        result = {
            'selected_features': ['feature1', 'feature2'],
            'strategy_used': strategy
        }

        if model:
            result['model_used'] = model
            result['model_result'] = f'trained_{model}_within_{strategy}'

        return result

    def _default_model(self, context: ExecutionContext, model: str, task: str = None, **kwargs):
        return {'model_result': f'trained_{model}', 'task': task}

    def _default_baseline(self, context: ExecutionContext, **kwargs):
        return {'baseline_result': 'baseline_complete'}

    def _default_imputation_strategy(self, context: ExecutionContext, strategy: str, model: str = None, **kwargs):
        """Default imputation strategy implementation that uses the model internally."""
        result = {
            'selected_features': ['feature1', 'feature3'],  # Features selected for imputation
            'strategy_used': strategy,
            'imputation_method': 'iterative'
        }

        if model:
            result['model_used'] = model
            result['model_result'] = f'trained_{model}_for_imputation'

        # Include imputation-specific parameters
        if 'tol' in kwargs:
            result['tolerance'] = kwargs['tol']
        if 'n_iterations' in kwargs:
            result['max_iterations'] = kwargs['n_iterations']

        return result

    def _default_augmentation(self, context: ExecutionContext, **kwargs):
        """Default augmentation implementation that concludes the pipeline."""
        # Collect results from all previous steps
        all_results = context.results.copy()

        # Create task-independent augmentation summary
        result = {
            'augmentation_type': 'final_step',
            'pipeline_summary': {
                'total_steps_completed': len([r for r in all_results.values() if r]),
                'completed_steps': list(all_results.keys())
            }
        }

        # Add configuration information that was passed
        result['ranking_method'] = kwargs.get('ranking')
        result['strategy_used'] = kwargs.get('strategy')
        result['model_used'] = kwargs.get('model')

        # Extract and combine results from key pipeline steps
        if 'ranking' in all_results:
            ranking_result = all_results['ranking']
            result['ranked_features'] = ranking_result.get('ranked_features', [])

        if 'collinearity_analysis' in all_results:
            collinearity_result = all_results['collinearity_analysis']
            result['recommended_features'] = collinearity_result.get('recommended_features', [])
            result['collinear_features'] = collinearity_result.get('collinear_features', [])
            result['vif_scores'] = collinearity_result.get('vif_scores', {})

        if 'strategy_with_model' in all_results:
            strategy_result = all_results['strategy_with_model']
            result['selected_features'] = strategy_result.get('selected_features', [])
            result['model_result'] = strategy_result.get('model_result')

        if 'explanation_model' in all_results:
            explanation_result = all_results['explanation_model']
            result['explanation_result'] = explanation_result.get('model_result')

        if 'direct_model' in all_results:
            direct_result = all_results['direct_model']
            result['direct_model_result'] = direct_result.get('model_result')

        # Add any additional config parameters
        if 'config_params' in kwargs:
            result['config_params'] = kwargs['config_params']

        # Create final recommendations based on available results
        recommendations = []

        if result.get('recommended_features') and result.get('selected_features'):
            # Compare collinearity recommendations with strategy selections
            recommended_set = set(result['recommended_features'])
            selected_set = set(result['selected_features'])

            if recommended_set == selected_set:
                recommendations.append("Strategy selection aligns with collinearity analysis")
            else:
                overlap = recommended_set.intersection(selected_set)
                recommendations.append(f"Feature overlap between collinearity and strategy: {list(overlap)}")

        elif result.get('recommended_features'):
            recommendations.append(f"Use recommended features: {result['recommended_features']}")

        if result.get('collinear_features'):
            recommendations.append(f"Consider removing collinear features: {result['collinear_features']}")

        result['recommendations'] = recommendations

        return result


class PlanExecutor:
    """Executes plans step by step."""

    def __init__(self):
        self.context = ExecutionContext()

    def execute_plan(self, plan: List[PlanStep], config: DiscoveryConfig, verbose: bool = False) -> ExecutionContext:
        """Execute a complete plan."""
        self.context.config = config

        if verbose:
            print(f'Executing plan with {len(plan)} steps...')
            print('=' * 50)

        for step in plan:
            if verbose:
                print(f'--- Step: {step.name} ({step.status}) ---')
            if not self._dependencies_satisfied(step, plan):
                step.status = StepStatus.FAILED
                step.error = Exception(f'Dependencies not satisfied: {step.dependencies}')
                if verbose:
                    print(f'❌ {step.name}: Dependencies not satisfied')
                continue
            elif step.status == StepStatus.COMPLETED:
                if verbose:
                    print(f'⏭️ Skipping: {step.name} ({step.step_type})')
                continue

            try:
                step.status = StepStatus.RUNNING
                if verbose:
                    print(f'🔄 Executing: {step.name} ({step.step_type})')
                    print(f'Dependencies: {step.dependencies}')

                # Execute the step function
                result = step.function(self.context, **step.params)

                step.result = result
                step.status = StepStatus.COMPLETED
                self.context.set_result(step.name, result)

                if verbose:
                    print(f'✅ Completed: {step.name}')
                    if result:
                        print(f'   Result: {result}')

            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = e
                if verbose:
                    print(f'❌ Failed: {step.name} - {str(e)}')
                raise e

        if verbose:
            print('=' * 50)
            print('Plan execution completed!')

        return self.context

    def _dependencies_satisfied(self, step: PlanStep, plan: List[PlanStep]) -> bool:
        """Check if all dependencies for a step are satisfied."""
        if not step.dependencies:
            return True

        completed_steps = {s.name for s in plan if s.status == StepStatus.COMPLETED}
        return all(dep in completed_steps for dep in step.dependencies)


class DiscoveryPlanner:
    """Main planner class that orchestrates plan building and execution."""

    def __init__(self, function_registry: Optional[Dict[str, Callable]] = None):
        """
        Initialize the planner.
        
        Args:
            function_registry: Optional dict of custom functions to use
        """
        self.function_registry = function_registry or {}
        self.builder = PlanBuilder(self.function_registry)
        self.executor = PlanExecutor()
        self.graph_generator = PlanGraphGenerator()

    def create_plan(self, config: DiscoveryConfig) -> List[PlanStep]:
        """Create an execution plan for the given configuration."""
        return self.builder.build_plan(config)

    def execute_plan(self, plan: List[PlanStep], config: DiscoveryConfig, verbose: bool = False) -> ExecutionContext:
        """Execute a plan."""
        return self.executor.execute_plan(plan, config, verbose)

    def create_and_execute_plan(self, config: DiscoveryConfig, verbose: bool = False) -> ExecutionContext:
        """Create and execute a plan for the given configuration."""
        # Build the plan
        plan = self.create_plan(config)

        if verbose:
            print(f'Generated plan for config: {config}')
            print(f'Plan steps: {[step.name for step in plan]}')
            print()

        # Execute the plan
        return self.execute_plan(plan, config, verbose)

    def register_function(self, name: str, function: Callable):
        """Register a custom function."""
        self.function_registry[name] = function
        self.builder.functions[name] = function

    def visualize_plan(self, plan: List[PlanStep], format: str = 'mermaid') -> str:
        """Generate a visual representation of the plan."""
        return self.graph_generator.generate_graph(plan, format)

    def create_and_visualize_plan(self, config: DiscoveryConfig, format: str = 'mermaid') -> str:
        """Create a plan and return its visual representation."""
        plan = self.create_plan(config)
        return self.visualize_plan(plan, format)


class PlanGraphGenerator:
    """Generates visual representations of execution plans."""

    def __init__(self):
        self.step_colors = {
            'query': '#e1f5fe',      # Light blue
            'ranking': '#f3e5f5',    # Light purple
            'collinearity': '#fff3e0', # Light orange
            'strategy': '#e8f5e8',   # Light green
            'model': '#fff8e1',      # Light yellow
            'baseline': '#f5f5f5',   # Light gray
            'augmentation': '#fce4ec' # Light pink
        }

    def generate_graph(self, plan: List[PlanStep], format: str = 'mermaid') -> str:
        """
        Generate a visual graph representation of the plan.
        
        Args:
            plan: List of plan steps
            format: Output format ('mermaid', 'dot', 'ascii')
            
        Returns:
            String representation of the graph
        """
        if format.lower() == 'mermaid':
            return self._generate_mermaid_graph(plan)
        elif format.lower() == 'dot':
            return self._generate_dot_graph(plan)
        elif format.lower() == 'ascii':
            return self._generate_ascii_graph(plan)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _generate_mermaid_graph(self, plan: List[PlanStep]) -> str:
        """Generate Mermaid flowchart syntax."""
        lines = ['graph TD']

        # Add nodes with styling
        for step in plan:
            node_id = step.name.replace(' ', '_')
            color = self.step_colors.get(step.step_type, '#f9f9f9')

            # Create node with type and status info
            status_icon = self._get_status_icon(step.status)
            node_label = f"{step.name}\\n({step.step_type}){status_icon}"

            lines.append(f'    {node_id}["{node_label}"]')
            lines.append(f'    {node_id} --> {node_id}')  # Self-reference for styling
            lines.append(f'    style {node_id} fill:{color},stroke:#333,stroke-width:2px')

        # Add dependencies (edges)
        for step in plan:
            node_id = step.name.replace(' ', '_')
            for dep in step.dependencies:
                dep_id = dep.replace(' ', '_')
                lines.append(f'    {dep_id} --> {node_id}')

        return '\n'.join(lines)

    def _generate_dot_graph(self, plan: List[PlanStep]) -> str:
        """Generate Graphviz DOT syntax."""
        lines = ['digraph PlanGraph {']
        lines.append('    rankdir=TD;')
        lines.append('    node [shape=rectangle, style=filled];')

        # Add nodes
        for step in plan:
            node_id = step.name.replace(' ', '_')
            color = self.step_colors.get(step.step_type, '#f9f9f9')
            status_icon = self._get_status_icon(step.status, unicode=False)

            label = f"{step.name}\\n({step.step_type}){status_icon}"
            lines.append(f'    {node_id} [label="{label}", fillcolor="{color}"];')

        # Add edges
        for step in plan:
            node_id = step.name.replace(' ', '_')
            for dep in step.dependencies:
                dep_id = dep.replace(' ', '_')
                lines.append(f'    {dep_id} -> {node_id};')

        lines.append('}')
        return '\n'.join(lines)

    def _generate_ascii_graph(self, plan: List[PlanStep]) -> str:
        """Generate ASCII art representation."""
        lines = ['Execution Plan Graph:', '=' * 50]

        # Create dependency map
        dep_map = {}
        for step in plan:
            dep_map[step.name] = step.dependencies

        # Topological sort for proper ordering
        ordered_steps = self._topological_sort(plan)

        # Generate ASCII representation
        level_map = self._calculate_levels(ordered_steps, dep_map)
        max_level = max(level_map.values()) if level_map else 0

        for level in range(max_level + 1):
            level_steps = [step for step in ordered_steps if level_map[step.name] == level]

            if level_steps:
                lines.append(f'\nLevel {level}:')
                lines.append('-' * 20)

                for step in level_steps:
                    status_icon = self._get_status_icon(step.status, unicode=False)
                    lines.append(f'  [{step.step_type.upper()}] {step.name}{status_icon}')

                    if step.dependencies:
                        lines.append(f'    Dependencies: {", ".join(step.dependencies)}')

        return '\n'.join(lines)

    def _get_status_icon(self, status: StepStatus, unicode: bool = True) -> str:
        """Get icon representation for step status."""
        if unicode:
            status_icons = {
                StepStatus.PENDING: ' ⏳',
                StepStatus.RUNNING: ' 🔄',
                StepStatus.COMPLETED: ' ✅',
                StepStatus.FAILED: ' ❌',
                StepStatus.SKIPPED: ' ⏭️'
            }
        else:
            status_icons = {
                StepStatus.PENDING: ' [PENDING]',
                StepStatus.RUNNING: ' [RUNNING]',
                StepStatus.COMPLETED: ' [DONE]',
                StepStatus.FAILED: ' [FAILED]',
                StepStatus.SKIPPED: ' [SKIPPED]'
            }

        return status_icons.get(status, '')

    def _topological_sort(self, plan: List[PlanStep]) -> List[PlanStep]:
        """Sort steps in topological order."""
        # Simple topological sort implementation
        result = []
        temp_mark = set()
        perm_mark = set()

        def visit(step):
            if step.name in perm_mark:
                return
            if step.name in temp_mark:
                return  # Skip circular dependencies

            temp_mark.add(step.name)

            # Visit dependencies first
            for dep_name in step.dependencies:
                dep_step = next((s for s in plan if s.name == dep_name), None)
                if dep_step:
                    visit(dep_step)

            temp_mark.remove(step.name)
            perm_mark.add(step.name)
            result.insert(0, step)  # Insert at beginning for reverse order

        for step in plan:
            if step.name not in perm_mark:
                visit(step)

        return list(reversed(result))

    def _calculate_levels(self, ordered_steps: List[PlanStep], dep_map: Dict[str, List[str]]) -> Dict[str, int]:
        """Calculate the level (depth) of each step in the graph."""
        levels = {}

        for step in ordered_steps:
            if not step.dependencies:
                levels[step.name] = 0
            else:
                max_dep_level = max(levels.get(dep, 0) for dep in step.dependencies)
                levels[step.name] = max_dep_level + 1

        return levels
