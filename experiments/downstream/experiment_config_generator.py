"""
Experiment Configuration Generator

This script generates experiment configurations as a Polars DataFrame based on
the provided lakes, tables, and algorithms from config.yml.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import yaml
import polars as pl


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file {config_path} not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration: {e}")
        sys.exit(1)


def get_available_options(config: Dict[str, Any]) -> Dict[str, Set[str]]:
    """Extract available lakes, tables, and algorithms from config."""
    lakes_info = config.get('lakes', {})
    
    available_lakes = set(lakes_info.keys())
    available_tables = set()
    available_algorithms = set()
    
    # Extract tables from all lakes
    for lake_data in lakes_info.values():
        for base_table in lake_data.get('base_tables', []):
            if isinstance(base_table, dict):
                available_tables.update(base_table.keys())
    
    # Extract algorithms from regression and classification
    for task_type in ['regression', 'classification']:
        if task_type in config:
            available_algorithms.update(config[task_type].keys())
    
    return {
        'lakes': available_lakes,
        'tables': available_tables,
        'algorithms': available_algorithms
    }


def filter_selections(requested: List[str], available: Set[str], option_type: str) -> List[str]:
    """Filter and validate user selections against available options."""
    if not requested:
        return list(available)
    
    # Check if all requested items are available
    invalid_items = set(requested) - available
    if invalid_items:
        print(f"Warning: The following {option_type} are not available: {invalid_items}")
        print(f"Available {option_type}: {sorted(available)}")
    
    # Return only valid items
    valid_items = [item for item in requested if item in available]
    
    if not valid_items:
        print(f"Error: No valid {option_type} selected.")
        sys.exit(1)
        
    return valid_items


def generate_experiment_configs(config: Dict[str, Any], 
                              selected_lakes: List[str],
                              selected_tables: List[str],
                              selected_algorithms: List[str]) -> pl.DataFrame:
    """Generate experiment configurations as a Polars DataFrame."""
    
    configurations = []
    lakes_info = config.get('lakes', {})
    
    for lake_name in selected_lakes:
        if lake_name not in lakes_info:
            continue
            
        lake_data = lakes_info[lake_name]
        lake_path = lake_data.get('data_lake_path', '')
        lake_sep = lake_data.get('lake_table_sep', ',')
        feature_selection_table = lake_data.get('feature_selection_table_name', '')
        overlap_table = lake_data.get('overlap_table_name', '')
        qcr_table = lake_data.get('qcr_table_name', '')
        
        # Process each base table in the lake
        for base_table_item in lake_data.get('base_tables', []):
            if not isinstance(base_table_item, dict):
                continue
                
            for table_name, table_configs in base_table_item.items():
                if table_name not in selected_tables:
                    continue
                    
                # Extract table properties
                table_task = None
                join_paths_df_path = None
                query_column_name = None
                target_column_name = None
                base_node_id = None
                
                for table_config in table_configs:
                    if isinstance(table_config, dict):
                        if 'task' in table_config:
                            table_task = table_config['task']
                        else:
                            join_paths_df_path = table_config.get('join_paths_df_path')
                            query_column_name = table_config.get('query_column_name')
                            target_column_name = table_config.get('target_column_name')
                            base_node_id = table_config.get('base_node_id')
                            query_table_path = table_config.get('query_table_path')
                if not table_task:
                    continue
                
                # Get algorithms for this task type
                task_algorithms = config.get(table_task, {})
                
                for algorithm_name in selected_algorithms:
                    if algorithm_name not in task_algorithms:
                        continue
                        
                    algorithm_config = task_algorithms[algorithm_name]
                    
                    # Create experiment configuration
                    exp_config = {
                        'lake': lake_name,
                        'data_lake_path': lake_path,
                        'lake_table_sep': lake_sep,
                        'feature_selection_table_name': feature_selection_table,
                        'overlap_table_name': overlap_table,
                        'qcr_table_name': qcr_table,
                        'table': table_name,
                        'task': table_task,
                        'algorithm': algorithm_name,
                        'join_paths_df_path': join_paths_df_path,
                        'query_column_name': query_column_name,
                        'target_column_name': target_column_name,
                        'base_node_id': base_node_id,
                        'baseline': algorithm_config.get('baseline', False),
                        'strategy': algorithm_config.get('strategy'),
                        'ranking': algorithm_config.get('ranking'),
                        'model': algorithm_config.get('model'),
                        'params': str(algorithm_config.get('params', {})),  # Convert dict to string for DataFrame
                        'query_table_path': query_table_path,
                    }
                    
                    configurations.append(exp_config)
    
    if not configurations:
        print("Warning: No experiment configurations generated with the current selections.")
        return pl.DataFrame()
    
    # Create Polars DataFrame
    df = pl.DataFrame(configurations)
    
    return df


def print_summary(df: pl.DataFrame, selected_lakes: List[str], 
                 selected_tables: List[str], selected_algorithms: List[str]):
    """Print a summary of the generated configurations."""
    print(f"\n🎯 Experiment Configuration Summary")
    print(f"{'='*50}")
    print(f"Selected Lakes: {', '.join(selected_lakes)}")
    print(f"Selected Tables: {', '.join(selected_tables)}")
    print(f"Selected Algorithms: {', '.join(selected_algorithms)}")
    print(f"\nGenerated {len(df)} experiment configurations")
    
    if len(df) > 0:
        print(f"\nBreakdown by:")
        print(f"  Lakes: {df['lake'].n_unique()} unique")
        print(f"  Tables: {df['table'].n_unique()} unique") 
        print(f"  Algorithms: {df['algorithm'].n_unique()} unique")
        print(f"  Tasks: {', '.join(df['task'].unique())}")
        print(f"  Columns: {len(df.columns)} total (including new lake-specific table names)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate experiment configurations from config.yml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all configurations
  python experiment_config_generator.py
  
  # Select specific lakes
  python experiment_config_generator.py --lakes nyc cuk
  
  # Select specific tables and algorithms
  python experiment_config_generator.py --tables crime inspections --algorithms forward lasso
  
  # Save to CSV file
  python experiment_config_generator.py --output experiments.csv
  
  # Show available options only
  python experiment_config_generator.py --list-options
        """)
    
    parser.add_argument('--config', 
                       type=Path,
                       default=Path('experiments/downstream/config.yml'),
                       help='Path to config.yml file (default: config.yml)')
    
    parser.add_argument('--lakes',
                       nargs='*',
                       help='Lakes to include (default: all available)')
    
    parser.add_argument('--tables', 
                       nargs='*',
                       help='Tables to include (default: all available)')
    
    parser.add_argument('--algorithms',
                       nargs='*', 
                       help='Algorithms to include (default: all available)')
    
    parser.add_argument('--output', '-o',
                       type=Path,
                       help='Output CSV file path (default: print to stdout)')
    
    parser.add_argument('--list-options',
                       action='store_true',
                       help='List all available lakes, tables, and algorithms')
    
    parser.add_argument('--format',
                       choices=['csv', 'json', 'parquet'],
                       default='csv',
                       help='Output format (default: csv)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    available_options = get_available_options(config)
    
    # List options and exit if requested
    if args.list_options:
        print("📋 Available Options:")
        print(f"\n🏞️  Lakes ({len(available_options['lakes'])}):")
        for lake in sorted(available_options['lakes']):
            print(f"  • {lake}")
        
        print(f"\n📊 Tables ({len(available_options['tables'])}):")
        for table in sorted(available_options['tables']):
            print(f"  • {table}")
            
        print(f"\n🔧 Algorithms ({len(available_options['algorithms'])}):")
        for algorithm in sorted(available_options['algorithms']):
            print(f"  • {algorithm}")
        
        return
    
    # Filter selections
    selected_lakes = filter_selections(args.lakes or [], available_options['lakes'], 'lakes')
    selected_tables = filter_selections(args.tables or [], available_options['tables'], 'tables') 
    selected_algorithms = filter_selections(args.algorithms or [], available_options['algorithms'], 'algorithms')
    
    # Generate configurations
    df = generate_experiment_configs(config, selected_lakes, selected_tables, selected_algorithms)
    
    if len(df) == 0:
        print("No configurations generated. Please check your selections.")
        return
    
    # Print summary
    print_summary(df, selected_lakes, selected_tables, selected_algorithms)
    
    # Output results
    if args.output:
        if args.format == 'csv':
            df.write_csv(args.output)
            print(f"\n💾 Saved {len(df)} configurations to {args.output}")
        elif args.format == 'json':
            df.write_json(args.output)
            print(f"\n💾 Saved {len(df)} configurations to {args.output}")
        elif args.format == 'parquet':
            df.write_parquet(args.output)
            print(f"\n💾 Saved {len(df)} configurations to {args.output}")
    else:
        print(f"\n📋 Experiment Configurations:")
        print("="*80)
        print(df)


if __name__ == '__main__':
    main()