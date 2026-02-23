#!/usr/bin/env python3
"""
Script to sample CSV files from data lakes, extract headers, and use AI to select
top datasets for ML tasks.
"""

import os
import json
import random
import csv
from pathlib import Path
from typing import Dict, List, Any

# Configuration
BASE_DIR = "/mnt/data1/lakes"
LAKES_CONFIG = {
    "nyc": {
        "path": "nyc/extracted",
        "separator": "\t"
    },
    "canada_us_uk_open_data": {
        "path": "canada_us_uk_open_data/extracted",
        "separator": ","
    }
}
SAMPLE_SIZE = 100
RANDOM_SEED = 42
OUTPUT_JSON = "sampled_lakes_headers.json"


def get_csv_files(directory: str) -> List[str]:
    """Get all CSV files from a directory recursively."""
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def sample_files(files: List[str], n: int, seed: int) -> List[str]:
    """Sample n files reproducibly from the list."""
    random.seed(seed)
    if len(files) <= n:
        return files
    return random.sample(files, n)


def get_header(filepath: str, separator: str) -> List[str]:
    """Extract header from a CSV file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f, delimiter=separator)
            header = next(reader, None)
            if header:
                return [col.strip() for col in header]
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return []


def get_table_name(filepath: str, lake_base: str) -> str:
    """Extract a meaningful table name from the filepath."""
    rel_path = os.path.relpath(filepath, lake_base)
    # Remove .csv extension and use path as name
    return rel_path.replace('.csv', '').replace(os.sep, '/')


def collect_lake_headers(lake_name: str, config: dict) -> Dict[str, List[str]]:
    """Collect headers from sampled files in a lake."""
    lake_path = os.path.join(BASE_DIR, config["path"])
    separator = config["separator"]
    
    print(f"\nProcessing lake: {lake_name}")
    print(f"  Path: {lake_path}")
    
    # Get all CSV files
    all_files = get_csv_files(lake_path)
    print(f"  Total CSV files found: {len(all_files)}")
    
    # Sample files
    sampled = sample_files(all_files, SAMPLE_SIZE, RANDOM_SEED)
    print(f"  Sampled files: {len(sampled)}")
    
    # Collect headers
    tables = {}
    for filepath in sampled:
        table_name = get_table_name(filepath, lake_path)
        header = get_header(filepath, separator)
        if header:
            tables[table_name] = header
    
    print(f"  Successfully extracted headers: {len(tables)}")
    return tables


def main():
    """Main function to collect headers and save to JSON."""
    print("=" * 60)
    print("Data Lake Header Extraction")
    print("=" * 60)
    
    result = {}
    
    for lake_name, config in LAKES_CONFIG.items():
        tables = collect_lake_headers(lake_name, config)
        result[lake_name] = tables
    
    # Save to JSON
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")
    
    # Print summary
    total_tables = sum(len(tables) for tables in result.values())
    print(f"\nSummary:")
    for lake_name, tables in result.items():
        print(f"  {lake_name}: {len(tables)} tables")
    print(f"  Total: {total_tables} tables")
    
    return result


def analyze_with_ai(json_path: str) -> str:
    """
    Send the collected headers to an AI agent for analysis.
    This function uses Google Gemini API - make sure GOOGLE_API_KEY is set.
    """
    try:
        import google.generativeai as genai
    except ImportError:
        print("Google Generative AI package not installed. Install with: pip install google-generativeai")
        return ""
    
    # Configure with API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    genai.configure(api_key=api_key)
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Prepare the prompt
    prompt = f"""You are a data science expert. I have sampled datasets from two data lakes.
Below is a JSON structure containing table names and their column headers.

{json.dumps(data, indent=2)}

Please analyze these datasets and select the TOP 10 datasets that would be most suitable 
for regression or classification machine learning tasks and have a meaningful join key to find other tables in the lake (not lat/lon coordinates).

For each selected dataset, provide:
1. The lake name and table name
2. The columns that could serve as join keys to connect with other tables in the lake (excluding lat/lon coordinates)
3. The columns that could serve as target variables
4. The columns that would be useful as features
5. What type of ML task (regression/classification) it's best suited for
6. A brief explanation of why this dataset is suitable

Format your response as a structured analysis."""

    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content(prompt)
    
    return response.text


if __name__ == "__main__":
    # Step 1-3: Collect headers and save to JSON
    main()
    
    # Step 4: Analyze with AI
    output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON)
    
    print("\n" + "=" * 60)
    print("AI Analysis")
    print("=" * 60)
    
    try:
        analysis = analyze_with_ai(output_path)
        if analysis:
            print(analysis)
            
            # Save AI analysis to file
            analysis_path = os.path.join(os.path.dirname(__file__), "ai_analysis_results.md")
            with open(analysis_path, 'w') as f:
                f.write(analysis)
            print(f"\nAI analysis saved to: {analysis_path}")
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        print("Make sure GOOGLE_API_KEY environment variable is set.")
