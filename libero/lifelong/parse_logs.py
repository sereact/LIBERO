#!/usr/bin/env python3
"""
Script to parse LIBERO experiment log files and generate a summary table.

Usage:
    python parse_logs.py <logs_folder>
    
Example:
    python parse_logs.py runs_libero_external_api
"""

import os
import re
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd


def extract_language_instruction(log_content: str) -> Optional[str]:
    """Extract the language instruction from the log content."""
    match = re.search(r'Language Instruction:\s+(.+)', log_content)
    if match:
        return match.group(1).strip()
    return None


def extract_success_rate(log_content: str) -> Optional[float]:
    """Extract the success rate from the log content."""
    # Look for the pattern "Success: <value>"
    match = re.search(r'Success:\s+([\d.]+)', log_content)
    if match:
        return float(match.group(1))
    return None


def parse_log_file(log_path: Path) -> Dict[str, any]:
    """Parse a single log file and extract relevant information."""
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        language_instruction = extract_language_instruction(content)
        success_rate = extract_success_rate(content)
        
        return {
            'language_instruction': language_instruction,
            'success_rate': success_rate
        }
    except Exception as e:
        print(f"Error parsing {log_path}: {e}", file=sys.stderr)
        return {
            'language_instruction': None,
            'success_rate': None
        }


def parse_logs_directory(logs_folder: str) -> List[Dict[str, any]]:
    """
    Parse all log files in the given directory structure.
    
    Expected structure:
    logs_folder/
        suite_name/
            task_0/
                seed_100.log
            task_1/
                seed_100.log
            ...
    """
    logs_path = Path(logs_folder)
    
    if not logs_path.exists():
        print(f"Error: Directory '{logs_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)
    
    results = []
    
    # Iterate through suite directories
    for suite_dir in sorted(logs_path.iterdir()):
        if not suite_dir.is_dir():
            continue
        
        suite_name = suite_dir.name
        
        # Iterate through task directories
        for task_dir in sorted(suite_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            
            # Extract task_id from directory name (e.g., "task_0" -> 0)
            task_match = re.match(r'task_(\d+)', task_dir.name)
            if not task_match:
                continue
            
            task_id = int(task_match.group(1))
            
            # Look for log files in the task directory
            for log_file in task_dir.glob('*.log'):
                log_data = parse_log_file(log_file)
                
                results.append({
                    'suite_name': suite_name,
                    'task_id': task_id,
                    'language_instruction': log_data['language_instruction'],
                    'success_rate': log_data['success_rate'],
                    'log_file': str(log_file.relative_to(logs_path))
                })
    
    return results


def create_summary_table(results: List[Dict[str, any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from the parsed results."""
    df = pd.DataFrame(results)
    
    # Sort by suite_name and task_id
    df = df.sort_values(['suite_name', 'task_id']).reset_index(drop=True)
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Parse LIBERO experiment log files and generate a summary table.'
    )
    parser.add_argument(
        'logs_folder',
        type=str,
        help='Path to the logs folder (e.g., runs_libero_external_api or runs_libero_external_api/logs)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path (optional, defaults to stdout)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'markdown', 'latex'],
        default='csv',
        help='Output format (default: csv)'
    )
    
    args = parser.parse_args()
    
    # Handle both "runs_libero_external_api" and "runs_libero_external_api/logs"
    logs_folder = args.logs_folder
    if not Path(logs_folder).joinpath('libero_10').exists():
        # Try appending 'logs' to the path
        potential_logs_path = Path(logs_folder) / 'logs'
        if potential_logs_path.exists():
            logs_folder = str(potential_logs_path)
    
    print(f"Parsing logs from: {logs_folder}", file=sys.stderr)
    
    # Parse all log files
    results = parse_logs_directory(logs_folder)
    
    if not results:
        print("No log files found or parsed successfully.", file=sys.stderr)
        sys.exit(1)
    
    # Create summary table
    df = create_summary_table(results)
    
    # Select only the requested columns
    output_df = df[['suite_name', 'task_id', 'language_instruction', 'success_rate']]
    
    # Output the results
    if args.output:
        if args.format == 'csv':
            output_df.to_csv(args.output, index=False)
        elif args.format == 'markdown':
            with open(args.output, 'w') as f:
                f.write(output_df.to_markdown(index=False))
        elif args.format == 'latex':
            with open(args.output, 'w') as f:
                f.write(output_df.to_latex(index=False))
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        # Print to stdout
        if args.format == 'csv':
            print(output_df.to_csv(index=False))
        elif args.format == 'markdown':
            print(output_df.to_markdown(index=False))
        elif args.format == 'latex':
            print(output_df.to_latex(index=False))
    
    # Print summary statistics
    print(f"\nSummary:", file=sys.stderr)
    print(f"Total records: {len(output_df)}", file=sys.stderr)
    print(f"Suites: {output_df['suite_name'].nunique()}", file=sys.stderr)
    print(f"Average success rate: {output_df['success_rate'].mean():.3f}", file=sys.stderr)
    print(f"\nSuccess rate by suite:", file=sys.stderr)
    suite_stats = output_df.groupby('suite_name')['success_rate'].agg(['mean', 'count'])
    print(suite_stats.to_string(), file=sys.stderr)


if __name__ == '__main__':
    main()
