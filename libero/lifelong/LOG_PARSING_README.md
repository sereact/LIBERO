# Log Parsing Script

## Overview

The `parse_logs.py` script parses LIBERO experiment log files and generates a summary table with the following columns:
- `suite_name`: Name of the test suite (e.g., libero_10, libero_goal, etc.)
- `task_id`: Task ID number
- `language_instruction`: The natural language instruction for the task
- `success_rate`: The success rate for the task (0.0 to 1.0)

## Usage

### Basic Usage (print to stdout)
```bash
python libero/lifelong/parse_logs.py runs_libero_external_api
```

### Save to CSV file
```bash
python libero/lifelong/parse_logs.py runs_libero_external_api --output results.csv
```

### Output in different formats
```bash
# CSV format (default)
python libero/lifelong/parse_logs.py runs_libero_external_api --format csv

# Markdown format (requires tabulate package)
python libero/lifelong/parse_logs.py runs_libero_external_api --format markdown

# LaTeX format
python libero/lifelong/parse_logs.py runs_libero_external_api --format latex
```

## Expected Directory Structure

The script expects the following directory structure:
```
logs_folder/
    suite_name/
        task_0/
            seed_100.log
        task_1/
            seed_100.log
        ...
```

For example:
```
runs_libero_external_api/
    logs/
        libero_10/
            task_0/
                seed_100.log
            task_1/
                seed_100.log
        libero_goal/
            task_0/
                seed_100.log
```

The script automatically detects whether you provide the base folder or the `logs` subfolder.

## Output

The script generates:
1. A table with the requested columns (suite_name, task_id, language_instruction, success_rate)
2. Summary statistics including:
   - Total number of records
   - Number of unique suites
   - Average success rate across all tasks
   - Average success rate per suite

## Example

```bash
$ python libero/lifelong/parse_logs.py runs_libero_external_api --output results.csv

Parsing logs from: runs_libero_external_api/logs
Results saved to: results.csv

Summary:
Total records: 40
Suites: 4
Average success rate: 0.745

Success rate by suite:
                 mean  count
suite_name                  
libero_10       0.645     10
libero_goal     0.715     10
libero_object   0.960     10
libero_spatial  0.660     10
```

## Requirements

- Python 3.6+
- pandas
- tabulate (optional, only required for markdown output)
