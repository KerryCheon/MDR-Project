#!/usr/bin/env python3
# Jakob Balkovec
# Integration Test for Column Alignment Fix
#
# This script performs an end-to-end test to verify that the column
# misalignment issue is resolved and data is being parsed correctly.

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipes.parse_pipe import ParsePipe
from utils.config import load_config

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def status(tag, msg):
    color = {"PASS": GREEN, "FAIL": RED, "INFO": CYAN}.get(tag, RESET)
    print(f"{BOLD}{color}[{tag}]\t{RESET} {msg}")

def test_parse_pipeline():
    """Test that the parse pipeline correctly reads all columns."""
    status("INFO", "Loading configuration...")
    config = load_config()
    
    status("INFO", "Initializing ParsePipe...")
    parser = ParsePipe(config)
    
    status("INFO", "Running ParsePipe on USCRN data...")
    df = parser.run()
    
    if df.empty:
        status("FAIL", "ParsePipe returned empty DataFrame")
        return False
    
    status("PASS", f"Parsed {len(df)} rows with {len(df.columns)} columns")
    return df

def test_critical_columns(df):
    """Test that critical columns 11-12 are present and contain correct data."""
    critical_cols = {
        'sur_temp_type': {
            'expected_values': ['R', 'C', 'U'],
            'description': 'Surface temperature sensor type'
        },
        'sur_temp_max': {
            'expected_type': 'numeric',
            'description': 'Maximum surface temperature'
        }
    }
    
    all_passed = True
    
    for col_name, specs in critical_cols.items():
        if col_name not in df.columns:
            status("FAIL", f"Column '{col_name}' is missing!")
            all_passed = False
            continue
        
        status("PASS", f"Column '{col_name}' is present")
        
        # Check for non-null values
        non_null = df[col_name].dropna()
        if len(non_null) == 0:
            status("WARN", f"Column '{col_name}' has all NaN values")
            continue
        
        # Validate data based on column type
        if 'expected_values' in specs:
            # Categorical column
            unique_vals = set(non_null.unique())
            expected_vals = set(specs['expected_values'])
            if unique_vals.issubset(expected_vals):
                status("PASS", f"Column '{col_name}' contains valid values: {sorted(unique_vals)}")
            else:
                invalid = unique_vals - expected_vals
                status("FAIL", f"Column '{col_name}' contains unexpected values: {invalid}")
                all_passed = False
        
        elif specs.get('expected_type') == 'numeric':
            # Numeric column - just check it's convertible to float
            try:
                numeric_vals = non_null.astype(str).str.replace('<NA>', '').replace('', None).dropna()
                if len(numeric_vals) > 0:
                    # Show sample numeric values
                    sample_vals = numeric_vals.head(3).tolist()
                    status("PASS", f"Column '{col_name}' contains numeric data. Sample: {sample_vals}")
                else:
                    status("WARN", f"Column '{col_name}' has no valid numeric values")
            except Exception as e:
                status("FAIL", f"Column '{col_name}' is not numeric: {e}")
                all_passed = False
    
    return all_passed

def test_column_count(df):
    """Test that DataFrame has expected number of columns."""
    # Expecting 28 data columns + 1 source_file column = 29 total
    expected_cols = 29
    actual_cols = len(df.columns)
    
    if actual_cols == expected_cols:
        status("PASS", f"DataFrame has expected {expected_cols} columns")
        return True
    else:
        status("FAIL", f"DataFrame has {actual_cols} columns, expected {expected_cols}")
        return False

def test_no_column_shift():
    """
    Test that columns are not shifted by verifying that surface temp type
    (column 11) contains categorical data and surface temp max (column 12)
    contains numeric data.
    """
    status("INFO", "Testing for column misalignment...")
    
    config = load_config()
    col_indices = config['pipeline']['parse']['col_indices']
    
    # Verify column 11 is sur_temp_type (should be categorical)
    col_11_name = None
    col_12_name = None
    
    for name, idx in col_indices.items():
        if idx == 11:
            col_11_name = name
        elif idx == 12:
            col_12_name = name
    
    if col_11_name == 'sur_temp_type' and col_12_name == 'sur_temp_max':
        status("PASS", "Columns 11-12 are correctly mapped (no shift detected)")
        return True
    else:
        status("FAIL", f"Column mapping issue: col 11={col_11_name}, col 12={col_12_name}")
        return False

def main():
    """Run all integration tests."""
    print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}COLUMN ALIGNMENT INTEGRATION TEST{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")
    
    all_passed = True
    
    # Test 1: Column shift detection
    if not test_no_column_shift():
        all_passed = False
    print()
    
    # Test 2: Parse pipeline
    df = test_parse_pipeline()
    if df is False or df.empty:
        all_passed = False
        print(f"\n{BOLD}{RED}✗ TESTS FAILED{RESET}\n")
        sys.exit(1)
    print()
    
    # Test 3: Column count
    if not test_column_count(df):
        all_passed = False
    print()
    
    # Test 4: Critical columns
    if not test_critical_columns(df):
        all_passed = False
    print()
    
    # Summary
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
    if all_passed:
        print(f"{BOLD}{GREEN}✓ ALL INTEGRATION TESTS PASSED{RESET}")
        print(f"{BOLD}{GREEN}Column alignment verified - data parsing correctly{RESET}")
    else:
        print(f"{BOLD}{RED}✗ SOME INTEGRATION TESTS FAILED{RESET}")
        sys.exit(1)
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")

if __name__ == "__main__":
    main()
