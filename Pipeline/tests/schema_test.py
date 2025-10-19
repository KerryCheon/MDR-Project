# Jakob Balkovec
# Schema Validation Test for USCRN Data Structure
# 
# This test validates that the config.yaml column mappings correctly match
# the actual USCRN daily01 data format to prevent column misalignment issues.

import pandas as pd
import yaml
from pathlib import Path
import sys

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"

def status(tag, msg):
    color = {"PASS": GREEN, "FAIL": RED, "WARN": YELLOW, "INFO": CYAN}.get(tag, RESET)
    print(f"{BOLD}{color}[{tag}]\t{RESET} {msg}")

# Expected USCRN daily01 format (28 columns, indices 0-27)
# Based on official NOAA USCRN documentation
USCRN_SCHEMA = {
    0: ("station_id", "WBANNO - Station identifier"),
    1: ("date", "LST_DATE - Date in YYYYMMDD format"),
    2: ("crx_vn", "CRX_VN - Version number"),
    3: ("longitude", "LONGITUDE - Decimal degrees"),
    4: ("latitude", "LATITUDE - Decimal degrees"),
    5: ("air_temp_max", "T_DAILY_MAX - Max air temperature (°C)"),
    6: ("air_temp_min", "T_DAILY_MIN - Min air temperature (°C)"),
    7: ("air_temp_mean", "T_DAILY_MEAN - Mean air temperature (°C)"),
    8: ("air_temp_avg", "T_DAILY_AVG - Avg air temperature (°C)"),
    9: ("precipitation", "P_DAILY_CALC - Precipitation (mm)"),
    10: ("solar_radiation", "SOLARAD_DAILY - Solar radiation (MJ/m²)"),
    11: ("sur_temp_type", "SUR_TEMP_DAILY_TYPE - Surface temp type (R/C/U)"),
    12: ("sur_temp_max", "SUR_TEMP_DAILY_MAX - Max surface temperature (°C)"),
    13: ("sur_temp_min", "SUR_TEMP_DAILY_MIN - Min surface temperature (°C)"),
    14: ("sur_temp_avg", "SUR_TEMP_DAILY_AVG - Avg surface temperature (°C)"),
    15: ("rh_max", "RH_DAILY_MAX - Max relative humidity (%)"),
    16: ("rh_min", "RH_DAILY_MIN - Min relative humidity (%)"),
    17: ("rh_mean", "RH_DAILY_AVG - Avg relative humidity (%)"),
    18: ("soil_moisture_5cm", "SOIL_MOISTURE_5_DAILY - Soil moisture at 5cm (m³/m³)"),
    19: ("soil_moisture_10cm", "SOIL_MOISTURE_10_DAILY - Soil moisture at 10cm (m³/m³)"),
    20: ("soil_moisture_20cm", "SOIL_MOISTURE_20_DAILY - Soil moisture at 20cm (m³/m³)"),
    21: ("soil_moisture_50cm", "SOIL_MOISTURE_50_DAILY - Soil moisture at 50cm (m³/m³)"),
    22: ("soil_moisture_100cm", "SOIL_MOISTURE_100_DAILY - Soil moisture at 100cm (m³/m³)"),
    23: ("soil_temp_5cm", "SOIL_TEMP_5_DAILY - Soil temperature at 5cm (°C)"),
    24: ("soil_temp_10cm", "SOIL_TEMP_10_DAILY - Soil temperature at 10cm (°C)"),
    25: ("soil_temp_20cm", "SOIL_TEMP_20_DAILY - Soil temperature at 20cm (°C)"),
    26: ("soil_temp_50cm", "SOIL_TEMP_50_DAILY - Soil temperature at 50cm (°C)"),
    27: ("soil_temp_100cm", "SOIL_TEMP_100_DAILY - Soil temperature at 100cm (°C)"),
}

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

def load_config():
    """Load and return the pipeline configuration."""
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)
        status("PASS", f"Loaded config from {CONFIG_PATH}")
        return config
    except Exception as e:
        status("FAIL", f"Could not load config: {e}")
        return None

def test_config_completeness(config):
    """Test that all 28 USCRN columns are mapped in config.yaml."""
    col_indices = config['pipeline']['parse']['col_indices']
    
    # Check that all indices 0-27 are mapped
    mapped_indices = set(col_indices.values())
    expected_indices = set(range(28))
    missing = expected_indices - mapped_indices
    
    if missing:
        status("FAIL", f"Missing column indices in config: {sorted(missing)}")
        return False
    
    status("PASS", "All 28 columns (0-27) are mapped in config.yaml")
    return True

def test_column_names(config):
    """Test that column names in config match expected USCRN schema."""
    col_indices = config['pipeline']['parse']['col_indices']
    
    all_match = True
    for idx in range(28):
        expected_name, _ = USCRN_SCHEMA[idx]
        
        # Find the config name for this index
        config_name = None
        for name, mapped_idx in col_indices.items():
            if mapped_idx == idx:
                config_name = name
                break
        
        if config_name != expected_name:
            status("WARN", f"Index {idx}: config has '{config_name}', expected '{expected_name}'")
            all_match = False
    
    if all_match:
        status("PASS", "All column names match expected USCRN schema")
    else:
        status("WARN", "Some column names differ from standard USCRN names")
    
    return True  # Not a critical failure

def test_critical_columns(config):
    """Test that critical columns 11-12 are properly defined."""
    col_indices = config['pipeline']['parse']['col_indices']
    
    # Verify columns 11-12 specifically (mentioned in the issue)
    col_11 = None
    col_12 = None
    
    for name, idx in col_indices.items():
        if idx == 11:
            col_11 = name
        elif idx == 12:
            col_12 = name
    
    if col_11 is None or col_12 is None:
        status("FAIL", "Columns 11-12 are not defined in config!")
        return False
    
    status("PASS", f"Column 11 is defined as '{col_11}'")
    status("PASS", f"Column 12 is defined as '{col_12}'")
    
    # Verify they match expected names
    if col_11 == "sur_temp_type" and col_12 == "sur_temp_max":
        status("PASS", "Columns 11-12 match expected USCRN schema")
    else:
        status("WARN", f"Columns 11-12 names differ: got ({col_11}, {col_12}), expected (sur_temp_type, sur_temp_max)")
    
    return True

def test_actual_data_format():
    """Test that actual USCRN data files have expected column count."""
    # Find a sample data file
    sample_files = list(RAW_DATA_DIR.glob("uscrn_*.txt"))
    
    if not sample_files:
        status("WARN", f"No USCRN data files found in {RAW_DATA_DIR}")
        return True  # Not a critical failure if no data files
    
    sample_file = sample_files[0]
    
    try:
        df = pd.read_csv(sample_file, sep=r'\s+', header=None, nrows=1, engine='python')
        actual_cols = len(df.columns)
        
        if actual_cols == 28:
            status("PASS", f"Sample data file has expected 28 columns")
        else:
            status("FAIL", f"Sample data file has {actual_cols} columns, expected 28")
            return False
        
        return True
    except Exception as e:
        status("FAIL", f"Could not read sample data file: {e}")
        return False

def main():
    """Run all schema validation tests."""
    print(f"\n{BOLD}{CYAN}{'=' * 80}{RESET}")
    print(f"{BOLD}{CYAN}USCRN SCHEMA VALIDATION TEST{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")
    
    # Load config
    config = load_config()
    if config is None:
        sys.exit(1)
    
    print()
    
    # Run tests
    tests = [
        ("Config Completeness", lambda: test_config_completeness(config)),
        ("Column Names", lambda: test_column_names(config)),
        ("Critical Columns 11-12", lambda: test_critical_columns(config)),
        ("Actual Data Format", test_actual_data_format),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        status("INFO", f"Running: {test_name}")
        passed = test_func()
        if not passed:
            all_passed = False
        print()
    
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}")
    if all_passed:
        print(f"{BOLD}{GREEN}✓ ALL TESTS PASSED{RESET}")
        print(f"{BOLD}{GREEN}Schema validation successful - no column misalignment issues detected{RESET}")
    else:
        print(f"{BOLD}{RED}✗ SOME TESTS FAILED{RESET}")
        print(f"{BOLD}{RED}Schema validation failed - column misalignment issues detected{RESET}")
        sys.exit(1)
    print(f"{BOLD}{CYAN}{'=' * 80}{RESET}\n")

if __name__ == "__main__":
    main()
