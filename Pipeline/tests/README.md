# Pipeline Tests

This directory contains tests for validating the data pipeline, particularly focusing on ensuring correct data parsing and column alignment.

## Test Files

### 1. schema_test.py
**Purpose**: Validates that the USCRN data schema configuration matches the actual data format.

**What it tests**:
- All 28 USCRN columns (indices 0-27) are defined in `config.yaml`
- Column names match the expected USCRN daily01 schema
- Critical columns 11-12 (sur_temp_type, sur_temp_max) are properly defined
- Actual data files have the expected 28-column structure

**Run**: 
```bash
python3 tests/schema_test.py
```

**Why it's important**: This test prevents column misalignment issues that can occur when column definitions are missing from the configuration.

---

### 2. integration_test.py
**Purpose**: End-to-end integration test that verifies the parse pipeline works correctly.

**What it tests**:
- Parse pipeline runs successfully and returns data
- DataFrame has expected number of columns (29: 28 data + 1 source_file)
- Critical columns contain correct data types and values
- No column shift/misalignment detected

**Run**:
```bash
python3 tests/integration_test.py
```

**Why it's important**: This test validates the entire parsing flow to ensure data is being read and processed correctly.

---

### 3. data_test.py
**Purpose**: Validates the final processed dataset (`data/processed/final.csv`).

**What it tests**:
- Required columns are present
- Coordinate values are plausible
- Date parsing works correctly
- Data values are within expected ranges
- Derived features are correctly calculated

**Run**:
```bash
python3 tests/data_test.py
```

**Note**: This test requires that the pipeline has been run and `final.csv` exists.

---

### 4. playground.py
A scratch file for testing and development.

---

## Running All Tests

To run all validation tests in sequence:

```bash
cd Pipeline

# 1. Schema validation
echo "=== Schema Validation ==="
python3 tests/schema_test.py

# 2. Integration test
echo "=== Integration Test ==="
python3 tests/integration_test.py

# 3. Data validation (if final.csv exists)
echo "=== Data Validation ==="
python3 tests/data_test.py
```

## Column Misalignment Issue

### Background
Previously, columns 11-12 in the USCRN data were missing from the configuration, causing all subsequent columns to be shifted. This resulted in incorrect data mapping (e.g., soil moisture values being read as soil temperature).

### Resolution
- All 28 USCRN columns are now properly defined in `config.yaml`
- Columns 11-12 are correctly mapped as:
  - Index 11: `sur_temp_type` (Surface temperature sensor type: R/C/U)
  - Index 12: `sur_temp_max` (Maximum surface temperature in °C)

### Prevention
- **schema_test.py** validates that all columns are defined
- **integration_test.py** verifies no column shift has occurred
- See `USCRN_SCHEMA.md` for the complete column reference

## Expected Test Results

When all tests pass, you should see:

```
✓ ALL TESTS PASSED
Schema validation successful - no column misalignment issues detected
```

If any test fails:
1. Check that `config.yaml` has all 28 column definitions
2. Verify data files are in the correct format
3. Review error messages for specific issues
4. Consult `USCRN_SCHEMA.md` for the correct schema

## References
- `../USCRN_SCHEMA.md` - Complete USCRN daily01 format documentation
- `../RESOLUTION_SUMMARY.md` - Details on the column misalignment issue resolution
- `../config.yaml` - Pipeline configuration with column mappings
