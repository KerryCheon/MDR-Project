# Jakob Balkovec
# test_data_validation.py

# Pytest version of MDR data validation tests

import pytest # type: ignore
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "final.csv"

@pytest.fixture(scope="session")
def df():
    """Load the final.csv once per test session."""
    if not DATA_PATH.exists():
        pytest.skip(f"Data file not found: {DATA_PATH}")
    try:
        return pd.read_csv(DATA_PATH)
    except Exception as e:
        pytest.fail(f"Could not read CSV: {e}")

# ---------------------------------------------------------------------
# Structural Tests
# ---------------------------------------------------------------------

def test_columns_present(df):
    expected = {
        "date", "station_id", "longitude", "latitude",
        "air_temp_mean", "precipitation", "soil_moisture_5cm",
        "DOY", "Rain_3d"
    }
    missing = expected - set(df.columns)
    assert not missing, f"Missing columns: {missing}"

def test_no_duplicate_rows(df):
    duplicates = df.duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate rows."

def test_no_all_null_columns(df):
    """Allow SM_label to be empty for now."""
    allowed_empty = {"SM_label"}
    all_null_cols = [
        c for c in df.columns
        if df[c].isna().all() and c not in allowed_empty
    ]
    assert not all_null_cols, f"Columns entirely null: {all_null_cols}"

# ---------------------------------------------------------------------
# Data Quality Tests
# ---------------------------------------------------------------------

def test_date_parsing_and_range(df):
    parsed = pd.to_datetime(df["date"], errors="coerce")
    assert parsed.notna().all(), "Some dates could not be parsed."
    min_year, max_year = parsed.dt.year.min(), parsed.dt.year.max()
    assert min_year <= 2007 and max_year >= 2024, f"Unexpected date range: {min_year}–{max_year}"

def test_coordinates_reasonable(df):
    assert df["longitude"].between(-180, 180).all(), "Invalid longitude values."
    assert df["latitude"].between(-90, 90).all(), "Invalid latitude values."
    lon_med, lat_med = df["longitude"].median(), df["latitude"].median()
    assert 116 < abs(lon_med) < 119 and 46 < lat_med < 48, (
        f"Median coordinates look off: lat={lat_med}, lon={lon_med}"
    )

@pytest.mark.parametrize(
    "col,low,high",
    [
        ("precipitation", 0, 200),
        ("air_temp_mean", -50, 60),
        ("soil_moisture_5cm", -99.0, 1.5),
    ],
)
def test_numeric_ranges(df, col, low, high):
    if col not in df:
        pytest.skip(f"{col} missing, skipping range check.")
    valid_ratio = df[col].dropna().between(low, high).mean()
    assert valid_ratio > 0.95, f"{col} has too many out-of-range values."

def test_derived_features_valid(df):
    assert "Rain_3d" in df and "DOY" in df, "Derived features missing."
    assert df["Rain_3d"].notna().all(), "Rain_3d contains NaNs."
    assert df["DOY"].between(1, 366).all(), "DOY out of valid range."

def test_missing_value_threshold(df):
    """Allow up to 30% missing data globally, except critical fields."""
    CRITICAL_FIELDS = {
        "date", "station_id", "longitude", "latitude",
        "air_temp_mean", "precipitation", "soil_moisture_5cm"
    }
    missing_ratio = df.isna().mean()
    too_many = missing_ratio[missing_ratio > 0.3].drop(labels=["SM_label"], errors="ignore")
    # Fail only if critical columns exceed threshold
    critical_fail = [c for c in too_many.index if c in CRITICAL_FIELDS]
    assert not critical_fail, f"Critical columns exceed missing threshold: {critical_fail}"

# ---------------------------------------------------------------------
# Schema / Type Checks
# ---------------------------------------------------------------------

def test_expected_dtypes(df):
    expected_types = {
        "station_id": ("int64", "object"),  # accept both
        "longitude": "float64",
        "latitude": "float64",
        "air_temp_mean": "float64",
        "precipitation": "float64",
        "soil_moisture_5cm": "float64",
        "DOY": "int64",
        "Rain_3d": "float64",
    }
    for col, expected_type in expected_types.items():
        if col not in df.columns:
            pytest.skip(f"{col} missing, skipping dtype check.")
        actual = str(df[col].dtype)
        if isinstance(expected_type, tuple):
            assert any(t in actual for t in expected_type), f"{col} wrong dtype: {actual}"
        else:
            assert expected_type in actual, f"{col} wrong dtype: {actual}, expected {expected_type}"
