# test_satellite_pipeline.py
# Comprehensive regression tests for the satellite pipeline optimizations

import pytest
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from src.pipeline.pipes.satellite_pipe import SatellitePipe
from src.pipeline.pipes.optimized_satellite_pipe import OptimizedSatellitePipe, SatellitePipeV2
from src.pipeline.pipes.temporal_fill_pipe import TemporalFillPipe
from src.pipeline.pipes.whittaker_pipe import WhittakerPipe
from src.pipeline.pipes.feature_pipe import FeaturePipe
from src.pipeline.utils.config import load_config


EXPECTED_SAT_COLUMNS = [
    "LST_modis", "NDVI_modis", "s1_vv", "s1_vh", "s1_vv_dB", "s1_vh_dB",
    "s2_b2", "s2_b3", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
    "elev", "slope", "aspect",
    "SMAP_sm_am", "SMAP_sm_pm", "SMAP_qual_am", "SMAP_qual_pm"
]


@pytest.fixture(autouse=True)
def mock_gee_init(monkeypatch):
    """Mocks GEE initialization so tests can run reliably offline or in CI."""
    monkeypatch.setenv("GEE_PROJECT_ID", "mock-gee-project")
    import ee
    monkeypatch.setattr(ee, "Initialize", lambda project=None: None)
    monkeypatch.setattr(ee, "Authenticate", lambda: None)


@pytest.fixture
def base_config():
    """Loads default configuration for testing."""
    return load_config("src/pipeline/config.yaml")


@pytest.fixture
def mock_station_df():
    """Creates a deterministic multi-week station DataFrame."""
    dates = pd.date_range("2016-01-01", periods=21, freq="D")
    df = pd.DataFrame({
        "station_id": 4237,
        "date": dates,
        "latitude": 47.51,
        "longitude": -123.81,
        "air_temp_mean": 5.0,
        "precipitation": 0.0,
        "soil_moisture_5cm": 0.25,
    })
    return df


@pytest.fixture
def sample_regression_fixtures():
    """Loads extracted reference regression fixtures if available."""
    fixture_path = Path("tests/fixtures/satellite_regression_samples.json")
    if fixture_path.exists():
        with open(fixture_path) as f:
            return json.load(f)
    return None


# -----------------------------------------------------------------------------
# Test 1: Class and Alias Integrity
# -----------------------------------------------------------------------------
def test_class_and_alias_integrity():
    assert OptimizedSatellitePipe is SatellitePipeV2
    assert hasattr(OptimizedSatellitePipe, "fetch_static_terrain")
    assert hasattr(OptimizedSatellitePipe, "fetch_single_week_unified")
    assert hasattr(OptimizedSatellitePipe, "fetch_satellite_batch_collection")
    assert hasattr(OptimizedSatellitePipe, "run")


# -----------------------------------------------------------------------------
# Test 2: Output Schema and Column Completeness
# -----------------------------------------------------------------------------
def test_satellite_output_schema(base_config, mock_station_df, tmp_path):
    cache_file = tmp_path / "test_sat_cache.json"
    cfg = json.loads(json.dumps(base_config))
    cfg["satellite"]["cache_path"] = str(cache_file)

    # Pre-populate cache with synthetic valid satellite data
    pre_cache = {}
    mock_station_df["week"] = mock_station_df["date"].dt.to_period("W-SUN").astype(str)
    for week, group in mock_station_df.groupby("week"):
        start = group["date"].min().strftime("%Y-%m-%d")
        end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        date_key = f"{start}_{end}"
        pre_cache[date_key] = {
            "LST_modis": 280.0,
            "NDVI_modis": 0.65,
            "s1_vv": 0.12,
            "s1_vh": 0.04,
            "s1_vv_dB": -9.2,
            "s1_vh_dB": -13.9,
            "s2_b2": 0.15,
            "s2_b3": 0.18,
            "s2_b4": 0.14,
            "s2_b8": 0.35,
            "s2_b11": 0.22,
            "s2_b12": 0.19,
            "elev": 105.0,
            "slope": 4.5,
            "aspect": 180.0,
            "SMAP_sm_am": 0.32,
            "SMAP_sm_pm": 0.34,
            "SMAP_qual_am": 0.0,
            "SMAP_qual_pm": 0.0,
        }

    with open(cache_file, "w") as f:
        json.dump(pre_cache, f)

    pipe = OptimizedSatellitePipe(config=cfg, station_name="test_station")
    result_df = pipe.run(mock_station_df.drop(columns=["week"]))

    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(mock_station_df)

    for col in EXPECTED_SAT_COLUMNS:
        assert col in result_df.columns, f"Expected satellite column '{col}' missing from result_df"

    # Verify input columns were retained
    assert "station_id" in result_df.columns
    assert "air_temp_mean" in result_df.columns
    assert "soil_moisture_5cm" in result_df.columns

    # Verify temporary 'week' column was dropped
    assert "week" not in result_df.columns
    # Verify deprecated 'Rain_sat' column is not present
    assert "Rain_sat" not in result_df.columns


# -----------------------------------------------------------------------------
# Test 3: Bidirectional Cache Compatibility
# -----------------------------------------------------------------------------
def test_bidirectional_cache_compatibility(base_config, mock_station_df, tmp_path):
    # Case A: V1 writes cache -> V2 reads cache
    cache_a = tmp_path / "cache_v1_to_v2.json"
    cfg_a1 = json.loads(json.dumps(base_config))
    cfg_a1["satellite"]["cache_path"] = str(cache_a)

    sample_entry = {}
    mock_station_df["week"] = mock_station_df["date"].dt.to_period("W-SUN").astype(str)
    for week, group in mock_station_df.groupby("week"):
        start = group["date"].min().strftime("%Y-%m-%d")
        end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        date_key = f"{start}_{end}"
        sample_entry[date_key] = {
            "LST_modis": 275.5,
            "NDVI_modis": 0.70,
            "s1_vv": 0.15,
            "s1_vh": 0.05,
            "s1_vv_dB": -8.2,
            "s1_vh_dB": -13.0,
            "s2_b2": 0.20,
            "s2_b3": 0.21,
            "s2_b4": 0.19,
            "s2_b8": 0.33,
            "s2_b11": 0.25,
            "s2_b12": 0.21,
            "elev": 96.0,
            "slope": 5.5,
            "aspect": 170.0,
            "SMAP_sm_am": 0.45,
            "SMAP_sm_pm": 0.53,
            "SMAP_qual_am": 1.0,
            "SMAP_qual_pm": 1.0,
        }

    with open(cache_a, "w") as f:
        json.dump(sample_entry, f, indent=2)

    cfg_a2 = json.loads(json.dumps(base_config))
    cfg_a2["satellite"]["cache_path"] = str(cache_a)

    pipe_v1_a = SatellitePipe(config=cfg_a1, station_name="compat_a")
    pipe_v2_a = OptimizedSatellitePipe(config=cfg_a2, station_name="compat_a")

    out_v1_a = pipe_v1_a.run(mock_station_df.drop(columns=["week"]))
    out_v2_a = pipe_v2_a.run(mock_station_df.drop(columns=["week"]))

    assert len(out_v1_a) == len(out_v2_a)
    for col in EXPECTED_SAT_COLUMNS:
        np.testing.assert_allclose(out_v1_a[col].dropna(), out_v2_a[col].dropna(), rtol=1e-5, atol=1e-5)

    # Case B: V2 writes cache -> V1 reads cache
    cache_b = tmp_path / "cache_v2_to_v1.json"
    with open(cache_b, "w") as f:
        json.dump(sample_entry, f, indent=2)

    cfg_b1 = json.loads(json.dumps(base_config))
    cfg_b1["satellite"]["cache_path"] = str(cache_b)
    cfg_b2 = json.loads(json.dumps(base_config))
    cfg_b2["satellite"]["cache_path"] = str(cache_b)

    pipe_v2_b = OptimizedSatellitePipe(config=cfg_b2, station_name="compat_b")
    pipe_v1_b = SatellitePipe(config=cfg_b1, station_name="compat_b")

    out_v2_b = pipe_v2_b.run(mock_station_df.drop(columns=["week"]))
    out_v1_b = pipe_v1_b.run(mock_station_df.drop(columns=["week"]))

    assert len(out_v2_b) == len(out_v1_b)
    for col in EXPECTED_SAT_COLUMNS:
        np.testing.assert_allclose(out_v2_b[col].dropna(), out_v1_b[col].dropna(), rtol=1e-5, atol=1e-5)


# -----------------------------------------------------------------------------
# Test 3B: Zero GEE Network Calls on Full Cache Hits
# -----------------------------------------------------------------------------
def test_zero_gee_calls_on_cache_hit(base_config, mock_station_df, tmp_path, monkeypatch):
    cache_file = tmp_path / "hit_cache.json"
    cfg = json.loads(json.dumps(base_config))
    cfg["satellite"]["cache_path"] = str(cache_file)

    sample_entry = {}
    mock_station_df["week"] = mock_station_df["date"].dt.to_period("W-SUN").astype(str)
    for week, group in mock_station_df.groupby("week"):
        start = group["date"].min().strftime("%Y-%m-%d")
        end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        date_key = f"{start}_{end}"
        sample_entry[date_key] = {
            "LST_modis": 275.5,
            "NDVI_modis": 0.70,
            "s1_vv": 0.15,
            "s1_vh": 0.05,
            "s1_vv_dB": -8.2,
            "s1_vh_dB": -13.0,
            "s2_b2": 0.20,
            "s2_b3": 0.21,
            "s2_b4": 0.19,
            "s2_b8": 0.33,
            "s2_b11": 0.25,
            "s2_b12": 0.21,
            "elev": 96.0,
            "slope": 5.5,
            "aspect": 170.0,
            "SMAP_sm_am": 0.45,
            "SMAP_sm_pm": 0.53,
            "SMAP_qual_am": 1.0,
            "SMAP_qual_pm": 1.0,
        }

    with open(cache_file, "w") as f:
        json.dump(sample_entry, f, indent=2)

    pipe_v2 = OptimizedSatellitePipe(config=cfg, station_name="hit_test")

    # Monkeypatch fetch_static_terrain to fail if called
    def should_not_be_called(*args, **kwargs):
        raise AssertionError("fetch_static_terrain should NOT be called on a 100% cache hit!")

    monkeypatch.setattr(pipe_v2, "fetch_static_terrain", should_not_be_called)

    out = pipe_v2.run(mock_station_df.drop(columns=["week"]))
    assert not out.empty
    assert out["elev"].notna().all()


# -----------------------------------------------------------------------------
# Test 3C: Failed Reductions (All-None) Are Not Cached
# -----------------------------------------------------------------------------
def test_invalid_all_none_not_cached(base_config, mock_station_df, tmp_path, monkeypatch):
    cache_file = tmp_path / "empty_cache.json"
    cfg = json.loads(json.dumps(base_config))
    cfg["satellite"]["cache_path"] = str(cache_file)

    pipe_v2 = OptimizedSatellitePipe(config=cfg, station_name="none_cache_test")

    # Mock batch and unified to return all-None (e.g. transient network failure)
    all_none_dict = {
        k: None for k in OptimizedSatellitePipe.DEFAULT_RES
        if k not in ("elev", "slope", "aspect")
    }
    monkeypatch.setattr(pipe_v2, "fetch_satellite_batch_collection", lambda items: {items[0][0]: all_none_dict})
    monkeypatch.setattr(pipe_v2, "fetch_single_week_unified", lambda *args: all_none_dict)

    out = pipe_v2.run(mock_station_df)
    assert not out.empty

    # Verify that the cache file on disk did NOT save the failed all-None week
    if cache_file.exists():
        with open(cache_file) as f:
            saved_cache = json.load(f)
        assert len(saved_cache) == 0, f"Expected empty cache after failed fetch, but found: {saved_cache}"


# -----------------------------------------------------------------------------
# Test 4: Regression Against Ground Truth Fixtures
# -----------------------------------------------------------------------------
def test_regression_against_ground_truth(base_config, sample_regression_fixtures, tmp_path):
    if not sample_regression_fixtures:
        pytest.skip("No fixture file found at tests/fixtures/satellite_regression_samples.json")

    cache_file = tmp_path / "fixture_cache.json"
    cfg = json.loads(json.dumps(base_config))
    cfg["satellite"]["cache_path"] = str(cache_file)

    # Populate cache from ground truth fixture
    fixture_cache = {}
    for dk, wdata in sample_regression_fixtures["weeks"].items():
        fixture_cache[dk] = wdata["expected_features"]

    with open(cache_file, "w") as f:
        json.dump(fixture_cache, f, indent=2)

    # Build dataframe for fixture weeks covering the exact weekly spans
    records = []
    for dk, wdata in sample_regression_fixtures["weeks"].items():
        week_dates = pd.date_range(
            start=wdata["start"],
            end=pd.to_datetime(wdata["end"]) - pd.Timedelta(days=1),
            freq="D"
        )
        for dt in week_dates:
            records.append({
                "station_id": 4237,
                "date": dt,
                "latitude": sample_regression_fixtures["latitude"],
                "longitude": sample_regression_fixtures["longitude"],
                "air_temp_mean": 5.0,
                "precipitation": 0.0,
                "soil_moisture_5cm": 0.25,
            })

    df = pd.DataFrame(records)
    pipe_v2 = OptimizedSatellitePipe(config=cfg, station_name="quinault_fixture")
    result_df = pipe_v2.run(df)

    # Verify against expected values
    for dk, wdata in sample_regression_fixtures["weeks"].items():
        expected = wdata["expected_features"]
        week_mask = (result_df["date"] >= wdata["start"]) & (result_df["date"] < wdata["end"])
        week_rows = result_df[week_mask]
        assert len(week_rows) > 0, f"No rows found for week {dk}"

        for col, exp_val in expected.items():
            actual_vals = week_rows[col]
            if exp_val is None:
                assert actual_vals.isna().all(), f"Expected NaN for {col}, got {actual_vals.tolist()}"
            else:
                assert not actual_vals.isna().any(), f"Expected {exp_val} for {col}, got NaNs"
                np.testing.assert_allclose(actual_vals.values, exp_val, rtol=1e-4, atol=1e-4)


# -----------------------------------------------------------------------------
# Test 5: End-to-End Downstream Pipeline Compatibility
# -----------------------------------------------------------------------------
def test_downstream_pipe_compatibility(base_config, mock_station_df, tmp_path):
    cache_file = tmp_path / "downstream_cache.json"
    cfg = json.loads(json.dumps(base_config))
    cfg["satellite"]["cache_path"] = str(cache_file)

    # Synthetic full cache
    pre_cache = {}
    mock_station_df["week"] = mock_station_df["date"].dt.to_period("W-SUN").astype(str)
    for week, group in mock_station_df.groupby("week"):
        start = group["date"].min().strftime("%Y-%m-%d")
        end = (group["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        date_key = f"{start}_{end}"
        pre_cache[date_key] = {
            "LST_modis": 278.0,
            "NDVI_modis": 0.72,
            "s1_vv": 0.15,
            "s1_vh": 0.04,
            "s1_vv_dB": -8.2,
            "s1_vh_dB": -13.9,
            "s2_b2": 0.20,
            "s2_b3": 0.20,
            "s2_b4": 0.18,
            "s2_b8": 0.33,
            "s2_b11": 0.25,
            "s2_b12": 0.21,
            "elev": 96.0,
            "slope": 5.5,
            "aspect": 170.0,
            "SMAP_sm_am": 0.45,
            "SMAP_sm_pm": 0.53,
            "SMAP_qual_am": 1.0,
            "SMAP_qual_pm": 1.0,
        }

    with open(cache_file, "w") as f:
        json.dump(pre_cache, f)

    # 1. Run Optimized Satellite Pipe
    pipe_v2 = OptimizedSatellitePipe(config=cfg, station_name="downstream_test")
    sat_df = pipe_v2.run(mock_station_df.drop(columns=["week"]))

    # Add mock weather columns required by downstream
    sat_df["rain_mm"] = 0.0
    sat_df["precip_mm"] = 0.0

    # 2. Run Temporal Fill Pipe
    filled_df = TemporalFillPipe(config=cfg.get("temporal_fill"), station_name="downstream_test").run(sat_df)
    assert not filled_df.empty
    assert "rain_mm_mask" in filled_df.columns, "rain_mm_mask should be created by TemporalFillPipe"

    # 3. Run Whittaker Pipe
    smooth_df = WhittakerPipe(config=cfg.get("whittaker"), station_name="downstream_test").run(filled_df)
    assert not smooth_df.empty

    # 4. Run Feature Pipe
    station_cfg = cfg.get("stations", {}).get("spokane_17_ssw", {})
    featured_df = FeaturePipe(config=station_cfg.get("feature", {})).run(smooth_df)
    assert not featured_df.empty

    # Assert engineered features created successfully
    assert "NDVI" in featured_df.columns or "NDVI_modis" in featured_df.columns
    assert "DOY" in featured_df.columns


# -----------------------------------------------------------------------------
# Test 6: Empty and None DataFrame Handling
# -----------------------------------------------------------------------------
def test_empty_and_none_dataframe_handling(base_config):
    pipe = OptimizedSatellitePipe(config=base_config, station_name="edge_test")
    assert pipe.run(None) is None
    empty_df = pd.DataFrame()
    assert pipe.run(empty_df).empty


# -----------------------------------------------------------------------------
# Test 7: Partial Sensor Properties Parsing
# -----------------------------------------------------------------------------
def test_partial_sensor_properties_parsing(base_config):
    pipe = OptimizedSatellitePipe(config=base_config, station_name="partial_test")
    # Simulate partial GEE response where only LST and SMAP are present, other sensors are None
    partial_props = {
        "lst_val": 14000.0,
        "smap_sm_am": 0.35,
        "smap_qual_am": 0.0,
        "s2_b2": 1500.0,
    }
    parsed = pipe._parse_dynamic_props(partial_props)
    assert parsed["LST_modis"] == pytest.approx(280.0)
    assert parsed["SMAP_sm_am"] == pytest.approx(0.35)
    assert parsed["s2_b2"] == pytest.approx(0.15)
    assert parsed["NDVI_modis"] is None
    assert parsed["s1_vv"] is None
    assert parsed["s1_vh"] is None
    assert parsed["s2_b3"] is None


def test_smap_quality_metadata_does_not_erase_finite_retrieval(base_config):
    pipe = OptimizedSatellitePipe(config=base_config, station_name="smap_quality_test")
    parsed = pipe._parse_dynamic_props({
        "smap_sm_am": 0.35,
        "smap_sm_pm": 0.37,
        "smap_qual_am": 1.0,
        "smap_qual_pm": 1.0,
    })

    assert parsed["SMAP_sm_am"] == pytest.approx(0.35)
    assert parsed["SMAP_sm_pm"] == pytest.approx(0.37)
    assert parsed["SMAP_qual_am"] == pytest.approx(1.0)
    assert parsed["SMAP_qual_pm"] == pytest.approx(1.0)
