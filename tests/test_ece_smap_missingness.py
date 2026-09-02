import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ECE_DIR = Path("data/splits/derived_8.4-ece").resolve()


def _load_module(name: str, path: Path):
    sys.path.insert(0, str(ECE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_prepare_smap_columns_preserves_retrieval_values_and_nan():
    builder = _load_module("make_derived_8_4_ece", ECE_DIR / "make_derived_8.4_ece.py")
    source = pd.DataFrame({
        "SMAP_sm_am": [0.21, 0.22, np.nan],
        "SMAP_sm_pm": [0.31, np.nan, np.nan],
        "SMAP_qual_am": [0.0, 1.0, 1.0],
        "SMAP_qual_pm": [0.0, 1.0, 1.0],
    })

    result = builder.prepare_smap_columns(source)

    assert result.loc[0, "SMAP_sm_am_interp"] == 0.21
    assert result.loc[1, "SMAP_sm_am_interp"] == 0.22
    assert result.loc[0, "SMAP_sm_pm_interp"] == 0.31
    assert result.loc[1:, "SMAP_sm_pm_interp"].isna().all()


def test_all_missing_smap_generates_zero_observation_masks_and_nan_features():
    features = _load_module(
        "ece_derived_features_all_math",
        ECE_DIR / "utils" / "derived_features_all_math.py",
    )
    source = pd.DataFrame({
        "station_id": ["ECE_TEST"] * 10,
        "date": pd.date_range("2026-07-01", periods=10),
        "SMAP_sm_am_interp": [np.nan] * 10,
        "SMAP_sm_pm_interp": [np.nan] * 10,
    })

    result = features.add_smap_features(source)

    assert result["SMAP_sm_am_interp_mask"].eq(0).all()
    assert result["SMAP_sm_pm_interp_mask"].eq(0).all()
    assert result["SMAP_sm_interp_mask"].eq(0).all()
    smap_value_columns = [c for c in result if c.startswith("SMAP_") and not c.endswith("_mask")]
    assert result[smap_value_columns].isna().all().all()


def test_recovery_builder_restores_all_smap_descendants_to_native_missing():
    recovery_dir = Path("data/splits/derived_8.4-ece-v2-native-missing").resolve()
    recovery = _load_module("ece_native_missing_builder", recovery_dir / "build_from_existing_split.py")
    source = pd.DataFrame({
        "SMAP_sm_am_interp": [0.0, 0.0],
        "SMAP_sm_am_interp_mask": [1, 1],
        "A_d_SMAP_sm_interp_kobs5": [0.0, 0.0],
        "soil_moisture_5cm": [0.1, 0.2],
    })

    corrected, smap_columns = recovery.restore_native_missing(source)

    assert len(smap_columns) == 3
    assert corrected["SMAP_sm_am_interp"].isna().all()
    assert corrected["A_d_SMAP_sm_interp_kobs5"].isna().all()
    assert corrected["SMAP_sm_am_interp_mask"].eq(0).all()
    assert corrected["soil_moisture_5cm"].equals(source["soil_moisture_5cm"])
