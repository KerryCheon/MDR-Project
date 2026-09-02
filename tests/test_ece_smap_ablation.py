import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


RUNNER = Path(
    "notebooks/experiment/derived_8.4-ece-smap-ablation-1.0/run_ablation.py"
).resolve()


def _module():
    spec = importlib.util.spec_from_file_location("ece_smap_ablation", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_training_month_climatology_uses_training_months_and_global_fallback():
    module = _module()
    train = pd.DataFrame({
        "date": pd.to_datetime(["2020-07-01", "2021-07-01", "2020-08-01"]),
        "SMAP_feature": [0.2, 0.4, 0.8],
    })
    evaluation = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-10", "2026-09-10"]),
        "SMAP_feature": [np.nan, np.nan],
    })

    result = module.training_month_climatology(
        train,
        evaluation,
        ["SMAP_feature"],
    )

    assert result.loc[0, "SMAP_feature"] == pytest.approx(0.3)
    assert result.loc[1, "SMAP_feature"] == pytest.approx(0.4)


def test_block_masking_is_seed_deterministic_and_only_changes_smap_features():
    module = _module()
    train = pd.DataFrame({
        "station_id": ["A"] * 20 + ["B"] * 20,
        "date": pd.date_range("2020-01-01", periods=20).tolist() * 2,
        "SMAP_feature": np.arange(40, dtype=float),
        "weather_feature": np.arange(40, dtype=float),
    })

    first, first_count = module.block_mask_training(
        train,
        ["SMAP_feature"],
        seed=42,
        block_days=5,
        fraction=0.25,
    )
    second, second_count = module.block_mask_training(
        train,
        ["SMAP_feature"],
        seed=42,
        block_days=5,
        fraction=0.25,
    )

    assert first_count == second_count == 10
    assert first["SMAP_feature"].equals(second["SMAP_feature"])
    assert first["weather_feature"].equals(train["weather_feature"])
