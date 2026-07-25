"""Stable paths and protocol constants for the isolated 2.1 experiment."""

from __future__ import annotations

from pathlib import Path


EXPERIMENT_NAME = "derived_8.3-feature-selection-2.1"
EXP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXP_DIR.parents[2]
GLOBAL_CONFIG_PATH = EXP_DIR / "global_config.yaml"
MOE_CONFIG_PATH = EXP_DIR / "moe_config.yaml"
BENCHMARK_REGISTRY_PATH = EXP_DIR / "benchmark_registry.yaml"
DEVELOPMENT_FREEZE_PATH = EXP_DIR / "development_freeze.json"

TARGET = "soil_moisture_5cm"
STATION_COL = "station_id"
TIME_COL = "date"
V0_SOURCE = (
    "data/splits/derived_8.3/dataset_metadata.py::"
    "OVERALL_SELECTED_FEATURES_V0"
)
V0_COUNT = 50
EXPECTED_PREDICTOR_COUNT = 496
OUTER_ORIGINS = (2020, 2021, 2022)
ENDPOINT_COUNTS = (150, 125, 100, 80, 65, 50, 40)
DELTA_COUNTS = (0, 5, 10, 15)

EXACT_LEARNER_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "min_child_weight": 10,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "tree_method": "hist",
    "n_jobs": 1,
}

LEDGER_COLUMNS = (
    "model",
    "candidate",
    "path_source",
    "endpoint",
    "actual_count",
    "ordered_feature_hash",
    "fold_family",
    "outer_origin",
    "fold_id",
    "station_partition_seed",
    "learner_seed",
    "station",
    "date",
    "year",
    "month",
    "truth",
    "prediction",
    "residual",
    "absolute_error",
    "squared_error",
    "beta",
    "model_config_id",
    "router_regime",
    "route_distance",
)

DEVELOPMENT_STAGE_NAMES = (
    "01_preflight",
    "02_fold_manifests",
    "03_control_ledgers",
    "04_path_screen",
    "05_robust_candidate_generation",
    "06_candidate_oof",
    "07_global_decision",
    "08_beta_decision",
    "09_consensus",
    "10_station_temporal_diagnostics",
    "11_moe_causal_matrix",
    "12_regime_delta_moe_decision",
    "13_development_freeze",
    "14_development_report",
)

