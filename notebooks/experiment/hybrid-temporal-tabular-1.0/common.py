from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Models.Temporal.lstm.train_v20 import FEATURE_COLS as TABULAR_FEATURES
from Models.Temporal.lstm.train_v23_baseline import TOP25_FEATURES

EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "splits" / "derived_9.0"
TARGET = "soil_moisture_5cm"
ID_COLUMNS = ["station_id", "date"]
SEED = 42

SMAP_ALTERNATIVE_FEATURES = [
    "SMAP_sm_interp",
    "V_rollmean_SMAP_sm_interp_kobs7",
    "V_rollstd_SMAP_sm_interp_kobs7",
    "V_rollmean_SMAP_sm_interp_kobs30",
]

NDVI_LST_FEATURES = [
    "F_NDVI",
    "V_ema_F_NDVI_kobs30",
    "V_rollmean_F_NDVI_kobs30",
    "LST_modis",
    "V_rollmean_LST_modis_kobs30",
]

SOIL_CONTINUOUS_FEATURES = [
    "K_sand_clay_ratio_b0",
    "K_clay_plus_sand_b0",
]

SOIL_TEXTURE_COLUMN = "J_soil_texture_usda_b0"


def unique(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


def load_splits() -> dict[str, pd.DataFrame]:
    splits = {
        name: pd.read_csv(DATA_DIR / f"{name}.csv")
        for name in ("train", "val", "test")
    }
    for name, frame in splits.items():
        frame["date"] = pd.to_datetime(frame["date"])
        if frame.duplicated(ID_COLUMNS).any():
            raise ValueError(f"Duplicate station/date keys in {name}")
    return splits


def add_soil_texture_one_hot(
    splits: dict[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    categories = sorted(
        pd.to_numeric(splits["train"][SOIL_TEXTURE_COLUMN], errors="coerce")
        .dropna()
        .unique()
        .tolist()
    )
    outputs = {}
    columns = [f"soil_texture_b0_{int(category)}" for category in categories]
    for split_name, frame in splits.items():
        output = frame.copy()
        values = pd.to_numeric(output[SOIL_TEXTURE_COLUMN], errors="coerce")
        for category, column in zip(categories, columns):
            output[column] = (values == category).astype(np.float32)
        outputs[split_name] = output
    return outputs, columns


def feature_variants(texture_columns: list[str]) -> dict[str, list[str]]:
    baseline = list(TABULAR_FEATURES)
    without_x_year = [feature for feature in baseline if feature != "SMAP_x_year"]
    smap = unique(without_x_year + SMAP_ALTERNATIVE_FEATURES)
    return {
        "baseline_38": baseline,
        "minus_smap_x_year": without_x_year,
        "alternative_smap_mean_std": smap,
        "alternative_smap_plus_ndvi_lst": unique(smap + NDVI_LST_FEATURES),
        "alternative_smap_plus_soil": unique(
            smap + SOIL_CONTINUOUS_FEATURES + texture_columns
        ),
        "alternative_smap_ndvi_lst_soil": unique(
            smap + NDVI_LST_FEATURES + SOIL_CONTINUOUS_FEATURES + texture_columns
        ),
    }


def temporal_feature_variants() -> dict[str, list[str]]:
    static = {"elev", "K_slope_sin", "K_slope_cos", "K_aspect_cos"}
    baseline = [feature for feature in TOP25_FEATURES if feature not in static]
    alternative = [feature for feature in baseline if feature != "SMAP_x_year"]
    alternative = unique(alternative + SMAP_ALTERNATIVE_FEATURES + NDVI_LST_FEATURES)
    return {
        "current_with_smap_x_year": baseline,
        "alternative_without_smap_x_year": alternative,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {
            "r2": float("nan"),
            "rmse": float("nan"),
            "ubrmse": float("nan"),
            "bias": float("nan"),
            "mae": float("nan"),
            "q90": float("nan"),
            "n": 0,
        }
    error = y_pred - y_true
    bias = float(np.mean(error))
    rmse = float(np.sqrt(np.mean(error**2)))
    ubrmse = float(np.sqrt(np.mean((error - bias) ** 2)))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan")
    return {
        "r2": r2,
        "rmse": rmse,
        "ubrmse": ubrmse,
        "bias": bias,
        "mae": float(np.mean(np.abs(error))),
        "q90": float(np.quantile(np.abs(error), 0.90)),
        "n": int(len(y_true)),
    }


def evaluate_predictions(frame: pd.DataFrame) -> dict:
    payload = {
        "overall": regression_metrics(frame["y_true"], frame["y_pred"]),
        "per_station": {},
        "per_year": {},
    }
    for station, group in frame.groupby("station_id"):
        payload["per_station"][str(station)] = regression_metrics(
            group["y_true"], group["y_pred"]
        )
    years = pd.to_datetime(frame["date"]).dt.year
    for year, group in frame.groupby(years):
        payload["per_year"][str(int(year))] = regression_metrics(
            group["y_true"], group["y_pred"]
        )
    station_r2 = [
        metrics["r2"]
        for metrics in payload["per_station"].values()
        if np.isfinite(metrics["r2"])
    ]
    payload["macro_station_r2"] = float(np.mean(station_r2))
    return payload


def fit_xgboost(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    seed: int = SEED,
) -> tuple[SimpleImputer, XGBRegressor]:
    missing = [column for column in feature_columns if column not in train_frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    imputer = SimpleImputer(strategy="median")
    train_x = imputer.fit_transform(train_frame[feature_columns])
    val_x = imputer.transform(val_frame[feature_columns])
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1200,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=10,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=4,
        tree_method="hist",
        early_stopping_rounds=50,
    )
    model.fit(
        train_x,
        train_frame[TARGET].to_numpy(dtype=float),
        eval_set=[(val_x, val_frame[TARGET].to_numpy(dtype=float))],
        verbose=False,
    )
    return imputer, model


def make_prediction_frame(
    source: pd.DataFrame,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    output = source[ID_COLUMNS].copy()
    output["y_true"] = source[TARGET].to_numpy(dtype=float)
    output["y_pred"] = np.asarray(y_pred, dtype=float)
    return output


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, default=json_default),
        encoding="utf-8",
    )
