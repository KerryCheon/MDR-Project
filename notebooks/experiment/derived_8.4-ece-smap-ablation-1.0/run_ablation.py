"""Reproducible SMAP missing-data ablation for the 2026 ECE evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor


EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
PARENT_EXP = EXP_DIR.parent / "derived_8.4-ece-additional-eval-1.0"
sys.path.insert(0, str(PARENT_EXP))
import eval_engine as engine  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seeds", default=None, help="Comma-separated seed override")
    return parser.parse_args()


def load_configuration() -> tuple[dict, dict]:
    with (EXP_DIR / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with (PARENT_EXP / "config.yaml").open(encoding="utf-8") as handle:
        parent = yaml.safe_load(handle)
    return config, parent


def load_data(config: dict, parent: dict) -> dict[str, pd.DataFrame | list[str] | str]:
    features = parent["feature_columns"]
    target = config["target_column"]
    required = list(dict.fromkeys(["station_id", "date", target, *features]))

    train_dir = PROJECT_ROOT / config["datasets"]["training"]
    train = pd.concat([
        pd.read_csv(train_dir / "train.csv", usecols=required, low_memory=False),
        pd.read_csv(train_dir / "val.csv", usecols=required, low_memory=False),
    ], ignore_index=True)
    zero = pd.read_csv(
        PROJECT_ROOT / config["datasets"]["ece_zero_filled"] / "test.csv",
        usecols=required,
        low_memory=False,
    )
    native = pd.read_csv(
        PROJECT_ROOT / config["datasets"]["ece_native_missing"] / "test.csv",
        usecols=required,
        low_memory=False,
    )
    for frame in (train, zero, native):
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    key_columns = ["station_id", "date", target]
    if not zero[key_columns].equals(native[key_columns]):
        raise ValueError("Corrected and parent ECE rows are not aligned")
    smap_features = [feature for feature in features if "SMAP" in feature]
    return {
        "train": train,
        "zero": zero,
        "native": native,
        "features": features,
        "smap_features": smap_features,
        "target": target,
    }


def training_month_climatology(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    smap_features: list[str],
) -> pd.DataFrame:
    """Fill evaluation-only SMAP gaps using training-only monthly medians."""
    filled = evaluation.copy()
    train_month = train["date"].dt.month
    eval_month = filled["date"].dt.month
    for feature in smap_features:
        monthly = train.groupby(train_month)[feature].median()
        fallback = train[feature].median()
        replacement = eval_month.map(monthly).fillna(fallback)
        filled[feature] = filled[feature].fillna(replacement)
    return filled


def block_mask_training(
    train: pd.DataFrame,
    smap_features: list[str],
    seed: int,
    block_days: int,
    fraction: float,
) -> tuple[pd.DataFrame, int]:
    """Mask deterministic contiguous station blocks to simulate outages."""
    masked = train.sort_values(["station_id", "date"]).copy()
    rng = np.random.default_rng(seed)
    masked_rows = pd.Series(False, index=masked.index)
    for _, station in masked.groupby("station_id", sort=True):
        block_ids = np.arange(len(station)) // block_days
        unique_blocks = np.unique(block_ids)
        n_mask = max(1, int(np.ceil(len(unique_blocks) * fraction)))
        selected = rng.choice(unique_blocks, size=n_mask, replace=False)
        station_mask = np.isin(block_ids, selected)
        masked_rows.loc[station.index[station_mask]] = True
    masked.loc[masked_rows, smap_features] = np.nan
    return masked.sort_index(), int(masked_rows.sum())


def fit_model(
    train: pd.DataFrame,
    features: list[str],
    target: str,
    params: dict,
    seed: int,
) -> XGBRegressor:
    model_params = dict(params)
    beta = model_params.pop("weight_beta")
    model_params.update({
        "tree_method": "hist",
        "device": "cpu",
        "n_jobs": -1,
        "random_state": seed,
    })
    weights = engine.compute_sample_weights(train, beta=beta)
    model = XGBRegressor(**model_params)
    model.fit(train[features], train[target], sample_weight=weights, verbose=0)
    return model


def metric_record(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    target: str,
    strategy: str,
    seed: int,
    fit_time: float,
) -> dict:
    record = engine.compute_metrics(frame[target], predictions)
    record.update({"strategy": strategy, "seed": seed, "fit_time_s": fit_time})
    return record


def main() -> None:
    args = parse_args()
    config, parent = load_configuration()
    data = load_data(config, parent)
    train = data["train"]
    zero = data["zero"]
    native = data["native"]
    features = data["features"]
    smap_features = data["smap_features"]
    target = data["target"]

    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else config["seeds"]
    params = dict(config["model_params"])
    if args.smoke:
        seeds = [42]
        params["n_estimators"] = 50

    climatology = training_month_climatology(train, native, smap_features)
    no_smap_features = [feature for feature in features if feature not in smap_features]
    records: list[dict] = []
    station_records: list[pd.DataFrame] = []
    masking_audit: list[dict] = []

    print("SMAP ABLATION INPUT AUDIT")
    print(f"train_rows={len(train)} ece_rows={len(native)} features={len(features)} smap_features={len(smap_features)}")
    print(f"zero_policy_finite_smap={int(zero[smap_features].notna().sum().sum())}")
    print(f"native_missing_finite_smap={int(native[smap_features].notna().sum().sum())}")
    print(f"climatology_remaining_missing={int(climatology[smap_features].isna().sum().sum())}")

    for seed in seeds:
        start = time.time()
        standard = fit_model(train, features, target, params, seed)
        standard_time = time.time() - start
        for strategy, evaluation in (
            ("zero_filled_existing_policy", zero),
            ("native_missing_existing_training", native),
            ("training_month_climatology", climatology),
        ):
            predictions = standard.predict(evaluation[features])
            records.append(metric_record(evaluation, predictions, target, strategy, seed, standard_time))
            station = engine.compute_per_station_metrics(evaluation, predictions, target_col=target)
            station["strategy"] = strategy
            station["seed"] = seed
            station_records.append(station)

        start = time.time()
        no_smap = fit_model(train, no_smap_features, target, params, seed)
        no_smap_time = time.time() - start
        predictions = no_smap.predict(native[no_smap_features])
        records.append(metric_record(native, predictions, target, "no_smap_retrained", seed, no_smap_time))
        station = engine.compute_per_station_metrics(native, predictions, target_col=target)
        station["strategy"] = "no_smap_retrained"
        station["seed"] = seed
        station_records.append(station)

        block_config = config["block_masking"]
        masked_train, masked_rows = block_mask_training(
            train,
            smap_features,
            seed,
            block_config["block_days"],
            block_config["fraction"],
        )
        start = time.time()
        block_model = fit_model(masked_train, features, target, params, seed)
        block_time = time.time() - start
        predictions = block_model.predict(native[features])
        records.append(metric_record(native, predictions, target, "block_masked_retrained", seed, block_time))
        station = engine.compute_per_station_metrics(native, predictions, target_col=target)
        station["strategy"] = "block_masked_retrained"
        station["seed"] = seed
        station_records.append(station)
        masking_audit.append({"seed": seed, "masked_rows": masked_rows, "training_rows": len(train)})
        print(f"seed={seed} complete")

    seed_metrics = pd.DataFrame(records)
    station_metrics = pd.concat(station_records, ignore_index=True)
    summary = seed_metrics.groupby("strategy", sort=False).agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        bias_mean=("bias", "mean"),
        ubrmse_mean=("ubrmse", "mean"),
        r2_mean=("r2", "mean"),
        pearson_r_mean=("pearson_r", "mean"),
    ).reset_index()
    zero_rmse = float(summary.loc[summary["strategy"].eq("zero_filled_existing_policy"), "rmse_mean"].iloc[0])
    summary["rmse_change_vs_zero"] = summary["rmse_mean"] - zero_rmse

    seed_metrics.to_csv(EXP_DIR / "seed_metrics.csv", index=False)
    station_metrics.to_csv(EXP_DIR / "station_metrics.csv", index=False)
    summary.to_csv(EXP_DIR / "summary.csv", index=False)
    with (EXP_DIR / "masking_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(masking_audit, handle, indent=2)

    print("\nSMAP ABLATION SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
