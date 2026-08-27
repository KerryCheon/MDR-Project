"""Pipeline execution for derived_8.4-ece-additional-eval-1.0."""

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from xgboost import XGBRegressor

import eval_engine as engine

EXP_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXP_DIR / "models"
PREDS_DIR = EXP_DIR / "predictions"
FIGURES_DIR = EXP_DIR / "figures"
ARTIFACTS_DIR = EXP_DIR / "artifacts"

for d in [MODELS_DIR, PREDS_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MDR-v25 ECE Additional Evaluation Pipeline")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode (1 seed, 50 trees)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list override")
    return parser.parse_args()


def load_datasets(config: Dict[str, Any], exp_dir: Path) -> Dict[str, Any]:
    """Load trainval and test splits for derived_8.0, derived_8.4, and derived_8.4-ece."""
    root = engine.find_data_root(config, exp_dir)
    target = config["target_column"]
    features = config["feature_columns"]
    meta = config.get("meta_columns", ["station_id", "date", "longitude", "latitude"])
    load_cols = list(set(meta + features + [target]))

    # 1. derived_8.0
    p80 = root / config["datasets"]["derived_8_0"]
    tr80 = pd.read_csv(p80 / "train.csv", usecols=load_cols)
    val80 = pd.read_csv(p80 / "val.csv", usecols=load_cols)
    te80 = pd.read_csv(p80 / "test.csv", usecols=load_cols)
    tv80 = pd.concat([tr80, val80], ignore_index=True)

    # 2. derived_8.4
    p84 = root / config["datasets"]["derived_8_4"]
    tr84 = pd.read_csv(p84 / "train.csv", usecols=load_cols)
    val84 = pd.read_csv(p84 / "val.csv", usecols=load_cols)
    te84 = pd.read_csv(p84 / "test.csv", usecols=load_cols)
    tv84 = pd.concat([tr84, val84], ignore_index=True)

    # 3. derived_8.4-ece
    pece = root / config["datasets"]["derived_8_4_ece"]
    teece = pd.read_csv(pece / "test.csv", usecols=load_cols)

    print("[Data] Loaded datasets:", flush=True)
    print(f"  derived_8.0: trainval={len(tv80)} rows ({len(tv80['station_id'].unique())} stations: {sorted(tv80['station_id'].unique())}), test={len(te80)} rows", flush=True)
    print(f"  derived_8.4: trainval={len(tv84)} rows ({len(tv84['station_id'].unique())} stations: {sorted(tv84['station_id'].unique())}), test={len(te84)} rows", flush=True)
    print(f"  derived_8.4-ece: test={len(teece)} rows ({len(teece['station_id'].unique())} stations: {sorted(teece['station_id'].unique())})", flush=True)

    return {
        "tv80": tv80,
        "te80": te80,
        "tv84": tv84,
        "te84": te84,
        "teece": teece,
        "features": features,
        "target": target,
    }


def main():
    args = parse_args()
    config = engine.load_config(EXP_DIR / "config.yaml")

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else config["seeds"]
    if args.smoke:
        seeds = [42]
        print("[Mode] Smoke test enabled: 1 seed (42), n_estimators=50", flush=True)

    # Detect device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] Using compute device: {device} (CUDA available: {torch.cuda.is_available()})", flush=True)

    data = load_datasets(config, EXP_DIR)
    features = data["features"]
    target = data["target"]
    teece = data["teece"]

    # Compute sample weights for trainval sets
    w_tv80 = engine.compute_sample_weights(data["tv80"], beta=config["model_params"]["weighted"]["weight_beta"])
    w_tv84 = engine.compute_sample_weights(data["tv84"], beta=config["model_params"]["weighted"]["weight_beta"])

    seed_records_ece = []
    seed_records_temporal = []
    station_seed_records_ece = []
    station_seed_records_temporal = []
    feature_importances = {}
    
    # Store predictions dataframe for detailed analysis
    preds_ece_df = teece[["station_id", "date", "longitude", "latitude", target]].copy()
    preds_ece_df.rename(columns={target: "y_true"}, inplace=True)

    configs_list = config["configurations"]

    print("\n" + "=" * 80, flush=True)
    print(f"STARTING EVALUATION: {len(configs_list)} Configurations x {len(seeds)} Seeds = {len(configs_list) * len(seeds)} Model Fits", flush=True)
    print("=" * 80, flush=True)

    start_all = time.time()

    for cfg in configs_list:
        cfg_id = cfg["config_id"]
        train_ds_name = cfg["train_dataset"]
        model_type = cfg["model_type"]
        desc = cfg["description"]

        if train_ds_name == "derived_8.0":
            trainval_df = data["tv80"]
            test_temp_df = data["te80"]
            sample_weight = w_tv80 if model_type == "weighted" else None
        else:
            trainval_df = data["tv84"]
            test_temp_df = data["te84"]
            sample_weight = w_tv84 if model_type == "weighted" else None

        X_trainval = trainval_df[features]
        y_trainval = trainval_df[target]
        X_test_temp = test_temp_df[features]
        y_test_temp = test_temp_df[target]
        X_test_ece = teece[features]
        y_test_ece = teece[target]

        base_params = dict(config["model_params"][model_type])
        base_params.pop("weight_beta", None)
        if args.smoke:
            base_params["n_estimators"] = 50

        # Set device & hist tree method
        base_params["tree_method"] = "hist"
        base_params["device"] = device
        if device == "cpu":
            base_params["n_jobs"] = -1

        importances_for_cfg = []

        print(f"\n--- Running [{cfg_id}] ({desc}) ---", flush=True)

        for seed in seeds:
            fit_params = dict(base_params)
            fit_params["random_state"] = seed

            t0 = time.time()
            model = XGBRegressor(**fit_params)
            model.fit(
                X_trainval,
                y_trainval,
                sample_weight=sample_weight,
                verbose=0,
            )
            fit_time = time.time() - t0

            # Save model
            model_file = MODELS_DIR / f"{cfg_id}__s{seed}.json"
            model.save_model(str(model_file))

            # Feature importances
            imp = model.feature_importances_
            imp_norm = imp / np.sum(imp) if np.sum(imp) > 0 else imp
            importances_for_cfg.append(imp_norm)

            # Predict on ECE spatial dataset
            pred_ece = np.asarray(model.predict(X_test_ece)).ravel()
            np.save(PREDS_DIR / f"{cfg_id}__s{seed}__ece_preds.npy", pred_ece)
            preds_ece_df[f"pred__{cfg_id}__s{seed}"] = pred_ece

            # Predict on in-distribution temporal test dataset
            pred_temp = np.asarray(model.predict(X_test_temp)).ravel()
            np.save(PREDS_DIR / f"{cfg_id}__s{seed}__temp_preds.npy", pred_temp)

            # Metrics: ECE
            m_ece = engine.compute_metrics(y_test_ece, pred_ece)
            m_ece["config_id"] = cfg_id
            m_ece["train_dataset"] = train_ds_name
            m_ece["model_type"] = model_type
            m_ece["seed"] = seed
            m_ece["fit_time_s"] = fit_time
            seed_records_ece.append(m_ece)

            # Metrics: Temporal
            m_temp = engine.compute_metrics(y_test_temp, pred_temp)
            m_temp["config_id"] = cfg_id
            m_temp["train_dataset"] = train_ds_name
            m_temp["model_type"] = model_type
            m_temp["seed"] = seed
            m_temp["fit_time_s"] = fit_time
            seed_records_temporal.append(m_temp)

            # Per-station metrics: ECE
            st_m_ece = engine.compute_per_station_metrics(teece, pred_ece, target_col=target)
            st_m_ece["config_id"] = cfg_id
            st_m_ece["train_dataset"] = train_ds_name
            st_m_ece["model_type"] = model_type
            st_m_ece["seed"] = seed
            station_seed_records_ece.append(st_m_ece)

            # Per-station metrics: Temporal
            st_m_temp = engine.compute_per_station_metrics(test_temp_df, pred_temp, target_col=target)
            st_m_temp["config_id"] = cfg_id
            st_m_temp["train_dataset"] = train_ds_name
            st_m_temp["model_type"] = model_type
            st_m_temp["seed"] = seed
            station_seed_records_temporal.append(st_m_temp)

            print(f"  Seed {seed:5d}: ECE R2 = {m_ece['r2']:+.4f} (RMSE = {m_ece['rmse']:.4f}) | Temp R2 = {m_temp['r2']:+.4f} (RMSE = {m_temp['rmse']:.4f}) [{fit_time:.2f}s]", flush=True)

        # Average feature importance for config across seeds
        avg_imp = np.mean(importances_for_cfg, axis=0)
        feature_importances[cfg_id] = pd.Series(avg_imp, index=features)

    total_time = time.time() - start_all
    print(f"\n[Done] All fits completed in {total_time:.2f}s", flush=True)

    # =========================================================================
    # Convert and save summary DataFrames
    # =========================================================================
    df_seed_ece = pd.DataFrame(seed_records_ece)
    df_seed_temp = pd.DataFrame(seed_records_temporal)
    df_st_seed_ece = pd.concat(station_seed_records_ece, ignore_index=True)
    df_st_seed_temp = pd.concat(station_seed_records_temporal, ignore_index=True)

    df_seed_ece.to_csv(EXP_DIR / "seed_summary_ece.csv", index=False)
    df_seed_temp.to_csv(EXP_DIR / "seed_summary_temporal.csv", index=False)
    df_st_seed_ece.to_csv(EXP_DIR / "station_seed_metrics_ece.csv", index=False)
    df_st_seed_temp.to_csv(EXP_DIR / "station_seed_metrics_temporal.csv", index=False)
    preds_ece_df.to_csv(EXP_DIR / "predictions_ece_df.csv", index=False)

    # Config-level summary tables
    summary_rows_ece = []
    for cfg_id in df_seed_ece["config_id"].unique():
        sub = df_seed_ece[df_seed_ece["config_id"] == cfg_id]
        sr2 = engine.seed_summary(sub["r2"])
        srmse = engine.seed_summary(sub["rmse"])
        smae = engine.seed_summary(sub["mae"])
        sbias = engine.seed_summary(sub["bias"])
        subrmse = engine.seed_summary(sub["ubrmse"])
        spr = engine.seed_summary(sub["pearson_r"])
        
        # Per-station average R2 across seeds
        sub_st = df_st_seed_ece[df_st_seed_ece["config_id"] == cfg_id]
        st_med_r2 = sub_st.groupby("station_id")["r2"].median()
        st_mean_r2 = sub_st.groupby("station_id")["r2"].mean()

        summary_rows_ece.append({
            "config_id": cfg_id,
            "train_dataset": sub["train_dataset"].iloc[0],
            "model_type": sub["model_type"].iloc[0],
            "n_seeds": len(sub),
            "r2_mean": sr2["mean"],
            "r2_std": sr2["std"],
            "r2_median": sr2["median"],
            "r2_min": sr2["min"],
            "r2_max": sr2["max"],
            "r2_ci": f"[{sr2['ci_low']:.4f}, {sr2['ci_high']:.4f}]",
            "rmse_mean": srmse["mean"],
            "rmse_std": srmse["std"],
            "mae_mean": smae["mean"],
            "bias_mean": sbias["mean"],
            "ubrmse_mean": subrmse["mean"],
            "pearson_r_mean": spr["mean"],
            "station_mean_r2": float(st_mean_r2.mean()),
            "station_median_r2": float(st_med_r2.median()),
        })
    df_cfg_ece = pd.DataFrame(summary_rows_ece)
    df_cfg_ece.to_csv(EXP_DIR / "config_summary_ece.csv", index=False)

    summary_rows_temp = []
    for cfg_id in df_seed_temp["config_id"].unique():
        sub = df_seed_temp[df_seed_temp["config_id"] == cfg_id]
        sr2 = engine.seed_summary(sub["r2"])
        srmse = engine.seed_summary(sub["rmse"])
        smae = engine.seed_summary(sub["mae"])
        sbias = engine.seed_summary(sub["bias"])
        subrmse = engine.seed_summary(sub["ubrmse"])
        spr = engine.seed_summary(sub["pearson_r"])

        summary_rows_temp.append({
            "config_id": cfg_id,
            "train_dataset": sub["train_dataset"].iloc[0],
            "model_type": sub["model_type"].iloc[0],
            "n_seeds": len(sub),
            "temp_r2_mean": sr2["mean"],
            "temp_r2_std": sr2["std"],
            "temp_r2_median": sr2["median"],
            "temp_rmse_mean": srmse["mean"],
            "temp_mae_mean": smae["mean"],
            "temp_bias_mean": sbias["mean"],
            "temp_ubrmse_mean": subrmse["mean"],
            "temp_pearson_r_mean": spr["mean"],
        })
    df_cfg_temp = pd.DataFrame(summary_rows_temp)
    df_cfg_temp.to_csv(EXP_DIR / "config_summary_temporal.csv", index=False)

    # Transfer Gap Table
    gap_df = pd.merge(df_cfg_temp, df_cfg_ece, on=["config_id", "train_dataset", "model_type", "n_seeds"])
    gap_df["transfer_gap_r2 (ECE - Temp)"] = gap_df["r2_mean"] - gap_df["temp_r2_mean"]
    gap_df["transfer_gap_rmse (ECE - Temp)"] = gap_df["rmse_mean"] - gap_df["temp_rmse_mean"]
    gap_df.to_csv(EXP_DIR / "transfer_gap_summary.csv", index=False)

    # Per-station median summary table across 5 ECE stations
    st_median_table = df_st_seed_ece.groupby(["config_id", "station_id"])[["r2", "rmse", "mae", "bias", "pearson_r"]].median().reset_index()
    st_median_table.to_csv(EXP_DIR / "station_median_summary_ece.csv", index=False)

    # Per-station R2 matrix (configs x stations)
    piv_r2 = st_median_table.pivot(index="config_id", columns="station_id", values="r2")
    piv_r2.to_csv(EXP_DIR / "station_matrix_ece_r2.csv")

    # Pairwise Hypothesis Tests
    comparisons = [
        # 1. 8.0 vs 8.4 (No Weights)
        ("d80_no_weights", "d84_no_weights", "derived_8.0 (5 st) vs derived_8.4 (7 st) [No Weights]"),
        # 2. 8.0 vs 8.4 (Weighted)
        ("d80_weighted", "d84_weighted", "derived_8.0 (5 st) vs derived_8.4 (7 st) [Weighted]"),
        # 3. Weighted vs No-Weights on derived_8.0
        ("d80_weighted", "d80_no_weights", "Weighted vs No-Weights [derived_8.0]"),
        # 4. Weighted vs No-Weights on derived_8.4
        ("d84_weighted", "d84_no_weights", "Weighted vs No-Weights [derived_8.4]"),
    ]
    pairwise_rows = []
    for a_id, b_id, label in comparisons:
        a_sub = df_seed_ece[df_seed_ece["config_id"] == a_id].sort_values("seed")
        b_sub = df_seed_ece[df_seed_ece["config_id"] == b_id].sort_values("seed")
        t_r2 = engine.paired_hypothesis_test(a_sub["r2"], b_sub["r2"], metric_name="r2")
        t_r2["comparison"] = label
        t_r2["config_A"] = a_id
        t_r2["config_B"] = b_id
        pairwise_rows.append(t_r2)

    df_pairwise = pd.DataFrame(pairwise_rows)
    df_pairwise.to_csv(EXP_DIR / "pairwise_hypothesis_tests.csv", index=False)

    # Feature Importance DataFrame
    df_fi = pd.DataFrame(feature_importances)
    df_fi["mean_all"] = df_fi.mean(axis=1)
    df_fi.sort_values("mean_all", ascending=False).to_csv(EXP_DIR / "feature_importances.csv")

    print("\n" + "=" * 80, flush=True)
    print("ALL SUMMARY ARTIFACTS GENERATED SUCCESSFULLY", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
