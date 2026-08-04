#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))

from lstm.train import (
    train_v21_and_extract_all, compute_pca_all, extract_all_representations,
    load_best_model, ALL_FEATURES, SEQ_LEN, PCA_LEVELS,
)
from eval_hybrid.data import load_hybrid_experiment_data
from eval_hybrid.evaluator import HybridStrategyEvaluator
from eval_hybrid.shap_analysis import run_full_shap_analysis

CONFIG_PATH = EXP_DIR / "config.yaml"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
MODELS_DIR = EXP_DIR / "models"


REPR_LABELS = {
    "ctx": ("160 CTX", "160 CTX PCA"),
    "hh": ("80 Head Hidden", "80 Head Hidden PCA"),
    "hp": ("80 Pre-ReLU", "80 Pre-ReLU PCA"),
}

PCA_DISPLAY = {
    "": "",
    "pca95": "-95%",
    "pca64": "-64",
    "pca32": "-32",
    "pca16": "-16",
}

# Build variants dynamically from PCA_LEVELS
PCA_KEYS = [lvl for lvl, _ in PCA_LEVELS]
VARIANTS = [(r, "") for r in ("ctx", "hh", "hp")]
VARIANTS += [(r, k) for r in ("ctx", "hh", "hp") for k in PCA_KEYS]


def _model_name(repr_type: str, pca_level: str, strategy: str) -> str:
    base_label, pca_label = REPR_LABELS[repr_type]
    suffix = PCA_DISPLAY[pca_level]
    if pca_level:
        display = f"{pca_label}{suffix}"
    else:
        display = base_label
    if strategy == "Global_Single":
        return f"Global Single (54 Backbone + {display})"
    return f"Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + {display})"


def _candidate_id(repr_type: str, pca_level: str, strategy: str) -> str:
    suffix = f"_{pca_level}" if pca_level else ""
    if strategy == "Global_Single":
        return f"Global_Single_{repr_type}{suffix}"
    return f"Clustering_V0_k2_{repr_type}{suffix}"


def _repr_suffix(pca_level: str) -> str:
    return f"_{pca_level}" if pca_level else ""


def compute_c1_gain_additions(config: dict | None = None) -> list[str]:
    rel_path = config.get("eval11_summary_path", "notebooks/experiment/derived_8.4-eval-1.1/metrics_summary.csv") if config else "notebooks/experiment/derived_8.4-eval-1.1/metrics_summary.csv"
    eval11_summary = PROJECT_ROOT / rel_path
    if eval11_summary.exists():
        df_eval11 = pd.read_csv(eval11_summary)
        row = df_eval11[df_eval11["model_name"] == "Clustering_V0_Full_k2 (Winner c0=0, c1=10)"]
        if not row.empty and pd.notna(row["cluster_1_additions"].values[0]):
            c1_str = str(row["cluster_1_additions"].values[0])
            additions = [f.strip() for f in c1_str.split(";") if f.strip()]
            return additions[:10]
    return [
        "V_rollmin_F_NDMI_kobs30", "V_rollmean_G_API_kobs14", "lia_mean_asc_deg",
        "J_bio_bio04", "DOY", "SMAP_sm_am_interp", "J_bio_bio07",
        "SMAP_sm_am_interp_lag1", "C_lag_F_NDMI_kobs30", "SMAP_sm_pm_interp_lag1",
    ]


def main():
    print("=" * 75, flush=True)
    print("Starting derived_8.0-optimization-2.0 Evaluation (V21 + PCA)", flush=True)
    print("=" * 75, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / config.get("data_dir", "data/splits/derived_8.0")

    lstm_checkpoint = MODELS_DIR / "best_lstm_model.pt"

    # Phase 1-3: Train V21, extract representations, compute PCA
    has_raw = all((ARTIFACTS_DIR / f"{n}_{s}.npy").exists()
                  for n in ("ctx", "head_hidden", "head_pre_relu")
                  for s in ("train", "val", "test"))
    pca_raw_names = ("ctx", "head_hidden", "head_pre_relu")
    pca_level_keys = [lvl for lvl, _ in PCA_LEVELS]
    has_pca = all((ARTIFACTS_DIR / f"{n}_{l}_{s}.npy").exists()
                  for n in pca_raw_names
                  for l in pca_level_keys
                  for s in ("train", "val", "test"))

    if has_raw and has_pca:
        print("[V21] Found existing raw + PCA representations. Skipping training.", flush=True)
    else:
        if has_raw and not has_pca:
            print("[V21] Raw found, PCA missing. Recomputing PCA only.", flush=True)
            reps_map = {
                "ctx": (np.load(ARTIFACTS_DIR / "ctx_train.npy"),
                        np.load(ARTIFACTS_DIR / "ctx_val.npy"),
                        np.load(ARTIFACTS_DIR / "ctx_test.npy")),
                "hh": (np.load(ARTIFACTS_DIR / "head_hidden_train.npy"),
                       np.load(ARTIFACTS_DIR / "head_hidden_val.npy"),
                       np.load(ARTIFACTS_DIR / "head_hidden_test.npy")),
                "hp": (np.load(ARTIFACTS_DIR / "head_pre_relu_train.npy"),
                       np.load(ARTIFACTS_DIR / "head_pre_relu_val.npy"),
                       np.load(ARTIFACTS_DIR / "head_pre_relu_test.npy")),
            }
            pca_info = compute_pca_all(reps_map, ARTIFACTS_DIR)
        else:
            best_seed_info, pca_info, lstm_metrics = train_v21_and_extract_all(data_dir, ARTIFACTS_DIR, MODELS_DIR)
            best_ckpt = MODELS_DIR / f"lstm_seed{best_seed_info['seed']}.pt"
            if best_ckpt.exists() and not lstm_checkpoint.exists():
                import shutil
                shutil.copy(best_ckpt, lstm_checkpoint)

    # Load LSTM metrics
    with open(ARTIFACTS_DIR / "lstm_metrics.json") as f:
        lstm_metrics = json.load(f)

    c1_additions = compute_c1_gain_additions(config)
    print(f"[Cluster 1 Deltas] {c1_additions}", flush=True)

    summary_records = []
    per_regime_records = []
    all_metrics = {}
    results_map = {}
    evaluator_map = {}

    # --- Tabular baselines (no PCA, no LSTM features) ---
    print("\n" + "=" * 75, flush=True)
    print("Training baselines (tabular only)", flush=True)
    print("=" * 75, flush=True)

    data_base = load_hybrid_experiment_data(PROJECT_ROOT, EXP_DIR, config, repr_type="ctx", pca_suffix="")
    backbone_54 = data_base.shared_backbone_54

    print(f"[Data] TrainVal={len(data_base.trainval)}, Test={len(data_base.test)}, Backbone={len(backbone_54)}", flush=True)

    # 1. Global Single Baseline
    eval_global_base = HybridStrategyEvaluator(data_base, config, "Global_Single", models_dir=MODELS_DIR)
    eval_v0_base = HybridStrategyEvaluator(data_base, config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR)

    res_g_base = eval_global_base.fit_and_evaluate(
        model_name="Global Single (54 Backbone)",
        candidate_id="Global_Single_54_Backbone",
        global_features=backbone_54,
    )
    rec = res_g_base.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec
    results_map[rec["model_name"]] = res_g_base
    evaluator_map[rec["model_name"]] = eval_global_base

    # 2. Clustering Baseline
    res_v0_base = eval_v0_base.fit_and_evaluate(
        model_name="Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)",
        candidate_id="Clustering_V0_k2_54_Backbone",
        global_features=backbone_54,
        cluster_additions={"0": [], "1": c1_additions},
    )
    rec = res_v0_base.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec
    results_map[rec["model_name"]] = res_v0_base
    evaluator_map[rec["model_name"]] = eval_v0_base

    for cl, m in res_v0_base.cluster_metrics.items():
        per_regime_records.append({
            "model_name": rec["model_name"], "cluster": cl,
            "n_train": m["n_train"], "n_test": m["n_test"],
            "r2": m["r2"], "rmse": m["rmse"], "ubrmse": m["ubrmse"],
            "bias": m["bias"], "mae": m["mae"], "pearson": m["pearson"],
        })

    # --- Hybrid models (all representation variants × 2 strategies) ---
    print("\n" + "=" * 75, flush=True)
    print("Training hybrid models (12 variants × 2 strategies)", flush=True)
    print("=" * 75, flush=True)

    for repr_type, pca_level in VARIANTS:
        suffix = _repr_suffix(pca_level)
        display_suffix = PCA_DISPLAY[pca_level]

        data_variant = load_hybrid_experiment_data(
            PROJECT_ROOT, EXP_DIR, config,
            repr_type=repr_type, pca_suffix=suffix,
        )
        hybrid_feats = data_variant.hybrid_features
        repr_count = data_variant.repr_feature_count

        print(f"[{repr_type}{display_suffix}] {repr_count} repr + {len(backbone_54)} backbone = {len(hybrid_feats)} tot", flush=True)

        eval_g = HybridStrategyEvaluator(data_variant, config, "Global_Single", models_dir=MODELS_DIR)
        eval_c = HybridStrategyEvaluator(data_variant, config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR)

        # Global Single variant
        mname_g = _model_name(repr_type, pca_level, "Global_Single")
        cid_g = _candidate_id(repr_type, pca_level, "Global_Single")
        res_g = eval_g.fit_and_evaluate(
            model_name=mname_g,
            candidate_id=cid_g,
            global_features=hybrid_feats,
        )
        rec = res_g.as_record()
        summary_records.append(rec)
        all_metrics[rec["model_name"]] = rec
        results_map[rec["model_name"]] = res_g
        evaluator_map[rec["model_name"]] = eval_g

        # Clustering variant
        mname_c = _model_name(repr_type, pca_level, "Clustering_V0_Full_k2")
        cid_c = _candidate_id(repr_type, pca_level, "Clustering_V0_Full_k2")
        res_c = eval_c.fit_and_evaluate(
            model_name=mname_c,
            candidate_id=cid_c,
            global_features=hybrid_feats,
            cluster_additions={"0": [], "1": c1_additions},
        )
        rec = res_c.as_record()
        summary_records.append(rec)
        all_metrics[rec["model_name"]] = rec
        results_map[rec["model_name"]] = res_c
        evaluator_map[rec["model_name"]] = eval_c

        for cl, m in res_c.cluster_metrics.items():
            per_regime_records.append({
                "model_name": rec["model_name"], "cluster": cl,
                "n_train": m["n_train"], "n_test": m["n_test"],
                "r2": m["r2"], "rmse": m["rmse"], "ubrmse": m["ubrmse"],
                "bias": m["bias"], "mae": m["mae"], "pearson": m["pearson"],
            })

    # LSTM-only baseline
    lstm_test = lstm_metrics["test"]
    lstm_record = {
        "model_name": "BiLSTM+Attn (LSTM-only, V21)",
        "strategy_name": "LSTM-only",
        "candidate_id": "LSTM-only",
        "pooled_r2": lstm_test["r2"], "pooled_rmse": lstm_test["rmse"],
        "pooled_ubrmse": lstm_test["ubrmse"], "pooled_bias": lstm_test["bias"],
        "pooled_mae": lstm_test["mae"], "pooled_pearson": float("nan"),
        "global_feature_count": len(ALL_FEATURES),
        "cluster_0_additions": "", "cluster_1_additions": "",
        "cluster_0_feature_count": 0, "cluster_1_feature_count": 0,
        "year_2023_r2": float("nan"), "year_2024_r2": float("nan"),
        "year_2025_r2": float("nan"), "train_time_s": float("nan"),
    }
    summary_records.append(lstm_record)
    all_metrics["BiLSTM+Attn (LSTM-only, V21)"] = lstm_record

    df_summary = pd.DataFrame(summary_records).sort_values("pooled_r2", ascending=False)
    df_summary.to_csv(ARTIFACTS_DIR / "summary_records.csv", index=False)
    df_regime = pd.DataFrame(per_regime_records)
    df_regime.to_csv(ARTIFACTS_DIR / "per_regime_records.csv", index=False)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save predictions for later analysis
    for mname, res in results_map.items():
        safe = mname.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "plus")
        np.save(ARTIFACTS_DIR / f"preds_{safe}.npy", res.predictions)

    # SHAP analysis
    shap_info = run_full_shap_analysis(evaluator_map, results_map, ARTIFACTS_DIR)

    print("\n" + "=" * 75, flush=True)
    print("FINAL MODEL LEADERBOARD (derived_8.0-optimization-2.0)", flush=True)
    print("=" * 75, flush=True)
    display_cols = ["model_name", "pooled_r2", "pooled_rmse", "pooled_ubrmse", "pooled_bias", "pooled_mae", "pooled_pearson"]
    print(df_summary[display_cols].to_string(index=False), flush=True)

    generate_readme(df_summary, df_regime, shap_info)


def generate_readme(df_summary: pd.DataFrame, df_regime: pd.DataFrame, shap_info: dict):
    df_top20 = shap_info["df_top20"]
    ctx_vs_tabular = shap_info["ctx_vs_tabular"]
    df_shap_summary = shap_info["df_summary"]

    combined_ranking = df_shap_summary.mean(axis=1).sort_values(ascending=False).head(10)
    top_features_table = pd.DataFrame({
        "Feature": combined_ranking.index,
        "Mean abs(SHAP)": combined_ranking.values.round(4),
    })

    ctx_tab_rows = []
    shap_results = shap_info["shap_results"]
    for model_name, info in ctx_vs_tabular.items():
        mean_abs = shap_results[model_name]["mean_abs_shap"]
        tab_mask = ~(mean_abs.index.str.startswith("ctx_") | mean_abs.index.str.startswith("hh_") | mean_abs.index.str.startswith("hp_"))
        lstm_mask = ~tab_mask
        tab_mean = mean_abs[tab_mask].mean()
        tab_median = mean_abs[tab_mask].median()
        lstm_mean = mean_abs[lstm_mask].mean() if lstm_mask.any() else 0.0
        lstm_median = mean_abs[lstm_mask].median() if lstm_mask.any() else 0.0
        ctx_tab_rows.append({
            "Model Name": model_name,
            "Tabular SHAP Sum": round(info["tabular_shap_sum"], 4),
            "Tabular Mean abs(SHAP)": round(tab_mean, 4),
            "Tabular Median abs(SHAP)": round(tab_median, 4),
            "Repr SHAP Sum": round(info["ctx_shap_sum"], 4),
            "Repr Mean abs(SHAP)": round(lstm_mean, 4),
            "Repr Median abs(SHAP)": round(lstm_median, 4),
            "Tabular % Share": f"{info['tabular_pct']:.2f}%",
            "Repr % Share": f"{info['ctx_pct']:.2f}%",
        })
    df_ctx_tab = pd.DataFrame(ctx_tab_rows)

    bench_rows = []
    for model_name, res in shap_info["shap_results"].items():
        b = res["benchmark"]
        bench_rows.append({
            "Model Name": model_name,
            "pred_contribs (s)": round(b["xgboost_pred_contribs_time_s"], 4),
        })
    df_bench = pd.DataFrame(bench_rows)

    n_xgb = len(shap_results)

    readme_content = f"""# Experiment: `derived_8.0-optimization-2.0` — V21 LSTM + PCA + XGBoost with Accelerated SHAP

## Objective
Re-run of `derived_8.4-hybrid-lstm-1.5` on the **`derived_8.0`** dataset split (5 Washington
stations) with the exact same feature sets (54-feature shared backbone, 58-feature V21
LSTM input, V0 router features, and the c1 additions from `derived_8.4-eval-1.1`).
The architecture uses **V21 BiLSTM+Attn** (Jakob 38 + V9-unique union, seq_len=30,
1-seed training). After training, PCA is applied to all three frozen
representations (160-dim ctx, 80-dim head_hidden, 80-dim pre-ReLU) at four compression levels:
- **95% variance** explained (auto component count)
- **64 components** fixed
- **32 components** fixed
- **16 components** fixed

All 15 representation variants (3 raw + 12 PCA) are compared against pure tabular baselines and the V21 LSTM-only baseline.

---

## Overall Leaderboard (2023–2025 Test Set)

{df_summary[["model_name", "pooled_r2", "pooled_rmse", "pooled_ubrmse", "pooled_bias", "pooled_mae", "pooled_pearson"]].to_markdown(index=False)}

---

## Per-Regime Performance Breakdown

{df_regime.to_markdown(index=False)}

---

## Year-by-Year $R^2$ Breakdown

{df_summary[["model_name", "pooled_r2", "year_2023_r2", "year_2024_r2", "year_2025_r2"]].to_markdown(index=False)}

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

{df_bench.to_markdown(index=False)}

### Feature Category Contribution (Tabular vs. LSTM Representations)

{df_ctx_tab.to_markdown(index=False)}

### Top 10 Features by Mean Absolute SHAP Value (averaged across all {n_xgb} XGBoost models)

{top_features_table.to_markdown(index=False)}

---

## Key Insights & Architecture Summary
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.0` (Jakob 38 + V9-unique features, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen representations extracted:
  - `ctx` (160-dim): Attention-pooled hidden state
  - `head_hidden` (80-dim): After head `Linear(160→80)→ReLU`
  - `head_pre_relu` (80-dim): After head `Linear(160→80)` BEFORE ReLU
- **Phase 3**: PCA reduces each representation at 3 levels (95% var, 64 comps, 32 comps).
- **Phase 4**: XGBoost fit on `[Tabular + Repr]` for all 12 representation variants × 2 strategies (Global + Clustering).
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
"""

    with open(EXP_DIR / "README.md", "w") as f:
        f.write(readme_content)
    print(f"\n[Generated] {EXP_DIR / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
