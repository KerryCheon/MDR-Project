#!/usr/bin/env python3
"""derived_8.4-hybrid-lstm-1.6: V21 BiLSTM+Attn hidden-size sweep (no PCA) + XGBoost + SHAP.

Trains the V21 BiLSTM+Attn at hidden sizes H in {40, 20, 16, 8, 4} (1 seed each),
extracts raw (non-PCA) ctx / head_hidden / head_pre_relu representations, and fits
hybrid [tabular + repr] XGBoost models under Global and Clustering strategies.

The goal is to check whether a smaller hidden size yields a compact enough
representation that PCA (required at H=80 in derived_8.4-hybrid-lstm-1.5) is no
longer needed, i.e. the tabular features are no longer diluted.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EXP_DIR))

from lstm.train import (
    train_lstm_for_hidden, ALL_FEATURES, SEQ_LEN, HIDDEN_SIZES,
)
from eval_hybrid.data import load_hybrid_experiment_data
from eval_hybrid.evaluator import HybridStrategyEvaluator
from eval_hybrid.shap_analysis import run_full_shap_analysis

CONFIG_PATH = EXP_DIR / "config.yaml"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
MODELS_DIR = EXP_DIR / "models"

# Representation naming: dims depend on the hidden size H (ctx = 2H, head = H).
REPR_LABELS = {
    "ctx": "CTX",
    "hh": "Head Hidden",
    "hp": "Pre-ReLU",
}


def repr_dim(hidden_size: int, repr_type: str) -> int:
    if repr_type == "ctx":
        return 2 * hidden_size
    return hidden_size


def load_hidden_sizes(config: dict) -> list[int]:
    return [int(h) for h in config.get("hidden_sizes", HIDDEN_SIZES)]


def build_variants(hidden_sizes: list[int]) -> list[tuple[int, str]]:
    return [(h, r) for h in hidden_sizes for r in ("ctx", "hh", "hp")]


def _model_name(hidden_size: int, repr_type: str, strategy: str) -> str:
    d = repr_dim(hidden_size, repr_type)
    label = REPR_LABELS[repr_type]
    display = f"{d} {label} [H{hidden_size}]"
    if strategy == "Global_Single":
        return f"Global Single (54 Backbone + {display})"
    return f"Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + {display})"


def _candidate_id(hidden_size: int, repr_type: str, strategy: str) -> str:
    if strategy == "Global_Single":
        return f"Global_Single_h{hidden_size}_{repr_type}"
    return f"Clustering_V0_k2_h{hidden_size}_{repr_type}"


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


def load_reference_1_5() -> pd.DataFrame:
    """Load derived_8.4-hybrid-lstm-1.5's published leaderboard as reference rows.

    Returns a DataFrame with model_name prefixed with '[1.5] ' and a 'reference'
    column set to True. Returns an empty frame if 1.5's summary is unavailable.
    """
    ref_path = EXP_DIR.parent / "derived_8.4-hybrid-lstm-1.5" / "artifacts" / "summary_records.csv"
    if not ref_path.exists():
        print(f"[Ref] 1.5 summary not found at {ref_path}; skipping reference rows.", flush=True)
        return pd.DataFrame()
    df_ref = pd.read_csv(ref_path).copy()
    if "model_name" not in df_ref.columns:
        return pd.DataFrame()
    df_ref["model_name"] = "[1.5] " + df_ref["model_name"].astype(str)
    df_ref["reference"] = True
    keep = ["model_name", "pooled_r2", "pooled_rmse", "pooled_ubrmse", "pooled_bias",
            "pooled_mae", "pooled_pearson", "year_2023_r2", "year_2024_r2", "year_2025_r2",
            "global_feature_count", "reference"]
    df_ref = df_ref[[c for c in keep if c in df_ref.columns]]
    print(f"[Ref] Loaded {len(df_ref)} reference rows from derived_8.4-hybrid-lstm-1.5.", flush=True)
    return df_ref


def main():
    print("=" * 75, flush=True)
    print("Starting derived_8.4-hybrid-lstm-1.6 Evaluation (V21 hidden-size sweep, no PCA)", flush=True)
    print("=" * 75, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / config.get("data_dir", "data/splits/derived_8.4")
    hidden_sizes = load_hidden_sizes(config)
    print(f"[Sweep] Hidden sizes: {hidden_sizes}", flush=True)

    # --- Phase 1-3: train LSTM + extract raw reps per hidden size (skip if done) ---
    print("\n" + "=" * 75, flush=True)
    print("Phase 1-3: LSTM training + raw representation extraction (per hidden size)", flush=True)
    print("=" * 75, flush=True)

    lstm_metrics_map = {}
    for h in hidden_sizes:
        h_dir = ARTIFACTS_DIR / f"h{h}"
        raw_names = ("ctx", "head_hidden", "head_pre_relu")
        has_raw = all((h_dir / f"{n}_{s}.npy").exists()
                      for n in raw_names for s in ("train", "val", "test"))
        has_metrics = (h_dir / "lstm_metrics.json").exists()
        if has_raw and has_metrics:
            print(f"[H{h}] Found existing representations + metrics. Skipping training.", flush=True)
            with open(h_dir / "lstm_metrics.json") as f:
                lstm_metrics_map[h] = json.load(f)
            continue
        h_dir.mkdir(parents=True, exist_ok=True)
        _, lstm_metrics = train_lstm_for_hidden(h, data_dir, h_dir, MODELS_DIR)
        lstm_metrics_map[h] = lstm_metrics

    c1_additions = compute_c1_gain_additions(config)
    print(f"[Cluster 1 Deltas] {c1_additions}", flush=True)

    summary_records = []
    per_regime_records = []
    all_metrics = {}
    results_map = {}
    evaluator_map = {}

    # --- Tabular baselines (no LSTM features) ---
    print("\n" + "=" * 75, flush=True)
    print("Training baselines (tabular only)", flush=True)
    print("=" * 75, flush=True)

    data_base = load_hybrid_experiment_data(PROJECT_ROOT, EXP_DIR, config, repr_type="ctx", hidden_size=hidden_sizes[0])
    backbone_54 = data_base.shared_backbone_54
    print(f"[Data] TrainVal={len(data_base.trainval)}, Test={len(data_base.test)}, Backbone={len(backbone_54)}", flush=True)

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

    # --- Hybrid models: (hidden_size, representation) x 2 strategies, raw only ---
    print("\n" + "=" * 75, flush=True)
    print("Training hybrid models (raw reps, no PCA)", flush=True)
    print("=" * 75, flush=True)

    for h, repr_type in build_variants(hidden_sizes):
        d = repr_dim(h, repr_type)
        data_variant = load_hybrid_experiment_data(
            PROJECT_ROOT, EXP_DIR, config, repr_type=repr_type, hidden_size=h,
        )
        hybrid_feats = data_variant.hybrid_features
        repr_count = data_variant.repr_feature_count
        print(f"[H{h} {REPR_LABELS[repr_type]}] {repr_count} repr ({d} dim) + {len(backbone_54)} backbone = {len(hybrid_feats)} tot", flush=True)

        eval_g = HybridStrategyEvaluator(data_variant, config, "Global_Single", models_dir=MODELS_DIR)
        eval_c = HybridStrategyEvaluator(data_variant, config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR)

        mname_g = _model_name(h, repr_type, "Global_Single")
        cid_g = _candidate_id(h, repr_type, "Global_Single")
        res_g = eval_g.fit_and_evaluate(model_name=mname_g, candidate_id=cid_g, global_features=hybrid_feats)
        rec = res_g.as_record()
        summary_records.append(rec)
        all_metrics[rec["model_name"]] = rec
        results_map[rec["model_name"]] = res_g
        evaluator_map[rec["model_name"]] = eval_g

        mname_c = _model_name(h, repr_type, "Clustering_V0_Full_k2")
        cid_c = _candidate_id(h, repr_type, "Clustering_V0_Full_k2")
        res_c = eval_c.fit_and_evaluate(
            model_name=mname_c, candidate_id=cid_c, global_features=hybrid_feats,
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

    # --- LSTM-only baselines (one per hidden size) ---
    for h in hidden_sizes:
        lstm_test = lstm_metrics_map[h]["test"]
        lstm_record = {
            "model_name": f"BiLSTM+Attn H{h} (LSTM-only, V21)",
            "strategy_name": "LSTM-only",
            "candidate_id": f"LSTM-only-h{h}",
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
        all_metrics[lstm_record["model_name"]] = lstm_record

    df_summary = pd.DataFrame(summary_records)

    # --- 1.5 reference rows (H80 +/- PCA) for direct comparison ---
    df_ref = load_reference_1_5()
    if not df_ref.empty:
        df_summary = pd.concat([df_summary, df_ref], ignore_index=True)

    df_summary = df_summary.sort_values("pooled_r2", ascending=False).reset_index(drop=True)
    df_summary.to_csv(ARTIFACTS_DIR / "summary_records.csv", index=False)
    df_regime = pd.DataFrame(per_regime_records)
    df_regime.to_csv(ARTIFACTS_DIR / "per_regime_records.csv", index=False)
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save predictions for later analysis
    for mname, res in results_map.items():
        safe = mname.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("+", "plus").replace("[", "").replace("]", "")
        np.save(ARTIFACTS_DIR / f"preds_{safe}.npy", res.predictions)

    # --- SHAP analysis ---
    shap_info = run_full_shap_analysis(evaluator_map, results_map, ARTIFACTS_DIR)

    print("\n" + "=" * 75, flush=True)
    print("FINAL MODEL LEADERBOARD (derived_8.4-hybrid-lstm-1.6)", flush=True)
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
    n_ref = int(df_summary["reference"].sum()) if "reference" in df_summary.columns else 0

    readme_content = f"""# Experiment: `derived_8.4-hybrid-lstm-1.6` — V21 BiLSTM+Attn Hidden-Size Sweep (no PCA)

## Objective
In `derived_8.4-hybrid-lstm-1.5` (H=80), the raw representations (160-dim ctx,
80-dim head_hidden, 80-dim pre-ReLU) diluted the 54 tabular features in the hybrid
XGBoost models, and PCA was required to restore the tabular share. This experiment
tests whether a **smaller LSTM hidden size** produces a compact raw representation
that no longer needs PCA. We sweep `hidden_size ∈ {{40, 20, 16, 8, 4}}` (1 seed each,
V21 BiLSTM+Attn, seq_len=30) and evaluate the hybrid models on **raw (non-PCA)**
representations. Reference rows from `derived_8.4-hybrid-lstm-1.5` (H=80 ± PCA) are
included in the leaderboard (marked `[1.5]`) for direct comparison.

Per hidden size H the representations are:
- `ctx` (2H-dim): attention-pooled bidirectional hidden state
- `head_hidden` (H-dim): after head `Linear(2H→H)→ReLU`
- `head_pre_relu` (H-dim): after head `Linear(2H→H)` BEFORE ReLU

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
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.4` for each hidden size
  `H ∈ {{40, 20, 16, 8, 4}}` (Jakob 38 + V9-unique features, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen raw representations extracted per hidden size:
  `ctx` (2H-dim), `head_hidden` (H-dim), `head_pre_relu` (H-dim).
- **Phase 3**: NO PCA — raw representations are used directly in the hybrid XGBoost models.
- **Phase 4**: XGBoost fit on `[Tabular + Repr]` for all 15 representation variants (5 hidden sizes × ctx/hh/hp) × 2 strategies (Global + Clustering) + 2 tabular baselines.
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
- Reference rows from `derived_8.4-hybrid-lstm-1.5` (H=80 ± PCA) are marked `[1.5]` ({n_ref} rows).
"""

    with open(EXP_DIR / "README.md", "w") as f:
        f.write(readme_content)
    print(f"\n[Generated] {EXP_DIR / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
