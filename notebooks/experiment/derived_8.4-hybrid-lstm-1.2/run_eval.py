#!/usr/bin/env python3
"""Main execution script for derived_8.4-hybrid-lstm-1.2 evaluation with SHAP analysis."""

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

from lstm.train import train_lstm_and_extract_ctx
from eval_hybrid.data import load_hybrid_experiment_data
from eval_hybrid.evaluator import HybridStrategyEvaluator
from eval_hybrid.shap_analysis import run_full_shap_analysis

CONFIG_PATH = EXP_DIR / "config.yaml"
ARTIFACTS_DIR = EXP_DIR / "artifacts"
MODELS_DIR = EXP_DIR / "models"


def compute_c1_gain_additions(config: dict | None = None) -> list[str]:
    """Extract top-10 delta additions for cluster 1 matching derived_8.4-eval-1.1 winning configuration."""
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
        "V_rollmin_F_NDMI_kobs30",
        "V_rollmean_G_API_kobs14",
        "lia_mean_asc_deg",
        "J_bio_bio04",
        "DOY",
        "SMAP_sm_am_interp",
        "J_bio_bio07",
        "SMAP_sm_am_interp_lag1",
        "C_lag_F_NDMI_kobs30",
        "SMAP_sm_pm_interp_lag1",
    ]


def main():
    print("=" * 75, flush=True)
    print("Starting derived_8.4-hybrid-lstm-1.2 Evaluation", flush=True)
    print("=" * 75, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1 & 2: Ensure CTX + Head Hidden vectors are extracted
    data_dir = PROJECT_ROOT / config.get("data_dir", "data/splits/derived_8.4")
    if not (ARTIFACTS_DIR / "ctx_test.npy").exists() or not (ARTIFACTS_DIR / "head_hidden_test.npy").exists():
        print("[LSTM] Extracting CTX + Head Hidden representations from BiLSTM+Attn...", flush=True)
        train_lstm_and_extract_ctx(data_dir, ARTIFACTS_DIR)
    else:
        print("[LSTM] Found existing extracted CTX + Head Hidden representations in artifacts.", flush=True)

    # Step 3: Load Hybrid Experiment Data (54 Backbone + 160 CTX + 80 CTX-head)
    data = load_hybrid_experiment_data(PROJECT_ROOT, EXP_DIR, config)
    print(f"[Data] TrainVal={len(data.trainval)} samples, Test={len(data.test)} samples.", flush=True)
    print(f"[Features] 54 Backbone + 160 CTX = {len(data.hybrid_backbone_214)} | + 80 CTX-head = {len(data.hybrid_backbone_134)}", flush=True)

    c1_additions = compute_c1_gain_additions(config)
    print(f"[Cluster 1 Deltas] Selected 10 Additions: {c1_additions}", flush=True)

    summary_records = []
    per_regime_records = []
    all_metrics = {}

    # 1. Global Single Baseline (54 Backbone)
    eval_global = HybridStrategyEvaluator(data, config, "Global_Single", models_dir=MODELS_DIR)
    res_g_base = eval_global.fit_and_evaluate(
        model_name="Global Single Model (54 Backbone)",
        candidate_id="Global_Single_54_Backbone",
        global_features=data.shared_backbone_54,
    )
    rec = res_g_base.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec

    # 2. Clustering_V0_Full_k2 Baseline (Winner c0=0, c1=10)
    eval_v0 = HybridStrategyEvaluator(data, config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR)
    res_v0_base = eval_v0.fit_and_evaluate(
        model_name="Clustering_V0_Full_k2 (Winner c0=0, c1=10)",
        candidate_id="Clustering_V0_Full_k2_c0_0_c1_10",
        global_features=data.shared_backbone_54,
        cluster_additions={"0": [], "1": c1_additions},
    )
    rec = res_v0_base.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec

    for cl, m in res_v0_base.cluster_metrics.items():
        per_regime_records.append({
            "model_name": rec["model_name"],
            "cluster": cl,
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "r2": m["r2"],
            "rmse": m["rmse"],
            "ubrmse": m["ubrmse"],
            "bias": m["bias"],
            "mae": m["mae"],
            "pearson": m["pearson"],
        })

    # 3. Global Single Hybrid (54 Backbone + 160 CTX)
    res_g_hybrid = eval_global.fit_and_evaluate(
        model_name="Global Single Model (54 Backbone + 160 CTX)",
        candidate_id="Global_Single_54_Backbone_160_CTX",
        global_features=data.hybrid_backbone_214,
    )
    rec = res_g_hybrid.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec

    # 4. Clustering_V0_Full_k2 Hybrid (Winner c0=0, c1=10 + 160 CTX)
    res_v0_hybrid = eval_v0.fit_and_evaluate(
        model_name="Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)",
        candidate_id="Clustering_V0_Full_k2_c0_0_c1_10_160_CTX",
        global_features=data.hybrid_backbone_214,
        cluster_additions={"0": [], "1": c1_additions},
    )
    rec = res_v0_hybrid.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec

    for cl, m in res_v0_hybrid.cluster_metrics.items():
        per_regime_records.append({
            "model_name": rec["model_name"],
            "cluster": cl,
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "r2": m["r2"],
            "rmse": m["rmse"],
            "ubrmse": m["ubrmse"],
            "bias": m["bias"],
            "mae": m["mae"],
            "pearson": m["pearson"],
        })

    # 5. Global Single Hybrid (54 Backbone + 80 CTX-head)
    res_g_hybrid_80 = eval_global.fit_and_evaluate(
        model_name="Global Single Model (54 Backbone + 80 CTX-head)",
        candidate_id="Global_Single_54_Backbone_80_CTXhead",
        global_features=data.hybrid_backbone_134,
    )
    rec = res_g_hybrid_80.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec

    # 6. Clustering_V0_Full_k2 Hybrid (Winner c0=0, c1=10 + 80 CTX-head)
    res_v0_hybrid_80 = eval_v0.fit_and_evaluate(
        model_name="Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)",
        candidate_id="Clustering_V0_Full_k2_c0_0_c1_10_80_CTXhead",
        global_features=data.hybrid_backbone_134,
        cluster_additions={"0": [], "1": c1_additions},
    )
    rec = res_v0_hybrid_80.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec

    for cl, m in res_v0_hybrid_80.cluster_metrics.items():
        per_regime_records.append({
            "model_name": rec["model_name"],
            "cluster": cl,
            "n_train": m["n_train"],
            "n_test": m["n_test"],
            "r2": m["r2"],
            "rmse": m["rmse"],
            "ubrmse": m["ubrmse"],
            "bias": m["bias"],
            "mae": m["mae"],
            "pearson": m["pearson"],
        })

    df_summary = pd.DataFrame(summary_records).sort_values("pooled_r2", ascending=False)
    df_summary.to_csv(ARTIFACTS_DIR / "summary_records.csv", index=False)

    df_regime = pd.DataFrame(per_regime_records)
    df_regime.to_csv(ARTIFACTS_DIR / "per_regime_records.csv", index=False)

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # Save test predictions for evaluation
    np.save(ARTIFACTS_DIR / "preds_global_single_baseline.npy", res_g_base.predictions)
    np.save(ARTIFACTS_DIR / "preds_clustering_v0_baseline.npy", res_v0_base.predictions)
    np.save(ARTIFACTS_DIR / "preds_global_single_hybrid.npy", res_g_hybrid.predictions)
    np.save(ARTIFACTS_DIR / "preds_clustering_v0_hybrid.npy", res_v0_hybrid.predictions)
    np.save(ARTIFACTS_DIR / "preds_global_single_hybrid_80.npy", res_g_hybrid_80.predictions)
    np.save(ARTIFACTS_DIR / "preds_clustering_v0_hybrid_80.npy", res_v0_hybrid_80.predictions)

    results_map = {
        "Global Single Model (54 Backbone)": res_g_base,
        "Clustering_V0_Full_k2 (Winner c0=0, c1=10)": res_v0_base,
        "Global Single Model (54 Backbone + 160 CTX)": res_g_hybrid,
        "Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)": res_v0_hybrid,
        "Global Single Model (54 Backbone + 80 CTX-head)": res_g_hybrid_80,
        "Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)": res_v0_hybrid_80,
    }

    # Step 4: Accelerated SHAP Feature Importance Analysis
    shap_info = run_full_shap_analysis(eval_global, eval_v0, results_map, ARTIFACTS_DIR)

    print("\n" + "=" * 75, flush=True)
    print("FINAL MODEL LEADERBOARD (derived_8.4-hybrid-lstm-1.2)", flush=True)
    print("=" * 75, flush=True)
    print(df_summary[["model_name", "pooled_r2", "pooled_rmse", "pooled_ubrmse", "pooled_bias", "pooled_mae", "pooled_pearson"]].to_string(index=False), flush=True)

    generate_readme(df_summary, df_regime, shap_info)


def generate_readme(df_summary: pd.DataFrame, df_regime: pd.DataFrame, shap_info: dict):
    df_top20 = shap_info["df_top20"]
    ctx_vs_tabular = shap_info["ctx_vs_tabular"]

    # Build CTX vs Tabular summary markdown table
    ctx_tab_rows = []
    for model_name, info in ctx_vs_tabular.items():
        ctx_tab_rows.append({
            "Model Name": model_name,
            "Tabular SHAP Sum": round(info["tabular_shap_sum"], 4),
            "CTX SHAP Sum": round(info["ctx_shap_sum"], 4),
            "Tabular % Share": f"{info['tabular_pct']:.2f}%",
            "CTX % Share": f"{info['ctx_pct']:.2f}%",
        })
    df_ctx_tab = pd.DataFrame(ctx_tab_rows)

    # Build SHAP timing table (pred_contribs only)
    bench_rows = []
    for model_name, res in shap_info["shap_results"].items():
        b = res["benchmark"]
        bench_rows.append({
            "Model Name": model_name,
            "pred_contribs (s)": round(b["xgboost_pred_contribs_time_s"], 4),
        })
    df_bench = pd.DataFrame(bench_rows)

    readme_content = f"""# Experiment: `derived_8.4-hybrid-lstm-1.2` — Hybrid LSTM Context Vector (`ctx` + `head_hidden`) + XGBoost with Accelerated SHAP

## Objective
Evaluate whether concatenating frozen temporal context vectors extracted from a converged **BiLSTM+Attn (v9)** model with tabular XGBoost features improves prediction performance on the `derived_8.4` test set (7 Washington stations). This experiment compares two CTX representations:
- **160-dim `ctx`**: Full attention-pooled hidden state (BiLSTM output)
- **80-dim `head_hidden`**: Intermediate representation after head `Linear(160→80)→ReLU`

Hypothesis: d=160 may be too large, diluting CTX feature importance. The 80-dim bottleneck may provide a more compressed, higher-signal representation.

Models evaluated:
1. **Global Single Model (54 Backbone)** — Pure tabular baseline
2. **Clustering_V0_Full_k2 (Winner c0=0, c1=10)** — Pure tabular MoE baseline
3. **Global Single Model (54 Backbone + 160 CTX)** — Hybrid global (214 features)
4. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 160 CTX)** — Hybrid MoE (214/224 features)
5. **Global Single Model (54 Backbone + 80 CTX-head)** — Hybrid global (134 features)
6. **Clustering_V0_Full_k2 (Winner c0=0, c1=10 + 80 CTX-head)** — Hybrid MoE (134/144 features)

---

## Overall Leaderboard (2023–2025 Test Set)

Evaluated on CUDA on the `derived_8.4` test set (6,620 samples across 7 WA stations):

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

### Feature Category Contribution (Tabular vs. LSTM Context Vectors `ctx` / `head_hidden`)

{df_ctx_tab.to_markdown(index=False)}

### Top 10 Features by Mean Absolute SHAP Value

```
{df_top20.head(10).to_string(index=False)}
```

---

## Key Insights & Architecture Summary
- **Phase 1**: BiLSTM+Attn model trained until validation convergence.
- **Phase 2**: Frozen representations extracted across `train`, `val`, and `test` splits:
  - `ctx` (160-dim): Attention-pooled hidden state
  - `head_hidden` (80-dim): Intermediate after head `Linear(160→80)→ReLU`
- **Phase 3**: XGBoost fit on `[Tabular + CTX]` and `[Tabular + head_hidden]` features.
- **Phase 4**: SHAP feature importance computed efficiently using XGBoost native `pred_contribs=True` with multi-threading / CUDA acceleration.
"""

    with open(EXP_DIR / "README.md", "w") as f:
        f.write(readme_content)
    print(f"\n[Generated] {EXP_DIR / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
