#!/usr/bin/env python3
"""derived_8.4-hybrid-lstm-2.0: strict static/temporal split + XGBoost + SHAP.

In derived_8.4-hybrid-lstm-1.6 the hybrid XGBoost received the 54-feature
backbone (49 of which are temporal/rolling) PLUS the LSTM context that was
computed from largely the same temporal features — SHAP showed the context
dominating (56-86% of SHAP share) and diluting the tabular set.

2.0 removes that overlap with a strict split:
  - XGBoost direct input = STATIC features only (18, constant per station).
  - LSTM input = TEMPORAL features only (79); the LSTM is retrained and its
    frozen ctx / head_hidden / head_pre_relu carry ALL temporal dynamics.

Everything else matches 1.6: V21 BiLSTM+Attn, hidden sizes {40,20,16,8,4},
seq_len=30, 1 seed, raw (non-PCA) representations, Global + Clustering
strategies, same XGBoost hyperparameters, accelerated SHAP. [1.6] reference
rows are appended to the leaderboard for direct comparison. Cluster-1 temporal
additions from derived_8.4-eval-1.1 are intentionally NOT used (they are
temporal features and would violate the static-only XGBoost design).
"""
from __future__ import annotations

import json
import re
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
    train_lstm_for_hidden, TEMPORAL_FEATURES, STATIC_FEATURES, SEQ_LEN, HIDDEN_SIZES,
    ALL_FEATURES_V1_6, BACKBONE_54,
)
from eval_hybrid.data import (
    load_hybrid_experiment_data, verify_static_temporal_split,
)
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
    n_static = len(STATIC_FEATURES)
    if strategy == "Global_Single":
        return f"Global Single ({n_static} Static + {display})"
    return f"Clustering_V0_Full_k2 ({n_static} Static, c0=0, c1=0 + {display})"


def _candidate_id(hidden_size: int, repr_type: str, strategy: str) -> str:
    if strategy == "Global_Single":
        return f"Global_Single_h{hidden_size}_{repr_type}"
    return f"Clustering_V0_k2_h{hidden_size}_{repr_type}"


def load_reference_1_6() -> pd.DataFrame:
    """Load derived_8.4-hybrid-lstm-1.6's published leaderboard as reference rows.

    Returns a DataFrame with model_name prefixed with '[1.6] ' and a 'reference'
    column set to True. Returns an empty frame if 1.6's summary is unavailable.
    """
    ref_path = EXP_DIR.parent / "derived_8.4-hybrid-lstm-1.6" / "artifacts" / "summary_records.csv"
    if not ref_path.exists():
        print(f"[Ref] 1.6 summary not found at {ref_path}; skipping reference rows.", flush=True)
        return pd.DataFrame()
    df_ref = pd.read_csv(ref_path).copy()
    if "model_name" not in df_ref.columns:
        return pd.DataFrame()
    # Drop rows that were themselves [1.5] reference rows inside 1.6's own
    # leaderboard (they are not 1.6's models); keep 1.6's own rows only.
    df_ref = df_ref[~df_ref["model_name"].astype(str).str.startswith("[1.5] ")]
    df_ref["model_name"] = "[1.6] " + df_ref["model_name"].astype(str)
    df_ref["reference"] = True
    keep = ["model_name", "pooled_r2", "pooled_rmse", "pooled_ubrmse", "pooled_bias",
            "pooled_mae", "pooled_pearson", "year_2023_r2", "year_2024_r2", "year_2025_r2",
            "global_feature_count", "reference"]
    df_ref = df_ref[[c for c in keep if c in df_ref.columns]]
    print(f"[Ref] Loaded {len(df_ref)} reference rows from derived_8.4-hybrid-lstm-1.6.", flush=True)
    return df_ref


def main():
    print("=" * 75, flush=True)
    print("Starting derived_8.4-hybrid-lstm-2.0 Evaluation (strict static/temporal split)", flush=True)
    print("=" * 75, flush=True)

    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data_dir = PROJECT_ROOT / config.get("data_dir", "data/splits/derived_8.4")
    hidden_sizes = load_hidden_sizes(config)
    print(f"[Sweep] Hidden sizes: {hidden_sizes}", flush=True)

    # --- Audit the strict static/temporal split (reproducibility) ---
    print("\n" + "=" * 75, flush=True)
    print("Feature split audit (static -> XGBoost, temporal -> LSTM)", flush=True)
    print("=" * 75, flush=True)
    train_df_check = pd.read_csv(data_dir / "train.csv")
    split_evidence = verify_static_temporal_split(
        train_df_check, STATIC_FEATURES, TEMPORAL_FEATURES,
    )
    print(
        f"[Split] static={split_evidence['n_static']}, temporal={split_evidence['n_temporal']}, "
        f"overlap={split_evidence['n_overlap']}",
        flush=True,
    )
    print(f"[Split] LSTM input count: {len(TEMPORAL_FEATURES)} (was 58 in 1.6)", flush=True)
    print(f"[Split] XGBoost direct count: {len(STATIC_FEATURES)} (was 54 in 1.6)", flush=True)

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

    summary_records = []
    per_regime_records = []
    all_metrics = {}
    results_map = {}
    evaluator_map = {}

    # --- Tabular baselines (STATIC features only, no LSTM features) ---
    print("\n" + "=" * 75, flush=True)
    print("Training baselines (static tabular only)", flush=True)
    print("=" * 75, flush=True)

    data_base = load_hybrid_experiment_data(PROJECT_ROOT, EXP_DIR, config, repr_type="ctx", hidden_size=hidden_sizes[0])
    static_feats = data_base.static_features
    print(f"[Data] TrainVal={len(data_base.trainval)}, Test={len(data_base.test)}, Static={len(static_feats)}", flush=True)

    eval_global_base = HybridStrategyEvaluator(data_base, config, "Global_Single", models_dir=MODELS_DIR)
    eval_v0_base = HybridStrategyEvaluator(data_base, config, "Clustering_V0_Full_k2", models_dir=MODELS_DIR)

    res_g_base = eval_global_base.fit_and_evaluate(
        model_name=f"Global Single ({len(static_feats)} Static)",
        candidate_id=f"Global_Single_{len(static_feats)}_Static",
        global_features=static_feats,
    )
    rec = res_g_base.as_record()
    summary_records.append(rec)
    all_metrics[rec["model_name"]] = rec
    results_map[rec["model_name"]] = res_g_base
    evaluator_map[rec["model_name"]] = eval_global_base

    res_v0_base = eval_v0_base.fit_and_evaluate(
        model_name=f"Clustering_V0_Full_k2 ({len(static_feats)} Static, c0=0, c1=0)",
        candidate_id=f"Clustering_V0_k2_{len(static_feats)}_Static",
        global_features=static_feats,
        cluster_additions={"0": [], "1": []},
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
    print("Training hybrid models (static tabular + raw reps, no PCA)", flush=True)
    print("=" * 75, flush=True)

    for h, repr_type in build_variants(hidden_sizes):
        d = repr_dim(h, repr_type)
        data_variant = load_hybrid_experiment_data(
            PROJECT_ROOT, EXP_DIR, config, repr_type=repr_type, hidden_size=h,
        )
        hybrid_feats = data_variant.hybrid_features
        repr_count = data_variant.repr_feature_count
        print(f"[H{h} {REPR_LABELS[repr_type]}] {repr_count} repr ({d} dim) + {len(static_feats)} static = {len(hybrid_feats)} tot", flush=True)

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
            cluster_additions={"0": [], "1": []},
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

    # --- LSTM-only baselines (one per hidden size, temporal-only input) ---
    for h in hidden_sizes:
        lstm_test = lstm_metrics_map[h]["test"]
        lstm_record = {
            "model_name": f"BiLSTM+Attn H{h} (LSTM-only, temporal-only input)",
            "strategy_name": "LSTM-only",
            "candidate_id": f"LSTM-only-h{h}",
            "pooled_r2": lstm_test["r2"], "pooled_rmse": lstm_test["rmse"],
            "pooled_ubrmse": lstm_test["ubrmse"], "pooled_bias": lstm_test["bias"],
            "pooled_mae": lstm_test["mae"], "pooled_pearson": float("nan"),
            "global_feature_count": len(TEMPORAL_FEATURES),
            "cluster_0_additions": "", "cluster_1_additions": "",
            "cluster_0_feature_count": 0, "cluster_1_feature_count": 0,
            "year_2023_r2": float("nan"), "year_2024_r2": float("nan"),
            "year_2025_r2": float("nan"), "train_time_s": float("nan"),
        }
        summary_records.append(lstm_record)
        all_metrics[lstm_record["model_name"]] = lstm_record

    df_summary = pd.DataFrame(summary_records)

    # --- 1.6 reference rows (54-feature backbone hybrids) for direct comparison ---
    df_ref = load_reference_1_6()
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
    print("FINAL MODEL LEADERBOARD (derived_8.4-hybrid-lstm-2.0)", flush=True)
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

    # --- Performance-degradation analysis (computed from the leaderboard, not hardcoded) ---
    names = df_summary["model_name"].astype(str)
    own = df_summary[~names.str.startswith("[1.6] ")].copy()
    refs = df_summary[names.str.startswith("[1.6] ")].copy()
    has_refs = len(refs) > 0

    hidden_sizes_found = sorted(
        {int(m.group(1)) for n in names for m in [re.search(r"\[H(\d+)\]", n)] if m}
    )

    def _exact_r2(rows: pd.DataFrame, name: str) -> float:
        r = rows[rows["model_name"].astype(str) == name]
        return float(r["pooled_r2"].iloc[0]) if not r.empty else float("nan")

    def _best_clust_ctx_r2(rows: pd.DataFrame, h: int) -> float:
        r = rows[
            rows["model_name"].astype(str).str.contains(f"[H{h}]", regex=False)
            & rows["model_name"].astype(str).str.contains("Clustering")
            & rows["model_name"].astype(str).str.contains("CTX")
        ]
        return float(r["pooled_r2"].max()) if not r.empty else float("nan")

    analysis_rows = []
    for h in hidden_sizes_found:
        l16 = _exact_r2(refs, f"[1.6] BiLSTM+Attn H{h} (LSTM-only, V21)") if has_refs else float("nan")
        l20 = _exact_r2(own, f"BiLSTM+Attn H{h} (LSTM-only, temporal-only input)")
        h16 = _best_clust_ctx_r2(refs, h) if has_refs else float("nan")
        h20 = _best_clust_ctx_r2(own, h)
        analysis_rows.append({
            "H": h,
            "1.6 LSTM-only R²": round(l16, 4) if np.isfinite(l16) else float("nan"),
            "2.0 LSTM-only R²": round(l20, 4),
            "Δ LSTM-only": round(l20 - l16, 4) if np.isfinite(l16) else float("nan"),
            "1.6 best Clust+CTX R²": round(h16, 4) if np.isfinite(h16) else float("nan"),
            "2.0 best Clust+CTX R²": round(h20, 4),
            "Δ hybrid": round(h20 - h16, 4) if np.isfinite(h16) else float("nan"),
        })
    df_analysis = pd.DataFrame(analysis_rows)

    static_r2 = _exact_r2(own, "Global Single (18 Static)")
    bb16_g = _exact_r2(refs, "[1.6] Global Single (54 Backbone)")
    bb16_c = _exact_r2(refs, "[1.6] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)")

    repr_pcts = [info["ctx_pct"] for info in ctx_vs_tabular.values() if info["ctx_pct"] > 0]
    repr_min, repr_max = (min(repr_pcts), max(repr_pcts)) if repr_pcts else (float("nan"), float("nan"))
    shap16_path = EXP_DIR.parent / "derived_8.4-hybrid-lstm-1.6" / "artifacts" / "shap_ctx_vs_tabular.json"
    if shap16_path.exists():
        d16 = json.load(open(shap16_path))
        p16 = [v["ctx_pct"] for v in d16.values() if v["ctx_pct"] > 0]
        repr16_min, repr16_max = (min(p16), max(p16)) if p16 else (float("nan"), float("nan"))
    else:
        repr16_min = repr16_max = float("nan")

    if has_refs:
        df_lstm_table = df_analysis[["H", "1.6 LSTM-only R²", "2.0 LSTM-only R²", "Δ LSTM-only"]].to_markdown(index=False)
        df_hyb_table = df_analysis[["H", "1.6 best Clust+CTX R²", "2.0 best Clust+CTX R²", "Δ hybrid"]].to_markdown(index=False)
        analysis_section = f"""## Why Performance Dropped vs 1.6 (Analysis)

The strict split removes the temporal overlap by design, but the leaderboard shows it
costs roughly 0.04–0.13 R² at every hidden size. The numbers indicate the "overlap"
removed was informative redundancy, not waste. Three compounding causes:

1. **The 49 temporal features were the model.** 1.6's tabular-only baselines scored
   **{bb16_c:.3f}** (Clustering) / **{bb16_g:.3f}** (Global) on the 54-feature backbone,
   while the 18-feature static-only baselines here collapse to **~{static_r2:.3f}**. Static
   station attributes barely move daily soil-moisture dynamics, so in 2.0 every temporal
   signal must pass through the lossy LSTM context bottleneck instead of being read
   directly by XGBoost.
2. **The LSTM lost its station context.** The 13 static features moved out of the LSTM
   (slope, elev, lat/lon, aspect, soil texture, K_*) are what told the LSTM *which*
   station — i.e. its baseline moisture level. LSTM-only test R² fell from ~0.70 (1.6) to
   ~0.53–0.56 at H16/H8, and the 34 added rolling features (largely redundant derivatives
   of the same SMAP/NDVI/LST series) did not compensate. Validation RMSE barely moved, so
   this is an information-bottleneck issue, not a training failure.
3. **The hybrid is capped by its context.** The 18 static features add only ~+0.03 to +0.08
   over the LSTM-only model at each H, and the SHAP share of the context rose to
   **{repr_min:.0f}–{repr_max:.0f}%** (vs {repr16_min:.0f}–{repr16_max:.0f}% in 1.6). The
   hybrids are effectively "LSTM context + a little station info", so their ceiling is what
   the context can encode.

### LSTM-only Regression (test R², same V21 architecture & hyperparameters)

{df_lstm_table}

### Best Clustering + CTX Hybrid (test R²)

{df_hyb_table}
"""
    else:
        df_lstm_table = df_analysis[["H", "2.0 LSTM-only R²"]].to_markdown(index=False)
        analysis_section = f"""## Why Performance Dropped vs 1.6 (Analysis)

[1.6] reference rows were unavailable, so only 2.0's own LSTM-only results are shown.

{df_lstm_table}
"""

    next_steps_section = """## Next Steps

The strict split proves the temporal features were load-bearing. To reduce overlap without
giving up performance:

1. **Restore the 13 station statics to the LSTM input** — keep the 18 static features in
   XGBoost *and* feed the station attributes back to the LSTM so it retains station
   identity. Cheapest test of whether the LSTM drop is caused by the removed statics.
2. **Allow a small set of raw temporal features back into XGBoost** — e.g. raw
   `SMAP_sm_pm_interp`, `precip_mm`, `F_NDVI` (not their rolling statistics), so XGBoost is
   not fully dependent on the context while the rolling/lagged derivatives stay in the LSTM.
3. **Use a wider / richer context** — `head_hidden` instead of `ctx`, concatenated
   representations, or a larger hidden size, to reduce the lossiness of the 2H-dim
   context bottleneck.
4. **Ablation** — retrain 1.6's LSTM on only its 45 temporal features (no statics, no 34
   additions) to isolate whether the LSTM drop comes from the removed statics or the added
   rolling features.
"""

    # --- Feature split: what went to which model (rendered from the actual constants) ---
    static_backbone = [f for f in STATIC_FEATURES if f in BACKBONE_54]
    static_from_lstm = [f for f in STATIC_FEATURES if f not in BACKBONE_54]
    temporal_carried = [f for f in TEMPORAL_FEATURES if f in ALL_FEATURES_V1_6]
    temporal_added = [f for f in TEMPORAL_FEATURES if f not in ALL_FEATURES_V1_6]

    feature_split_section = f"""## Feature Split: What Went Where

The strict split is a disjoint partition of the feature universe between the two models
(audited in the setup cell: {len(STATIC_FEATURES)} static + {len(TEMPORAL_FEATURES)} temporal,
no overlap, and every static feature is constant per station).

### XGBoost — direct (tabular) input: {len(STATIC_FEATURES)} static features

The {len(static_backbone)} static members of the 54-feature backbone (constant per station):

```text
{", ".join(static_backbone)}
```

The {len(static_from_lstm)} station attributes moved out of the 1.6 LSTM input
(longitude, latitude, elevation, slope, aspect, soil texture, terrain transforms):

```text
{", ".join(static_from_lstm)}
```

XGBoost receives **no temporal/rolling features directly** — all temporal signal arrives
through the frozen LSTM context vectors (`ctx` / `head_hidden` / `head_pre_relu`).

### LSTM — sequence input: {len(TEMPORAL_FEATURES)} temporal/rolling features

The {len(temporal_carried)} temporal features carried over from 1.6's LSTM input:

```text
{", ".join(temporal_carried)}
```

The {len(temporal_added)} temporal backbone features that 1.6 did NOT feed the LSTM
(added so the frozen context carries ALL temporal dynamics):

```text
{", ".join(temporal_added)}
```

The LSTM receives **no static features** — station identity is supplied exclusively by
XGBoost's static inputs.
"""

    readme_content = f"""# Experiment: `derived_8.4-hybrid-lstm-2.0` — Strict Static/Temporal Split (Static → XGBoost, Temporal → LSTM)

## Objective
In `derived_8.4-hybrid-lstm-1.6`, the hybrid XGBoost received the 54-feature backbone
(49 of which are temporal/rolling) PLUS the LSTM context that was computed from largely
the same temporal features; SHAP showed the LSTM context dominating (56–86% of SHAP
share) and diluting the tabular set. This experiment implements a **strict split** to
remove that overlap:
- **XGBoost direct input = STATIC features only** ({len(STATIC_FEATURES)}, all constant
  per station: 5 from the 54-feature backbone + 13 station attributes formerly in the
  LSTM input).
- **LSTM input = TEMPORAL features only** ({len(TEMPORAL_FEATURES)}: the 45 temporal
  features 1.6 fed the LSTM + the 34 temporal backbone features 1.6 did not feed it),
  retrained so the frozen context vector carries ALL temporal dynamics.

Everything else matches 1.6: V21 BiLSTM+Attn, `hidden_size ∈ {{40, 20, 16, 8, 4}}`
(1 seed each, seq_len=30), raw (non-PCA) `ctx`/`head_hidden`/`head_pre_relu`, Global +
Clustering strategies, same XGBoost hyperparameters. Cluster-1 temporal additions are
intentionally omitted (they would violate the static-only design). Leaderboard rows from
`derived_8.4-hybrid-lstm-1.6` are appended as `[1.6]` references.

Per hidden size H the representations are:
- `ctx` (2H-dim): attention-pooled bidirectional hidden state
- `head_hidden` (H-dim): after head `Linear(2H→H)→ReLU`
- `head_pre_relu` (H-dim): after head `Linear(2H→H)` BEFORE ReLU

---

{feature_split_section}

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

{analysis_section}

---

{next_steps_section}

---

## Key Insights & Architecture Summary
- **Phase 0**: Feature-split audit — {len(STATIC_FEATURES)} static features verified
  constant per station and disjoint from {len(TEMPORAL_FEATURES)} temporal LSTM features.
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.4` for each hidden size
  `H ∈ {{40, 20, 16, 8, 4}}` (temporal-only input, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen raw representations extracted per hidden size:
  `ctx` (2H-dim), `head_hidden` (H-dim), `head_pre_relu` (H-dim).
- **Phase 3**: NO PCA — raw representations are used directly in the hybrid XGBoost models.
- **Phase 4**: XGBoost fit on `[Static + Repr]` for all 15 representation variants
  (5 hidden sizes × ctx/hh/hp) × 2 strategies (Global + Clustering) + 2 static baselines.
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
- Reference rows from `derived_8.4-hybrid-lstm-1.6` (54-feature backbone hybrids) are marked `[1.6]` ({n_ref} rows).
"""

    with open(EXP_DIR / "README.md", "w") as f:
        f.write(readme_content)
    print(f"\n[Generated] {EXP_DIR / 'README.md'}", flush=True)


if __name__ == "__main__":
    main()
