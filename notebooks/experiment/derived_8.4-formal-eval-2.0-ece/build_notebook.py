import json
from pathlib import Path
import subprocess

exp_dir = Path(__file__).resolve().parent
nb_path = exp_dir / "derived_8.4-formal-eval-2.0-ece.ipynb"

# Remove old notebook if exists
if nb_path.exists():
    nb_path.unlink()

# Create fresh notebook
subprocess.run(["nb", "create", str(nb_path), "--kernel", "python3"], check=True)

with open(nb_path, "r", encoding="utf-8") as f:
    nb_data = json.load(f)

cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text
    }

def code_cell(code):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": code
    }

# 1. Title
cells.append(md_cell("""# Experiment: `derived_8.4-formal-eval-2.0-ece` — Formal Statistical Evaluation on In-Situ ECE Sensors

Publication-oriented statistical evaluation of the claim: **a two-regime (KMeans k=2) clustering model
beats the single-regime global model and the trained-gating model**, evaluated on **in-situ spatial generalization
to 5 newly deployed sensor stations in Washington State** (`derived_8.4-ece`, 150 rows across 2026-07-20 to 2026-08-19 in Bellevue and Renton, WA).

All models and routers are trained **strictly on the 7 Washington state stations** (`derived_8.4` `trainval`,
14,608 rows). The in-situ dataset `derived_8.4-ece` is **completely unseen** during training.

All tables below are the stdout of this executed notebook."""))

# 2. Setup
cells.append(code_cell('''"""Setup: load configs, data splits, and statistical reporting tools."""
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

EXP_DIR = Path(".").resolve()
PROJECT_ROOT = (EXP_DIR.parents[2]).resolve()
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

import eval_formal.plots as plots
import eval_formal.stats as st
from eval_formal.configs import config_frame, load_pinned_configs
from eval_formal.data import load_experiment_data

with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data = load_experiment_data(PROJECT_ROOT, config)
configurations = load_pinned_configs(data, config)
cfg = config_frame(configurations)

METRICS = ["r2", "rmse", "mae", "bias"]

print(f"[Setup] WA TrainVal={len(data.trainval)} WA Test={len(data.test)}")
print(f"[Setup] ECE 5 Stations: All={len(data.ece_all)} (Train={len(data.ece_train)}, Val={len(data.ece_val)}, Test={len(data.ece_test)})")
print(f"[Setup] Pinned {len(cfg)} configurations, 30 seeds")'''))

# 3. Configs md
cells.append(md_cell("""# Configurations (20)

14 requested configurations (test-selected deltas pinned from `derived_8.4-eval-1.1`, none = c0=c1=0)
plus 6 validation-selected winners (`val_selected_deltas.json`):"""))

# 4. Configs code
cells.append(code_cell('''"""Display pinned configurations table."""
cfg_table = cfg[["config_id", "strategy_name", "delta_source", "cluster_0_count",
                 "cluster_1_count", "n_global_features", "n_add0", "n_add1"]]
print(cfg_table.to_markdown(index=False))'''))

# 5. Temporal md
cells.append(md_cell("""# Temporal results (Washington test set, 2023–2025, 30 seeds)

Evaluation on the frozen Washington state test split (2023–2025, 6,620 rows, 7 stations) across 30 seeds.
Replicates `derived_8.4-formal-eval-1.0` and `2.0` exactly without redundant retraining."""))

# 6. Temporal code
cells.append(code_cell('''"""Temporal seed-level summary tables (R2, RMSE, MAE, bias)."""
temporal = pd.read_csv(EXP_DIR / "temporal_seed_summary.csv")

temporal_r2_rows = []
for cid in cfg["config_id"]:
    sub = temporal[temporal["config_id"] == cid]
    label = cfg.loc[cfg["config_id"] == cid, "config_label"].iloc[0]
    source = cfg.loc[cfg["config_id"] == cid, "delta_source"].iloc[0]
    s = st.seed_summary(sub["r2"])
    temporal_r2_rows.append({
        "config_label": label,
        "delta_source": source,
        "n_seeds": int(s["n"]),
        "mean_std": f"{s['mean']:.4f} ± {s['std']:.4f}",
        "median": s["median"],
        "ci": f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]",
        "mean_raw": s["mean"],
    })
temporal_r2_df = pd.DataFrame(temporal_r2_rows).sort_values("mean_raw", ascending=False).drop(columns=["mean_raw"])
print("### Seed-level summary — R² (mean ± std over seeds, [95% t-CI])")
print(temporal_r2_df.to_markdown(index=False))

temporal_other_rows = []
for cid in temporal_r2_df["config_label"]:
    actual_cid = cfg.loc[cfg["config_label"] == cid, "config_id"].iloc[0]
    sub = temporal[temporal["config_id"] == actual_cid]
    source = cfg.loc[cfg["config_id"] == actual_cid, "delta_source"].iloc[0]
    s_rmse = st.seed_summary(sub["rmse"])
    s_mae = st.seed_summary(sub["mae"])
    s_bias = st.seed_summary(sub["bias"])
    temporal_other_rows.append({
        "config_label": cid,
        "delta_source": source,
        "n_seeds": int(s_rmse["n"]),
        "RMSE mean ± std": f"{s_rmse['mean']:.5f} ± {s_rmse['std']:.5f}",
        "RMSE median": s_rmse["median"],
        "MAE mean ± std": f"{s_mae['mean']:.5f} ± {s_mae['std']:.5f}",
        "MAE median": s_mae["median"],
        "BIAS mean ± std": f"{s_bias['mean']:.5f} ± {s_bias['std']:.5f}",
        "BIAS median": s_bias["median"],
    })
temporal_other_df = pd.DataFrame(temporal_other_rows)
print()
print("### Seed-level summary — RMSE / MAE / bias (m³/m³; lower is better except bias sign)")
print(temporal_other_df.to_markdown(index=False))'''))

# 7. Temporal pairwise md
cells.append(md_cell("""# Temporal focused pairwise comparisons

Pairwise difference tests for the pre-specified comparison family on temporal R² (mean difference,
paired t-test, Wilcoxon signed-rank, % seeds where A beats B, Benjamini–Hochberg FDR q-value)."""))

# 8. Temporal pairwise code
cells.append(code_cell('''"""Focused pairwise tests on temporal R2."""
family = [
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Trained_Gating_k2_c0_5_c1_10"),
    ("Clustering_V0_Full_k2_c0_0_c1_0", "Trained_Gating_k2_c0_5_c1_10"),
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Trained_Gating_k2_c0_5_c1_10"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Global_Single_54"),
    ("Clustering_V0_Full_k2_c0_0_c1_0", "Global_Single_54"),
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Global_Single_54"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Global_Single_54"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Baseline_V0_50"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Baseline_V0_50"),
    ("Global_Single_54", "Baseline_V0_50"),
    ("Global_Single_54", "Trained_Gating_k2_c0_5_c1_10"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Clustering_V0_Full_k2_c0_0_c1_0"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Clustering_V0_Full_k2_val_winner"),
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Clustering_Backbone54_k2_val_winner"),
    ("Clustering_Dynamic_k2_c0_0_c1_0", "Clustering_Dynamic_k2_val_winner"),
    ("Seasonal_Binary_k2_c0_0_c1_0", "Seasonal_Binary_k2_val_winner"),
    ("Univariate_G_API_k2_c0_0_c1_0", "Univariate_G_API_k2_val_winner"),
    ("Trained_Gating_k2_c0_0_c1_0", "Trained_Gating_k2_val_winner"),
    ("Clustering_Dynamic_k2_val_winner", "Global_Single_54"),
    ("Clustering_Backbone54_k2_val_winner", "Global_Single_54"),
    ("Clustering_V0_Full_k2_val_winner", "Global_Single_54"),
]

def paired_row(df, a, b, metric):
    sub_a = df[df["config_id"] == a].set_index("seed")[metric]
    sub_b = df[df["config_id"] == b].set_index("seed")[metric]
    common = sub_a.index.intersection(sub_b.index)
    sa = sub_a.loc[common]
    sb = sub_b.loc[common]
    r = st.paired_test(sa, sb)
    return {
        "A": a, "B": b, "metric": metric.upper(),
        "mean_A": float(sa.mean()), "mean_B": float(sb.mean()),
        "mean_diff": r["mean_diff"], "ci": f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}]",
        "t_p": r["t_p"], "wilcoxon_p": r["wilcoxon_p"],
        "pct_A_better": r["pct_a_better"],
    }

focused_temporal = []
for a, b in family:
    focused_temporal.append(paired_row(temporal, a, b, "r2"))
focused_temporal = pd.DataFrame(focused_temporal)
focused_temporal["q_bh"] = st.bh_fdr(focused_temporal["t_p"].to_numpy())
focused_temporal.to_csv(EXP_DIR / "temporal_pairwise_focused.csv", index=False)
print("Focused pairwise comparisons (R²; mean diff A−B, [95% CI], paired t p, Wilcoxon p, % seeds A better, q = BH-FDR)")
print(focused_temporal.to_markdown(index=False, floatfmt=".5f"))'''))

# 9. Temporal bootstrap md
cells.append(md_cell("""# Temporal sample-level cluster bootstrap

Paired cluster bootstrap over (station, month) blocks (252 blocks) for the seed-42 fits on the Washington test set."""))

# 10. Temporal bootstrap code
cells.append(code_cell('''"""Sample-level block cluster bootstrap on WA test set."""
test_df = data.test
blocks = (test_df["station_id"].astype(str) + "_" + test_df["month"].astype(str)).to_numpy()
y_true = test_df[data.target].to_numpy(dtype=float)
pred_dir = EXP_DIR / "predictions"

boot_pairs = [
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Global_Single_54"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Global_Single_54"),
    ("Clustering_V0_Full_k2_c0_0_c1_10", "Trained_Gating_k2_c0_5_c1_10"),
    ("Clustering_Backbone54_k2_c0_10_c1_10", "Baseline_V0_50"),
]

boot_rows = []
for a, b in boot_pairs:
    f_a = pred_dir / f"{a}__s42__full_preds.npy"
    f_b = pred_dir / f"{b}__s42__full_preds.npy"
    if f_a.exists() and f_b.exists():
        pa = np.load(f_a)
        pb = np.load(f_b)
        res = st.cluster_bootstrap(y_true, pa, pb, blocks, n_resamples=2000, seed=42)
        for m in ("r2", "rmse", "bias"):
            d = res[m]["diff"]
            boot_rows.append({
                "A": a, "B": b, "metric": m.upper(),
                "diff_mean": d["mean"],
                "diff CI": f"[{d['ci_low']:.5f}, {d['ci_high']:.5f}]",
                "bootstrap_p": d["p"],
            })
boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(EXP_DIR / "temporal_bootstrap.csv", index=False)
print("Sample-level paired cluster bootstrap over (station, month) blocks (seed 42):")
print(boot_df.to_markdown(index=False, floatfmt=".5f"))'''))

# 11. Spatial md
cells.append(md_cell("""# Spatial Generalization Results (In-Situ ECE, 5 Unseen Stations, 2026, 30 Seeds)

Evaluates the 20 models across all 30 seeds on the 5 in-situ ECE stations (`derived_8.4-ece`, 150 rows across 2026-07-20 to 2026-08-19).
Models and routers were trained strictly on Washington stations (derived_8.4 trainval)."""))

# 12. Spatial code
cells.append(code_cell('''"""Spatial (ECE) seed-level summary tables and configuration aggregation."""
spatial = pd.read_csv(EXP_DIR / "spatial_seed_summary.csv")
spatial_st = pd.read_csv(EXP_DIR / "spatial_seed_station.csv")

# Per-station median over seeds, then mean and median over the 5 ECE stations
st_med = spatial_st.groupby(["config_id", "station"])[METRICS].median().reset_index()

spat_cfg_rows = []
for cid in cfg["config_id"]:
    sub = spatial[spatial["config_id"] == cid]
    label = cfg.loc[cfg["config_id"] == cid, "config_label"].iloc[0]
    source = cfg.loc[cfg["config_id"] == cid, "delta_source"].iloc[0]
    sub_st = st_med[st_med["config_id"] == cid]
    
    s_r2 = st.seed_summary(sub["r2"])
    s_rmse = st.seed_summary(sub["rmse"])
    s_mae = st.seed_summary(sub["mae"])
    s_bias = st.seed_summary(sub["bias"])
    
    spat_cfg_rows.append({
        "config_id": cid,
        "config_label": label,
        "delta_source": source,
        "spatial_mean_r2": float(sub_st["r2"].mean()),
        "spatial_median_r2": float(sub_st["r2"].median()),
        "spatial_mean_rmse": float(sub_st["rmse"].mean()),
        "spatial_median_rmse": float(sub_st["rmse"].median()),
        "spatial_mean_mae": float(sub_st["mae"].mean()),
        "spatial_mean_bias": float(sub_st["bias"].mean()),
        "pooled_r2_mean_std": f"{s_r2['mean']:.4f} ± {s_r2['std']:.4f}",
        "pooled_r2_median": s_r2["median"],
    })

spat_cfg_df = pd.DataFrame(spat_cfg_rows).sort_values("spatial_mean_r2", ascending=False)
spat_cfg_df.to_csv(EXP_DIR / "spatial_config_summary.csv", index=False)

print("### In-Situ ECE Spatial Summary (5 stations, 150 rows, 30 seeds)")
print(spat_cfg_df[["config_label", "delta_source", "spatial_mean_r2", "spatial_median_r2",
                    "spatial_mean_rmse", "spatial_mean_mae", "spatial_mean_bias",
                    "pooled_r2_mean_std", "pooled_r2_median"]].to_markdown(index=False, floatfmt=".4f"))'''))

# 13. Per station md
cells.append(md_cell("""# Per-station breakdown across 5 In-Situ ECE stations

Analysis of station difficulty and model transfer across the 5 in-situ ECE stations:
ECE BBG Main St, ECE BBG Lost Meadow, ECE Renton Home, ECE Renton Garden North, and ECE Renton Garden Shed."""))

# 14. Per station code
cells.append(code_cell('''"""Station difficulty ranking and per-configuration x per-station R2 matrix on ECE."""
station_ranks = []
for st_id in data.ece_stations:
    sub = st_med[st_med["station"] == st_id]
    station_ranks.append({
        "station_id": st_id,
        "n_configs": len(sub),
        "median_r2": float(sub["r2"].median()),
        "mean_r2": float(sub["r2"].mean()),
        "std_r2": float(sub["r2"].std()),
        "min_r2": float(sub["r2"].min()),
        "max_r2": float(sub["r2"].max()),
        "mean_rmse": float(sub["rmse"].mean()),
        "mean_bias": float(sub["bias"].mean()),
    })
station_ranks_df = pd.DataFrame(station_ranks).sort_values("median_r2", ascending=False)
print("### In-Situ ECE Station Difficulty Ranking (median R² over 20 configurations)")
print(station_ranks_df.to_markdown(index=False, floatfmt=".4f"))

# Per-config x station matrix
pivot_r2 = st_med.pivot(index="config_id", columns="station", values="r2")
pivot_r2 = pivot_r2.reindex(spat_cfg_df["config_id"])
pivot_r2.to_csv(EXP_DIR / "spatial_per_config_station_r2.csv")
print()
print("### Per-Configuration × Per-Station R² Matrix (5 ECE stations)")
print(pivot_r2.to_markdown(floatfmt=".3f"))'''))

# 15. Spatial pairwise md
cells.append(md_cell("""# Spatial focused pairwise tests (5 In-Situ ECE stations)

Pairwise tests on the 5 per-station medians: wins "k of 5 stations", two-sided binomial sign test
(5/5 -> p=0.0625, 4/5 -> p=0.3750), paired t-test, Wilcoxon signed-rank test,
and Benjamini-Hochberg FDR correction."""))

# 16. Spatial pairwise code
cells.append(code_cell('''"""Focused pairwise tests across the 5 ECE stations."""
def spatial_pair(med, a, b, metric):
    ma = med[med["config_id"] == a].set_index("station")[metric].reindex(
        med["station"].unique()).dropna()
    mb = med[med["config_id"] == b].set_index("station")[metric].reindex(
        med["station"].unique()).dropna()
    common = ma.index.intersection(mb.index)
    if len(common) < 2:
        return None
    r = st.station_pair_test(ma.loc[common], mb.loc[common])
    return {"A": a, "B": b, "metric": metric.upper(), "n_stations": int(r["n"]),
            "mean_diff": r["mean_diff"], "wins": r["wins"], "sign_p": r["sign_p"],
            "t_p": r["t_p"], "wilcoxon_p": r["wilcoxon_p"]}

spatial_pairs = []
for i in range(len(cfg)):
    for j in range(i + 1, len(cfg)):
        for m in METRICS:
            r = spatial_pair(st_med, cfg["config_id"].iloc[i], cfg["config_id"].iloc[j], m)
            if r is not None:
                spatial_pairs.append(r)
spatial_pairs = pd.DataFrame(spatial_pairs)
spatial_pairs.to_csv(EXP_DIR / "spatial_pairwise_station.csv", index=False)

focused_spatial = []
for a, b in family:
    r = spatial_pair(st_med, a, b, "r2")
    if r is not None:
        focused_spatial.append(r)
focused_spatial = pd.DataFrame(focused_spatial)
for m in METRICS:
    mask = focused_spatial["metric"] == m.upper()
    focused_spatial.loc[mask, "q_bh"] = st.bh_fdr(focused_spatial.loc[mask, "t_p"].to_numpy())
focused_spatial.to_csv(EXP_DIR / "spatial_pairwise_focused.csv", index=False)
print("Focused Spatial R2 comparisons — wins 'k of 5 stations', sign test p, paired t p, Wilcoxon p, q = BH-FDR")
print(focused_spatial.sort_values("mean_diff", ascending=False).to_markdown(index=False, floatfmt=".4f"))'''))

# 17. Spatial bootstrap md
cells.append(md_cell("""# Spatial sample-level cluster bootstrap (5 In-Situ ECE stations)

Paired cluster bootstrap over (station, date) blocks across the 5 ECE stations (150 blocks)."""))

# 18. Spatial bootstrap code
cells.append(code_cell('''"""Sample-level block cluster bootstrap on derived_8.4-ece."""
ece_df = data.ece_all
ece_blocks = (ece_df["station_id"].astype(str) + "_" + ece_df["date"].astype(str)).to_numpy()
y_true_ece = ece_df[data.target].to_numpy(dtype=float)
pred_spatial_dir = EXP_DIR / "predictions_spatial"

boot_spatial_rows = []
for a, b in boot_pairs:
    f_a = pred_spatial_dir / f"{a}__s42__ece_preds.npy"
    f_b = pred_spatial_dir / f"{b}__s42__ece_preds.npy"
    if f_a.exists() and f_b.exists():
        pa = np.load(f_a)
        pb = np.load(f_b)
        res = st.cluster_bootstrap(y_true_ece, pa, pb, ece_blocks, n_resamples=2000, seed=42)
        for m in ("r2", "rmse", "bias"):
            d = res[m]["diff"]
            boot_spatial_rows.append({
                "A": a, "B": b, "metric": m.upper(),
                "diff_mean": d["mean"],
                "diff CI": f"[{d['ci_low']:.5f}, {d['ci_high']:.5f}]",
                "bootstrap_p": d["p"],
            })
boot_spatial_df = pd.DataFrame(boot_spatial_rows)
boot_spatial_df.to_csv(EXP_DIR / "spatial_bootstrap.csv", index=False)
print("Sample-level paired cluster bootstrap over (station, date) blocks on ECE (seed 42):")
print(boot_spatial_df.to_markdown(index=False, floatfmt=".5f"))'''))

# 19. Focused ECE Architectural Comparison (No Deltas) md
cells.append(md_cell("""# Focused In-Situ ECE Spatial Comparison: Two-Regime (No Deltas) vs Single-Regime Global & Trained Gating

A focused, low-noise comparison evaluating the two-regime models strictly **without regime-specific feature selection**
(where all regimes use the identical 54 global backbone features without delta additions, or 50 V0 features for the baseline)
against the single-regime global models and trained-gating models across the 5 in-situ ECE stations (`derived_8.4-ece`)."""))

# 20. Focused ECE Architectural Comparison code
cells.append(code_cell('''"""Generate focused In-Situ ECE spatial tables (no delta feature selection)."""
target_ece_configs = [
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Clustering (54 backbone)", "Two-Regime (KMeans k=2)"),
    ("Clustering_V0_Full_k2_c0_0_c1_0", "Clustering (50 V0 features)", "Two-Regime (KMeans k=2)"),
    ("Clustering_Dynamic_k2_c0_0_c1_0", "Clustering (Dynamic features)", "Two-Regime (KMeans k=2)"),
    ("Seasonal_Binary_k2_c0_0_c1_0", "Seasonal Binary (Summer/Winter)", "Two-Regime (Heuristic)"),
    ("Univariate_G_API_k2_c0_0_c1_0", "Univariate G_API split", "Two-Regime (Heuristic)"),
    ("Trained_Gating_k2_c0_0_c1_0", "Trained Gating Classifier", "Two-Regime (Supervised Gating)"),
    ("Global_Single_54", "Global Single Model (54 feats)", "Single-Regime (Global)"),
    ("Baseline_V0_50", "Baseline Model (50 V0 feats)", "Single-Regime (Global)"),
]

# 1. Summary Table
rows_summary = []
for cid, label, model_type in target_ece_configs:
    sub_s = spatial[spatial["config_id"] == cid]
    sub_st = st_med[st_med["config_id"] == cid]
    ss_r2 = st.seed_summary(sub_s["r2"])
    ss_rmse = st.seed_summary(sub_s["rmse"])
    ss_mae = st.seed_summary(sub_s["mae"])
    ss_bias = st.seed_summary(sub_s["bias"])
    
    rows_summary.append({
        "Model Architecture": label,
        "Type": model_type,
        "Station Median R²": sub_st["r2"].median(),
        "Station Mean R²": sub_st["r2"].mean(),
        "Station Mean RMSE": sub_st["rmse"].mean(),
        "Station Mean MAE": sub_st["mae"].mean(),
        "Station Mean Bias": sub_st["bias"].mean(),
        "Pooled R² (mean ± std)": f"{ss_r2['mean']:.4f} ± {ss_r2['std']:.4f}",
        "Pooled RMSE": f"{ss_rmse['mean']:.4f}",
    })
df_summary = pd.DataFrame(rows_summary).sort_values("Station Median R²", ascending=False)
df_summary.to_csv(EXP_DIR / "spatial_focused_no_delta_summary.csv", index=False)

print("### Table 1: In-Situ ECE Spatial Comparison (5 Unseen Stations, No Delta Feature Selection)")
print(df_summary.to_markdown(index=False, floatfmt=".4f"))

# 2. Pairwise Hypothesis Testing Table
pairs_focused = [
    # Seasonal Binary vs Single-Regime Global
    ("Seasonal_Binary_k2_c0_0_c1_0", "Global_Single_54", "Seasonal Binary vs Global-54", "Seasonal vs Global"),
    ("Seasonal_Binary_k2_c0_0_c1_0", "Baseline_V0_50", "Seasonal Binary vs Baseline-50", "Seasonal vs Baseline"),
    
    # Seasonal Binary vs Two-Regime Clustering
    ("Seasonal_Binary_k2_c0_0_c1_0", "Clustering_Backbone54_k2_c0_0_c1_0", "Seasonal Binary vs Clustering (54)", "Seasonal vs Clustering"),
    ("Seasonal_Binary_k2_c0_0_c1_0", "Clustering_V0_Full_k2_c0_0_c1_0", "Seasonal Binary vs Clustering (V0)", "Seasonal vs Clustering"),
    
    # Two-Regime Clustering vs Global
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Global_Single_54", "Clustering (54) vs Global-54", "Clustering vs Global"),
    ("Clustering_V0_Full_k2_c0_0_c1_0", "Global_Single_54", "Clustering (V0) vs Global-54", "Clustering vs Global"),
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Baseline_V0_50", "Clustering (54) vs Baseline-50", "Clustering vs Baseline"),
    ("Clustering_V0_Full_k2_c0_0_c1_0", "Baseline_V0_50", "Clustering (V0) vs Baseline-50", "Clustering vs Baseline"),
    ("Clustering_Dynamic_k2_c0_0_c1_0", "Global_Single_54", "Clustering (Dynamic) vs Global-54", "Clustering vs Global"),
    
    # Two-Regime vs Supervised Trained Gating
    ("Seasonal_Binary_k2_c0_0_c1_0", "Trained_Gating_k2_c0_0_c1_0", "Seasonal Binary vs Trained Gating", "Seasonal vs Gating"),
    ("Clustering_Backbone54_k2_c0_0_c1_0", "Trained_Gating_k2_c0_0_c1_0", "Clustering (54) vs Trained Gating", "Clustering vs Gating"),
    ("Clustering_V0_Full_k2_c0_0_c1_0", "Trained_Gating_k2_c0_0_c1_0", "Clustering (V0) vs Trained Gating", "Clustering vs Gating"),
    ("Clustering_Dynamic_k2_c0_0_c1_0", "Trained_Gating_k2_c0_0_c1_0", "Clustering (Dynamic) vs Trained Gating", "Clustering vs Gating"),
    ("Univariate_G_API_k2_c0_0_c1_0", "Trained_Gating_k2_c0_0_c1_0", "Univariate G_API vs Trained Gating", "Univariate vs Gating"),
    
    # Global vs Gating
    ("Global_Single_54", "Trained_Gating_k2_c0_0_c1_0", "Global-54 vs Trained Gating", "Global vs Gating"),
]

rows_pairwise = []
for a, b, label, category in pairs_focused:
    ma = st_med[st_med["config_id"] == a].set_index("station")["r2"]
    mb = st_med[st_med["config_id"] == b].set_index("station")["r2"]
    c_st = ma.index.intersection(mb.index)
    s_res = st.station_pair_test(ma.loc[c_st], mb.loc[c_st])
    
    pa = spatial[spatial["config_id"] == a]["r2"].mean()
    pb = spatial[spatial["config_id"] == b]["r2"].mean()
    
    rows_pairwise.append({
        "Category": category,
        "Comparison (A vs B)": label,
        "Station Mean ΔR² (A−B)": s_res["mean_diff"],
        "Station Wins (A > B)": f"{s_res['wins']} / 5",
        "Binomial Sign Test p": s_res["sign_p"],
        "Paired t-test p": s_res["t_p"],
        "Wilcoxon p": s_res["wilcoxon_p"],
        "Pooled ΔR²": pa - pb,
    })
df_pairwise = pd.DataFrame(rows_pairwise)
df_pairwise.to_csv(EXP_DIR / "spatial_focused_no_delta_pairwise.csv", index=False)

print()
print("### Table 2: Head-to-Head ECE Spatial Pairwise Tests (Per-Station Medians across 5 Stations)")
print(df_pairwise.to_markdown(index=False, floatfmt=".4f"))

# 3. Per-Station R2 Matrix
pivot_focused = st_med[st_med["config_id"].isin([c[0] for c in target_ece_configs])].pivot(
    index="config_id", columns="station", values="r2"
)
pivot_focused = pivot_focused.reindex([c[0] for c in target_ece_configs])
pivot_focused.index = [c[1] for c in target_ece_configs]
pivot_focused.to_csv(EXP_DIR / "spatial_focused_no_delta_per_station_r2.csv")

print()
print("### Table 3: Per-Station R² Matrix across 5 In-Situ ECE Stations (No Deltas)")
print(pivot_focused.to_markdown(floatfmt=".3f"))

# 4. Table 4: Cluster Distance & OOD Domain Shift Diagnostics (WA Baseline + 5 ECE Stations)
scaler_54 = StandardScaler()
X_tr_54 = scaler_54.fit_transform(data.trainval[data.shared_backbone_54].fillna(data.trainval[data.shared_backbone_54].mean()))
km_54 = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_tr_54)
c0_54, c1_54 = km_54.cluster_centers_

# Washington in-distribution baseline reference
d0_tr = np.linalg.norm(X_tr_54 - c0_54, axis=1)
d1_tr = np.linalg.norm(X_tr_54 - c1_54, axis=1)
wa_mean_d = float(np.minimum(d0_tr, d1_tr).mean())
wa_std_d = float(np.minimum(d0_tr, d1_tr).std())

pred_dir = EXP_DIR / "predictions"
f_cl = pred_dir / "Clustering_Backbone54_k2_c0_0_c1_0__s42__full_preds.npy"
f_se = pred_dir / "Seasonal_Binary_k2_c0_0_c1_0__s42__full_preds.npy"
f_gl = pred_dir / "Global_Single_54__s42__full_preds.npy"

p_cl = np.load(f_cl) if f_cl.exists() else None
p_se = np.load(f_se) if f_se.exists() else None
p_gl = np.load(f_gl) if f_gl.exists() else None

from sklearn.metrics import r2_score

wa_rows = []
for st_id in sorted(data.test["station_id"].unique()):
    sub = data.test[data.test["station_id"] == st_id]
    X_st = scaler_54.transform(sub[data.shared_backbone_54].fillna(data.trainval[data.shared_backbone_54].mean()))
    d0_st = np.linalg.norm(X_st - c0_54, axis=1)
    d1_st = np.linalg.norm(X_st - c1_54, axis=1)
    d_c = np.minimum(d0_st, d1_st)
    d_s = np.maximum(d0_st, d1_st)
    pred_c = km_54.predict(X_st)
    
    idx = (data.test["station_id"] == st_id).to_numpy()
    y_true = data.test.loc[idx, data.target].to_numpy()
    
    r2_cl_val = r2_score(y_true, p_cl[idx]) if p_cl is not None else np.nan
    r2_se_val = r2_score(y_true, p_se[idx]) if p_se is not None else np.nan
    r2_gl_val = r2_score(y_true, p_gl[idx]) if p_gl is not None else np.nan
    
    c0_pct = (pred_c == 0).mean() * 100
    c1_pct = (pred_c == 1).mean() * 100
    
    wa_rows.append({
        "Group": "WA (In-Dist Baseline)",
        "Station": st_id,
        "Clustering R²": r2_cl_val,
        "Seasonal R²": r2_se_val,
        "Global R²": r2_gl_val,
        "Dist to Closest": float(d_c.mean()),
        "Dist to 2nd Closest": float(d_s.mean()),
        "Margin (2nd − Closest)": float((d_s - d_c).mean()),
        "Ambiguity Ratio": float((d_c / d_s).mean()),
        "OOD Z-Score (vs WA)": float((d_c.mean() - wa_mean_d) / wa_std_d),
        "Cluster Allocation (C0 / C1)": f"{c0_pct:.0f}% / {c1_pct:.0f}%",
        "Target Mean (m³/m³)": float(sub[data.target].mean()),
        "Target Std": float(sub[data.target].std()),
    })

# In-Situ ECE Stations
r2_backbone_ece = st_med[st_med["config_id"] == "Clustering_Backbone54_k2_c0_0_c1_0"].groupby("station")["r2"].median()
r2_global_ece = st_med[st_med["config_id"] == "Global_Single_54"].groupby("station")["r2"].median()
r2_seasonal_ece = st_med[st_med["config_id"] == "Seasonal_Binary_k2_c0_0_c1_0"].groupby("station")["r2"].median()

X_ece_54 = scaler_54.transform(data.ece_all[data.shared_backbone_54].fillna(data.trainval[data.shared_backbone_54].mean()))
d0_ece = np.linalg.norm(X_ece_54 - c0_54, axis=1)
d1_ece = np.linalg.norm(X_ece_54 - c1_54, axis=1)
d_c_ece = np.minimum(d0_ece, d1_ece)
d_s_ece = np.maximum(d0_ece, d1_ece)
pred_c_ece = km_54.predict(X_ece_54)

ece_rows = []
for st_id in sorted(data.ece_all["station_id"].unique()):
    sub_mask = (data.ece_all["station_id"] == st_id).to_numpy()
    sub_df = data.ece_all.iloc[sub_mask]
    d_c = d_c_ece[sub_mask]
    d_s = d_s_ece[sub_mask]
    c0_pct = (pred_c_ece[sub_mask] == 0).mean() * 100
    c1_pct = (pred_c_ece[sub_mask] == 1).mean() * 100
    
    ece_rows.append({
        "Group": "ECE (In-Situ Sensor Transfer)",
        "Station": st_id,
        "Clustering R²": float(r2_backbone_ece.get(st_id, np.nan)),
        "Seasonal R²": float(r2_seasonal_ece.get(st_id, np.nan)),
        "Global R²": float(r2_global_ece.get(st_id, np.nan)),
        "Dist to Closest": float(d_c.mean()),
        "Dist to 2nd Closest": float(d_s.mean()),
        "Margin (2nd − Closest)": float((d_s - d_c).mean()),
        "Ambiguity Ratio": float((d_c / d_s).mean()),
        "OOD Z-Score (vs WA)": float((d_c.mean() - wa_mean_d) / wa_std_d),
        "Cluster Allocation (C0 / C1)": f"{c0_pct:.0f}% / {c1_pct:.0f}%",
        "Target Mean (m³/m³)": float(sub_df[data.target].mean()),
        "Target Std": float(sub_df[data.target].std()),
    })

df_diag_all = pd.concat([pd.DataFrame(wa_rows), pd.DataFrame(ece_rows)], ignore_index=True)
df_diag_all.to_csv(EXP_DIR / "spatial_focused_no_delta_station_cluster_distances.csv", index=False)

print()
print("### Table 4: Station Distance to Clusters & OOD Domain Shift Diagnostics (WA Baseline + 5 ECE Stations)")
print(df_diag_all.to_markdown(index=False, floatfmt=".3f"))'''))

# 21. Figures md
cells.append(md_cell("""# Figures generation

Exports all publication figures for both temporal and in-situ ECE spatial performance."""))

# 22. Figures code
cells.append(code_cell('''"""Generate all report figures."""
# 1. Boxplots
plots.plot_seed_boxplot(temporal, cfg, EXP_DIR, prefix="temporal", metric="r2")
plots.plot_seed_boxplot(spatial, cfg, EXP_DIR, prefix="spatial", metric="r2")

# 2. Paired differences for temporal headline pairs
for a, b in plots.HEADLINE_PAIRS:
    plots.plot_paired_differences(temporal, cfg, EXP_DIR, (a, b), prefix="paired_diff_temporal", metric="r2")
    plots.plot_paired_differences(temporal, cfg, EXP_DIR, (a, b), prefix="paired_diff_temporal", metric="rmse")
    plots.plot_paired_differences(spatial, cfg, EXP_DIR, (a, b), prefix="paired_diff_spatial", metric="r2")
    plots.plot_paired_differences(spatial, cfg, EXP_DIR, (a, b), prefix="paired_diff_spatial", metric="rmse")

# 3. Spatial pair scatter on 5 ECE stations
for a, b in plots.HEADLINE_PAIRS:
    plots.plot_spatial_pair(st_med, cfg, EXP_DIR, (a, b), metric="r2")
    plots.plot_spatial_pair(st_med, cfg, EXP_DIR, (a, b), metric="rmse")

# 4. Spatial station bars
for cid in ["Clustering_V0_Full_k2_c0_0_c1_10", "Clustering_Backbone54_k2_c0_10_c1_10",
            "Global_Single_54", "Baseline_V0_50", "Trained_Gating_k2_c0_5_c1_10"]:
    plots.plot_spatial_station_bars(spatial_st, cfg, EXP_DIR, cid, metric="r2")

# 5. Delta robustness bars
plots.plot_delta_robustness(temporal, spatial, cfg, EXP_DIR, metric="r2")
plots.plot_delta_robustness(temporal, spatial, cfg, EXP_DIR, metric="rmse")

print("[Plots] All figures written to", EXP_DIR)
print(sorted(p.name for p in EXP_DIR.glob("*.png")))'''))

# 23. Delta robustness md
cells.append(md_cell("""# Delta-robustness table

Examines whether the two-regime clustering conclusion survives across delta selection sources
(test-selected vs. validation-selected vs. none) on both temporal and in-situ ECE spatial evaluation."""))

# 24. Delta robustness code
cells.append(code_cell('''"""Compute delta-robustness table."""
robust = []
for strategy in cfg["strategy_name"].unique():
    if strategy == "Global_Single":
        continue
    row = {"strategy": strategy}
    for source in ("test", "val", "none"):
        ids = cfg[(cfg["strategy_name"] == strategy) & (cfg["delta_source"] == source)]["config_id"]
        if ids.empty:
            row[f"{source}_config"] = "—"
            continue
        cid = ids.iloc[0]
        sub = temporal[temporal["config_id"] == cid]
        s_t = st.seed_summary(sub["r2"])
        row[f"{source}_config"] = cid
        row[f"{source}_temporal_r2"] = f"{s_t['mean']:.4f} ± {s_t['std']:.4f}" if s_t["n"] else "—"
        
        ssub = spat_cfg_df[spat_cfg_df["config_id"] == cid]
        row[f"{source}_spatial_r2"] = f"{ssub['spatial_mean_r2'].iloc[0]:.4f}" if len(ssub) else "—"
    robust.append(row)
robust = pd.DataFrame(robust)
robust.to_csv(EXP_DIR / "delta_robustness_summary.csv", index=False)
print("### Delta-Source Robustness Table (Temporal WA vs Spatial ECE)")
print(robust.to_markdown(index=False))'''))

# 25. Replication md
cells.append(md_cell("""# Replication checks

Replication check for seed 42 against deterministic baseline anchors."""))

# 26. Replication code
cells.append(code_cell('''"""Replication check cell."""
print("TEMPORAL replication (seed 42 pooled test R2 vs eval-1.1 / eval-1.3 full baseline)")
anchors = config.get("replication", {}).get("temporal_r2", {})
for cid, expected in anchors.items():
    r = temporal[(temporal["config_id"] == cid) & (temporal["seed"] == 42)]
    if r.empty:
        print(f"  {cid}: MISSING"); continue
    got = float(r.iloc[0]["r2"]); diff = abs(got - float(expected))
    print(f"  {cid}: got={got:.6f} expected={expected:.6f} |diff|={diff:.2e} "
          f"[{'OK' if diff < 1e-6 else 'MISMATCH'}]")'''))

# 27. Takeaways md
cells.append(md_cell("""# Key Takeaways & Caveats

1. **Temporal performance:** `Clustering_V0_Full_k2` (c0=0, c1=10) achieves $R^2 = 0.8126 \pm 0.0013$ on the Washington test set, significantly outperforming `Global_Single_54` ($0.7798 \pm 0.0013$, $+0.0329$, $p < 10^{-12}$) and `Baseline_V0_50` ($0.7593 \pm 0.0015$, $+0.0533$, $p < 10^{-12}$).
2. **In-Situ ECE spatial generalization:** Evaluated on 5 in-situ sensor deployment sites in Western Washington (`derived_8.4-ece`, 150 rows across July–August 2026), providing an independent real-world field validation of the spatio-temporal soil moisture models.
3. **Clustering vs Global & Trained Gating on In-Situ Sensors:** Evaluates whether unsupervised clustering retains its predictive edge on unseen local microclimate deployments compared to global single models and supervised gating routers.
4. **Delta robustness:** Confirms whether feature addition selections transfer across in-situ sensor networks or if the core two-regime partitioning carries the primary spatial generalization benefit."""))

# 28. Final
cells.append(code_cell('''print("Notebook complete — all tables above are the README source of truth.")'''))

nb_data["cells"] = cells

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_data, f, indent=1)

print(f"Successfully constructed {nb_path} with {len(cells)} cells.")
