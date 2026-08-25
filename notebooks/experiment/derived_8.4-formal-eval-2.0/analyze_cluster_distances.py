"""Reproducible analysis script for Table 4: Cluster Distances & OOD Domain Shift Diagnostics.

Computes cluster centroid distances, boundary margins, ambiguity ratios, OOD Z-scores,
and soil moisture target parameters for both the 7 Washington baseline stations
and the 10 Out-of-State unseen transfer stations.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import yaml

from eval_formal.data import load_experiment_data

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2].resolve()

with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data = load_experiment_data(PROJECT_ROOT, config)
trainval = data.trainval
test_wa = data.test
oos = data.oos_all
feats_54 = data.shared_backbone_54
target = data.target

# Fit KMeans on WA trainval
scaler = StandardScaler()
X_tr = scaler.fit_transform(trainval[feats_54].fillna(trainval[feats_54].mean()))
km = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_tr)
c0, c1 = km.cluster_centers_

# WA in-distribution baseline reference
d0_tr = np.linalg.norm(X_tr - c0, axis=1)
d1_tr = np.linalg.norm(X_tr - c1, axis=1)
d_closest_tr = np.minimum(d0_tr, d1_tr)
wa_mean_d = float(d_closest_tr.mean())
wa_std_d = float(d_closest_tr.std())

pred_dir = EXP_DIR / "predictions"
f_cl = pred_dir / "Clustering_Backbone54_k2_c0_0_c1_0__s42__full_preds.npy"
f_se = pred_dir / "Seasonal_Binary_k2_c0_0_c1_0__s42__full_preds.npy"
f_gl = pred_dir / "Global_Single_54__s42__full_preds.npy"

p_cl = np.load(f_cl) if f_cl.exists() else None
p_se = np.load(f_se) if f_se.exists() else None
p_gl = np.load(f_gl) if f_gl.exists() else None

# 1. Washington Baseline Stations
wa_rows = []
for st_id in sorted(test_wa["station_id"].unique()):
    sub = test_wa[test_wa["station_id"] == st_id]
    X_st = scaler.transform(sub[feats_54].fillna(trainval[feats_54].mean()))
    d0_st = np.linalg.norm(X_st - c0, axis=1)
    d1_st = np.linalg.norm(X_st - c1, axis=1)
    d_c = np.minimum(d0_st, d1_st)
    d_s = np.maximum(d0_st, d1_st)
    pred_c = km.predict(X_st)
    
    idx = (test_wa["station_id"] == st_id).to_numpy()
    y_true = test_wa.loc[idx, target].to_numpy()
    
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
        "Target Mean (m³/m³)": float(sub[target].mean()),
        "Target Std": float(sub[target].std()),
    })

# 2. Out-of-State Stations
spat_st = pd.read_csv(EXP_DIR / "spatial_seed_station.csv")
r2_backbone_oos = spat_st[spat_st["config_id"] == "Clustering_Backbone54_k2_c0_0_c1_0"].groupby("station")["r2"].median()
r2_global_oos = spat_st[spat_st["config_id"] == "Global_Single_54"].groupby("station")["r2"].median()
r2_seasonal_oos = spat_st[spat_st["config_id"] == "Seasonal_Binary_k2_c0_0_c1_0"].groupby("station")["r2"].median()

X_oos = scaler.transform(oos[feats_54].fillna(trainval[feats_54].mean()))
d0_oos = np.linalg.norm(X_oos - c0, axis=1)
d1_oos = np.linalg.norm(X_oos - c1, axis=1)
d_c_oos = np.minimum(d0_oos, d1_oos)
d_s_oos = np.maximum(d0_oos, d1_oos)
pred_c_oos = km.predict(X_oos)

oos_rows = []
for st_id in sorted(oos["station_id"].unique()):
    sub_mask = (oos["station_id"] == st_id).to_numpy()
    sub_df = oos.iloc[sub_mask]
    d_c = d_c_oos[sub_mask]
    d_s = d_s_oos[sub_mask]
    c0_pct = (pred_c_oos[sub_mask] == 0).mean() * 100
    c1_pct = (pred_c_oos[sub_mask] == 1).mean() * 100
    
    oos_rows.append({
        "Group": "OOS (Unseen Transfer)",
        "Station": st_id,
        "Clustering R²": float(r2_backbone_oos.get(st_id, np.nan)),
        "Seasonal R²": float(r2_seasonal_oos.get(st_id, np.nan)),
        "Global R²": float(r2_global_oos.get(st_id, np.nan)),
        "Dist to Closest": float(d_c.mean()),
        "Dist to 2nd Closest": float(d_s.mean()),
        "Margin (2nd − Closest)": float((d_s - d_c).mean()),
        "Ambiguity Ratio": float((d_c / d_s).mean()),
        "OOD Z-Score (vs WA)": float((d_c.mean() - wa_mean_d) / wa_std_d),
        "Cluster Allocation (C0 / C1)": f"{c0_pct:.0f}% / {c1_pct:.0f}%",
        "Target Mean (m³/m³)": float(sub_df[target].mean()),
        "Target Std": float(sub_df[target].std()),
    })

df_all = pd.concat([pd.DataFrame(wa_rows), pd.DataFrame(oos_rows)], ignore_index=True)
out_csv = EXP_DIR / "spatial_focused_no_delta_station_cluster_distances.csv"
df_all.to_csv(out_csv, index=False)
print(f"Saved distance diagnostic table to {out_csv}")
print()
print("### Table 4: Station Distance to Clusters & OOD Domain Shift Diagnostics (WA Baseline + 10 OOS Stations)")
print(df_all.to_markdown(index=False, floatfmt=".3f"))
