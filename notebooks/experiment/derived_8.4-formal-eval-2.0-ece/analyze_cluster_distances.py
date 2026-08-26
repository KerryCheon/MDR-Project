"""Reproducible analysis script for Table 4: Cluster Distances & OOD Domain Shift Diagnostics.

Computes cluster centroid distances, boundary margins, ambiguity ratios, OOD Z-scores,
and soil moisture target parameters for both the 7 Washington baseline stations
and the 5 In-Situ ECE sensor transfer stations.
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
ece = data.ece_all
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

# 2. In-Situ ECE Stations
spat_st_path = EXP_DIR / "spatial_seed_station.csv"
if spat_st_path.exists():
    spat_st = pd.read_csv(spat_st_path)
    r2_backbone_ece = spat_st[spat_st["config_id"] == "Clustering_Backbone54_k2_c0_0_c1_0"].groupby("station")["r2"].median()
    r2_global_ece = spat_st[spat_st["config_id"] == "Global_Single_54"].groupby("station")["r2"].median()
    r2_seasonal_ece = spat_st[spat_st["config_id"] == "Seasonal_Binary_k2_c0_0_c1_0"].groupby("station")["r2"].median()
else:
    r2_backbone_ece = {}
    r2_global_ece = {}
    r2_seasonal_ece = {}

X_ece = scaler.transform(ece[feats_54].fillna(trainval[feats_54].mean()))
d0_ece = np.linalg.norm(X_ece - c0, axis=1)
d1_ece = np.linalg.norm(X_ece - c1, axis=1)
d_c_ece = np.minimum(d0_ece, d1_ece)
d_s_ece = np.maximum(d0_ece, d1_ece)
pred_c_ece = km.predict(X_ece)

ece_rows = []
for st_id in sorted(ece["station_id"].unique()):
    sub_mask = (ece["station_id"] == st_id).to_numpy()
    sub_df = ece.iloc[sub_mask]
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
        "Target Mean (m³/m³)": float(sub_df[target].mean()),
        "Target Std": float(sub_df[target].std()),
    })

df_all = pd.concat([pd.DataFrame(wa_rows), pd.DataFrame(ece_rows)], ignore_index=True)
out_csv = EXP_DIR / "spatial_focused_no_delta_station_cluster_distances.csv"
df_all.to_csv(out_csv, index=False)
print(f"Saved distance diagnostic table to {out_csv}")
print()
print("### Table 4: Station Distance to Clusters & OOD Domain Shift Diagnostics (WA Baseline + 5 ECE Stations)")
print(df_all.to_markdown(index=False, floatfmt=".3f"))
