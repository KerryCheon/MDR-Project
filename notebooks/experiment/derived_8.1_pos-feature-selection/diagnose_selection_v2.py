import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from Modeling.Utils.config import load_config
from Modeling.Src.soilmoist_fl.Data.load import load_splits
from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
from Modeling.Src.soilmoist_fl.Selectors.mi import select_mi
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet

TRACK_FEATURES = [
    'elev', 'slope', 'year_frac', 'sin_year', 'cos_year',
    'G_API', 'G_rain_sum_3d', 'G_rain_sum_7d', 'G_DSLR',
    'J_clay_wfrac_b0', 'J_sand_wfrac_b0', 'K_slope_sin',
    'K_slope_cos', 'K_aspect_cos', 'API_x_year', 'SMAP_x_year',
    'D_sin_DOY', 'SMAP_sm_pm_interp_ema02', 'V_rollmin_LST_modis_kobs30'
]

def main():
    config_path = "notebooks/experiment/derived_8.1_pos-feature-selection/config.yaml"
    cfg = load_config(config_path)
    
    loaded = load_splits(cfg)
    fold = loaded.folds[0]
    
    data_cfg = cfg.get("data", {})
    target = data_cfg.get("target")
    id_cols = list(data_cfg.get("id_cols", []) or [])
    time_col = data_cfg.get("time_col")
    
    drop_cols = list(id_cols)
    if time_col:
        drop_cols.append(time_col)
        
    X_tr, y_tr, _, _ = preprocess_split(fold.train, target, drop_cols=drop_cols)
    
    print(f"X_tr shape: {X_tr.shape}")
    
    # --- Experiment 1: ElasticNet directly on all 496 features (No MI filter) ---
    print("\n--- Running ElasticNet on ALL 496 features ---")
    enet_all = select_elasticnet(X_tr, y_tr, k=60)
    feats_all = enet_all["selected"]
    scores_all = enet_all["scores"]
    ranked_all = enet_all["ranked"]
    
    # --- Experiment 2: MI with k=300 followed by ElasticNet ---
    print("\n--- Running MI with k=300 followed by ElasticNet ---")
    mi_300 = select_mi(X_tr, y_tr, k=300)
    mi_300_feats = mi_300["selected"]
    
    X_mi_300 = X_tr[mi_300_feats]
    enet_mi_300 = select_elasticnet(X_mi_300, y_tr, k=60)
    feats_mi_300 = enet_mi_300["selected"]
    scores_mi_300 = enet_mi_300["scores"]
    ranked_mi_300 = enet_mi_300["ranked"]
    
    print("\n" + "="*80)
    print("COMPARATIVE FEATURE SELECTION REPORT")
    print("="*80)
    print(f"{'Feature Name':<30} | {'EN (No MI) Coef':<16} {'EN (No MI) Rank':<16} {'EN (No MI) Sel':<15} | {'EN (MI=300) Coef':<16} {'EN (MI=300) Rank':<16} {'EN (MI=300) Sel':<15}")
    print("-"*135)
    
    for f in TRACK_FEATURES:
        # EN (No MI)
        coef_all = scores_all.get(f, 0.0)
        rank_all = ranked_all.index(f) + 1 if f in ranked_all else -1
        sel_all = "Yes" if f in feats_all else "No"
        
        # EN (MI=300)
        coef_mi = scores_mi_300.get(f, 0.0)
        rank_mi = ranked_mi_300.index(f) + 1 if f in ranked_mi_300 else -1
        sel_mi = "Yes" if f in feats_mi_300 else "No"
        
        print(f"{f:<30} | {coef_all:<16.5f} {rank_all:<16} {sel_all:<15} | {coef_mi:<16.5f} {rank_mi:<16} {sel_mi:<15}")
        
    print("="*80)
    
    print("Top 15 selected features EN (No MI):")
    for i, f in enumerate(ranked_all[:15]):
        print(f"  {i+1:<2}: {f:<35} (Coef: {scores_all[f]:.5f})")
        
    print("\nTop 15 selected features EN (MI=300):")
    for i, f in enumerate(ranked_mi_300[:15]):
        print(f"  {i+1:<2}: {f:<35} (Coef: {scores_mi_300[f]:.5f})")

if __name__ == "__main__":
    main()
