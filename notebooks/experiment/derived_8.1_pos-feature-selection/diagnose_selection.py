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
from Modeling.Src.soilmoist_fl.Selectors.stability import stability_bootstrap_elasticnet

# Features from Set A to track
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
    
    # 1. MI Selection
    mi_out = select_mi(X_tr, y_tr, k=120)
    mi_feats = mi_out["selected"]
    mi_scores = mi_out["scores"]
    mi_ranked = mi_out["ranked"]
    
    # 2. ElasticNet Selection
    X_mi = X_tr[mi_feats]
    enet_out = select_elasticnet(X_mi, y_tr, k=60)
    enet_feats = enet_out["selected"]
    enet_scores = enet_out["scores"]
    enet_ranked = enet_out["ranked"]
    
    # 3. Stability Selection
    stab_out = stability_bootstrap_elasticnet(
        X_mi,
        y_tr,
        n_boot=5,
        sample_frac=0.8,
        min_freq=0.6,
        top_k=40,
        random_state=42,
        enet_k=60,
        enet_kwargs={},
    )
    stable_feats = stab_out["selected"]
    stable_scores = stab_out["scores"]
    stable_ranked = stab_out["ranked"]
    
    print("\n" + "="*80)
    print("DIAGNOSTIC REPORT FOR FEATURE SELECTION STAGES")
    print("="*80)
    print(f"{'Feature Name':<30} | {'In X_tr':<7} | {'MI Score':<10} {'MI Rank':<7} {'MI Sel':<6} | {'EN Coef':<10} {'EN Rank':<7} {'EN Sel':<6} | {'Stab Freq':<10} {'Stab Sel':<6}")
    print("-"*125)
    
    for f in TRACK_FEATURES:
        in_xtr = f in X_tr.columns
        if not in_xtr:
            print(f"{f:<30} | {'No':<7} | {'-':<10} {'-':<7} {'-':<6} | {'-':<10} {'-':<7} {'-':<6} | {'-':<10} {'-':<6}")
            continue
            
        # MI
        mi_score = mi_scores.get(f, 0.0)
        mi_rank = mi_ranked.index(f) + 1 if f in mi_ranked else -1
        mi_sel = "Yes" if f in mi_feats else "No"
        
        # EN
        en_score = enet_scores.get(f, 0.0)
        en_rank = enet_ranked.index(f) + 1 if f in enet_ranked else -1
        en_sel = "Yes" if f in enet_feats else "No"
        
        # Stab
        stab_freq = stable_scores.get(f, 0.0)
        stab_sel = "Yes" if f in stable_feats else "No"
        
        print(f"{f:<30} | {'Yes':<7} | {mi_score:<10.5f} {mi_rank:<7} {mi_sel:<6} | {en_score:<10.5f} {en_rank:<7} {en_sel:<6} | {stab_freq:<10.3f} {stab_sel:<6}")
    
    print("="*80)
    print("Top 15 MI Features:")
    for i, f in enumerate(mi_ranked[:15]):
        print(f"  {i+1:<2}: {f:<35} (Score: {mi_scores[f]:.5f})")
        
    print("\nTop 15 ElasticNet Features:")
    for i, f in enumerate(enet_ranked[:15]):
        print(f"  {i+1:<2}: {f:<35} (Coef: {enet_scores[f]:.5f})")

    print("\nTop 15 Stable Features:")
    for i, f in enumerate(stable_ranked[:15]):
        print(f"  {i+1:<2}: {f:<35} (Freq: {stable_scores[f]:.3f})")

if __name__ == "__main__":
    main()
