import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

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

def select_features_for_data(X, y, label, mi_k, enet_k, top_k, n_boot, min_freq):
    print(f"\n--- Running feature selection for: {label} (X shape: {X.shape}) ---")
    print(f"Parameters: mi_k={mi_k}, enet_k={enet_k}, top_k={top_k}, n_boot={n_boot}, min_freq={min_freq}")
    
    # Identify bypass features to prevent Mutual Information starvation
    bypass_prefixes = ('J_', 'K_', 'D_', 'G_')
    bypass_exact = {'longitude', 'latitude', 'elev', 'slope', 'aspect', 'DOY', 'precip_mm', 'sin_year', 'cos_year'}
    
    bypass_cols = [
        c for c in X.columns 
        if c.startswith(bypass_prefixes) or 'year' in c or c in bypass_exact
    ]
    ts_cols = [c for c in X.columns if c not in bypass_cols]
    
    # 1. MI on dynamic time-series features only
    mi_out = select_mi(X[ts_cols], y, k=mi_k)
    mi_feats = mi_out["selected"]
    print(f"MI stage done: selected {len(mi_feats)} TS features")
    
    # Combine selected TS features with bypass features
    enet_candidate_feats = list(set(mi_feats + bypass_cols))
    # Double check they exist in X
    enet_candidate_feats = [f for f in enet_candidate_feats if f in X.columns]
    print(f"Candidate pool for ElasticNet: {len(enet_candidate_feats)} features")
    
    # 2. ElasticNet
    X_mi = X[enet_candidate_feats]
    enet_out = select_elasticnet(X_mi, y, k=enet_k)
    enet_feats = enet_out["selected"]
    print(f"ElasticNet stage done: selected {len(enet_feats)} features")
    
    # 3. Stability Selection
    stab_out = stability_bootstrap_elasticnet(
        X_mi,
        y,
        n_boot=n_boot,
        sample_frac=0.8,
        min_freq=min_freq,
        top_k=top_k,
        random_state=42,
        enet_k=enet_k,
        enet_kwargs={},
    )
    selected = stab_out["selected"]
    print(f"Stability stage done: selected {len(selected)} features")
    return selected

def main():
    config_path = "notebooks/experiment/derived_8.2-feature-selection/config.yaml"
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
    
    # Parse parameters from config.yaml
    sel_cfg = cfg.get("selection", {})
    top_k = int(sel_cfg.get("top_k", 40))
    n_boot = int(sel_cfg.get("stability_n_boot", 10))
    stages = list(sel_cfg.get("stages", []) or [])
    
    mi_k = 300
    enet_k = 60
    min_freq = 0.6
    
    for st in stages:
        kind = str(st.get("kind", "")).lower()
        if kind == "mi":
            mi_k = int(st.get("k", mi_k))
        elif kind == "elasticnet":
            enet_k = int(st.get("k", enet_k))
        elif kind == "stability":
            min_freq = float(st.get("min_freq", min_freq))
            
    # Run Overall Selection (only for a single global model)
    overall_feats = select_features_for_data(X_tr, y_tr, "Overall", mi_k, enet_k, top_k, n_boot, min_freq)
    
    # Write the output file
    dest_path = "data/splits/derived_8.2/dataset_metadata.py"
    print(f"\nWriting selected features and metadata to {dest_path}...")
    
    content = f"""# Metadata and configuration constants for derived_8.2 splits
# Note: Regime thresholds T1 and T2 are not used for regime separation since derived_8.2 focuses on a single global model.

# Selected features for overall global model
OVERALL_SELECTED_FEATURES_V1 = {repr(list(overall_feats))}
# Note: V1 is just for versioning, its generated the same wat as other OVERALL_SELECTED_FEATURES.
#       But DO NOT alias it to OVERALL_SELECTED_FEATURES as I want to compare different feature sets without changing models used the older feature sets.
"""
    
    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(content)
        
    print("Done!")

if __name__ == "__main__":
    main()
