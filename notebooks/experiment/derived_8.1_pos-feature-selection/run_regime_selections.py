import os
import sys
import argparse
from pathlib import Path
import pandas as pd

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import dataset_metadata to get T1 and T2
sys.path.append(str(project_root / "data" / "splits" / "derived_8.1_pos"))
from dataset_metadata import T1, T2

from Modeling.Utils.config import load_config
from Modeling.Src.soilmoist_fl.Data.load import load_splits
from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
from Modeling.Src.soilmoist_fl.Selectors.mi import select_mi
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet
from Modeling.Src.soilmoist_fl.Selectors.stability import stability_bootstrap_elasticnet

def select_features_for_data(X, y, label):
    print(f"\n--- Running feature selection for: {label} (X shape: {X.shape}) ---")
    
    # 1. MI
    mi_out = select_mi(X, y, k=120)
    mi_feats = mi_out["selected"]
    print(f"MI stage done: selected {len(mi_feats)} features")
    
    # 2. ElasticNet
    X_mi = X[mi_feats]
    enet_out = select_elasticnet(X_mi, y, k=60)
    enet_feats = enet_out["selected"]
    print(f"ElasticNet stage done: selected {len(enet_feats)} features")
    
    # 3. Stability Selection
    stab_out = stability_bootstrap_elasticnet(
        X_mi,
        y,
        n_boot=5,
        sample_frac=0.8,
        min_freq=0.6,
        top_k=40,
        random_state=42,
        enet_k=60,
        enet_kwargs={},
    )
    selected = stab_out["selected"]
    print(f"Stability stage done: selected {len(selected)} features")
    return selected

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
    
    # Run Overall Selection
    overall_feats = select_features_for_data(X_tr, y_tr, "Overall")
    
    # Split into Dry, Transition, Wet regimes using thresholds from dataset_metadata
    mask_dry = y_tr < T1
    mask_trans = (y_tr >= T1) & (y_tr < T2)
    mask_wet = y_tr >= T2
    
    dry_feats = select_features_for_data(X_tr[mask_dry], y_tr[mask_dry], "Dry Regime")
    trans_feats = select_features_for_data(X_tr[mask_trans], y_tr[mask_trans], "Transition Regime")
    wet_feats = select_features_for_data(X_tr[mask_wet], y_tr[mask_wet], "Wet Regime")
    
    print("\n--- Summary of Selected Features ---")
    print(f"Overall (40): {list(overall_feats)[:5]}...")
    print(f"Dry (40): {list(dry_feats)[:5]}...")
    print(f"Transition (40): {list(trans_feats)[:5]}...")
    print(f"Wet (40): {list(wet_feats)[:5]}...")
    
    # Print the code block to be copied/pasted into dataset_metadata.py
    print("\n" + "="*60)
    print("CODE BLOCK FOR dataset_metadata.py")
    print("="*60)
    print(f"""# Selected features for each regime and overall
OVERALL_SELECTED_FEATURES = {repr(list(overall_feats))}

DRY_SELECTED_FEATURES = {repr(list(dry_feats))}

TRANSITION_SELECTED_FEATURES = {repr(list(trans_feats))}

WET_SELECTED_FEATURES = {repr(list(wet_feats))}""")
    print("="*60)

if __name__ == "__main__":
    main()
