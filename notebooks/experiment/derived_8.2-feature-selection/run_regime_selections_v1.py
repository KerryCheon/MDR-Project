import os
import sys
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Import dataset_metadata to get thresholds
metadata_path = project_root / "data" / "splits" / "derived_8.2" / "dataset_metadata.py"
spec = importlib.util.spec_from_file_location("dataset_metadata", metadata_path)
dataset_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_metadata)

T_DRY = dataset_metadata.TERNARY_REGIME_DRY_THRESHOLD
T_WET = dataset_metadata.TERNARY_REGIME_WET_THRESHOLD
T_BINARY = dataset_metadata.BINARY_REGIME_THRESHOLD

from Modeling.Utils.config import load_config
from Modeling.Src.soilmoist_fl.Data.load import load_splits
from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
from Modeling.Src.soilmoist_fl.cli import select_features, DEFAULT_RUNS_DIR
from Modeling.Src.soilmoist_fl.Tracking.artifacts import ensure_run_dir
from Modeling.Src.soilmoist_fl.Tracking.registry import register_run

def update_metadata_file(metadata_path, selected_features_dict):
    print(f"\nUpdating {metadata_path}...")
    with open(metadata_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.splitlines()
    updated_vars = set()
    new_lines = []
    
    for line in lines:
        matched = False
        for var_name, feats in selected_features_dict.items():
            if line.startswith(f"{var_name} ="):
                new_lines.append(f"{var_name} = {repr(list(feats))}")
                updated_vars.add(var_name)
                matched = True
                break
        if not matched:
            new_lines.append(line)
            
    # For any variables not updated, append them to the end
    for var_name, feats in selected_features_dict.items():
        if var_name not in updated_vars:
            new_lines.append("")
            new_lines.append(f"# Selected features for {var_name}")
            new_lines.append(f"{var_name} = {repr(list(feats))}")
            
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")
    print("Metadata file updated successfully!")

def main():
    config_path = "notebooks/experiment/derived_8.2-feature-selection/config_v3.yaml"
    print(f"Loading V3 config: {config_path}")
    cfg = load_config(config_path)
    
    # Enable console log
    if "logging" not in cfg:
        cfg["logging"] = {}
    cfg["logging"]["console"] = True
    cfg["logging"]["level"] = "INFO"
    
    print("Loading data splits...")
    loaded = load_splits(cfg)
    fold = loaded.folds[0]
    
    data_cfg = cfg.get("data", {})
    target = data_cfg.get("target")
    id_cols = list(data_cfg.get("id_cols", []) or [])
    time_col = data_cfg.get("time_col")
    
    drop_cols = list(id_cols)
    if time_col:
        drop_cols.append(time_col)
        
    print("Preprocessing train/val/test splits...")
    X_tr, y_tr, _, _ = preprocess_split(fold.train, target, drop_cols=drop_cols)
    X_va, y_va, _, _ = preprocess_split(fold.val, target, drop_cols=drop_cols)
    X_te, y_te, _, _ = preprocess_split(fold.test, target, drop_cols=drop_cols)
    
    print(f"Total processed train samples: {len(y_tr)}")
    print(f"Ternary dry threshold: {T_DRY}")
    print(f"Ternary wet threshold: {T_WET}")
    print(f"Binary threshold: {T_BINARY}")
    
    regimes = {
        "TERNARY_REGIME_DRY_SELECTED_FEATURES_V1": {
            "mask_tr": y_tr < T_DRY,
            "mask_va": y_va < T_DRY,
            "mask_te": y_te < T_DRY,
            "desc": "Ternary Dry"
        },
        "TERNARY_REGIME_TRANSITION_SELECTED_FEATURES_V1": {
            "mask_tr": (y_tr >= T_DRY) & (y_tr < T_WET),
            "mask_va": (y_va >= T_DRY) & (y_va < T_WET),
            "mask_te": (y_te >= T_DRY) & (y_te < T_WET),
            "desc": "Ternary Transition"
        },
        "TERNARY_REGIME_WET_SELECTED_FEATURES_V1": {
            "mask_tr": y_tr >= T_WET,
            "mask_va": y_va >= T_WET,
            "mask_te": y_te >= T_WET,
            "desc": "Ternary Wet"
        },
        "BINARY_REGIME_DRY_SELECTED_FEATURES_V1": {
            "mask_tr": y_tr < T_BINARY,
            "mask_va": y_va < T_BINARY,
            "mask_te": y_te < T_BINARY,
            "desc": "Binary Dry"
        },
        "BINARY_REGIME_WET_SELECTED_FEATURES_V1": {
            "mask_tr": y_tr >= T_BINARY,
            "mask_va": y_va >= T_BINARY,
            "mask_te": y_te >= T_BINARY,
            "desc": "Binary Wet"
        }
    }
    
    selected_features_dict = {}
    
    for var_name, info in regimes.items():
        desc = info["desc"]
        mask_tr = info["mask_tr"]
        mask_va = info["mask_va"]
        mask_te = info["mask_te"]
        
        n_tr = mask_tr.sum()
        n_va = mask_va.sum() if len(mask_va) > 0 else 0
        n_te = mask_te.sum() if len(mask_te) > 0 else 0
        
        print(f"\n==================================================")
        print(f"Starting Selection for {desc} ({var_name})")
        print(f"Samples: Train={n_tr}, Val={n_va}, Test={n_te}")
        print(f"==================================================")
        
        X_tr_r = X_tr[mask_tr]
        y_tr_r = y_tr[mask_tr]
        
        X_va_r = X_va[mask_va] if n_va > 0 else None
        y_va_r = y_va[mask_va] if n_va > 0 else None
        
        X_te_r = X_te[mask_te] if n_te > 0 else None
        y_te_r = y_te[mask_te] if n_te > 0 else None
        
        # Determine runs dir
        run_id = f"derived_8.2_regime_{var_name.lower().replace('_selected_features_v1', '')}_v1"
        run_dir, run_id = ensure_run_dir(DEFAULT_RUNS_DIR, run_id=run_id)
        register_run(run_dir, run_id=run_id, meta={"config": str(config_path), "regime": desc})
        
        # We will try running with the configured config first, but without models to handle fallback gracefully
        cfg_no_models = dict(cfg)
        if "models" in cfg_no_models:
            cfg_no_models["models"] = []
            
        res = select_features(
            X_train=X_tr_r,
            y_train=y_tr_r,
            X_val=X_va_r,
            y_val=y_va_r,
            X_test=X_te_r,
            y_test=y_te_r,
            config=cfg_no_models,
            run_dir=run_dir,
            run_id=run_id,
            verbose=True
        )
        selected_feats = res["selected_features"]
        
        # If 0 features selected, retry with min_freq=0.0 to fallback to top-k by frequency
        use_fallback = False
        if len(selected_feats) == 0:
            print(f"\n[WARNING] 0 features selected for {desc} with min_freq=0.6. Retrying with min_freq=0.0 (top-k fallback)...")
            use_fallback = True
            
        # Now run the final selection with model evaluation enabled
        cfg_final = dict(cfg)
        if use_fallback:
            # We copy selection config and adjust min_freq
            cfg_final_selection = dict(cfg_final.get("selection", {}))
            stages_fallback = []
            for stage in cfg_final_selection.get("stages", []):
                if stage.get("kind") == "stability":
                    stage_copy = dict(stage)
                    stage_copy["min_freq"] = 0.0
                    stages_fallback.append(stage_copy)
                else:
                    stages_fallback.append(stage)
            cfg_final_selection["stages"] = stages_fallback
            cfg_final["selection"] = cfg_final_selection
            
        res = select_features(
            X_train=X_tr_r,
            y_train=y_tr_r,
            X_val=X_va_r,
            y_val=y_va_r,
            X_test=X_te_r,
            y_test=y_te_r,
            config=cfg_final,
            run_dir=run_dir,
            run_id=run_id,
            verbose=True
        )
        selected_feats = res["selected_features"]
        selected_features_dict[var_name] = selected_feats
        print(f"Finished {desc}. Selected {len(selected_feats)} features: {selected_feats[:5]}...")
        
    # Update dataset_metadata.py
    update_metadata_file(metadata_path, selected_features_dict)

if __name__ == "__main__":
    main()
