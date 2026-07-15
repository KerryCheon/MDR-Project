import os
import sys
import json
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set up project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Load dataset_metadata to get OVERALL_SELECTED_FEATURES_V3
metadata_path = project_root / "data" / "splits" / "derived_8.2" / "dataset_metadata.py"
spec = importlib.util.spec_from_file_location("dataset_metadata", metadata_path)
dataset_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_metadata)
features_v3 = dataset_metadata.OVERALL_SELECTED_FEATURES_V3

from Modeling.Utils.config import load_config
from Modeling.Src.soilmoist_fl.Data.load import load_splits
from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
from Modeling.Src.soilmoist_fl.cli import select_features, DEFAULT_RUNS_DIR
from Modeling.Src.soilmoist_fl.Tracking.artifacts import ensure_run_dir
from Modeling.Src.soilmoist_fl.Tracking.registry import register_run

# Quantile binner helper
class QuantileBinner:
    def __init__(self, K):
        self.K = K
        self.thresholds = []
        
    def fit(self, series):
        val = series.fillna(series.mean())
        self.thresholds = [val.quantile(i / self.K) for i in range(1, self.K)]
        
    def predict(self, series):
        val = series.fillna(series.mean())
        if self.K == 2:
            return np.where(val < self.thresholds[0], 0, 1)
        elif self.K == 3:
            return np.where(val < self.thresholds[0], 0, np.where(val < self.thresholds[1], 1, 2))
        else:
            raise NotImplementedError("Only K=2 and K=3 implemented")

# KMeans clustering helper
class KMeansClusterer:
    def __init__(self, cols, K):
        self.cols = cols
        self.K = K
        self.means = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        
    def fit(self, df):
        X = df[self.cols].copy()
        self.means = X.mean()
        X = X.fillna(self.means)
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans.fit(X_scaled)
        
    def predict(self, df):
        X = df[self.cols].copy()
        X = X.fillna(self.means)
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

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
    # Get original DataFrames to extract month and apply clustering rules
    # In load_splits, fold.train is a DataFrame, let's parse date/month on them
    for df in [fold.train, fold.val, fold.test]:
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month.astype(int)
        df["year"] = df["date"].dt.year.astype(float)
        
    X_tr, y_tr, _, _ = preprocess_split(fold.train, target, drop_cols=drop_cols)
    X_va, y_va, _, _ = preprocess_split(fold.val, target, drop_cols=drop_cols)
    X_te, y_te, _, _ = preprocess_split(fold.test, target, drop_cols=drop_cols)
    
    # Define clustering columns
    cols_dyn = ["SMAP_sm_pm_interp_lag1", "G_API", "LST_modis"]
    
    # Define gating strategies
    gating_strategies = {
        "Univariate_G_API_k2": {
            "K": 2,
            "type": "quantile",
            "col": "G_API"
        },
        "Clustering_Dynamic_k2": {
            "K": 2,
            "type": "kmeans",
            "cols": cols_dyn
        },
        "Seasonal_Binary_k2": {
            "K": 2,
            "type": "seasonal_binary"
        },
        "Clustering_V3_Full_k3": {
            "K": 3,
            "type": "kmeans",
            "cols": features_v3
        },
        "Clustering_Dynamic_k3": {
            "K": 3,
            "type": "kmeans",
            "cols": cols_dyn
        },
        "Univariate_G_API_k3": {
            "K": 3,
            "type": "quantile",
            "col": "G_API"
        }
    }
    
    selected_features_by_strategy = {}
    
    for strat_name, strat_info in gating_strategies.items():
        K = strat_info["K"]
        stype = strat_info["type"]
        print(f"\n=== Fitting strategy: {strat_name} ===")
        
        # Determine labels for train, val, test
        if stype == "quantile":
            binner = QuantileBinner(K)
            binner.fit(fold.train[strat_info["col"]])
            labels_tr = binner.predict(fold.train[strat_info["col"]])
            labels_va = binner.predict(fold.val[strat_info["col"]])
            labels_te = binner.predict(fold.test[strat_info["col"]])
        elif stype == "seasonal_binary":
            cond_k2_tr = [
                fold.train['month'].isin([5, 6, 7, 8, 9, 10]),
                fold.train['month'].isin([11, 12, 1, 2, 3, 4])
            ]
            labels_tr = np.select(cond_k2_tr, [0, 1], default=0)
            
            cond_k2_va = [
                fold.val['month'].isin([5, 6, 7, 8, 9, 10]),
                fold.val['month'].isin([11, 12, 1, 2, 3, 4])
            ]
            labels_va = np.select(cond_k2_va, [0, 1], default=0)
            
            cond_k2_te = [
                fold.test['month'].isin([5, 6, 7, 8, 9, 10]),
                fold.test['month'].isin([11, 12, 1, 2, 3, 4])
            ]
            labels_te = np.select(cond_k2_te, [0, 1], default=0)
        elif stype == "kmeans":
            clusterer = KMeansClusterer(strat_info["cols"], K)
            clusterer.fit(fold.train)
            labels_tr = clusterer.predict(fold.train)
            labels_va = clusterer.predict(fold.val)
            labels_te = clusterer.predict(fold.test)
            
        selected_features_by_strategy[strat_name] = {}
        
        # Run selection for each cluster
        for c in range(K):
            print(f"\n--------------------------------------------------")
            print(f"Starting Selection for {strat_name} | Cluster {c}")
            print(f"--------------------------------------------------")
            
            mask_tr = (labels_tr == c)
            mask_va = (labels_va == c)
            mask_te = (labels_te == c)
            
            n_tr = mask_tr.sum()
            n_va = mask_va.sum() if len(mask_va) > 0 else 0
            n_te = mask_te.sum() if len(mask_te) > 0 else 0
            print(f"Samples: Train={n_tr}, Val={n_va}, Test={n_te}")
            
            # Sub-sample datasets
            X_tr_r = X_tr[mask_tr]
            y_tr_r = y_tr[mask_tr]
            
            X_va_r = X_va[mask_va] if n_va > 0 else None
            y_va_r = y_va[mask_va] if n_va > 0 else None
            
            X_te_r = X_te[mask_te] if n_te > 0 else None
            y_te_r = y_te[mask_te] if n_te > 0 else None
            
            # If training subset is too small (e.g. less than 10 samples), skip and use all features
            if n_tr < 10:
                print(f"[WARNING] Cluster {c} has only {n_tr} samples. Skipping feature selection and using all features.")
                selected_features_by_strategy[strat_name][str(c)] = list(X_tr.columns)
                continue
                
            # Determine runs dir
            run_id = f"derived_8.2_cluster_{strat_name.lower()}_c{c}"
            run_dir, run_id = ensure_run_dir(DEFAULT_RUNS_DIR, run_id=run_id)
            register_run(run_dir, run_id=run_id, meta={"config": str(config_path), "strategy": strat_name, "cluster": c})
            
            # Programmatically adjust config: keep stability_n_boot=100 and models=[]
            cfg_no_models = dict(cfg)
            if "selection" in cfg_no_models:
                selection_cfg = dict(cfg_no_models["selection"])
                selection_cfg["stability_n_boot"] = 100
                cfg_no_models["selection"] = selection_cfg
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
            
            # Handle fallback if 0 features selected
            if len(selected_feats) == 0:
                print(f"\n[WARNING] 0 features selected for Cluster {c}. Retrying with min_freq=0.0...")
                cfg_fallback = dict(cfg_no_models)
                selection_fallback = dict(cfg_fallback.get("selection", {}))
                stages_fallback = []
                for stage in selection_fallback.get("stages", []):
                    if stage.get("kind") == "stability":
                        stage_copy = dict(stage)
                        stage_copy["min_freq"] = 0.0
                        stages_fallback.append(stage_copy)
                    else:
                        stages_fallback.append(stage)
                selection_fallback["stages"] = stages_fallback
                cfg_fallback["selection"] = selection_fallback
                
                res = select_features(
                    X_train=X_tr_r,
                    y_train=y_tr_r,
                    X_val=X_va_r,
                    y_val=y_va_r,
                    X_test=X_te_r,
                    y_test=y_te_r,
                    config=cfg_fallback,
                    run_dir=run_dir,
                    run_id=run_id,
                    verbose=True
                )
                selected_feats = res["selected_features"]
                
            selected_features_by_strategy[strat_name][str(c)] = list(selected_feats)
            print(f"Finished {strat_name} Cluster {c}. Selected {len(selected_feats)} features: {selected_feats[:5]}")
            
    # Save the selected features by strategy to a json file
    out_file = project_root / "notebooks" / "experiment" / "derived_8.2-eval-3.0" / "cluster_features.json"
    with open(out_file, "w") as f:
        json.dump(selected_features_by_strategy, f, indent=4)
    print(f"\nSuccessfully wrote all selected features to {out_file}!")

if __name__ == "__main__":
    main()
