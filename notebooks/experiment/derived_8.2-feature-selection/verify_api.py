import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from Modeling.Src.soilmoist_fl import select_features

def main():
    print("Generating synthetic data for API validation...")
    np.random.seed(42)
    n_samples = 100
    
    # Generate random features
    data = {
        # Spatial/bypass features
        "J_spatial_1": np.random.randn(n_samples),
        "longitude": np.random.randn(n_samples),
        "latitude": np.random.randn(n_samples),
        # Dynamic TS features
        "ts_feat_1": np.random.randn(n_samples),
        "ts_feat_2": np.random.randn(n_samples),
        "ts_feat_3": np.random.randn(n_samples),
        "ts_feat_4": np.random.randn(n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Target value depends strongly on J_spatial_1 and ts_feat_1
    y = df["J_spatial_1"] * 2.0 + df["ts_feat_1"] * 1.5 + np.random.randn(n_samples) * 0.1
    
    print("Calling select_features API with in-memory DataFrames...")
    # Define custom stages to run quickly (correlation and elasticnet, no bootstrap to save time)
    stages = [
        {"kind": "correlation", "threshold": 0.95},
        {"kind": "elasticnet", "k": 3}
    ]
    
    config = {
        "selection": {
            "top_k": 3,
            "stages": stages
        },
        "logging": {
            "level": "INFO",
            "console": True,
            "log_to_file": False
        }
    }
    
    res = select_features(
        X_train=df,
        y_train=y,
        config=config,
        verbose=True
    )
    
    selected = res["selected_features"]
    print("\nVerification successful!")
    print(f"Selected features: {selected}")
    
    # Assertions
    assert len(selected) > 0, "No features selected!"
    assert "J_spatial_1" in selected or "ts_feat_1" in selected, "Expected strong features to be selected!"
    print("All assertions passed!")

if __name__ == "__main__":
    main()
