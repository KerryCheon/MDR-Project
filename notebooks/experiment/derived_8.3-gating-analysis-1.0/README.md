# derived_8.3-gating-analysis-1.0

Diagnostic analysis and parameter export for $K=2$ gating strategies on the `derived_8.3` dataset split.

## Overview & Purpose

This experiment evaluates unsupervised clustering and univariate gating strategies on Washington state soil moisture observations (`derived_8.3` split) using `OVERALL_SELECTED_FEATURES_V0` (50 features).

Key objectives:
1. **$K=2$ Regimes**: Focus evaluation on 2-group regime partitions.
2. **Feature Set**: Use `OVERALL_SELECTED_FEATURES_V0` loaded from `data/splits/derived_8.3/dataset_metadata.py`.
3. **Clustering Parameter Export**: Export fitted scaling, imputation, and centroid parameters to JSON and Joblib formats for direct loading by external models and experiments.

## Evaluated Gating Strategies ($K=2$)

Quantitative summary of the 4 evaluated $K=2$ gating strategies computed on `train.csv` (12,678 samples):

| Strategy Name | K | Group Sizes | Top 20 Divergence | Max Drift | Max Drift Feature |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Univariate_G_API` | 2 | 6339, 6339 | 0.1553 | 0.5233 | `V_ema_LST_modis_kobs30` |
| `Clustering_Dynamic` | 2 | 5723, 6955 | 0.1694 | 0.6669 | `V_ema_F_NDVI_kobs30` |
| `Seasonal_Binary` | 2 | 6526, 6152 | 0.1761 | 0.6650 | `V_ema_LST_modis_kobs30` |
| `Clustering_V0_Full` | 2 | 10031, 2647 | 0.1343 | 1.0089 | `SMAP_sm_pm_interp_rollmean30` |

### Key Takeaways
- **`Seasonal_Binary`** achieves the highest correlation divergence across top features (**0.1761**).
- **`Clustering_V0_Full`** exhibits the largest individual feature correlation drift (**1.0089** on `SMAP_sm_pm_interp_rollmean30`).
- **`Clustering_Dynamic`** creates balanced physical clusters while achieving strong top-20 feature divergence (**0.1694**).

## Unsupervised Clustering Quality Metrics ($K=2$)

Diagnostic cohesion and separation metrics computed on standard scaled feature spaces:

| Strategy Name | Features Used | WSS Inertia | Avg Silhouette Score | Key Observation |
| :--- | :--- | :--- | :--- | :--- |
| `Clustering_Dynamic` | 3 dynamic features (`SMAP lag1`, `G_API`, `LST`) | 24,439.34 | **0.3465** | Well-separated clusters in lower-dimensional physical space |
| `Clustering_V0_Full` | 50 features (`OVERALL_SELECTED_FEATURES_V0`) | 510,845.29 | **0.2200** | High-dimensional feature space, higher inertia and boundary overlap |

## Exported Clustering Parameters

Fitted parameters on `train.csv` are exported in the experiment root directory:

| Artifact File | Format | Description |
| :--- | :--- | :--- |
| `clustering_params_dynamic_k2.json` | JSON | Parameter dict (`features`, `impute_means`, `scaler_mean`, `scaler_scale`, `cluster_centers`) for `Clustering_Dynamic` ($K=2$) |
| `clustering_params_dynamic_k2.joblib` | Joblib | Pre-fitted scikit-learn pipeline bundle (`scaler`, `kmeans`, `impute_means`, `features`) for `Clustering_Dynamic` ($K=2$) |
| `clustering_params_v0_full_k2.json` | JSON | Parameter dict (`features`, `impute_means`, `scaler_mean`, `scaler_scale`, `cluster_centers`) for `Clustering_V0_Full` ($K=2$) |
| `clustering_params_v0_full_k2.joblib` | Joblib | Pre-fitted scikit-learn pipeline bundle (`scaler`, `kmeans`, `impute_means`, `features`) for `Clustering_V0_Full` ($K=2$) |
| `clustering_params_k2_combined.json` | JSON | Combined registry containing parameter dictionaries for both $K=2$ strategies |

### Usage Example (JSON Loading)

```python
import json
import numpy as np
import pandas as pd

# Load exported parameters
with open("notebooks/experiment/derived_8.3-gating-analysis-1.0/clustering_params_v0_full_k2.json") as f:
    params = json.load(f)

# Predict cluster labels on new data
def predict_cluster_labels(df, params):
    feats = params["features"]
    imp_means = pd.Series(params["impute_means"])
    sc_mean = np.array(params["scaler_mean"])
    sc_scale = np.array(params["scaler_scale"])
    centroids = np.array(params["cluster_centers"])
    
    # 1. Fill missing values
    X_filled = df[feats].copy().fillna(imp_means).values
    
    # 2. Standardize
    X_scaled = (X_filled - sc_mean) / sc_scale
    
    # 3. Nearest centroid assignment
    dists = np.linalg.norm(X_scaled[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(dists, axis=1)

# Usage
labels = predict_cluster_labels(test_df, params)
```

### Usage Example (Joblib Loading)

```python
import joblib

bundle = joblib.load("notebooks/experiment/derived_8.3-gating-analysis-1.0/clustering_params_v0_full_k2.joblib")

# Impute, scale, and predict
X_filled = test_df[bundle["features"]].fillna(bundle["impute_means"])
X_scaled = bundle["scaler"].transform(X_filled)
labels = bundle["kmeans"].predict(X_scaled)
```

## Generated Visual Artifacts

- `gating_target_distributions_grid.png`: Soil moisture target KDE distributions across groups.
- `gating_correlation_drift_grid.png`: Heatmaps of feature-target correlation shifts across top 20 features.
- `clustering_dynamic_k2_pairplot.png`: Pairplot for dynamic clustering feature space.
- `clustering_v0_full_k2_pca.png`: 2D PCA projection for `Clustering_V0_Full` ($K=2$).
- `gating_geographic_distribution.png`: Washington state station maps colored by dominant gating regime.
- `clustering_metrics_comparison.png`: Elbow and Silhouette quality curves ($K=2\dots 6$).
- `clustering_silhouette_profiles.png`: Silhouette coefficient profiles for $K=2$.
- `clustering_tsne_projection.png`: Non-linear t-SNE 2D projections.
- `clustering_centroid_distances.png`: Centroid distance ratio ($d_{\text{own}} / d_{\text{other}}$) and boundary overlap maps.
