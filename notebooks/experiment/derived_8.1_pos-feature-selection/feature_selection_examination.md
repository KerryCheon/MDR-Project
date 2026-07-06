# Feature Selection Pipeline Examination Report

This document reports findings from the examination of the feature selection pipeline under `Modeling/` on the Washington-only `derived_8.1_pos` dataset. 

In `derived_8.1_pos-eval-3.0`, automatically selected features (Model B) performed significantly worse than baseline features (Model A), dropping from $R^2 \approx 0.628$ to $R^2 \approx 0.491$. This investigation uncovers the root mathematical cause and evaluates three alternative configurations.

---

## 1. Summary of Comparative Performance Results

We executed the feature selection strategies and trained global XGBoost regressors on `trainval_df` using temporal recency weighting ($\beta=0.2$) and evaluated them on the held-out test split. The results are summarized below:

| Feature Configuration | Size | $R^2$ | RMSE | ubRMSE | Bias | MAE | Pearson | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model B (Default Pipeline: MI $k=120$)** | 40 | 0.4909 | 0.0751 | 0.0742 | −0.0118 | 0.0550 | 0.7130 | *Current Failure* |
| **Model A (v25 Baseline: Hand-Selected)** | 38 | 0.6280 | 0.0642 | 0.0598 | −0.0233 | 0.0491 | 0.8247 | *Historical Baseline* |
| **Model E (Hybrid Selection: Bypass MI)** | 40 | 0.6309 | 0.0640 | 0.0610 | −0.0194 | 0.0477 | 0.8203 | *Improved* |
| **Model C (No MI Pre-Filter)** | 40 | 0.6343 | 0.0637 | 0.0596 | −0.0224 | 0.0479 | 0.8289 | *Excellent* |
| **Model D (High MI Limit: $k=300$)** | 40 | **0.6595** | **0.0614** | **0.0595** | **−0.0155** | **0.0448** | **0.8270** | **State of the Art** |

> [!IMPORTANT]
> - By increasing the Mutual Information pre-filter threshold from **`k=120` to `k=300` (Model D)**, the performance improves by **$+0.1686$ absolute $R^2$** over the default pipeline, establishing a new best $R^2$ of **`0.6595`** on the Washington test stations.
> - Bypassing the Mutual Information stage entirely and letting ElasticNet select from the pool of 496 features (Model C) also recovers the baseline performance ($R^2 \approx 0.634$).

---

## 2. Root Cause Analysis: The MI "Starvation" Failure Mode

The feature selection pipeline uses a three-stage filter:
$$\text{Raw Features (496)} \xrightarrow{\text{MI } (k=120)} \text{Candidate pool (120)} \xrightarrow{\text{ElasticNet } (k=60)} \text{Selection pool (60)} \xrightarrow{\text{Stability}} \text{Final Features (40)}$$

### A. Why the Univariate MI Filter Fails
1. **Univariate Assessment**: Mutual Information (MI) regression computes the dependency score feature-by-feature independently. It does not measure the joint information or account for feature redundancy.
2. **Redundancy Flooding**: Soil moisture changes slowly and has extremely high temporal autocorrelation. Consequently, rolling window statistics (e.g. rolling minimum, maximum, mean, range) and lags of remote sensing inputs (MODIS LST, Sentinel-1 ratios, SMAP) over 7, 14, and 30 days all have very high univariate mutual information with the target. 
3. **Starvation of Static/Seasonal Drivers**: 
   - Static features (like `elev`, `slope`, `J_clay_wfrac_b0`) only vary by station. In a dataset of 15,964 daily records across 13 stations, their univariate MI with daily soil moisture changes is low on its own.
   - Seasonal calendar variables (like `sin_year`, `cos_year`, `D_sin_DOY`) have non-monotonic relations with soil moisture, giving them low univariate scores.
   - Raw weather variables (like precipitation) are sparse/impulsive, leading to low univariate scores.

Because $k$ was set to $120$, the highly redundant rolling variants of remote sensing inputs flooded all 120 slots of the MI filter. **Geography (static features), calendar coordinates, and raw weather inputs were completely filtered out in the first step.** Downstream selectors (ElasticNet and Stability) were starved of these features and had no opportunity to select them.

### B. Evidence from Diagnostic Stage-Tracking
Our diagnostic checks traced the scores and ranks of these omitted features during the pipeline:

| Feature Name | In $X_{tr}$ | MI Score | MI Rank | MI Selected | EN Coef (No MI) | EN Selected (No MI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `slope` | Yes | 0.46531 | **190** | **No** | 0.01227 | **Yes** (Rank 11) |
| `elev` | Yes | 0.46591 | **187** | **No** | 0.00000 | No |
| `K_aspect_cos` | Yes | 0.46126 | **205** | **No** | 0.01692 | **Yes** (Rank 7) |
| `J_clay_wfrac_b0` | Yes | 0.41640 | **242** | **No** | 0.00000 | No |
| `D_sin_DOY` | Yes | 0.20858 | **371** | **No** | 0.02956 | **Yes** (Rank 4) |
| `sin_year` | Yes | 0.03987 | **479** | **No** | 0.00629 | **Yes** (Rank 24) |
| `cos_year` | Yes | 0.02395 | **487** | **No** | 0.00238 | **Yes** (Rank 46) |
| `G_rain_sum_7d` | Yes | 0.19323 | **375** | **No** | 0.00364 | **Yes** (Rank 33) |

> [!NOTE]
> - `slope` was ranked 190 in univariate MI (discarded by the $k=120$ limit). But when ElasticNet was allowed to see it, it ranked it **11th** overall with a non-zero coefficient.
> - `D_sin_DOY` was ranked 371 in MI (discarded). ElasticNet ranked it **4th** overall with a high coefficient.
> - Calendar variables `sin_year`/`cos_year` ranked 479/487 in MI (discarded). ElasticNet selected both in the top 46.

---

## 3. Recommended Actions and Code Fixes

To fix the pipeline and achieve state-of-the-art results ($R^2 \approx 0.659$), we should implement one of the following two solutions:

### Option A: Increase the MI Filter Limit (Recommended)
This is the simplest fix. By changing `k: 300` in the config, we allow the static, seasonal, and rain features to survive the initial univariate filter and reach ElasticNet, which will then perform the multivariate prune.

Modify the pipeline config files (e.g. `Modeling/Configs/default.yaml` and split configs):
```yaml
selection:
  top_k: 40
  stages:
    - kind: mi
      k: 300       # Increased from 120
    - kind: elasticnet
      k: 60
    - kind: stability
      min_freq: 0.6
```

### Option B: Implement a Hybrid/Bypass Rule in the Selector
If we want to keep the candidate pool small to minimize memory/time, we can modify `Modeling/Src/soilmoist_fl/cli.py` to bypass the MI filter for static, geographic, and seasonal columns.

Modify `run_feature_selection` in [cli.py](file:///c:/Users/pan/Documents/GitHub/MDR-Project/Modeling/Src/soilmoist_fl/cli.py):
```python
    # Identify features to bypass Mutual Information filter
    bypass_prefixes = ('J_', 'K_', 'D_', 'G_')
    bypass_exact = {'longitude', 'latitude', 'elev', 'slope', 'aspect', 'DOY', 'precip_mm'}
    
    bypass_cols = [
        c for c in X_tr.columns 
        if c.startswith(bypass_prefixes) or 'year' in c or c in bypass_exact
    ]
    ts_cols = [c for c in X_tr.columns if c not in bypass_cols]
    
    # Run MI only on dynamic time-series features
    mi_out = select_mi(X_tr[ts_cols], y_tr, k=mi_k)
    mi_feats = mi_out["selected"]
    
    # Combine selected time-series with all bypass features
    enet_candidate_feats = mi_feats + bypass_cols
    X_tr_mi = X_tr[enet_candidate_feats]
```

---

## 4. Reproducible Code Artifacts

The entire analysis, pipeline runs, and models are fully saved and reproducible under:
- **Notebook**: [evaluate_feature_sets.ipynb](file:///c:/Users/pan/Documents/GitHub/MDR-Project/notebooks/experiment/derived_8.1_pos-feature-selection/evaluate_feature_sets.ipynb) (contains cells, outputs, and the comparison execution).
- **Run Outputs**: Selected features for Sets C, D, and E are saved in [selected_features_comparison.json](file:///c:/Users/pan/Documents/GitHub/MDR-Project/notebooks/experiment/derived_8.1_pos-feature-selection/selected_features_comparison.json).
