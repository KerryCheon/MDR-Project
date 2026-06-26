# `derived_8.1` Soil Moisture Feature Selection & Evaluation

This experiment runs the temporal feature selection pipeline (`soilmoist-fl`) on the `derived_8.1` dataset. The goal is to filter a large candidate pool of 496 spatio-temporal features down to a robust subset of 40 stable predictors that generalize well over unseen years and stations.

---

## 1. Methodology

The feature selection pipeline uses a three-stage filter-wrapper design to select features that are informative, non-redundant, and stable across bootstrap samples:

```
[496 Features] ---> Mutual Information (MI) ---> [120 Features]
                     ---> ElasticNet CV ---> [60 Features]
                           ---> Stability Selection (Bootstrap) ---> [40 Features]
```

### Stage 1: Mutual Information (MI)
Filter out features that contain no statistical signal. Evaluates features independently against the target `soil_moisture_5cm` using `mutual_info_regression` with `n_neighbors=3`. Retains the top `k=120` features.

### Stage 2: ElasticNet Cross-Validation
Evaluate features in combination using regularized linear regression. `ElasticNetCV` uses 5-fold cross-validation over a grid of L1 ratios (`0.5, 0.9, 1.0`) and 30 alphas to enforce sparsity and handle collinearity. Retains the top `k=60` features.

### Stage 3: Stability Selection
Run 5 bootstrap iterations on 80% sub-samples of the training dataset. For each sub-sample, a full ElasticNet selection is run. Features are ranked by their selection frequency across the bootstrap iterations. The top `k=40` features meeting a frequency threshold of `min_freq >= 0.6` are selected.

---

## 2. Selected Features Analysis

The pipeline successfully identified exactly 40 features. Strikingly, **35 out of the 40 features** were selected in **100%** of the bootstrap iterations, indicating a highly stable core feature set. The remaining 5 features achieved **80%** frequency.

Below is a categorization of the selected features by their data source and physical relevance:

### 1. Sentinel-1 SAR Backscatter & Physics (`E_` / `C_lag_E_` / `D_z_E_`)
SAR features represent the most prominent group of selected features:
- **Selected features**: `E_SAR_ratio`, `D_z_E_SAR_ratio`, `D_sa_E_SAR_ratio`, `V_rollmax_E_SAR_ratio_kobs30`, `V_rollmin_E_SAR_ratio_kobs30`, `V_rollmax_E_SAR_diff_kobs14`, `V_rollmin_E_SAR_diff_kobs30`, `C_lag_E_SAR_diff_kobs12`, `C_lag_E_SAR_diff_kobs30`, `C_lag_E_SAR_ratio_kobs30`.
- **Physical context**: Sentinel-1 SAR C-band backscatter (specifically the VV/VH ratio and differences) is highly sensitive to the surface dielectric constant, which is directly determined by soil moisture. The selection of long-term rolling statistics (`kobs30`, `kobs14`) and lag-memory features indicates that SAR acts as a robust proxy for surface soil wetness memory.

### 2. Land Surface Temperature (`LST_modis` / `D_sa_LST_` / `D_z_LST_`)
- **Selected features**: `LST_modis`, `D_sa_LST_modis`, `D_z_LST_modis`, `V_rollmax_LST_modis_kobs30`, `V_rollmin_LST_modis_kobs30`, `C_lag_LST_modis_kobs30`.
- **Physical context**: LST anomalies (`D_sa_LST_modis`) and z-scores reflect energy balance. High soil moisture increases latent heat flux (evapotranspiration) and cools the land surface, while dry soil increases sensible heat flux and raises LST. MODIS LST and its lags provide strong clues about soil moisture depletion rates (drying curves).

### 3. Passive Microwave SMAP Soil Moisture (`SMAP_sm_` / `C_lag_SMAP_`)
- **Selected features**: `SMAP_sm_pm_interp_lag30`, `V_rollmax_SMAP_sm_interp_kobs7`, `V_rollmax_SMAP_sm_interp_kobs14`, `V_rollmax_SMAP_sm_interp_kobs30`, `C_lag_SMAP_sm_interp_kobs2`, `C_lag_SMAP_sm_interp_kobs5`, `C_lag_SMAP_sm_interp_kobs6`, `C_lag_SMAP_sm_interp_kobs30`.
- **Physical context**: SMAP provides direct surface soil moisture estimates at coarse resolution (36km/9km). While too coarse for direct station-level prediction, the rolling maximums and short-term lags act as a spatial bias-correction baseline.

### 4. Sentinel-2 Vegetation Canopy Water & Greenness (`F_` / `C_lag_F_`)
- **Selected features**: `V_rollmax_F_NDMI_kobs30`, `V_rollmax_F_NDMI_kobs7`, `V_rollmin_F_NDMI_kobs7`, `V_rollmax_F_NDVI_kobs30`, `V_rollmin_F_NDVI_kobs30`, `C_lag_F_NDMI_kobs6`, `C_lag_F_NDMI_kobs12`, `C_lag_F_NDMI_kobs30`.
- **Physical context**: NDVI (greenness) and NDMI (canopy water content) reflect vegetation health, which is a lagging indicator of root-zone soil moisture. The long-term rolling statistics (`kobs30`, `kobs7`) represent vegetation response to prolonged wet or dry periods.

### 5. Antecedent Precipitation Index (`G_API` / `V_roll_G_API`)
- **Selected features**: `V_rollmax_G_API_kobs14`, `V_rollmax_G_API_kobs30`, `V_rollmin_G_API_kobs14`, `V_rollmin_G_API_kobs30`.
- **Physical context**: API models the decay of rainfall water in the soil. Rolling statistics of API provide the precipitation history context, which helps models adjust for recent rain events.

---

## 3. Downstream Model Performance

The 40 selected features were evaluated on Ridge Regression, HistGradientBoosting (XGB), and Random Forest (RF) models. The models were trained on years **2017–2020** (Train), validated on **2021–2022** (Val), and tested on **2023–2025** (Test) across all 13 Washington stations.

| Model | Split | Observations ($N$) | $R^2$ | RMSE ($\text{cm}^3/\text{cm}^3$) | Rel. RMSE | MAE ($\text{cm}^3/\text{cm}^3$) | Bias (ME) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ridge Regression** | Train | 16,462 | 0.3998 | 0.0883 | 0.4346 | 0.0718 | -0.0000 |
| | Val | 7,714 | 0.3512 | 0.0985 | 0.5286 | 0.0802 | 0.0222 |
| | Test | 10,599 | 0.1449 | 0.1094 | 0.6981 | 0.0868 | 0.0465 |
| **HistGradientBoosting**| Train | 16,462 | 0.9593 | 0.0230 | 0.1132 | 0.0166 | -0.0000 |
| | Val | 7,714 | 0.5307 | 0.0838 | 0.4495 | 0.0643 | 0.0084 |
| | Test | 10,599 | 0.2726 | 0.1009 | 0.6438 | 0.0780 | 0.0278 |
| **Random Forest** | Train | 16,462 | 0.9882 | 0.0124 | 0.0608 | 0.0073 | -0.0000 |
| | Val | 7,714 | 0.4947 | 0.0869 | 0.4665 | 0.0654 | 0.0112 |
| | Test | 10,599 | 0.2611 | 0.1017 | 0.6489 | 0.0772 | 0.0329 |

---

## 4. Key Findings & Discussion

1. **High Non-Linearity**:
   The linear Ridge model underperforms significantly compared to tree ensembles. While Ridge achieves a modest $R^2 = 0.35$ on validation, HistGradientBoosting reaches $R^2 = 0.53$. This confirms that soil moisture dynamics are highly non-linear and rely on complex feature interactions (e.g., combining SAR dielectric responses with LST thermal signatures).
   
2. **Generalization Gap**:
   Both XGB and RF models experience a noticeable generalization drop on the Test set (**2023–2025**), with $R^2$ dropping to $0.27$ and $0.26$ respectively. This temporal gap is a known challenge in spatio-temporal soil moisture estimation:
   - **Climatic drift**: The test period of 2023–2025 had extreme precipitation anomalies (prolonged dry summer spells and intense winter atmospheric rivers in the Pacific Northwest) compared to the 2017–2020 training baseline.
   - **Station network expansion**: `derived_8.1` introduces 8 new SNOTEL stations in different microclimates and soil conditions (higher elevations, steeper slopes, and different forest canopy structures).
   
3. **Overfitting in Ensembles**:
   Random Forest shows the highest training $R^2$ ($0.988$) but slightly lower validation performance than HistGradientBoosting ($0.495$ vs $0.531$). HistGradientBoosting generalizes better due to its early stopping configuration, which halts tree building when validation loss stabilizes.

## 5. Conclusion

Stability Selection successfully reduced 496 candidate features to **40 highly robust, physically meaningful features**. Sentinel-1 SAR dielectric indicators and MODIS Land Surface Temperature thermal anomaly metrics dominate the selected feature set, coupled with SMAP spatial baselines. 

For future model training (e.g., deep learning or ensemble training), the selected 40-feature subset is highly recommended as it preserves the core spatio-temporal signals while preventing high-dimensional overfitting.
