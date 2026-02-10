# Global SHAP Feature Importance + Correlation Report

## Quick Summary

- Dataset: `MDR/Temporal/Pipeline/data/splits/derived_new_updated/test_derived_updated.csv`
- Selected features (after mapping): 15
- Missing after mapping: 0
- Feature set 10 handling: `xgb`
- MI estimator: scikit-learn (symmetrized)
- Dataset selection: `max_coverage`

## Method (casual)

We grab the top-10 mean |SHAP| table for each of the 11 feature sets, then compute:

- Frequency: \( f = \frac{\#\text{sets with feature in top-10}}{11} \)
- Avg SHAP: \( \overline{\lvert SHAP \rvert} \) over the sets where the feature appears
- Global importance: \( G = f \times \overline{\lvert SHAP \rvert} \)

Feature selection for correlation uses `top 15 by global_importance`.
Correlations are computed on `MDR/Temporal/Pipeline/data/splits/derived_new_updated/test_derived_updated.csv` (rows: 1967, dropped NaNs: 862).

## Top Features by Global Importance

| Feature                      | Freq Count | Frequency | Avg SHAP | Global Importance |
| ---------------------------- | ---------- | --------- | -------- | ----------------- |
| DOY                          | 9          | 0.818     | 0.030560 | 0.025003          |
| air_temp_mean                | 4          | 0.364     | 0.038941 | 0.014160          |
| s1_vv                        | 7          | 0.636     | 0.007951 | 0.005060          |
| V_ema_LST_modis_kobs30       | 2          | 0.182     | 0.024560 | 0.004465          |
| rh_mean                      | 3          | 0.273     | 0.015523 | 0.004234          |
| C_smm_G_API_alpha0.85_n5     | 2          | 0.182     | 0.020411 | 0.003711          |
| precip_mm                    | 5          | 0.455     | 0.008061 | 0.003664          |
| s2_b8                        | 5          | 0.455     | 0.007492 | 0.003406          |
| s1_vh                        | 6          | 0.545     | 0.006226 | 0.003396          |
| API                          | 1          | 0.091     | 0.036759 | 0.003342          |
| slope                        | 3          | 0.273     | 0.011967 | 0.003264          |
| SAR_ratio                    | 5          | 0.455     | 0.006948 | 0.003158          |
| C_smm_LST_modis_alpha0.85_n5 | 2          | 0.182     | 0.017209 | 0.003129          |
| aspect                       | 3          | 0.273     | 0.010882 | 0.002968          |
| NDVI                         | 5          | 0.455     | 0.006030 | 0.002741          |

![](global_importance_top15.png)

## Consistent Features (>=6 of 11)

| Feature (>=6/11) |
| ---------------- |
| DOY              |
| s1_vv            |
| s1_vh            |

## Feature Column Mapping

Some features appear with a leading family letter (e.g., `NDVI` -> `F_NDVI`).
| Original | Resolved Column | Status | Candidates |
| --- | --- | --- | --- |
| DOY | DOY | exact | DOY |
| air_temp_mean | air_temp_mean | exact | air_temp_mean |
| s1_vv | s1_vv | exact | s1_vv |
| V_ema_LST_modis_kobs30 | V_ema_LST_modis_kobs30 | exact | V_ema_LST_modis_kobs30 |
| rh_mean | rh_mean | exact | rh_mean |
| C_smm_G_API_alpha0.85_n5 | C_smm_G_API_alpha0.85_n5 | exact | C_smm_G_API_alpha0.85_n5 |
| precip_mm | precip_mm | exact | precip_mm |
| s2_b8 | s2_b8 | exact | s2_b8 |
| s1_vh | s1_vh | exact | s1_vh |
| API | G_API | alias | G_API |
| slope | slope | exact | slope |
| SAR_ratio | SAR_ratio | exact | SAR_ratio |
| C_smm_LST_modis_alpha0.85_n5 | C_smm_LST_modis_alpha0.85_n5 | exact | C_smm_LST_modis_alpha0.85_n5 |
| aspect | aspect | exact | aspect |
| NDVI | NDVI | exact | NDVI |

## Correlation Thresholds

- Pearson |r| >= 0.8
- Spearman |rho| >= 0.8
- Mutual information >= 0.1

## Correlation Heatmaps

These are for the selected features (after mapping).
![](correlation_pearson.png)
![](correlation_spearman.png)
![](correlation_mutual_info.png)

## Top Correlated Pairs (max 30)

Sorted by max(|Pearson|, |Spearman|, MI).
| Feature A | Feature B | Pearson | Spearman | Mutual Info |
| --- | --- | --- | --- | --- |
| s1_vv | s1_vh | 0.976 | 0.899 | 5.554 |
| s1_vh | SAR_ratio | -0.857 | -0.773 | 5.550 |
| s1_vv | SAR_ratio | -0.757 | -0.518 | 5.549 |
| s2_b8 | NDVI | 0.293 | 0.431 | 4.285 |
| NDVI | SAR_ratio | -0.410 | -0.330 | 4.260 |
| s2_b8 | SAR_ratio | -0.109 | 0.107 | 4.241 |
| s1_vh | NDVI | 0.583 | 0.458 | 4.207 |
| s1_vv | NDVI | 0.641 | 0.570 | 4.173 |
| s1_vv | s2_b8 | 0.398 | 0.366 | 4.145 |
| s1_vh | s2_b8 | 0.291 | 0.144 | 4.139 |
| G_API | C_smm_G_API_alpha0.85_n5 | 0.955 | 0.975 | 2.029 |
| s2_b8 | DOY | -0.033 | 0.050 | 2.004 |
| DOY | V_ema_LST_modis_kobs30 | 0.168 | 0.170 | 1.959 |
| DOY | SAR_ratio | 0.075 | 0.118 | 1.931 |
| DOY | NDVI | -0.006 | -0.039 | 1.885 |
| DOY | C_smm_LST_modis_alpha0.85_n5 | 0.009 | 0.007 | 1.772 |
| V_ema_LST_modis_kobs30 | C_smm_LST_modis_alpha0.85_n5 | 0.959 | 0.960 | 1.758 |
| s1_vv | DOY | -0.026 | 0.055 | 1.727 |
| s1_vh | DOY | -0.049 | -0.009 | 1.667 |
| V_ema_LST_modis_kobs30 | SAR_ratio | 0.461 | 0.584 | 1.619 |
| s2_b8 | V_ema_LST_modis_kobs30 | 0.394 | 0.518 | 1.509 |
| V_ema_LST_modis_kobs30 | NDVI | 0.016 | 0.047 | 1.480 |
| C_smm_LST_modis_alpha0.85_n5 | SAR_ratio | 0.442 | 0.574 | 1.344 |
| s1_vv | V_ema_LST_modis_kobs30 | -0.260 | -0.134 | 1.280 |
| s1_vh | V_ema_LST_modis_kobs30 | -0.370 | -0.355 | 1.276 |
| s2_b8 | C_smm_LST_modis_alpha0.85_n5 | 0.385 | 0.497 | 1.270 |
| C_smm_LST_modis_alpha0.85_n5 | NDVI | 0.068 | 0.091 | 1.260 |
| C_smm_G_API_alpha0.85_n5 | SAR_ratio | -0.634 | -0.827 | 1.137 |
| slope | aspect | -0.875 | -0.541 | 1.099 |
| slope | SAR_ratio | -0.416 | -0.494 | 1.096 |

## Missing Features After Mapping

- None

## Notes

- Pearson captures linear relationships.
- Spearman captures monotonic relationships.
- Mutual information captures nonlinear dependencies.
