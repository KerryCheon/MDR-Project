# Global SHAP Feature Importance + Correlation Report

## Method (brief)
- Parsed top-10 mean |SHAP| tables from `MDR/Temporal/Pipeline/data/features/review/SHAP_ANALYSIS.md`.
- Feature set 10 handling: `xgb`.
- Computed global importance: frequency (appears in top-10 across 11 sets) * avg SHAP.
- Selected features for correlation using `top 15 by global_importance`.
- Correlations computed on dataset: `MDR/Temporal/Pipeline/data/splits/derived_all/test_derived_all.csv` (rows: 3408, dropped NaN rows: 0).
- Mutual information estimated via binned histogram (bins=20) on z-scored features.

## Top Features by Global Importance
| Feature | Freq Count | Frequency | Avg SHAP | Global Importance |
| --- | --- | --- | --- | --- |
| DOY | 9 | 0.818 | 0.030560 | 0.025003 |
| air_temp_mean | 4 | 0.364 | 0.038941 | 0.014160 |
| s1_vv | 7 | 0.636 | 0.007951 | 0.005060 |
| V_ema_LST_modis_kobs30 | 2 | 0.182 | 0.024560 | 0.004465 |
| rh_mean | 3 | 0.273 | 0.015523 | 0.004234 |
| C_smm_G_API_alpha0.85_n5 | 2 | 0.182 | 0.020411 | 0.003711 |
| precip_mm | 5 | 0.455 | 0.008061 | 0.003664 |
| s2_b8 | 5 | 0.455 | 0.007492 | 0.003406 |
| s1_vh | 6 | 0.545 | 0.006226 | 0.003396 |
| API | 1 | 0.091 | 0.036759 | 0.003342 |
| slope | 3 | 0.273 | 0.011967 | 0.003264 |
| SAR_ratio | 5 | 0.455 | 0.006948 | 0.003158 |
| C_smm_LST_modis_alpha0.85_n5 | 2 | 0.182 | 0.017209 | 0.003129 |
| aspect | 3 | 0.273 | 0.010882 | 0.002968 |
| NDVI | 5 | 0.455 | 0.006030 | 0.002741 |

## High-Correlation Thresholds
- Pearson |r| >= 0.8
- Spearman |rho| >= 0.8
- Mutual information >= 0.1

## Top Correlated Pairs (max 30)
| Feature A | Feature B | Pearson | Spearman | Mutual Info |
| --- | --- | --- | --- | --- |
| slope | aspect | 0.151 | 0.260 | 1.366 |
| V_ema_LST_modis_kobs30 | C_smm_LST_modis_alpha0.85_n5 | 0.964 | 0.965 | 1.311 |
| DOY | V_ema_LST_modis_kobs30 | 0.260 | 0.233 | 1.181 |
| s1_vv | s1_vh | 0.943 | 0.918 | 1.175 |
| DOY | C_smm_LST_modis_alpha0.85_n5 | 0.114 | 0.086 | 1.029 |
| s1_vv | slope | 0.139 | 0.103 | 0.867 |
| s1_vh | slope | -0.096 | -0.109 | 0.782 |
| s1_vv | aspect | -0.393 | -0.365 | 0.633 |
| s1_vh | C_smm_G_API_alpha0.85_n5 | 0.566 | 0.610 | 0.384 |
| C_smm_G_API_alpha0.85_n5 | C_smm_LST_modis_alpha0.85_n5 | -0.436 | -0.595 | 0.360 |
| C_smm_G_API_alpha0.85_n5 | V_ema_LST_modis_kobs30 | -0.422 | -0.592 | 0.357 |
| s1_vh | aspect | -0.459 | -0.361 | 0.570 |
| s1_vv | C_smm_G_API_alpha0.85_n5 | 0.444 | 0.527 | 0.331 |
| s1_vh | V_ema_LST_modis_kobs30 | -0.182 | -0.141 | 0.473 |
| s2_b8 | DOY | -0.137 | -0.099 | 0.444 |
| s1_vv | V_ema_LST_modis_kobs30 | -0.211 | -0.147 | 0.439 |
| s2_b8 | aspect | -0.344 | -0.432 | 0.303 |
| s1_vh | DOY | 0.059 | 0.059 | 0.417 |
| s1_vh | C_smm_LST_modis_alpha0.85_n5 | -0.178 | -0.136 | 0.410 |
| s1_vv | DOY | 0.040 | 0.057 | 0.407 |
| s1_vv | s2_b8 | 0.289 | 0.344 | 0.406 |
| precip_mm | C_smm_G_API_alpha0.85_n5 | 0.377 | 0.374 | 0.101 |
| s2_b8 | V_ema_LST_modis_kobs30 | -0.048 | 0.007 | 0.372 |
| s1_vh | s2_b8 | 0.278 | 0.314 | 0.360 |
| s1_vv | C_smm_LST_modis_alpha0.85_n5 | -0.207 | -0.149 | 0.359 |
| s2_b8 | slope | -0.148 | -0.277 | 0.359 |
| s2_b8 | C_smm_LST_modis_alpha0.85_n5 | -0.046 | 0.012 | 0.318 |
| slope | C_smm_G_API_alpha0.85_n5 | -0.186 | -0.018 | 0.303 |
| DOY | C_smm_G_API_alpha0.85_n5 | 0.051 | -0.050 | 0.282 |
| slope | V_ema_LST_modis_kobs30 | -0.090 | -0.067 | 0.281 |

## Consistency
- Features appearing in >=6 of 11 sets: 3
- List: DOY, s1_vv, s1_vh

## Missing Features
- Missing from dataset columns: API, NDVI, SAR_ratio, air_temp_mean, rh_mean

## Notes
- Pearson captures linear relationships.
- Spearman captures monotonic relationships.
- Mutual information captures nonlinear dependencies.