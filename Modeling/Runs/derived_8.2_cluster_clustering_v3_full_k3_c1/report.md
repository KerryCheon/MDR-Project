# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_v3_full_k3_c1
- Generated: 2026-07-14T16:30:44
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 8 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | D_sin_DOY |
| 2 | V_ema_F_NDVI_kobs30 |
| 3 | V_ema_G_API_kobs14 |
| 4 | V_rollmean_LST_modis_kobs30 |
| 5 | V_rollmin_G_API_kobs30 |
| 6 | C_lag_E_SAR_ratio_kobs30 |
| 7 | V_rollmin_LST_modis_kobs30 |
| 8 | C_lag_G_API_kobs1 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
