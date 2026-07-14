# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_dynamic_k3_c1
- Generated: 2026-07-14T16:33:39
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 16 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_G_API_kobs1 |
| 3 | D_sin_DOY |
| 4 | J_bio_bio12 |
| 5 | J_lc_code |
| 6 | K_aspect_cos |
| 7 | V_rollmax_G_API_kobs30 |
| 8 | V_rollmin_LST_modis_kobs30 |
| 9 | J_aspect_deg |
| 10 | C_lag_E_SAR_ratio_kobs30 |
| 11 | V_ema_F_NDVI_kobs30 |
| 12 | G_DSLR |
| 13 | V_rollmax_LST_modis_kobs30 |
| 14 | G_API |
| 15 | V_rollrng_E_SAR_ratio_kobs30 |
| 16 | V_ema_LST_modis_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
