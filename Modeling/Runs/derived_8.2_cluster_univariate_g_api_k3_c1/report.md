# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_univariate_g_api_k3_c1
- Generated: 2026-07-14T16:35:59
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 3 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | D_sin_DOY |
| 2 | J_aspect_deg |
| 3 | C_lag_F_NDVI_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
