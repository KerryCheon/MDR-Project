# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T18:42:50
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 9 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | C_lag_LST_modis_kobs30 |
| 2 | J_aspect_deg |
| 3 | V_ema_G_API_kobs30 |
| 4 | V_ema_LST_modis_kobs30 |
| 5 | V_rollmean_LST_modis_kobs30 |
| 6 | V_rollmin_LST_modis_kobs30 |
| 7 | lia_std_asc_deg |
| 8 | V_rollrng_F_NDMI_kobs30 |
| 9 | V_ema_G_API_kobs14 |

## Metrics

_No metrics provided._
