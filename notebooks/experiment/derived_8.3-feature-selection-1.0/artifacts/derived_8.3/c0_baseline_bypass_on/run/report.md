# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T18:41:48
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 14 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | C_lag_LST_modis_kobs30 |
| 2 | D_sin_DOY |
| 3 | G_API |
| 4 | G_DSLR |
| 5 | J_aspect_deg |
| 6 | V_ema_LST_modis_kobs30 |
| 7 | V_rollmean_LST_modis_kobs30 |
| 8 | V_rollmin_LST_modis_kobs30 |
| 9 | lia_std_asc_deg |
| 10 | V_rollmax_G_API_kobs7 |
| 11 | V_rollrng_F_NDMI_kobs30 |
| 12 | V_ema_G_API_kobs14 |
| 13 | V_ema_G_API_kobs30 |
| 14 | C_lag_LST_modis_kobs12 |

## Metrics

_No metrics provided._
