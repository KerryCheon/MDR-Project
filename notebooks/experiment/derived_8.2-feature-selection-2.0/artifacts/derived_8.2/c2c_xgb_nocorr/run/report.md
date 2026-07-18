# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:28:52
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 55 |
| Stages | xgb_importance, family_coverage, stability |
| Top-k target | 55 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_E_SAR_diff_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_LST_modis_kobs30 |
| 5 | DOY |
| 6 | D_cos_DOY |
| 7 | D_sin_DOY |
| 8 | J_aspect_deg |
| 9 | J_bio_bio04 |
| 10 | J_bio_bio09 |
| 11 | K_aspect_cos |
| 12 | V_ema_G_API_kobs14 |
| 13 | V_ema_G_API_kobs30 |
| 14 | V_ema_LST_modis_kobs30 |
| 15 | V_rollmax_F_NDVI_kobs30 |
| 16 | V_rollmax_LST_modis_kobs30 |
| 17 | V_rollmean_F_NDVI_kobs30 |
| 18 | V_rollmean_LST_modis_kobs30 |
| 19 | V_rollmin_G_API_kobs30 |
| 20 | V_rollmin_LST_modis_kobs30 |
| 21 | aspect |
| 22 | cos_year |
| 23 | elev |
| 24 | latitude |
| 25 | J_bio_bio05 |
| 26 | J_bio_bio10 |
| 27 | J_bio_bio12 |
| 28 | V_rollmax_F_NDMI_kobs30 |
| 29 | V_rollmax_G_API_kobs30 |
| 30 | V_rollmin_F_NDVI_kobs7 |
| 31 | V_rollrng_F_NDVI_kobs30 |
| 32 | longitude |
| 33 | year |
| 34 | G_rain_sum_30d |
| 35 | V_ema_F_NDVI_kobs30 |
| 36 | V_ema_G_API_kobs7 |
| 37 | V_rollmin_E_SAR_diff_kobs30 |
| 38 | C_lag_G_API_kobs1 |
| 39 | V_rollmax_F_NDVI_kobs14 |
| 40 | V_rollmin_G_API_kobs14 |

## Metrics

_No metrics provided._
