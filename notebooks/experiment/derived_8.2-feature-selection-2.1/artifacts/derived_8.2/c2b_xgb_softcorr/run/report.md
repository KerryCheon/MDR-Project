# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:27:22
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 55 |
| Stages | correlation, xgb_importance, family_coverage, stability |
| Top-k target | 55 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_E_SAR_diff_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_G_API_kobs1 |
| 5 | C_lag_LST_modis_kobs30 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_30d |
| 10 | J_aspect_deg |
| 11 | J_bio_bio04 |
| 12 | J_bio_bio05 |
| 13 | J_bio_bio09 |
| 14 | K_aspect_cos |
| 15 | SMAP_sm_pm_interp_lag1 |
| 16 | V_ema_G_API_kobs14 |
| 17 | V_ema_G_API_kobs30 |
| 18 | V_rollmax_E_SAR_ratio_kobs30 |
| 19 | V_rollmax_F_NDMI_kobs30 |
| 20 | V_rollmax_F_NDVI_kobs14 |
| 21 | V_rollmax_F_NDVI_kobs30 |
| 22 | V_rollmax_G_API_kobs30 |
| 23 | V_rollmax_LST_modis_kobs30 |
| 24 | V_rollmean_F_NDVI_kobs30 |
| 25 | V_rollmean_LST_modis_kobs30 |
| 26 | V_rollmin_E_SAR_diff_kobs30 |
| 27 | V_rollmin_F_NDVI_kobs30 |
| 28 | V_rollmin_G_API_kobs30 |
| 29 | V_rollmin_LST_modis_kobs30 |
| 30 | aspect |
| 31 | cos_year |
| 32 | elev |
| 33 | latitude |
| 34 | longitude |
| 35 | G_rain_sum_7d |
| 36 | V_rollmean_E_SAR_ratio_kobs30 |
| 37 | V_rollmin_F_NDVI_kobs7 |
| 38 | V_rollmin_G_API_kobs14 |
| 39 | V_rollmin_s2_b11_kobs30 |
| 40 | V_rollrng_F_NDVI_kobs30 |

## Metrics

_No metrics provided._
