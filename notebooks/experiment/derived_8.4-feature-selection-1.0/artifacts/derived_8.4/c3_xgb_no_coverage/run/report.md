# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:05:03
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | correlation, xgb_importance, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_F_NDMI_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_G_API_kobs1 |
| 5 | C_lag_LST_modis_kobs30 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_3d |
| 10 | G_rain_sum_7d |
| 11 | J_bio_bio02 |
| 12 | J_bio_bio15 |
| 13 | SMAP_x_year |
| 14 | V_ema_G_API_kobs30 |
| 15 | V_rollmax_E_SAR_diff_kobs14 |
| 16 | V_rollmax_E_SAR_ratio_kobs30 |
| 17 | V_rollmax_F_NDMI_kobs30 |
| 18 | V_rollmax_F_NDVI_kobs14 |
| 19 | V_rollmax_F_NDVI_kobs30 |
| 20 | V_rollmax_LST_modis_kobs30 |
| 21 | V_rollmin_E_SAR_diff_kobs30 |
| 22 | V_rollmin_E_SAR_ratio_kobs30 |
| 23 | V_rollmin_F_NDVI_kobs30 |
| 24 | V_rollmin_G_API_kobs14 |
| 25 | V_rollmin_G_API_kobs30 |
| 26 | V_rollmin_LST_modis_kobs30 |
| 27 | V_rollmin_s2_b11_kobs30 |
| 28 | V_rollrng_E_SAR_diff_kobs30 |
| 29 | elev |
| 30 | latitude |
| 31 | lia_mean_asc_deg |
| 32 | sin_year |
| 33 | C_lag_LST_modis_kobs6 |
| 34 | V_ema_E_SAR_ratio_kobs30 |
| 35 | V_rollmax_F_NDMI_kobs14 |
| 36 | C_smm_F_NDVI_alpha0.85_n5 |
| 37 | V_rollmean_F_NDVI_kobs30 |
| 38 | cos_year |
| 39 | J_bio_bio03 |
| 40 | J_bio_bio14 |

## Metrics

_No metrics provided._
