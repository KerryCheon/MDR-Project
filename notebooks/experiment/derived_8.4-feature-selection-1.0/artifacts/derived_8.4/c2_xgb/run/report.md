# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:10:34
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | correlation, xgb_importance, family_coverage, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_F_NDMI_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_G_API_kobs1 |
| 5 | C_lag_LST_modis_kobs12 |
| 6 | C_lag_LST_modis_kobs30 |
| 7 | C_smm_F_NDVI_alpha0.85_n5 |
| 8 | DOY |
| 9 | D_cos_DOY |
| 10 | D_sin_DOY |
| 11 | G_rain_sum_3d |
| 12 | G_rain_sum_7d |
| 13 | J_bio_bio02 |
| 14 | J_bio_bio03 |
| 15 | J_bio_bio15 |
| 16 | SMAP_sm_am_interp_lag30 |
| 17 | SMAP_x_year |
| 18 | V_ema_E_SAR_ratio_kobs30 |
| 19 | V_ema_G_API_kobs30 |
| 20 | V_rollmax_E_SAR_diff_kobs14 |
| 21 | V_rollmax_E_SAR_ratio_kobs30 |
| 22 | V_rollmax_F_NDMI_kobs30 |
| 23 | V_rollmax_F_NDVI_kobs14 |
| 24 | V_rollmax_F_NDVI_kobs30 |
| 25 | V_rollmax_LST_modis_kobs30 |
| 26 | V_rollmax_s2_b11_kobs30 |
| 27 | V_rollmean_F_NDVI_kobs30 |
| 28 | V_rollmin_E_SAR_diff_kobs30 |
| 29 | V_rollmin_E_SAR_ratio_kobs30 |
| 30 | V_rollmin_F_NDMI_kobs30 |
| 31 | V_rollmin_F_NDVI_kobs30 |
| 32 | V_rollmin_G_API_kobs14 |
| 33 | V_rollmin_G_API_kobs30 |
| 34 | V_rollmin_LST_modis_kobs30 |
| 35 | V_rollmin_s2_b11_kobs14 |
| 36 | V_rollmin_s2_b11_kobs30 |
| 37 | V_rollrng_E_SAR_diff_kobs30 |
| 38 | V_rollstd_F_NDVI_kobs30 |
| 39 | cos_year |
| 40 | elev |

## Metrics

_No metrics provided._
