# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:05:39
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | correlation, mi, xgb_importance, family_coverage, stability |
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
| 11 | J_bio_bio10 |
| 12 | J_bio_bio14 |
| 13 | J_bio_bio19 |
| 14 | SMAP_x_year |
| 15 | V_ema_G_API_kobs30 |
| 16 | V_rollmax_F_NDMI_kobs30 |
| 17 | V_rollmax_F_NDVI_kobs14 |
| 18 | V_rollmax_F_NDVI_kobs30 |
| 19 | V_rollmax_LST_modis_kobs30 |
| 20 | V_rollmean_F_NDVI_kobs30 |
| 21 | V_rollmin_E_SAR_diff_kobs30 |
| 22 | V_rollmin_F_NDVI_kobs30 |
| 23 | V_rollmin_G_API_kobs14 |
| 24 | V_rollmin_G_API_kobs30 |
| 25 | V_rollmin_LST_modis_kobs30 |
| 26 | V_rollmin_s2_b11_kobs30 |
| 27 | V_rollrng_E_SAR_diff_kobs30 |
| 28 | latitude |
| 29 | lia_mean_asc_deg |
| 30 | sin_year |
| 31 | C_lag_LST_modis_kobs6 |
| 32 | SMAP_sm_am_interp_lag30 |
| 33 | V_ema_E_SAR_ratio_kobs30 |
| 34 | V_rollmax_E_SAR_ratio_kobs30 |
| 35 | cos_year |
| 36 | s2_b8 |
| 37 | C_lag_E_SAR_ratio_kobs30 |
| 38 | K_aspect_cos |
| 39 | V_rollmean_F_NDMI_kobs30 |
| 40 | V_rollmax_E_SAR_diff_kobs14 |

## Metrics

_No metrics provided._
