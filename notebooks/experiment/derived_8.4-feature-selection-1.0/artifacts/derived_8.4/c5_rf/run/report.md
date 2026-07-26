# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-26T16:08:50
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | correlation, rf_importance, family_coverage, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | C_lag_E_SAR_ratio_kobs30 |
| 3 | C_lag_F_NDMI_kobs30 |
| 4 | C_lag_F_NDVI_kobs30 |
| 5 | C_lag_G_API_kobs1 |
| 6 | C_lag_LST_modis_kobs30 |
| 7 | C_lag_LST_modis_kobs6 |
| 8 | C_lag_SMAP_sm_interp_kobs12 |
| 9 | DOY |
| 10 | D_cos_DOY |
| 11 | D_sin_DOY |
| 12 | G_rain_sum_3d |
| 13 | G_rain_sum_7d |
| 14 | J_bio_bio02 |
| 15 | J_bio_bio15 |
| 16 | J_bio_bio19 |
| 17 | SMAP_sm_am_interp_lag30 |
| 18 | SMAP_x_year |
| 19 | V_ema_E_SAR_ratio_kobs30 |
| 20 | V_ema_G_API_kobs30 |
| 21 | V_rollcv_E_SAR_diff_kobs30 |
| 22 | V_rollcv_SMAP_sm_interp_kobs30 |
| 23 | V_rollmax_E_SAR_diff_kobs30 |
| 24 | V_rollmax_E_SAR_ratio_kobs30 |
| 25 | V_rollmax_F_NDMI_kobs14 |
| 26 | V_rollmax_F_NDMI_kobs30 |
| 27 | V_rollmax_F_NDVI_kobs30 |
| 28 | V_rollmax_LST_modis_kobs30 |
| 29 | V_rollmean_F_NDVI_kobs30 |
| 30 | V_rollmin_E_SAR_diff_kobs30 |
| 31 | V_rollmin_E_SAR_ratio_kobs30 |
| 32 | V_rollmin_F_NDVI_kobs30 |
| 33 | V_rollmin_G_API_kobs14 |
| 34 | V_rollmin_G_API_kobs30 |
| 35 | V_rollmin_LST_modis_kobs30 |
| 36 | V_rollmin_s2_b11_kobs30 |
| 37 | V_rollrng_E_SAR_diff_kobs30 |
| 38 | V_rollstd_F_NDVI_kobs30 |
| 39 | lia_mean_asc_deg |
| 40 | precip_mm |

## Metrics

_No metrics provided._
