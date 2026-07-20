# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:04:53
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
| 2 | A_grad_F_NDMI_kobs30 |
| 3 | C_lag_E_SAR_diff_kobs6 |
| 4 | C_lag_E_SAR_ratio_kobs30 |
| 5 | C_lag_LST_modis_kobs12 |
| 6 | C_lag_LST_modis_kobs30 |
| 7 | DOY |
| 8 | D_sin_DOY |
| 9 | G_API |
| 10 | G_rain_sum_3d |
| 11 | G_rain_sum_7d |
| 12 | J_bio_bio09 |
| 13 | J_clay_wfrac_b200 |
| 14 | SMAP_x_year |
| 15 | V_ema_G_API_kobs14 |
| 16 | V_rollcv_F_NDMI_kobs30 |
| 17 | V_rollcv_SMAP_sm_interp_kobs30 |
| 18 | V_rollmax_E_SAR_ratio_kobs30 |
| 19 | V_rollmax_F_NDMI_kobs30 |
| 20 | V_rollmax_G_API_kobs30 |
| 21 | V_rollmax_LST_modis_kobs30 |
| 22 | V_rollmax_SMAP_sm_interp_kobs30 |
| 23 | V_rollmin_E_SAR_diff_kobs30 |
| 24 | V_rollmin_F_NDVI_kobs30 |
| 25 | V_rollmin_G_API_kobs30 |
| 26 | V_rollmin_LST_modis_kobs30 |
| 27 | V_rollmin_s2_b12_kobs14 |
| 28 | V_rollrng_F_NDVI_kobs30 |
| 29 | aspect |
| 30 | cos_year |
| 31 | s2_b8 |
| 32 | sin_year |
| 33 | year |
| 34 | A_grad_LST_modis_kobs30 |
| 35 | C_lag_E_SAR_diff_kobs30 |
| 36 | C_smm_LST_modis_alpha0.85_n5 |
| 37 | D_z_E_SAR_ratio |
| 38 | F_MSI |
| 39 | V_rollmax_F_NDVI_kobs30 |
| 40 | V_rollmax_s2_b12_kobs30 |

## Metrics

_No metrics provided._
