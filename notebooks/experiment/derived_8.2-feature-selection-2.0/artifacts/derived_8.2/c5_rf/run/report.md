# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-17T21:18:14
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
| 2 | C_lag_E_SAR_diff_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_LST_modis_kobs30 |
| 5 | C_lag_LST_modis_kobs6 |
| 6 | C_lag_SMAP_sm_interp_kobs5 |
| 7 | C_smm_G_API_alpha0.85_n5 |
| 8 | DOY |
| 9 | D_cos_DOY |
| 10 | D_sin_DOY |
| 11 | G_rain_sum_3d |
| 12 | G_rain_sum_7d |
| 13 | J_aspect_deg |
| 14 | J_bio_bio01 |
| 15 | J_bio_bio02 |
| 16 | J_sand_wfrac_b0 |
| 17 | K_aspect_cos |
| 18 | SMAP_sm_am_interp_lag30 |
| 19 | SMAP_sm_pm_interp_lag1 |
| 20 | SMAP_sm_pm_interp_rollrange30 |
| 21 | SMAP_x_year |
| 22 | V_ema_G_API_kobs30 |
| 23 | V_rollcv_E_SAR_diff_kobs30 |
| 24 | V_rollmax_E_SAR_diff_kobs30 |
| 25 | V_rollmax_E_SAR_ratio_kobs30 |
| 26 | V_rollmax_F_NDMI_kobs30 |
| 27 | V_rollmax_F_NDVI_kobs30 |
| 28 | V_rollmax_LST_modis_kobs30 |
| 29 | V_rollmean_E_SAR_diff_kobs30 |
| 30 | V_rollmean_E_SAR_ratio_kobs30 |
| 31 | V_rollmean_F_NDVI_kobs30 |
| 32 | V_rollmin_E_SAR_diff_kobs30 |
| 33 | V_rollmin_E_SAR_ratio_kobs30 |
| 34 | V_rollmin_F_NDVI_kobs30 |
| 35 | V_rollmin_G_API_kobs14 |
| 36 | V_rollmin_G_API_kobs30 |
| 37 | V_rollmin_LST_modis_kobs30 |
| 38 | V_rollstd_F_NDVI_kobs30 |
| 39 | elev |
| 40 | latitude |

## Metrics

_No metrics provided._
