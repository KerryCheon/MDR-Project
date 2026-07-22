# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T19:13:38
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
| 2 | C_lag_E_SAR_diff_kobs30 |
| 3 | C_lag_F_NDVI_kobs30 |
| 4 | C_lag_LST_modis_kobs30 |
| 5 | C_lag_LST_modis_kobs6 |
| 6 | C_smm_G_API_alpha0.85_n5 |
| 7 | DOY |
| 8 | D_cos_DOY |
| 9 | D_sa_F_NDMI |
| 10 | D_sin_DOY |
| 11 | D_z_E_SAR_ratio |
| 12 | G_rain_sum_3d |
| 13 | G_rain_sum_7d |
| 14 | J_aspect_deg |
| 15 | J_bio_bio02 |
| 16 | J_bio_bio03 |
| 17 | J_bio_bio04 |
| 18 | J_bio_bio05 |
| 19 | K_slope_sin |
| 20 | SMAP_sm_interp_lag1 |
| 21 | SMAP_sm_pm_interp_rollmean30 |
| 22 | SMAP_x_year |
| 23 | V_ema_E_SAR_diff_kobs30 |
| 24 | V_ema_G_API_kobs30 |
| 25 | V_rollmax_E_SAR_diff_kobs30 |
| 26 | V_rollmax_F_NDMI_kobs30 |
| 27 | V_rollmax_F_NDVI_kobs14 |
| 28 | V_rollmax_F_NDVI_kobs30 |
| 29 | V_rollmax_LST_modis_kobs30 |
| 30 | V_rollmax_s2_b11_kobs14 |
| 31 | V_rollmax_s2_b11_kobs30 |
| 32 | V_rollmean_E_SAR_ratio_kobs30 |
| 33 | V_rollmean_F_NDVI_kobs30 |
| 34 | V_rollmin_E_SAR_diff_kobs30 |
| 35 | V_rollmin_E_SAR_ratio_kobs30 |
| 36 | V_rollmin_F_NDMI_kobs30 |
| 37 | V_rollmin_G_API_kobs14 |
| 38 | V_rollmin_G_API_kobs30 |
| 39 | V_rollmin_LST_modis_kobs30 |
| 40 | V_rollmin_SMAP_sm_interp_kobs30 |

## Metrics

_No metrics provided._
