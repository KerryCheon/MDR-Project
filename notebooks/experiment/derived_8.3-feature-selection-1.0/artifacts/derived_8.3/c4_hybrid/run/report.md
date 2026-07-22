# Feature Selection Report

## Run Info
- Run ID: run
- Generated: 2026-07-21T18:57:42
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
| 2 | C_lag_F_NDVI_kobs30 |
| 3 | C_lag_LST_modis_kobs30 |
| 4 | C_smm_G_API_alpha0.85_n5 |
| 5 | DOY |
| 6 | D_cos_DOY |
| 7 | D_sa_F_NDMI |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_3d |
| 10 | G_rain_sum_7d |
| 11 | J_bio_bio01 |
| 12 | J_bio_bio04 |
| 13 | J_bio_bio09 |
| 14 | K_aspect_cos |
| 15 | K_slope_sin |
| 16 | SMAP_sm_interp_lag1 |
| 17 | SMAP_sm_pm_interp_rollmean30 |
| 18 | SMAP_x_year |
| 19 | V_ema_G_API_kobs30 |
| 20 | V_rollmax_F_NDMI_kobs30 |
| 21 | V_rollmax_F_NDVI_kobs30 |
| 22 | V_rollmax_LST_modis_kobs30 |
| 23 | V_rollmean_E_SAR_ratio_kobs30 |
| 24 | V_rollmean_F_NDVI_kobs30 |
| 25 | V_rollmin_E_SAR_ratio_kobs30 |
| 26 | V_rollmin_G_API_kobs14 |
| 27 | V_rollmin_G_API_kobs30 |
| 28 | V_rollmin_LST_modis_kobs30 |
| 29 | V_rollmin_SMAP_sm_interp_kobs30 |
| 30 | V_rollmin_s2_b11_kobs30 |
| 31 | aspect |
| 32 | elev |
| 33 | latitude |
| 34 | lia_mean_asc_deg |
| 35 | lia_std_asc_deg |
| 36 | lia_std_desc_deg |
| 37 | C_lag_LST_modis_kobs6 |
| 38 | D_z_E_SAR_ratio |
| 39 | J_bio_bio14 |
| 40 | V_ema_E_SAR_diff_kobs30 |

## Metrics

_No metrics provided._
