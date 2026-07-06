# Feature Selection Report

## Run Info
- Run ID: 2026-07-06_140155
- Generated: 2026-07-06T14:06:34
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 40 |
| Stages | mi, correlation, rf_importance, stability |
| Top-k target | 40 |
| Score | 0.3814 |
| Mean R2 | 0.5057 |
| Std R2 | 0.0511 |
| Train-Val Gap | 0.2939 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | A_d_G_API_kobs14 |
| 3 | A_d_G_API_kobs2 |
| 4 | A_d_G_API_kobs30 |
| 5 | C_lag_E_SAR_ratio_kobs30 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | G_rain_sum_3d |
| 10 | J_bio_bio02 |
| 11 | J_clay_wfrac_b0 |
| 12 | J_clay_wfrac_b200 |
| 13 | J_elev_m |
| 14 | J_lc_code |
| 15 | J_sand_wfrac_b0 |
| 16 | J_sand_wfrac_b200 |
| 17 | K_aspect_cos |
| 18 | K_clay_plus_sand_b0 |
| 19 | K_sand_clay_ratio_b0 |
| 20 | SMAP_x_year |
| 21 | V_ema_F_NDVI_kobs30 |
| 22 | V_rollmax_G_API_kobs30 |
| 23 | V_rollmax_SMAP_sm_interp_kobs30 |
| 24 | V_rollmean_s2_b11_kobs30 |
| 25 | V_rollmin_E_SAR_ratio_kobs30 |
| 26 | V_rollmin_F_NDVI_kobs30 |
| 27 | V_rollmin_LST_modis_kobs30 |
| 28 | V_rollmin_LST_modis_kobs7 |
| 29 | V_rollrng_G_API_kobs14 |
| 30 | V_rollrng_G_API_kobs30 |
| 31 | V_rollrng_G_API_kobs7 |
| 32 | V_rollrng_LST_modis_kobs30 |
| 33 | V_rollrng_s2_b11_kobs30 |
| 34 | aspect |
| 35 | cos_year |
| 36 | latitude |
| 37 | lia_std_desc_deg |
| 38 | longitude |
| 39 | s2_b8 |
| 40 | sin_year |

## Metrics

> Note: These models have not been tuned or optimized in any way

| split   |     n |   dropped_nonfinite |     r2 |   rmse |   rel_rmse |    mae |   bias_me | model   |   n_features |
|:--------|------:|--------------------:|-------:|-------:|-----------:|-------:|----------:|:--------|-------------:|
| train   | 29362 |                   0 | 0.5403 | 0.0742 |     0.448  | 0.0596 |   -0      | linear  |           40 |
| val     | 13637 |                   0 | 0.4374 | 0.0856 |     0.5431 | 0.0661 |    0.0179 | linear  |           40 |
| test    | 20166 |                   0 | 0.2876 | 0.0949 |     0.5845 | 0.0729 |    0.0175 | linear  |           40 |
| train   | 29362 |                   0 | 0.907  | 0.0334 |     0.2015 | 0.0243 |    0      | xgb     |           40 |
| val     | 13637 |                   0 | 0.5603 | 0.0756 |     0.4802 | 0.0551 |    0.0051 | xgb     |           40 |
| test    | 20166 |                   0 | 0.4818 | 0.0809 |     0.4985 | 0.0581 |    0.0052 | xgb     |           40 |
| train   | 29362 |                   0 | 0.9514 | 0.0241 |     0.1457 | 0.0145 |   -0      | rf      |           40 |
| val     | 13637 |                   0 | 0.5195 | 0.0791 |     0.502  | 0.0564 |    0.0097 | rf      |           40 |
| test    | 20166 |                   0 | 0.4545 | 0.083  |     0.5115 | 0.0584 |    0.0057 | rf      |           40 |
