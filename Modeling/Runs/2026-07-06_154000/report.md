# Feature Selection Report

## Run Info
- Run ID: 2026-07-06_154000
- Generated: 2026-07-06T15:42:30
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 40 |
| Stages | mi, elasticnet, stability |
| Top-k target | 40 |
| Score | 0.4939 |
| Mean R2 | 0.6246 |
| Std R2 | 0.0777 |
| Train-Val Gap | 0.2590 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | A_grad_E_SAR_diff_kobs30 |
| 4 | A_grad_E_SAR_ratio_kobs30 |
| 5 | C_lag_F_NDVI_kobs30 |
| 6 | C_lag_LST_modis_kobs30 |
| 7 | DOY |
| 8 | D_cos_DOY |
| 9 | D_sin_DOY |
| 10 | G_API |
| 11 | G_DSLR |
| 12 | J_aspect_deg |
| 13 | J_bio_bio15 |
| 14 | J_bio_bio16 |
| 15 | J_bio_bio19 |
| 16 | J_lc_code |
| 17 | J_soil_texture_usda_b0 |
| 18 | J_soil_texture_usda_b200 |
| 19 | K_aspect_cos |
| 20 | SMAP_sm_pm_interp_rollrange30 |
| 21 | SMAP_x_year |
| 22 | V_rollmax_LST_modis_kobs14 |
| 23 | V_rollmean_LST_modis_kobs30 |
| 24 | V_rollmin_LST_modis_kobs30 |
| 25 | V_rollmin_s2_b11_kobs30 |
| 26 | V_rollrng_E_SAR_ratio_kobs30 |
| 27 | V_rollrng_G_API_kobs7 |
| 28 | latitude |
| 29 | lia_std_asc_deg |
| 30 | s2_b8 |
| 31 | sin_year |
| 32 | slope |
| 33 | J_soil_texture_usda_b10 |
| 34 | V_rollmax_F_NDVI_kobs14 |
| 35 | V_rollmax_G_API_kobs30 |
| 36 | V_rollrng_F_NDVI_kobs30 |
| 37 | s2_b4 |
| 38 | A_d_SMAP_sm_interp_kobs5 |
| 39 | E_rough_s1_vh_kobs7 |
| 40 | V_ema_LST_modis_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

> Note: These models have not been tuned or optimized in any way

| split   |     n |   dropped_nonfinite |     r2 |   rmse |   rel_rmse |    mae |   bias_me | model   |   n_features |
|:--------|------:|--------------------:|-------:|-------:|-----------:|-------:|----------:|:--------|-------------:|
| train   | 15704 |                   0 | 0.69   | 0.0604 |     0.2914 | 0.0482 |   -0      | linear  |           40 |
| val     |  7149 |                   0 | 0.5147 | 0.08   |     0.3976 | 0.0604 |    0.0316 | linear  |           40 |
| test    |  8902 |                   0 | 0.2343 | 0.0921 |     0.4939 | 0.0727 |    0.0544 | linear  |           40 |
| train   | 15704 |                   0 | 0.9708 | 0.0185 |     0.0894 | 0.0129 |    0      | xgb     |           40 |
| val     |  7149 |                   0 | 0.6785 | 0.0651 |     0.3237 | 0.0439 |    0.0073 | xgb     |           40 |
| test    |  8902 |                   0 | 0.5534 | 0.0704 |     0.3772 | 0.0528 |    0.0248 | xgb     |           40 |
| train   | 15704 |                   0 | 0.9899 | 0.0109 |     0.0527 | 0.0063 |    0.0001 | rf      |           40 |
| val     |  7149 |                   0 | 0.6805 | 0.0649 |     0.3227 | 0.0437 |    0.0072 | rf      |           40 |
| test    |  8902 |                   0 | 0.6189 | 0.065  |     0.3484 | 0.0485 |    0.0197 | rf      |           40 |
