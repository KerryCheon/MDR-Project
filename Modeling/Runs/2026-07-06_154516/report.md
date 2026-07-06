# Feature Selection Report

## Run Info
- Run ID: 2026-07-06_154516
- Generated: 2026-07-06T15:50:09
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
| Score | 0.5002 |
| Mean R2 | 0.6274 |
| Std R2 | 0.0734 |
| Train-Val Gap | 0.2527 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | A_grad_E_SAR_diff_kobs30 |
| 4 | A_grad_E_SAR_ratio_kobs30 |
| 5 | C_lag_F_NDVI_kobs30 |
| 6 | DOY |
| 7 | D_cos_DOY |
| 8 | D_sin_DOY |
| 9 | J_aspect_deg |
| 10 | J_bio_bio15 |
| 11 | J_bio_bio16 |
| 12 | J_bio_bio19 |
| 13 | J_lc_code |
| 14 | J_soil_texture_usda_b0 |
| 15 | J_soil_texture_usda_b10 |
| 16 | J_soil_texture_usda_b200 |
| 17 | K_aspect_cos |
| 18 | SMAP_sm_pm_interp_rollrange30 |
| 19 | V_ema_LST_modis_kobs30 |
| 20 | V_rollmean_LST_modis_kobs30 |
| 21 | V_rollmin_LST_modis_kobs30 |
| 22 | V_rollmin_s2_b11_kobs30 |
| 23 | latitude |
| 24 | lia_std_asc_deg |
| 25 | s2_b8 |
| 26 | sin_year |
| 27 | slope |
| 28 | G_API |
| 29 | J_bio_bio13 |
| 30 | SMAP_x_year |
| 31 | V_rollmax_LST_modis_kobs14 |
| 32 | V_rollrng_G_API_kobs7 |
| 33 | G_DSLR |
| 34 | V_rollrng_E_SAR_ratio_kobs30 |
| 35 | V_rollmax_F_NDVI_kobs14 |
| 36 | C_lag_LST_modis_kobs30 |
| 37 | V_rollmax_G_API_kobs30 |
| 38 | A_d_SMAP_sm_interp_kobs5 |
| 39 | V_rollrng_F_NDVI_kobs30 |
| 40 | J_bio_bio12 |

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
| train   | 15704 |                   0 | 0.6802 | 0.0613 |     0.296  | 0.0492 |    0      | linear  |           40 |
| val     |  7149 |                   0 | 0.5236 | 0.0792 |     0.394  | 0.0601 |    0.0324 | linear  |           40 |
| test    |  8902 |                   0 | 0.2223 | 0.0929 |     0.4977 | 0.0732 |    0.0539 | linear  |           40 |
| train   | 15704 |                   0 | 0.9705 | 0.0186 |     0.0899 | 0.0129 |    0      | xgb     |           40 |
| val     |  7149 |                   0 | 0.6776 | 0.0652 |     0.3241 | 0.0438 |    0.0074 | xgb     |           40 |
| test    |  8902 |                   0 | 0.5608 | 0.0698 |     0.374  | 0.0522 |    0.0238 | xgb     |           40 |
| train   | 15704 |                   0 | 0.9897 | 0.011  |     0.0531 | 0.0063 |    0      | rf      |           40 |
| val     |  7149 |                   0 | 0.6811 | 0.0648 |     0.3224 | 0.0435 |    0.0075 | rf      |           40 |
| test    |  8902 |                   0 | 0.6227 | 0.0647 |     0.3467 | 0.0483 |    0.0198 | rf      |           40 |
