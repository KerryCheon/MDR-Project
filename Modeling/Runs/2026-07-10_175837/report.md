# Feature Selection Report

## Run Info
- Run ID: 2026-07-10_175837
- Generated: 2026-07-10T18:04:02
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 47 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |
| Score | 0.5111 |
| Mean R2 | 0.6383 |
| Std R2 | 0.0606 |
| Train-Val Gap | 0.2491 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_E_SAR_diff_kobs30 |
| 2 | A_grad_E_SAR_diff_kobs30 |
| 3 | A_grad_E_SAR_ratio_kobs30 |
| 4 | C_lag_F_NDVI_kobs30 |
| 5 | DOY |
| 6 | D_cos_DOY |
| 7 | D_sin_DOY |
| 8 | J_aspect_deg |
| 9 | J_bio_bio15 |
| 10 | J_lc_code |
| 11 | J_soil_texture_usda_b10 |
| 12 | J_soil_texture_usda_b200 |
| 13 | K_aspect_cos |
| 14 | SMAP_sm_pm_interp_rollrange30 |
| 15 | V_rollmean_LST_modis_kobs30 |
| 16 | V_rollmin_LST_modis_kobs30 |
| 17 | V_rollmin_s2_b11_kobs30 |
| 18 | latitude |
| 19 | lia_std_asc_deg |
| 20 | s2_b8 |
| 21 | sin_year |
| 22 | A_d_E_SAR_ratio_kobs30 |
| 23 | G_API |
| 24 | J_bio_bio19 |
| 25 | SMAP_x_year |
| 26 | V_rollmax_LST_modis_kobs14 |
| 27 | V_rollrng_G_API_kobs7 |
| 28 | G_DSLR |
| 29 | V_rollrng_E_SAR_ratio_kobs30 |
| 30 | slope |
| 31 | J_soil_texture_usda_b0 |
| 32 | V_rollrng_F_NDVI_kobs30 |
| 33 | V_rollmax_F_NDVI_kobs14 |
| 34 | A_d_SMAP_sm_interp_kobs5 |
| 35 | V_rollmax_G_API_kobs30 |
| 36 | V_rollrng_s2_b11_kobs30 |
| 37 | C_lag_LST_modis_kobs30 |
| 38 | E_rough_s1_vh_kobs14 |
| 39 | s2_b4 |
| 40 | A_grad_SMAP_sm_interp_kobs30 |

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
| train   | 15704 |                   0 | 0.6995 | 0.0595 |     0.2869 | 0.0473 |    0      | linear  |           47 |
| val     |  7149 |                   0 | 0.5525 | 0.0768 |     0.3818 | 0.0577 |    0.0261 | linear  |           47 |
| test    |  8902 |                   0 | 0.2846 | 0.0891 |     0.4774 | 0.0699 |    0.0487 | linear  |           47 |
| train   | 15704 |                   0 | 0.9725 | 0.018  |     0.0868 | 0.0125 |   -0      | xgb     |           47 |
| val     |  7149 |                   0 | 0.6835 | 0.0646 |     0.3211 | 0.0442 |    0.0034 | xgb     |           47 |
| test    |  8902 |                   0 | 0.5709 | 0.069  |     0.3697 | 0.0515 |    0.0171 | xgb     |           47 |
| train   | 15704 |                   0 | 0.9901 | 0.0108 |     0.0521 | 0.0062 |    0      | rf      |           47 |
| val     |  7149 |                   0 | 0.6787 | 0.0651 |     0.3236 | 0.0447 |    0.0037 | rf      |           47 |
| test    |  8902 |                   0 | 0.6096 | 0.0658 |     0.3526 | 0.0494 |    0.0171 | rf      |           47 |
