# Feature Selection Report

## Run Info
- Run ID: derived_8.2_regime_binary_regime_wet_v1
- Generated: 2026-07-14T14:47:42
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 45 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |
| Score | 0.1148 |
| Mean R2 | 0.2854 |
| Std R2 | 0.0257 |
| Train-Val Gap | 0.5637 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | DOY |
| 2 | D_cos_DOY |
| 3 | D_sa_F_NDMI |
| 4 | D_z_E_SAR_ratio |
| 5 | D_z_F_NDMI |
| 6 | D_z_LST_modis |
| 7 | F_MSI |
| 8 | J_aspect_deg |
| 9 | J_bio_bio15 |
| 10 | K_aspect_cos |
| 11 | V_rollmax_F_NDVI_kobs30 |
| 12 | cos_year |
| 13 | latitude |
| 14 | lia_std_asc_deg |
| 15 | s2_b4 |
| 16 | s2_b8 |
| 17 | slope |
| 18 | E_SAR_ratio |
| 19 | G_API |
| 20 | V_ema_LST_modis_kobs30 |
| 21 | V_rollcv_E_SAR_diff_kobs30 |
| 22 | V_rollmean_F_NDVI_kobs30 |
| 23 | V_rollstd_E_SAR_ratio_kobs30 |
| 24 | D_sa_LST_modis |
| 25 | V_ema_E_SAR_ratio_kobs30 |
| 26 | V_ema_s2_b11_kobs30 |
| 27 | V_rollcv_E_SAR_ratio_kobs30 |
| 28 | V_rollmean_E_SAR_ratio_kobs30 |
| 29 | V_ema_s2_b12_kobs30 |
| 30 | V_ema_E_SAR_diff_kobs30 |
| 31 | F_NDMI |
| 32 | J_lc_code |
| 33 | V_rollrng_E_SAR_ratio_kobs30 |
| 34 | V_rollstd_s2_b12_kobs30 |
| 35 | V_rollmin_LST_modis_kobs14 |
| 36 | G_DSLR |
| 37 | V_rollmean_F_NDMI_kobs30 |
| 38 | J_soil_texture_usda_b200 |
| 39 | K_aspect_sin |
| 40 | V_rollstd_F_NDMI_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

> Note: These models have not been tuned or optimized in any way

| split   |     n |   dropped_nonfinite |      r2 |   rmse |   rel_rmse |    mae |   bias_me | model   |   n_features |
|:--------|------:|--------------------:|--------:|-------:|-----------:|-------:|----------:|:--------|-------------:|
| train   | 10183 |                   0 |  0.6259 | 0.0404 |     0.1474 | 0.0312 |   -0      | linear  |           45 |
| val     |  4210 |                   0 |  0.2532 | 0.0524 |     0.1836 | 0.0396 |   -0.0115 | linear  |           45 |
| test    |  5545 |                   0 | -0.2216 | 0.0617 |     0.2396 | 0.0491 |    0.0065 | linear  |           45 |
| train   | 10183 |                   0 |  0.9463 | 0.0153 |     0.0558 | 0.0107 |    0      | xgb     |           45 |
| val     |  4210 |                   0 |  0.2868 | 0.0512 |     0.1794 | 0.0379 |   -0.0157 | xgb     |           45 |
| test    |  5545 |                   0 | -0.1169 | 0.059  |     0.2291 | 0.0452 |   -0      | xgb     |           45 |
| train   | 10183 |                   0 |  0.9749 | 0.0104 |     0.0381 | 0.0063 |    0.0001 | rf      |           45 |
| val     |  4210 |                   0 |  0.3161 | 0.0502 |     0.1757 | 0.0365 |   -0.0107 | rf      |           45 |
| test    |  5545 |                   0 | -0.069  | 0.0577 |     0.2241 | 0.0433 |    0.0064 | rf      |           45 |
