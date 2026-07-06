# Feature Selection Report

## Run Info
- Run ID: 2026-07-06_140117
- Generated: 2026-07-06T14:31:28
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
| Score | 0.3972 |
| Mean R2 | 0.5268 |
| Std R2 | 0.0646 |
| Train-Val Gap | 0.2867 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | C_lag_G_API_kobs30 |
| 2 | C_smm_G_API_alpha0.85_n5 |
| 3 | DOY |
| 4 | D_cos_DOY |
| 5 | D_sin_DOY |
| 6 | D_z_F_NDMI |
| 7 | F_NDMI |
| 8 | G_DSLR |
| 9 | J_bio_bio01 |
| 10 | J_bio_bio05 |
| 11 | J_clay_plus_sand_b0 |
| 12 | J_clay_wfrac_b60 |
| 13 | J_lc_code |
| 14 | J_sand_clay_ratio_b0 |
| 15 | J_sand_wfrac_b0 |
| 16 | J_sand_wfrac_b10 |
| 17 | J_sand_wfrac_b100 |
| 18 | J_sand_wfrac_b200 |
| 19 | J_sand_wfrac_b30 |
| 20 | J_sand_wfrac_b60 |
| 21 | K_clay_plus_sand_b0 |
| 22 | K_sand_clay_ratio_b0 |
| 23 | K_slope_sin |
| 24 | SMAP_ampm_diff_interp |
| 25 | SMAP_x_year |
| 26 | V_rollmin_G_API_kobs30 |
| 27 | V_rollrng_F_NDVI_kobs30 |
| 28 | V_rollrng_G_API_kobs30 |
| 29 | aspect |
| 30 | latitude |
| 31 | longitude |
| 32 | sin_year |
| 33 | s2_b8 |
| 34 | J_slope_deg |
| 35 | G_rain_sum_7d |
| 36 | API_x_year |
| 37 | year |
| 38 | E_rough_s1_vv_kobs14 |
| 39 | A_grad_E_SAR_ratio_kobs30 |
| 40 | J_clay_wfrac_b100 |

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
| train   | 29362 |                   0 | 0.552  | 0.0733 |     0.4423 | 0.0586 |   -0      | linear  |           40 |
| val     | 13637 |                   0 | 0.4357 | 0.0857 |     0.544  | 0.0663 |    0.0219 | linear  |           40 |
| test    | 20166 |                   0 | 0.2649 | 0.0964 |     0.5937 | 0.0755 |    0.0265 | linear  |           40 |
| train   | 29362 |                   0 | 0.9055 | 0.0337 |     0.2031 | 0.0245 |    0      | xgb     |           40 |
| val     | 13637 |                   0 | 0.5776 | 0.0741 |     0.4706 | 0.0547 |    0.0171 | xgb     |           40 |
| test    | 20166 |                   0 | 0.4546 | 0.083  |     0.5114 | 0.061  |    0.013  | xgb     |           40 |
| train   | 29362 |                   0 | 0.983  | 0.0143 |     0.0861 | 0.0084 |   -0      | rf      |           40 |
| val     | 13637 |                   0 | 0.5671 | 0.0751 |     0.4764 | 0.0534 |    0.0162 | rf      |           40 |
| test    | 20166 |                   0 | 0.4278 | 0.085  |     0.5239 | 0.0602 |    0.012  | rf      |           40 |
