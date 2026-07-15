# Feature Selection Report

## Run Info
- Run ID: 2026-07-14_193434
- Generated: 2026-07-14T19:36:26
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 50 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |
| Score | 0.7423 |
| Mean R2 | 0.8166 |
| Std R2 | 0.0109 |
| Train-Val Gap | 0.0941 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | DOY |
| 2 | D_fft_ent_E_SAR_ratio_kobs30 |
| 3 | D_sin_DOY |
| 4 | D_z_F_NDMI |
| 5 | F_NDMI |
| 6 | G_DSLR |
| 7 | J_bio_bio05 |
| 8 | J_clay_wfrac_b200 |
| 9 | J_lc_code |
| 10 | SMAP_ampm_diff_interp |
| 11 | V_ema_LST_modis_kobs30 |
| 12 | V_rollmin_F_NDMI_kobs30 |
| 13 | V_rollrng_F_NDVI_kobs30 |
| 14 | precip_mm |
| 15 | s2_b8 |
| 16 | sin_year |
| 17 | A_grad_E_SAR_diff_kobs30 |
| 18 | C_lag_LST_modis_kobs30 |
| 19 | D_cos_DOY |
| 20 | SMAP_x_year |
| 21 | V_rollrng_G_API_kobs7 |
| 22 | A_grad_E_SAR_ratio_kobs30 |
| 23 | G_rain_sum_3d |
| 24 | E_rough_s1_vv_kobs14 |
| 25 | V_rollmin_F_NDVI_kobs14 |
| 26 | V_rollmax_E_SAR_ratio_kobs14 |
| 27 | A_d_SMAP_sm_interp_kobs5 |
| 28 | V_rollrng_LST_modis_kobs30 |
| 29 | G_rain_sum_7d |
| 30 | D_fft_dom_F_NDMI_kobs30 |
| 31 | V_rollrng_E_SAR_diff_kobs7 |
| 32 | V_rollrng_s2_b11_kobs30 |
| 33 | V_rollrng_E_SAR_ratio_kobs7 |
| 34 | A_d_s2_b11_kobs14 |
| 35 | A_grad_F_NDMI_kobs30 |
| 36 | V_rollrng_s2_b11_kobs7 |
| 37 | V_rollmean_E_SAR_ratio_kobs14 |
| 38 | D_fft_dom_LST_modis_kobs30 |
| 39 | D_fft_ent_LST_modis_kobs30 |
| 40 | V_rollmax_SMAP_sm_interp_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

> Note: These models have not been tuned or optimized in any way

| split   |    n |   dropped_nonfinite |     r2 |   rmse |   rel_rmse |    mae |   bias_me | model   |   n_features |
|:--------|-----:|--------------------:|-------:|-------:|-----------:|-------:|----------:|:--------|-------------:|
| train   | 6868 |                   0 | 0.7583 | 0.0492 |     0.251  | 0.0384 |   -0      | linear  |           50 |
| val     | 2720 |                   0 | 0.8058 | 0.0463 |     0.2091 | 0.0351 |   -0.0101 | linear  |           50 |
| test    | 4016 |                   0 | 0.6567 | 0.0552 |     0.2612 | 0.043  |    0.0057 | linear  |           50 |
| train   | 6868 |                   0 | 0.9839 | 0.0127 |     0.0647 | 0.0086 |    0      | xgb     |           50 |
| val     | 2720 |                   0 | 0.8125 | 0.0455 |     0.2054 | 0.0362 |   -0.0202 | xgb     |           50 |
| test    | 4016 |                   0 | 0.649  | 0.0558 |     0.2641 | 0.0442 |   -0.0214 | xgb     |           50 |
| train   | 6868 |                   0 | 0.9897 | 0.0101 |     0.0517 | 0.0061 |    0      | rf      |           50 |
| val     | 2720 |                   0 | 0.8315 | 0.0431 |     0.1948 | 0.0337 |   -0.0175 | rf      |           50 |
| test    | 4016 |                   0 | 0.7455 | 0.0475 |     0.2248 | 0.0366 |   -0.0112 | rf      |           50 |
