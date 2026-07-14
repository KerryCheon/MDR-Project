# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_dynamic_k3_c0
- Generated: 2026-07-14T16:33:10
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 34 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | A_d_F_NDMI_kobs30 |
| 2 | A_grad_F_NDMI_kobs30 |
| 3 | D_cos_DOY |
| 4 | D_sin_DOY |
| 5 | J_aspect_deg |
| 6 | J_bio_bio13 |
| 7 | K_aspect_cos |
| 8 | V_rollrng_LST_modis_kobs30 |
| 9 | cos_year |
| 10 | latitude |
| 11 | lia_std_asc_deg |
| 12 | sin_year |
| 13 | slope |
| 14 | V_rollrng_E_SAR_ratio_kobs30 |
| 15 | SMAP_sm_pm_interp_rollmean30 |
| 16 | G_API |
| 17 | V_rollmin_SMAP_sm_interp_kobs30 |
| 18 | DOY |
| 19 | J_bio_bio19 |
| 20 | V_rollmin_G_API_kobs30 |
| 21 | E_rough_s1_vv_kobs14 |
| 22 | G_rain_sum_7d |
| 23 | V_rollmax_s2_b12_kobs14 |
| 24 | J_bio_bio03 |
| 25 | V_ema_s2_b12_kobs30 |
| 26 | V_rollrng_s2_b12_kobs14 |
| 27 | E_rough_s1_vv_kobs7 |
| 28 | V_rollmax_s2_b12_kobs30 |
| 29 | A_d_SMAP_sm_interp_kobs14 |
| 30 | A_grad_SMAP_sm_interp_kobs14 |
| 31 | A_d_E_SAR_diff_kobs30 |
| 32 | A_grad_E_SAR_diff_kobs30 |
| 33 | J_bio_bio09 |
| 34 | C_lag_s2_b11_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
