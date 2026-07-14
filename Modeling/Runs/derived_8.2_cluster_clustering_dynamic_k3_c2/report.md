# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_dynamic_k3_c2
- Generated: 2026-07-14T16:34:40
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 46 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | DOY |
| 4 | D_cos_DOY |
| 5 | G_DSLR |
| 6 | G_rain_sum_3d |
| 7 | J_aspect_deg |
| 8 | J_soil_texture_usda_b0 |
| 9 | J_soil_texture_usda_b10 |
| 10 | SMAP_sm_pm_interp_rollrange30 |
| 11 | V_ema_LST_modis_kobs30 |
| 12 | V_rollmax_F_NDVI_kobs30 |
| 13 | latitude |
| 14 | lia_std_asc_deg |
| 15 | lia_std_desc_deg |
| 16 | sin_year |
| 17 | C_lag_LST_modis_kobs30 |
| 18 | V_ema_F_NDVI_kobs30 |
| 19 | D_fft_dom_LST_modis_kobs30 |
| 20 | s1_vh |
| 21 | E_rough_s1_vh_kobs7 |
| 22 | V_rollmin_s2_b11_kobs14 |
| 23 | D_sa_LST_modis |
| 24 | G_API |
| 25 | J_bio_bio15 |
| 26 | V_rollmin_s2_b11_kobs30 |
| 27 | A_grad_E_SAR_ratio_kobs14 |
| 28 | A_grad_SMAP_sm_interp_kobs30 |
| 29 | J_soil_texture_usda_b200 |
| 30 | V_rollmin_LST_modis_kobs30 |
| 31 | s2_b4 |
| 32 | V_ema_E_SAR_ratio_kobs30 |
| 33 | year_frac |
| 34 | C_lag_F_NDVI_kobs30 |
| 35 | F_NDMI |
| 36 | A_d_SMAP_sm_interp_kobs5 |
| 37 | V_rollrng_F_NDVI_kobs30 |
| 38 | C_lag_s2_b11_kobs30 |
| 39 | G_rain_sum_7d |
| 40 | V_ema_s2_b11_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
