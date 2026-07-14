# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_v3_full_k3_c2
- Generated: 2026-07-14T16:32:24
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 48 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | A_d_E_SAR_ratio_kobs30 |
| 3 | A_grad_E_SAR_ratio_kobs30 |
| 4 | D_cos_DOY |
| 5 | D_z_E_SAR_ratio |
| 6 | J_aspect_deg |
| 7 | J_bio_bio15 |
| 8 | K_aspect_cos |
| 9 | K_aspect_sin |
| 10 | SMAP_sm_pm_interp_rollrange30 |
| 11 | V_ema_E_SAR_ratio_kobs30 |
| 12 | V_ema_LST_modis_kobs30 |
| 13 | V_rollmax_F_NDVI_kobs30 |
| 14 | V_rollmin_E_SAR_ratio_kobs30 |
| 15 | V_rollmin_s2_b11_kobs14 |
| 16 | V_rollmin_s2_b12_kobs30 |
| 17 | latitude |
| 18 | sin_year |
| 19 | C_lag_LST_modis_kobs30 |
| 20 | G_rain_sum_3d |
| 21 | J_soil_texture_usda_b10 |
| 22 | LST_modis |
| 23 | SMAP_x_year |
| 24 | V_rollrng_G_API_kobs7 |
| 25 | precip_mm |
| 26 | C_lag_F_NDVI_kobs30 |
| 27 | V_ema_F_NDVI_kobs30 |
| 28 | lia_std_desc_deg |
| 29 | s2_b4 |
| 30 | D_sin_DOY |
| 31 | DOY |
| 32 | A_d_SMAP_sm_interp_kobs5 |
| 33 | J_soil_texture_usda_b0 |
| 34 | V_rollstd_G_API_kobs14 |
| 35 | V_rollmax_F_NDVI_kobs14 |
| 36 | V_rollrng_SMAP_sm_interp_kobs14 |
| 37 | V_rollmax_LST_modis_kobs14 |
| 38 | V_rollmean_LST_modis_kobs30 |
| 39 | D_fft_ent_LST_modis_kobs30 |
| 40 | E_rough_s1_vv_kobs14 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
