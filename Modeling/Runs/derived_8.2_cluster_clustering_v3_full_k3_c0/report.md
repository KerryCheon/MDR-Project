# Feature Selection Report

## Run Info
- Run ID: derived_8.2_cluster_clustering_v3_full_k3_c0
- Generated: 2026-07-14T16:30:10
- Model: feature_selection
- Target: soil_moisture_5cm
- Time column: date
- ID columns: station_id

## Selection Summary

| Item | Value |
| --- | --- |
| Selected features | 44 |
| Stages | mi, elasticnet, stability |
| Top-k target | 50 |

## Top Selected Features

| # | Feature |
| --- | --- |
| 1 | API_x_year |
| 2 | DOY |
| 3 | D_cos_DOY |
| 4 | D_z_E_SAR_ratio |
| 5 | J_aspect_deg |
| 6 | J_soil_texture_usda_b0 |
| 7 | J_soil_texture_usda_b10 |
| 8 | K_aspect_cos |
| 9 | V_rollcv_F_NDMI_kobs30 |
| 10 | V_rollmin_SMAP_sm_interp_kobs30 |
| 11 | V_rollrng_E_SAR_ratio_kobs30 |
| 12 | cos_year |
| 13 | latitude |
| 14 | lia_std_asc_deg |
| 15 | sin_year |
| 16 | slope |
| 17 | D_fft_ent_F_NDMI_kobs30 |
| 18 | D_sa_F_NDMI |
| 19 | V_ema_LST_modis_kobs30 |
| 20 | E_SAR_ratio |
| 21 | K_slope_cos |
| 22 | SMAP_sm_pm_interp_lag30 |
| 23 | SMAP_sm_interp_rollrange30 |
| 24 | V_rollmin_G_API_kobs30 |
| 25 | V_rollmax_G_API_kobs7 |
| 26 | V_rollrng_F_NDMI_kobs14 |
| 27 | C_lag_LST_modis_kobs30 |
| 28 | D_sin_DOY |
| 29 | SMAP_sm_am_interp_rollrange30 |
| 30 | V_rollstd_G_API_kobs30 |
| 31 | V_rollmean_E_SAR_diff_kobs30 |
| 32 | C_lag_E_SAR_diff_kobs30 |
| 33 | lia_mean_desc_deg |
| 34 | E_rough_s1_vv_kobs14 |
| 35 | J_elev_m |
| 36 | SMAP_sm_pm_interp_rollmean7 |
| 37 | V_rollmax_E_SAR_ratio_kobs7 |
| 38 | V_rollcv_E_SAR_diff_kobs30 |
| 39 | V_rollrng_SMAP_sm_interp_kobs30 |
| 40 | V_rollmin_E_SAR_diff_kobs30 |

## Score Weights

| Metric | Weight |
| --- | --- |
| gap | -0.2000 |
| k_penalty | -0.0010 |
| mean_r2 | 1.0000 |
| std_r2 | -0.5000 |

## Metrics

_No metrics provided._
