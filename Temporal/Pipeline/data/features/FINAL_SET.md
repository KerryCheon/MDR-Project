# Final Set of Features

**Last Updated:** 2026-02-23

| Index | Name | Explanation |
| ---: | --- | --- |
| 1 | `SMAP_sm_pm_interp_ema02` | Exponential moving average (2) of interpolated SMAP PM soil moisture. |
| 2 | `V_rollmin_LST_modis_kobs30` | Rolling minimum of MODIS land-surface temperature (LST) over 30 observations. |
| 3 | `D_sin_DOY` | Sine transform of day-of-year for seasonal phase. |
| 4 | `G_rain_sum_3d` | Rainfall accumulation over 3 days. |
| 5 | `V_ema_G_API_kobs7` | Exponential moving average of antecedent precipitation index (API) over 7 observations. |
| 6 | `V_rollmin_G_API_kobs30` | Rolling minimum of antecedent precipitation index (API) over 30 observations. |
| 7 | `G_rain_sum_7d` | Rainfall accumulation over 7 days. |
| 8 | `C_lag_LST_modis_kobs30` | Lagged value of MODIS land-surface temperature (LST) by 30 observations. |
| 9 | `C_lag_G_API_kobs1` | Lagged value of antecedent precipitation index (API) by 1 observation. |
| 10 | `V_ema_G_API_kobs14` | Exponential moving average of antecedent precipitation index (API) over 14 observations. |
| 11 | `V_rollmean_G_API_kobs14` | Rolling mean of antecedent precipitation index (API) over 14 observations. |
| 12 | `G_API` | Antecedent precipitation index (API). |
| 13 | `A_pct_G_API` | Percent change of antecedent precipitation index (API). |
| 14 | `V_rollcv_G_API_kobs30` | Rolling coefficient of variation of antecedent precipitation index (API) over 30 observations. |
| 15 | `G_DSLR` | Days since last rain. |
| 16 | `SMAP_ampm_diff_interp` | Difference between interpolated SMAP AM and PM moisture. |
| 17 | `V_rollmax_G_API_kobs30` | Rolling maximum of antecedent precipitation index (API) over 30 observations. |
| 18 | `V_rollmin_G_API_kobs7` | Rolling minimum of antecedent precipitation index (API) over 7 observations. |
| 19 | `V_ema_G_API_kobs30` | Exponential moving average of antecedent precipitation index (API) over 30 observations. |
| 20 | `V_rollmean_s2_b11_kobs7` | Rolling mean of Sentinel-2 band 11 (SWIR) over 7 observations. |
| 21 | `V_ema_LST_modis_kobs7` | Exponential moving average of MODIS land-surface temperature (LST) over 7 observations. |
| 22 | `C_smm_G_API_alpha0.85_n5` | Soil-moisture memory index of antecedent precipitation index (API) (alpha=0.85, n=5). |
| 23 | `C_lag_G_API_kobs5` | Lagged value of antecedent precipitation index (API) by 5 observations. |
| 24 | `V_rollmean_G_API_kobs7` | Rolling mean of antecedent precipitation index (API) over 7 observations. |
| 25 | `C_lag_s2_b11_kobs30` | Lagged value of Sentinel-2 band 11 (SWIR) by 30 observations. |
| 26 | `D_z_LST_modis` | Z-score anomaly of MODIS land-surface temperature (LST). |
| 27 | `A_d_G_API_kobs1` | Difference in antecedent precipitation index (API) from current value to 1 observation ago. |
| 28 | `V_rollcv_LST_modis_kobs30` | Rolling coefficient of variation of MODIS land-surface temperature (LST) over 30 observations. |
| 29 | `V_rollcv_G_API_kobs7` | Rolling coefficient of variation of antecedent precipitation index (API) over 7 observations. |
| 30 | `V_rollstd_LST_modis_kobs30` | Rolling standard deviation of MODIS land-surface temperature (LST) over 30 observations. |
| 31 | `A_d_E_SAR_diff_kobs14` | Difference in SAR backscatter difference (VV-VH) from current value to 14 observations ago. |
| 32 | `C_lag_G_API_kobs6` | Lagged value of antecedent precipitation index (API) by 6 observations. |
| 33 | `V_rollrng_F_NDMI_kobs7` | Rolling range of NDMI over 7 observations. |
| 34 | `V_rollcv_G_API_kobs14` | Rolling coefficient of variation of antecedent precipitation index (API) over 14 observations. |
| 35 | `C_lag_LST_modis_kobs6` | Lagged value of MODIS land-surface temperature (LST) by 6 observations. |
| 36 | `A_d_E_SAR_diff_kobs30` | Difference in SAR backscatter difference (VV-VH) from current value to 30 observations ago. |
| 37 | `A_d_LST_modis_kobs14` | Difference in MODIS land-surface temperature (LST) from current value to 14 observations ago. |
| 38 | `SMAP_sm_am_interp_rollrange7` | Rolling range of interpolated SMAP AM soil moisture over 7 steps. |
| 39 | `V_rollstd_LST_modis_kobs14` | Rolling standard deviation of MODIS land-surface temperature (LST) over 14 observations. |
| 40 | `D_fft_ent_E_SAR_ratio_kobs30` | Spectral entropy of SAR backscatter ratio (VV/VH) over 30 observations. |
| 41 | `A_d_E_SAR_diff_kobs5` | Difference in SAR backscatter difference (VV-VH) from current value to 5 observations ago. |
| 42 | `SMAP_sm_pm_interp_rollrange7` | Rolling range of interpolated SMAP PM soil moisture over 7 steps. |
| 43 | `V_rollstd_F_NDMI_kobs7` | Rolling standard deviation of NDMI over 7 observations. |
| 44 | `V_rollstd_E_SAR_ratio_kobs7` | Rolling standard deviation of SAR backscatter ratio (VV/VH) over 7 observations. |
| 45 | `V_rollrng_E_SAR_diff_kobs7` | Rolling range of SAR backscatter difference (VV-VH) over 7 observations. |
| 46 | `V_rollstd_s2_b12_kobs7` | Rolling standard deviation of Sentinel-2 band 12 (SWIR) over 7 observations. |
| 47 | `A_grad_E_SAR_diff_kobs14` | Average gradient of SAR backscatter difference (VV-VH) over 14 observations. |
| 48 | `D_fft_dom_LST_modis_kobs30` | Dominant FFT frequency of MODIS land-surface temperature (LST) over 30 observations. |
| 49 | `V_rollcv_s2_b12_kobs7` | Rolling coefficient of variation of Sentinel-2 band 12 (SWIR) over 7 observations. |
| 50 | `A_d_E_SAR_ratio_kobs5` | Difference in SAR backscatter ratio (VV/VH) from current value to 5 observations ago. |
| 51 | `D_fft_ent_LST_modis_kobs30` | Spectral entropy of MODIS land-surface temperature (LST) over 30 observations. |
| 52 | `V_rollstd_F_NDVI_kobs7` | Rolling standard deviation of NDVI over 7 observations. |
| 53 | `A_grad_s2_b12_kobs7` | Average gradient of Sentinel-2 band 12 (SWIR) over 7 observations. |
| 54 | `A_pct_F_NDVI` | Percent change of NDVI. |
| 55 | `A_d_s2_b12_kobs2` | Difference in Sentinel-2 band 12 (SWIR) from current value to 2 observations ago. |
| 56 | `A_grad_E_SAR_diff_kobs30` | Average gradient of SAR backscatter difference (VV-VH) over 30 observations. |
| 57 | `A_d_F_NDVI_kobs2` | Difference in NDVI from current value to 2 observations ago. |
| 58 | `A_grad_E_SAR_diff_kobs7` | Average gradient of SAR backscatter difference (VV-VH) over 7 observations. |
| 59 | `SMAP_sm_interp_rollrange7` | Rolling range of interpolated SMAP soil moisture over 7 steps. |
| 60 | `A_d_s2_b12_kobs7` | Difference in Sentinel-2 band 12 (SWIR) from current value to 7 observations ago. |
| 61 | `A_d_F_NDVI_kobs1` | Difference in NDVI from current value to 1 observation ago. |
| 62 | `V_rollcv_LST_modis_kobs14` | Rolling coefficient of variation of MODIS land-surface temperature (LST) over 14 observations. |
| 63 | `SMAP_sm_am_interp_rollstd7` | Rolling standard deviation of interpolated SMAP AM soil moisture over 7 steps. |
| 64 | `V_rollstd_SMAP_sm_interp_kobs7` | Rolling standard deviation of interpolated SMAP soil moisture over 7 observations. |
| 65 | `A_d_s2_b12_kobs5` | Difference in Sentinel-2 band 12 (SWIR) from current value to 5 observations ago. |
| 66 | `A_pct_SMAP_sm_interp` | Percent change of interpolated SMAP soil moisture. |
| 67 | `SMAP_sm_am_interp_pctchg` | Percent change of interpolated SMAP AM soil moisture. |
| 68 | `V_rollrng_SMAP_sm_interp_kobs7` | Rolling range of interpolated SMAP soil moisture over 7 observations. |
| 69 | `SMAP_sm_interp_pctchg` | Percent change of interpolated SMAP soil moisture. |
| 70 | `A_d_E_SAR_diff_kobs2` | Difference in SAR backscatter difference (VV-VH) from current value to 2 observations ago. |
| 71 | `G_DSLR_isnan` | Indicator that DSLR is missing. |
| 72 | `SMAP_sm_pm_interp_mask` | Validity mask for interpolated SMAP PM soil moisture. |
| 73 | `SMAP_sm_interp_mask` | Validity mask for interpolated SMAP soil moisture. |
| 74 | `SMAP_sm_am_interp_mask` | Validity mask for interpolated SMAP AM soil moisture. |
| 75 | `SMAP_sm_am_interp_diff1` | First difference of interpolated SMAP AM soil moisture. |
| 76 | `SMAP_sm_interp_rollstd7` | Rolling standard deviation of interpolated SMAP soil moisture over 7 steps. |
| 77 | `SMAP_sm_interp_diff1` | First difference of interpolated SMAP soil moisture. |
| 78 | `V_rollstd_E_SAR_diff_kobs7` | Rolling standard deviation of SAR backscatter difference (VV-VH) over 7 observations. |
| 79 | `A_grad_s2_b12_kobs14` | Average gradient of Sentinel-2 band 12 (SWIR) over 14 observations. |
| 80 | `A_d_E_SAR_ratio_kobs7` | Difference in SAR backscatter ratio (VV/VH) from current value to 7 observations ago. |
| 81 | `A_grad_LST_modis_kobs14` | Average gradient of MODIS land-surface temperature (LST) over 14 observations. |
| 82 | `A_d_SMAP_sm_interp_kobs1` | Difference in interpolated SMAP soil moisture from current value to 1 observation ago. |
| 83 | `SMAP_sm_pm_interp_pctchg` | Percent change of interpolated SMAP PM soil moisture. |
| 84 | `A_grad_E_SAR_ratio_kobs7` | Average gradient of SAR backscatter ratio (VV/VH) over 7 observations. |
| 85 | `A_d_E_SAR_diff_kobs1` | Difference in SAR backscatter difference (VV-VH) from current value to 1 observation ago. |
| 86 | `A_d_E_SAR_diff_kobs7` | Difference in SAR backscatter difference (VV-VH) from current value to 7 observations ago. |
| 87 | `SMAP_sm_pm_interp_diff1` | First difference of interpolated SMAP PM soil moisture. |
| 88 | `A_d_F_NDVI_kobs5` | Difference in NDVI from current value to 5 observations ago. |
| 89 | `A_d_E_SAR_ratio_kobs2` | Difference in SAR backscatter ratio (VV/VH) from current value to 2 observations ago. |
| 90 | `A_d_G_API_kobs5` | Difference in antecedent precipitation index (API) from current value to 5 observations ago. |
| 91 | `A_d_SMAP_sm_interp_kobs2` | Difference in interpolated SMAP soil moisture from current value to 2 observations ago. |
| 92 | `D_fft_dom_E_SAR_ratio_kobs30` | Dominant FFT frequency of SAR backscatter ratio (VV/VH) over 30 observations. |
| 93 | `SMAP_sm_pm_interp_rollstd7` | Rolling standard deviation of interpolated SMAP PM soil moisture over 7 steps. |
| 94 | `V_rollrng_s2_b12_kobs7` | Rolling range of Sentinel-2 band 12 (SWIR) over 7 observations. |
| 95 | `V_rollrng_F_NDVI_kobs7` | Rolling range of NDVI over 7 observations. |
| 96 | `A_d_SMAP_sm_interp_kobs14` | Difference in interpolated SMAP soil moisture from current value to 14 observations ago. |
| 97 | `A_pct_E_SAR_ratio` | Percent change of SAR backscatter ratio (VV/VH). |
| 98 | `V_rollstd_SMAP_sm_interp_kobs30` | Rolling standard deviation of interpolated SMAP soil moisture over 30 observations. |
| 99 | `A_d_E_SAR_ratio_kobs1` | Difference in SAR backscatter ratio (VV/VH) from current value to 1 observation ago. |
| 100 | `A_pct_LST_modis` | Percent change of MODIS land-surface temperature (LST). |
| 101 | `A_grad_SMAP_sm_interp_kobs14` | Average gradient of interpolated SMAP soil moisture over 14 observations. |
| 102 | `A_pct_E_SAR_diff` | Percent change of SAR backscatter difference (VV-VH). |
| 103 | `SMAP_sm_interp_grad7` | Gradient of interpolated SMAP soil moisture over 7 steps. |
| 104 | `A_grad_SMAP_sm_interp_kobs7` | Average gradient of interpolated SMAP soil moisture over 7 observations. |
| 105 | `A_d_LST_modis_kobs1` | Difference in MODIS land-surface temperature (LST) from current value to 1 observation ago. |
| 106 | `V_rollcv_E_SAR_diff_kobs7` | Rolling coefficient of variation of SAR backscatter difference (VV-VH) over 7 observations. |
| 107 | `A_d_s2_b11_kobs5` | Difference in Sentinel-2 band 11 (SWIR) from current value to 5 observations ago. |
| 108 | `V_rollstd_LST_modis_kobs7` | Rolling standard deviation of MODIS land-surface temperature (LST) over 7 observations. |
| 109 | `slope` | Station terrain slope. |
| 110 | `elev` | Station elevation. |
| 111 | `K_slope_sin` | Sine transform of slope angle. |
| 112 | `K_slope_cos` | Cosine transform of slope angle. |
| 113 | `K_aspect_cos` | Cosine transform of terrain aspect. |
| 114 | `J_clay_wfrac_b0` | Clay mass fraction at top soil layer (b0). |
| 115 | `J_sand_wfrac_b0` | Sand mass fraction at top soil layer (b0). |
| 116 | `J_sand_clay_ratio_b0` | Sand-to-clay ratio at top soil layer (b0). |
