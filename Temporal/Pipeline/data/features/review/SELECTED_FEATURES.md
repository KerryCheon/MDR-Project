# Temporal Feature Sets Used in the Models

**Authors:** Jakob Balkovec, Kerry Cheon
**Last updated:** Fri Jan 23

This is a simple reference for which feature sets were used by each model run. Scores are **Test** set metrics from `Models/Temporal/RESULTS_JAKOB.md` + Kerry's results log. Feature names are the exact code labels used in the models (kept as-is)

## How to read the scores

- **$R^2$**: higher is better (explained variance)
- **RMSE / MAE**: lower is better (error size)

## Notes

These sets essentially form a timeline of our modeling decisions. The early ones are raw baselines. From there, we layer in derived features once leakage is addressed, and finally we prune or expand based on what actually moves the needle.

We’d be happy to share additional details, data, or reports if helpful. Correlation and contribution analysis is not included in this notebook, but it is documented elsewhere. Our analysis shows that the features are very stable and provide a fairly strong signal. We validated this by removing features one by one and tracking changes in both error and $R^2$. That process made it clear that we are past the “add a clever feature to boost performance” phase. At this point, gains are coming less from new features and more from how specific parts of the data are modeled.

Most of our recent focus has been on understanding and stabilizing these harder regimes. Early on, rain appeared to be a major source of noise. Rain features fluctuate heavily and are difficult to model, which led us to experiment with modeling rain separately, potentially using a dedicated model to predict rain impact on soil moisture and then feeding that signal back in as a feature. That experiment did not improve performance, but it was still informative. It showed us that rain itself was not the core issue.

We then shifted our attention to seasonality. Initially, we assumed the problem was a simple summer versus winter split. What we found instead was that winter alone is the real challenge. I can’t fully explain the physical reasons behind this, as I’m not a domain expert and don’t yet have enough soil science background, but empirically the winter data is much harder to model and tends to inject noisy or misleading signal. We are intentionally not removing winter data, since doing so would inflate metrics and reduce generalization. We’re actively exploring ways to model winter behavior more effectively, but it has proven more difficult than we initially expected.

The next major direction is time series modeling. This is where we deliberately pumped the brakes. Building a proper time series model is nontrivial and would likely require more data. Given that, our current hypothesis is that adding spatial context could improve performance within the existing framework, effectively moving toward a spatio-temporal model rather than a purely temporal one.

**Motivation in plain terms**

- `v1.x`: start simple (raw bands + metadata) to get a clean baseline and sanity check the split.
- `v2.x`: fix leakage, then try a small, handpicked derived bundle to see if temporal dynamics help.
- `v3.x`: go all-in on derived families, then prune by importance to keep the signal and drop the noise.
- `v5.1`: swap in ElasticNet for a diagnostic pass (interpretability + feature selection).
- `v7.x`: pipeline-driven feature set, tuned models, and some careful pruning passes.
- `v8.x`: add rain impulse features to test explicit rainfall effects.

**What changed, and why**

- `Feature set 01-05`: raw inputs + light tweaks (baseline + early ablations like dropping `DOY`).
- `Feature set 06`: first curated derived set (handpicked, temporally valid).
- `Feature set 07`: derived set with one risky feature removed (`I_ts_spike_s1_vv`).
- `Feature set 08`: 40-feature derived subset (pruned from the full set).
- `Feature set 09`: rain impulse additions on top of the temporal core.
- `Feature set 10`: expanded temporal set used in the v7.x tuned runs.
- `Feature set 11`: the full derived family (89 features).

**Selection notes (where choices came from)**

- Ablation-driven tweaks: dropping `DOY` in v1.1, pruning low-importance families in v3.2, and removing `I_ts_spike_s1_vv` in v3.3.
- ElasticNet diagnostic: v5.1 runs on the same 40-feature core as v3.2, mainly for interpretability and feature selection signals.
- Feature pipeline: the derived families (A/B/C/D/E/F/G/H/I) powering v2.3 onward come straight from the temporal feature pipeline, and the v7.x feature bundle is pipeline-generated and then tuned.

## Feature set 01 (9 features)

Context: Baseline feature set built mostly from raw satellite bands and basic metadata.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v1.0.0 | `Models/Temporal/v1/v1.0/mdr_ts_v2_3_20260108_115420` | 0.520847 | 0.064362 | 0.049829 |
| v2.2.0 | `Models/Temporal/v2/v2.2/mdr_ts_v2_2_20260107_162107` | 0.506571 | 0.065313 | 0.050461 |

Features used:

```yaml
features:
  - rain_mm
  - precip_mm
  - NDVI
  - NDMI
  - MSI
  - s1_vv
  - s1_vh
  - SAR_ratio
  - DOY
```

![Feature set 01 family breakdown](piecharts/feature_set_01.svg)

## Feature set 02 (12 features)

Context: Baseline feature set built mostly from raw satellite bands and basic metadata.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v1.2.0 | `Models/Temporal/v1/v1.2/mdr_ts_v1_2_20251223_113808` | 0.201255 | 0.068399 | 0.057298 |

Features used:

```yaml
features:
  - rain_mm
  - precip_mm
  - air_temp_mean
  - solar_radiation
  - NDVI
  - NDMI
  - MSI
  - s1_vv
  - s1_vh
  - SAR_ratio
  - elev
  - slope
```

![Feature set 02 family breakdown](piecharts/feature_set_02.svg)

## Feature set 03 (12 features)

Context: Baseline feature set built mostly from raw satellite bands and basic metadata.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v2.1.0 | `Models/Temporal/v2/v2.1/mdr_ts_v2_1_20260106_211015` | 0.712817 | 0.052080 | 0.039524 |
| v2.2.0 | `Models/Temporal/v2/v2.2/mdr_ts_v2_2_20260107_161235` | 0.506571 | 0.065313 | 0.050461 |

Features used:

```yaml
features:
  - rain_mm
  - precip_mm
  - air_temp_mean
  - rh_mean
  - solar_radiation
  - NDVI
  - NDMI
  - MSI
  - s1_vv
  - s1_vh
  - SAR_ratio
  - DOY
```

![Feature set 03 family breakdown](piecharts/feature_set_03.svg)

## Feature set 04 (13 features)

Context: Baseline feature set built mostly from raw satellite bands and basic metadata.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v1.1.0 | `Models/Temporal/v1/v1.1/mdr_ts_v1_1_20251223_113030` | 0.017134 | 0.075874 | 0.064540 |

Features used:

```yaml
features:
  - rain_mm
  - precip_mm
  - air_temp_mean
  - rh_mean
  - solar_radiation
  - NDVI
  - NDMI
  - MSI
  - s1_vv
  - s1_vh
  - SAR_ratio
  - elev
  - slope
```

![Feature set 04 family breakdown](piecharts/feature_set_04.svg)

## Feature set 05 (14 features)

Context: Baseline feature set built mostly from raw satellite bands and basic metadata.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v1.0.0 | `Models/Temporal/v1/v1.0/mdr_ts_v1_0_20251223_105722` | 0.520847 | 0.064362 | 0.049829 |

Features used:

```yaml
features:
  - rain_mm
  - precip_mm
  - air_temp_mean
  - rh_mean
  - solar_radiation
  - NDVI
  - NDMI
  - MSI
  - s1_vv
  - s1_vh
  - SAR_ratio
  - elev
  - slope
  - DOY
```

![Feature set 05 family breakdown](piecharts/feature_set_05.svg)

## Feature set 06 (22 features)

Context: Compact, handpicked derived set focused on short-term dynamics and precipitation memory.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v2.3.0 | `Models/Temporal/v2/v2.3/mdr_ts_v2_2_20260107_170214` | 0.671525 | 0.053289 | 0.041054 |

Features used:

```yaml
features:
  - precip_mm
  - s1_vv
  - s1_vh
  - s2_b8
  - s2_b11
  - LST_modis
  - DOY
  - NDMI
  - SAR_ratio
  - API
  - DSLR
  - grad_API_7
  - rollstd_API_7
  - grad_NDMI_7
  - rollstd_NDMI_7
  - grad_SAR_ratio_7
  - rollstd_SAR_ratio_7
  - grad_LST_modis_7
  - rollstd_LST_modis_7
  - NDMI_sa
  - SAR_ratio_sa
  - LST_modis_sa
```

![Feature set 06 family breakdown](piecharts/feature_set_06.svg)

## Feature set 07 (39 features)

Context: Derived temporal features (lags, rolling stats, and seasonal anomalies) built on top of core inputs.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v3.3.0 | `Models/Temporal/v3/v3.3/mdr_ts_v3_3_20260109_113817` | 0.714397 | 0.049690 | 0.038198 |
| v6.1.0 | `Models/Temporal/v6/v6.1/mdr_ts_rf_v6_1_20260111_012146` | N/A | N/A | N/A |

Features used:

```yaml
features:
  - A_d_E_SAR_ratio_kobs1
  - A_d_F_NDMI_kobs1
  - A_d_G_API_kobs1
  - A_d_LST_modis_kobs1
  - C_smm_E_SAR_ratio_alpha0.85_n5
  - C_smm_F_NDMI_alpha0.85_n5
  - C_smm_G_API_alpha0.85_n5
  - C_smm_LST_modis_alpha0.85_n5
  - DOY
  - D_sa_E_SAR_ratio
  - D_sa_F_NDMI
  - D_sa_LST_modis
  - D_z_E_SAR_ratio
  - D_z_F_NDMI
  - D_z_LST_modis
  - E_SAR_diff
  - E_SAR_ratio
  - F_MSI
  - F_NDMI
  - F_NDVI
  - G_API
  - G_DSLR
  - G_DSLR_isnan
  - G_rain_sum_30d
  - G_rain_sum_3d
  - G_rain_sum_7d
  - LST_modis
  - V_rollstd_E_SAR_ratio_kobs7
  - V_rollstd_F_NDMI_kobs7
  - V_rollstd_G_API_kobs7
  - V_rollstd_LST_modis_kobs7
  - aspect
  - precip_mm
  - s1_vh
  - s1_vv
  - s2_b11
  - s2_b12
  - s2_b4
  - s2_b8
```

![Feature set 07 family breakdown](piecharts/feature_set_07.svg)

## Feature set 08 (40 features)

Context: Derived temporal features (lags, rolling stats, and seasonal anomalies) built on top of core inputs.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v3.1.0 | `Models/Temporal/v3/v3.1/mdr_ts_v3_1_20260108_151234` | 0.632453 | 0.056370 | 0.043802 |
| v3.2.0 | `Models/Temporal/v3/v3.2/mdr_ts_v3_2_20260108_151018` | 0.628125 | 0.056701 | 0.043800 |

Features used:

```yaml
features:
  - A_d_E_SAR_ratio_kobs1
  - A_d_F_NDMI_kobs1
  - A_d_G_API_kobs1
  - A_d_LST_modis_kobs1
  - C_smm_E_SAR_ratio_alpha0.85_n5
  - C_smm_F_NDMI_alpha0.85_n5
  - C_smm_G_API_alpha0.85_n5
  - C_smm_LST_modis_alpha0.85_n5
  - DOY
  - D_sa_E_SAR_ratio
  - D_sa_F_NDMI
  - D_sa_LST_modis
  - D_z_E_SAR_ratio
  - D_z_F_NDMI
  - D_z_LST_modis
  - E_SAR_diff
  - E_SAR_ratio
  - F_MSI
  - F_NDMI
  - F_NDVI
  - G_API
  - G_DSLR
  - G_DSLR_isnan
  - G_rain_sum_30d
  - G_rain_sum_3d
  - G_rain_sum_7d
  - I_ts_spike_s1_vv
  - LST_modis
  - V_rollstd_E_SAR_ratio_kobs7
  - V_rollstd_F_NDMI_kobs7
  - V_rollstd_G_API_kobs7
  - V_rollstd_LST_modis_kobs7
  - aspect
  - precip_mm
  - s1_vh
  - s1_vv
  - s2_b11
  - s2_b12
  - s2_b4
  - s2_b8
```

![Feature set 08 family breakdown](piecharts/feature_set_08.svg)

## Feature set 09 (41 features)

Context: Adds explicit rain-impulse signals on top of the core temporal features.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v8.1.0 | `Models/Temporal/v8/v8.1/mdr_ts_v8_1_20260121_105425` | 0.674228 | 0.053383 | 0.041720 |

Features used:

```yaml
features:
  - precip_mm
  - G_rain_sum_30d
  - C_lag_E_SAR_diff_kobs12
  - C_lag_E_SAR_diff_kobs30
  - C_lag_E_SAR_ratio_kobs30
  - C_lag_F_NDVI_kobs30
  - C_lag_LST_modis_kobs12
  - C_lag_LST_modis_kobs30
  - DOY
  - D_sa_E_SAR_ratio
  - D_sa_F_NDMI
  - D_z_E_SAR_ratio
  - D_z_F_NDMI
  - E_SAR_diff
  - E_SAR_ratio
  - F_MSI
  - F_NDMI
  - V_ema_LST_modis_kobs30
  - V_rollmax_E_SAR_diff_kobs14
  - V_rollmax_E_SAR_diff_kobs30
  - V_rollmax_F_NDVI_kobs30
  - V_rollmax_LST_modis_kobs7
  - V_rollmax_s2_b11_kobs30
  - V_rollmin_E_SAR_diff_kobs30
  - V_rollmin_E_SAR_ratio_kobs30
  - V_rollmin_F_NDMI_kobs30
  - V_rollmin_s2_b11_kobs30
  - V_rollmin_s2_b12_kobs30
  - s1_vh
  - s2_b8
  - A_d_LST_modis_kobs7
  - A_grad_LST_modis_kobs14
  - C_lag_E_SAR_diff_kobs6
  - V_rollmax_E_SAR_diff_kobs7
  - V_rollmax_E_SAR_ratio_kobs14
  - V_rollmax_F_NDVI_kobs14
  - s2_b12
  - D_sa_LST_modis
  - rain_event_impulse_0_7
  - rain_mm_impulse_0_7
  - days_since_rain_event
```

![Feature set 09 family breakdown](piecharts/feature_set_09.svg)

## Feature set 10 (46 features)

Context: Derived temporal features (lags, rolling stats, and seasonal anomalies) built on top of core inputs.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v7.1.0 | `Models/Temporal/v7/v7.1/mdr_ts_v7_1_20260117_161048` | 0.757695 | 0.046039 | 0.036493 |
| v7.3.0 | `Models/Temporal/v7/v7.3/mdr_ts_v7_3_20260117_165132` | 0.757695 | 0.046039 | 0.036493 |
| v7.5.0 | `Models/Temporal/v7/v7.5/mdr_ts_v7_5_20260118_135014` | 0.709898 | 0.050376 | 0.039043 |
| v7.6.0 | `Models/Temporal/v7/v7.6/mdr_ts_v7_6_20260118_141451` | N/A | N/A | N/A |

Features used:

```yaml
features:
  - precip_mm
  - G_rain_sum_3d
  - G_rain_sum_7d
  - G_rain_sum_30d
  - G_API
  - G_DSLR
  - C_lag_E_SAR_diff_kobs12
  - C_lag_E_SAR_diff_kobs30
  - C_lag_E_SAR_ratio_kobs30
  - C_lag_F_NDVI_kobs30
  - C_lag_LST_modis_kobs12
  - C_lag_LST_modis_kobs30
  - DOY
  - D_sa_E_SAR_ratio
  - D_sa_F_NDMI
  - D_z_E_SAR_ratio
  - D_z_F_NDMI
  - E_SAR_diff
  - E_SAR_ratio
  - F_MSI
  - F_NDMI
  - V_ema_LST_modis_kobs30
  - V_rollmax_E_SAR_diff_kobs14
  - V_rollmax_E_SAR_diff_kobs30
  - V_rollmax_F_NDVI_kobs30
  - V_rollmax_G_API_kobs30
  - V_rollmax_G_API_kobs7
  - V_rollmax_LST_modis_kobs7
  - V_rollmax_s2_b11_kobs30
  - V_rollmean_G_API_kobs30
  - V_rollmin_E_SAR_diff_kobs30
  - V_rollmin_E_SAR_ratio_kobs30
  - V_rollmin_F_NDMI_kobs30
  - V_rollmin_G_API_kobs7
  - V_rollmin_s2_b11_kobs30
  - V_rollmin_s2_b12_kobs30
  - s1_vh
  - s2_b8
  - A_d_LST_modis_kobs7
  - A_grad_LST_modis_kobs14
  - C_lag_E_SAR_diff_kobs6
  - V_rollmax_E_SAR_diff_kobs7
  - V_rollmax_E_SAR_ratio_kobs14
  - V_rollmax_F_NDVI_kobs14
  - s2_b12
  - D_sa_LST_modis
```

![Feature set 10 family breakdown](piecharts/feature_set_10.svg)

## Feature set 11 (89 features)

Context: Derived temporal features (lags, rolling stats, and seasonal anomalies) built on top of core inputs.

Model runs:
| Version | Run folder | Test $R^2$ | Test RMSE | Test MAE |
| --- | --- | --- | --- | --- |
| v3.1.0 | `Models/Temporal/v3/v3.1/mdr_ts_v3_1_20260108_144926` | 0.632453 | 0.056370 | 0.043802 |
| v3.2.0 | `Models/Temporal/v3/v3.2/mdr_ts_v3_2_20260108_151136` | 0.628125 | 0.056701 | 0.043800 |

Features used:

```yaml
features:
  - precip_mm
  - s1_vv
  - s1_vh
  - s2_b4
  - s2_b8
  - s2_b11
  - s2_b12
  - LST_modis
  - aspect
  - DOY
  - F_NDVI
  - F_NDMI
  - F_MSI
  - E_SAR_ratio
  - E_SAR_diff
  - E_dVV_1
  - G_API
  - G_DSLR
  - G_DSLR_isnan
  - G_rain_sum_3d
  - G_rain_sum_7d
  - G_rain_sum_30d
  - A_d_G_API_kobs1
  - A_d_G_API_kobs2
  - A_grad_G_API_kobs7
  - A_pct_G_API
  - V_rollstd_G_API_kobs7
  - V_rollrng_G_API_kobs7
  - V_rollcv_G_API_kobs7
  - V_rollmean_G_API_kobs7
  - V_rollmin_G_API_kobs7
  - V_rollmax_G_API_kobs7
  - V_ema_G_API_kobs7
  - C_lag_G_API_kobs1
  - C_lag_G_API_kobs2
  - C_lag_G_API_kobs5
  - C_smm_G_API_alpha0.85_n5
  - A_d_F_NDMI_kobs1
  - A_d_F_NDMI_kobs2
  - A_grad_F_NDMI_kobs7
  - A_pct_F_NDMI
  - V_rollstd_F_NDMI_kobs7
  - V_rollrng_F_NDMI_kobs7
  - V_rollcv_F_NDMI_kobs7
  - V_rollmean_F_NDMI_kobs7
  - V_rollmin_F_NDMI_kobs7
  - V_rollmax_F_NDMI_kobs7
  - V_ema_F_NDMI_kobs7
  - C_lag_F_NDMI_kobs1
  - C_lag_F_NDMI_kobs2
  - C_lag_F_NDMI_kobs5
  - C_smm_F_NDMI_alpha0.85_n5
  - A_d_E_SAR_ratio_kobs1
  - A_d_E_SAR_ratio_kobs2
  - A_grad_E_SAR_ratio_kobs7
  - A_pct_E_SAR_ratio
  - V_rollstd_E_SAR_ratio_kobs7
  - V_rollrng_E_SAR_ratio_kobs7
  - V_rollcv_E_SAR_ratio_kobs7
  - V_rollmean_E_SAR_ratio_kobs7
  - V_rollmin_E_SAR_ratio_kobs7
  - V_rollmax_E_SAR_ratio_kobs7
  - V_ema_E_SAR_ratio_kobs7
  - C_lag_E_SAR_ratio_kobs1
  - C_lag_E_SAR_ratio_kobs2
  - C_lag_E_SAR_ratio_kobs5
  - C_smm_E_SAR_ratio_alpha0.85_n5
  - A_d_LST_modis_kobs1
  - A_d_LST_modis_kobs2
  - A_grad_LST_modis_kobs7
  - A_pct_LST_modis
  - V_rollstd_LST_modis_kobs7
  - V_rollrng_LST_modis_kobs7
  - V_rollcv_LST_modis_kobs7
  - V_rollmean_LST_modis_kobs7
  - V_rollmin_LST_modis_kobs7
  - V_rollmax_LST_modis_kobs7
  - V_ema_LST_modis_kobs7
  - C_lag_LST_modis_kobs1
  - C_lag_LST_modis_kobs2
  - C_lag_LST_modis_kobs5
  - C_smm_LST_modis_alpha0.85_n5
  - I_ts_spike_s1_vv
  - D_sa_F_NDMI
  - D_z_F_NDMI
  - D_sa_E_SAR_ratio
  - D_z_E_SAR_ratio
  - D_sa_LST_modis
  - D_z_LST_modis
```

![Feature set 11 family breakdown](piecharts/feature_set_11.svg)

---

_Jakob Balkovec & Kerry Cheon_
