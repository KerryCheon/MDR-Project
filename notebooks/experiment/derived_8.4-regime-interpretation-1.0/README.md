# derived_8.4-regime-interpretation-1.0

Physical & environmental interpretation of the two KMeans regimes of the winning
`Clustering_V0_Full_k2` model from `derived_8.4-eval-1.1`, plus a first-time-reader
explanation of the model (clustering mechanism, feature-selection provenance, end-to-end
mermaid diagram). The notebook reproduces the exact eval-1.1 router (`V0FullRouter` from
`derived_8.4-eval-1.1/eval11/routers.py`, fitted on `derived_8.4` train+val) and is verified
against eval-1.1's per-regime sample counts.

## Run

From `notebooks/`:

```bash
nb execute experiment/derived_8.4-regime-interpretation-1.0/derived_8.4_regime_interpretation_1.0.ipynb --uv --timeout 300
```

## Router reproduction (sanity check)

| split    | expected (from eval-1.1) | got |
| :--- | :--- | :--- |
| trainval | (10624, 3984) | (10624, 3984) → OK |
| test | (4817, 1803) | (4817, 1803) → OK |

## Regime sizes

| split    | cluster | n | share |
|:---------|--------:|--:|:------|
| trainval | 0 | 10624 | 72.7% |
| trainval | 1 | 3984 | 27.3% |
| test     | 0 | 4817 | 72.8% |
| test     | 1 | 1803 | 27.2% |

## Target (soil_moisture_5cm) statistics per regime (trainval)

| cluster | n | mean | std | p10 | p50 | p90 | mean_shift_vs_global |
|--------:|--:|-----:|----:|----:|----:|----:|---------------------:|
| 0 | 10624 | 0.2216 | 0.107 | 0.041 | 0.239 | 0.347 | 0.0033 |
| 1 | 3984 | 0.2094 | 0.1098 | 0.044 | 0.244 | 0.335 | -0.0089 |

The target distributions overlap heavily: the regimes are **not** a wet/dry split of the target.

## Seasonality: share of rows in cluster 1 by month

| month | test | trainval |
|:------|-----:|---------:|
| Jan | 0.242 | 0.234 |
| Feb | 0.219 | 0.221 |
| Mar | 0.272 | 0.267 |
| Apr | 0.300 | 0.286 |
| May | 0.300 | 0.286 |
| Jun | 0.293 | 0.288 |
| Jul | 0.272 | 0.286 |
| Aug | 0.287 | 0.294 |
| Sep | 0.277 | 0.301 |
| Oct | 0.274 | 0.295 |
| Nov | 0.265 | 0.273 |
| Dec | 0.250 | 0.226 |

Cluster-1 share is nearly uniform across months: the regimes are **not seasonal**.

## Cluster-1 share per year

| split | year | n | share_cluster1 |
|:------|-----:|---:|---------------:|
| trainval | 2017 | 2383 | 0.26773 |
| trainval | 2018 | 2479 | 0.271077 |
| trainval | 2019 | 2422 | 0.263832 |
| trainval | 2020 | 2519 | 0.277094 |
| trainval | 2021 | 2471 | 0.281263 |
| trainval | 2022 | 2334 | 0.275064 |
| test | 2023 | 2346 | 0.266837 |
| test | 2024 | 2332 | 0.29717 |
| test | 2025 | 1942 | 0.249228 |

Stable across years: the regime definition does not drift over time.

## Station composition (trainval) — the regimes are 100% station-geographic

| station_id | n | share_cluster1 | dominant_cluster | longitude | latitude |
|:-----------|--:|---------------:|-----------------:|----------:|---------:|
| Spokane | 1793 | 1 | 1 | -117.53 | 47.42 |
| SourdoughGulch_WA_985 | 2191 | 1 | 1 | -117.4 | 46.2333 |
| BeaverPass_WA_990 | 2185 | 0 | 0 | -121.255 | 48.8793 |
| Darrington | 2048 | 0 | 0 | -121.45 | 48.54 |
| CayusePass_WA | 2042 | 0 | 0 | -121.534 | 46.8696 |
| Quinault | 2160 | 0 | 0 | -123.81 | 47.51 |
| Paradise_WA | 2189 | 0 | 0 | -121.748 | 46.7827 |

Mean station purity (trainval): **1.000** — every station is entirely in one regime.

## Station purity on the test split (out-of-sample)

| station_id | n_test | share_cluster1_test | purity_test |
|:-----------|-------:|--------------------:|------------:|
| Spokane | 897 | 1 | 1 |
| SourdoughGulch_WA_985 | 906 | 1 | 1 |
| BeaverPass_WA_990 | 626 | 0 | 1 |
| Darrington | 999 | 0 | 1 |
| CayusePass_WA | 1081 | 0 | 1 |
| Quinault | 1044 | 0 | 1 |
| Paradise_WA | 1067 | 0 | 1 |

Mean test purity: **1.000**. Cluster 1 = Spokane + SourdoughGulch (eastern, inland WA);
cluster 0 = the five western/mountain stations.

## Weather & dynamic drivers (trainval; rank_biserial_r > 0 ⇒ higher in cluster 1)

| feature | median_cluster0 | median_cluster1 | rank_biserial_r |
|:--------|----------------:|----------------:|----------------:|
| precip_mm | 0.7 | 0 | 0.216 |
| G_API | 47.4587 | 14.9358 | 0.551 |
| G_rain_sum_7d | 30.1 | 9.5 | 0.424 |
| LST_modis | 280.944 | 291.242 | -0.384 |
| SMAP_sm_pm_interp | 0.4141 | 0.1923 | 0.932 |
| G_DSLR | 0 | 1 | -0.169 |
| F_NDVI | 0.532 | 0.3296 | 0.326 |

Cluster 1 = hotter (~+10 K LST), much drier by satellite SMAP (0.19 vs 0.41), less rain,
less vegetation → **semi-arid**.

## Static / environmental attributes (trainval)

| feature | median_cluster0 | median_cluster1 | rank_biserial_r |
|:--------|----------------:|----------------:|----------------:|
| latitude | 47.51 | 46.233 | 0.642 |
| longitude | -121.534 | -117.4 | -1.0 |
| elev | 1205.09 | 1160.53 | 0.208 |
| slope | 18.149 | 20.225 | 0.112 |
| J_soil_texture_usda_b0 | 7 | 7 | 0.398 |
| J_lc_code | 10 | 30 | -0.55 |
| J_bio_bio13 | 391 | 70 | 1.0 |

Longitude (r = −1.0) and biome class (r = 1.0) separate the regimes perfectly.

## Top-15 V0 features separating the regimes (trainval)

| feature | median_cluster0 | median_cluster1 | rank_biserial_r |
|:--------|----------------:|----------------:|----------------:|
| K_aspect_cos | -0.5 | -0.891 | 1.0 |
| J_bio_bio15 | 61 | 34 | 1.0 |
| J_clay_wfrac_b100 | 9 | 23 | -1.0 |
| J_bio_bio02 | 97 | 133 | -1.0 |
| SMAP_sm_pm_interp_rollmean30 | 0.4147 | 0.1945 | 0.951 |
| V_ema_SMAP_sm_interp_kobs30 | 0.3702 | 0.1838 | 0.946 |
| SMAP_sm_pm_interp_lag30 | 0.4142 | 0.1931 | 0.93 |
| V_rollmin_s2_b12_kobs30 | 0.0458 | 0.137 | -0.917 |
| V_rollmean_F_NDMI_kobs30 | 0.3964 | 0.0124 | 0.893 |
| D_sa_F_NDMI | 0.0605 | -0.3353 | 0.882 |
| C_lag_F_NDMI_kobs30 | 0.3935 | 0.0096 | 0.861 |
| F_MSI | 0.4352 | 0.9809 | -0.859 |
| D_z_E_SAR_ratio | -0.2926 | 1.4568 | -0.848 |
| V_rollmax_E_SAR_ratio_kobs7 | 4.5249 | 5.9967 | -0.796 |
| V_rollmax_E_SAR_ratio_kobs30 | 4.7217 | 6.3671 | -0.793 |

Static climate/soil/terrain attributes (annual mean temperature `J_bio_bio02`: 133 vs 97;
clay fraction: 23 vs 9; aspect cosine) and long-window SMAP/NDMI means dominate — i.e. the
clustering separates on **regional climate & land-surface state**, not day-to-day weather.

## Key takeaway

The two KMeans regimes are **regional (spatial) specialists, not temporal wet/dry regimes**:
cluster 1 = the two semi-arid eastern-WA stations (hotter, drier SMAP/NDVI, more clay, no
rain most days), cluster 0 = the five wet, cool, forested western/mountain stations. In
`derived_8.4-eval-1.1` the cluster-1 expert (54 + 10 delta features, test R² = 0.8440) models
the eastern pair and the cluster-0 expert (54 features, test R² = 0.8025) models the western
five — i.e. the routing implements **within-state spatial specialization**, which is exactly
the generalization mode needed for the ECE team's new in-situ stations.

## Generated artifacts

| File | Description |
| :--- | :--- |
| `regime_target_distributions.png` | Soil-moisture KDE per regime |
| `regime_seasonality.png` | Cluster-1 share by month (trainval vs test) |
| `regime_geographic_distribution.png` | Dominant regime per station (WA map) |
| `regime_weather_drivers.png` | Weather/dynamic driver boxen plots per regime |
| `regime_static_attributes.png` | Static attribute boxen plots per regime |
| `regime_profile_summary.csv` | Per-regime medians + rank-biserial for target/drivers/static |
| `regime_station_composition.csv` | Per-station regime shares, purity, coordinates |
| `derived_8.4_regime_interpretation_1.0.ipynb` | Notebook (explanation + experiment) |

## References

- `notebooks/experiment/derived_8.4-eval-1.1/` — the two-regime MoE evaluation (winner:
  `Clustering_V0_Full_k2`, pooled test R² = 0.8150)
- `notebooks/experiment/derived_8.4-feature-selection-2.0/` — 54-feature backbone greedy search
- `notebooks/experiment/derived_8.3-gating-analysis-1.0/` — K=2 clustering diagnostics
  (elbow/silhouette/PCA/t-SNE) and exported clustering parameters
