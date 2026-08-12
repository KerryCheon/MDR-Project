# derived_8.4-gating-analysis-1.0

Diagnostic analysis and parameter export for gating strategies on the `derived_8.4` dataset
split, extended from `derived_8.3-gating-analysis-1.0` (K=2 strategies, clustering
diagnostics, parameter export) with the post-clustering per-regime physical interpretation of
`derived_8.4-regime-interpretation-1.0`, and with a K-sweep (K = 2, 3, 4) designed to answer
**"why 2 regimes?"**.

## Overview & Purpose

- **Dataset**: `derived_8.4` split (7 WA stations; train 9,803 / val 4,805 / test 6,620 rows).
- **Primary new strategy**: `Clustering_Backbone54` — KMeans on the exact 54-feature
  `shared_backbone_54` used by the single global model and by both MoE experts (loaded from
  `derived_8.4-eval-1.1/selected_features.json`, not redefined). The two-regime model is thus
  directly describable as built from the single-regime global model: same features, one shared
  backbone, two regional specialists.
- **Scaling**: every clustering pipeline mean-imputes missing values, then standardizes with
  `StandardScaler` (z-scores) before KMeans — same recipe as the base notebook and the eval-1.1
  winning router; a dedicated diagnostic (§5) demonstrates why standardization is required.
- **K-sweep**: all five clustering strategies are run at K = 2, 3, 4, with quality metrics for
  K = 2…6, and the full per-regime interpretation at K = 2, 3, 4 for `Clustering_Backbone54`
  (plus K = 2 for the retained `Clustering_V0_Full`).
- **Additional feature sets**: `Clustering_Static` (58 per-station constants) and
  `Clustering_Weather` (16 dynamic drivers) in addition to the retained `Clustering_Dynamic`
  (3 features) and `Clustering_V0_Full` (50 V0 features).
- All clustering is fitted on **train + val ("trainval")** and applied to test, exactly like the
  eval-1.1 router. (The base `derived_8.3` notebook fitted on train only; this difference is
  intentional and lets the interpretation verify regimes out-of-sample.)

## Run

From `notebooks/`:

```bash
nb execute experiment/derived_8.4-gating-analysis-1.0/derived_8.4_gating_analysis_1.0.ipynb --uv --timeout 900
```

No GPU training is involved (clustering + diagnostics only).

## Feature sets used for clustering

| Strategy | # Features | Definition |
| :--- | ---: | :--- |
| `Clustering_Dynamic` (retained) | 3 | `SMAP_sm_pm_interp_lag1`, `G_API`, `LST_modis` |
| `Clustering_V0_Full` (retained) | 50 | `OVERALL_SELECTED_FEATURES_V0` from `data/splits/derived_8.4/dataset_metadata.py` |
| `Clustering_Backbone54` (new, primary) | 54 | `shared_backbone_54` from `derived_8.4-eval-1.1/selected_features.json` |
| `Clustering_Static` (new) | 58 | Columns with zero within-station variance (coordinates, terrain, soil, land cover, bioclimatic normals); derived programmatically; data-availability `*_mask` / `*_isnan` flags excluded |
| `Clustering_Weather` (new) | 16 | Core dynamic drivers: precip, LST, SMAP (3 products + AM/PM diff), NDVI, NDMI, MSI, SAR ratio/diff, G_API, G_DSLR, 3/7/30-day rain sums |

## Evaluated gating strategies (trainval; K = 2, 3, 4)

| Strategy | K | Features Used | Group Sizes | Top 20 Divergence | Max Drift | Drift Feature |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Univariate_G_API | 2 | - | 7304, 7304 | 0.1715 | 0.5864 | V_rollmax_G_API_kobs7 |
| Seasonal_Binary | 2 | - | 7559, 7049 | 0.1661 | 0.5757 | V_ema_G_API_kobs30 |
| Clustering_Dynamic | 2 | 3 | 7974, 6634 | 0.1599 | 0.5026 | V_ema_F_NDVI_kobs30 |
| Clustering_Dynamic | 3 | 3 | 4717, 3891, 6000 | 0.2019 | 0.7641 | V_ema_G_API_kobs30 |
| Clustering_Dynamic | 4 | 3 | 2173, 2350, 5048, 5037 | 0.2187 | 0.9025 | latitude |
| Clustering_V0_Full | 2 | 50 | 10624, 3984 | 0.1133 | 1.0725 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_V0_Full | 3 | 50 | 4247, 3968, 6393 | 0.1318 | 1.3793 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_V0_Full | 4 | 50 | 4247, 3968, 4158, 2235 | 0.1622 | 1.3045 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_Backbone54 | 2 | 54 | 10624, 3984 | 0.1133 | 1.0725 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_Backbone54 | 3 | 54 | 6513, 3857, 4238 | 0.2386 | 0.9488 | V_ema_SMAP_sm_interp_kobs30 |
| Clustering_Backbone54 | 4 | 54 | 2142, 4053, 3826, 4587 | 0.2209 | 1.1132 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_Static | 2 | 58 | 8192, 6416 | 0.0822 | 0.5829 | slope |
| Clustering_Static | 3 | 58 | 4208, 6416, 3984 | 0.178 | 1.3249 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_Static | 4 | 58 | 4231, 3984, 2185, 4208 | 0.155 | 1.3249 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_Weather | 2 | 16 | 3870, 10738 | 0.103 | 0.9687 | SMAP_sm_pm_interp_rollmean30 |
| Clustering_Weather | 3 | 16 | 3794, 6230, 4584 | 0.2007 | 0.9901 | SMAP_sm_pm_interp_lag30 |
| Clustering_Weather | 4 | 16 | 5695, 3517, 1694, 3702 | 0.2179 | 1.0907 | V_ema_SMAP_sm_interp_kobs30 |

`Clustering_Backbone54` K=2 produces the **exact same partition as the eval-1.1 winning
`Clustering_V0_Full` K=2 router** (group sizes 10,624 / 3,984; ARI = 1.0000), so routing on the
54-feature backbone preserves the proven split.

## Why standardization matters (Backbone54, K=2)

Without `StandardScaler`, a single high-range feature dominates the KMeans distance:

| Metric | Value |
| :--- | :--- |
| Silhouette (unscaled) | 0.5264 |
| Silhouette (scaled) | 0.2171 |
| ARI between unscaled/scaled partitions | 0.4886 |

The unscaled centroid separation is dominated by `J_bio_bio13` (annual precipitation; std ≈ 176),
i.e. the unscaled solution is effectively a univariate threshold; after z-scoring, separation is
spread across many features (SMAP roll statistics, LST/NDMI windows, terrain). The higher raw
silhouette is an artifact of near-univariate thresholding, not evidence of a better partition.
All exported routers therefore standardize (`clustering_scaling_diagnostic.png`).

## Cluster quality metrics vs K (Backbone54; full table in `cluster_quality_metrics.csv`)

| K | WSS Inertia | Avg Silhouette | Calinski-Harabasz | Davies-Bouldin |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 623,236 | 0.2171 | 3,880.86 | 1.5792 |
| 3 | 528,875 | 0.2234 | 3,589.38 | 1.8155 |
| 4 | 497,221 | 0.1832 | 2,855.00 | 2.0150 |
| 5 | 472,821 | 0.1658 | 2,439.99 | 1.9943 |
| 6 | 450,825 | 0.1312 | 2,189.58 | 2.0726 |

Silhouette peaks (marginally) at K=3 while Calinski-Harabasz and Davies-Bouldin are best at K=2;
K=4 is worse than K=2 on every index. Note `Clustering_Static`'s silhouette keeps rising with K
(0.51 → 0.93): with 7 stations, more clusters trivially separate station points, which is exactly
why internal quality indices alone cannot justify large K.

## Why 2 regimes? (Clustering_Backbone54 across K)

| K | silhouette | calinski_harabasz | davies_bouldin | div_top20 | max_drift | station_purity_trainval |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0.2171 | 3,880.86 | 1.5792 | 0.1133 | 1.0725 | 1.000 |
| 3 | 0.2234 | 3,589.38 | 1.8155 | 0.2386 | 0.9488 | 0.833 |
| 4 | 0.1832 | 2,855.00 | 2.0150 | 0.2209 | 1.1132 | 0.695 |

Adjusted Rand index between partitions (trainval):
ARI(K=2, K=3) = 0.5090 · ARI(K=2, K=4) = 0.3633 · ARI(K=3, K=4) = 0.7541

Dominant cluster per station across K (trainval):

| station_id | K2_dominant | K2_purity | K3_dominant | K3_purity | K4_dominant | K4_purity |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| BeaverPass_WA_990 | 0 | 1 | 0 | 0.500229 | 1 | 0.475973 |
| CayusePass_WA | 0 | 1 | 2 | 0.743389 | 1 | 0.713516 |
| Darrington | 0 | 1 | 0 | 0.982422 | 3 | 0.609863 |
| Paradise_WA | 0 | 1 | 2 | 0.668799 | 1 | 0.62677 |
| Quinault | 0 | 1 | 0 | 0.999537 | 0 | 0.518056 |
| SourdoughGulch_WA_985 | 1 | 1 | 1 | 0.966225 | 2 | 0.955272 |
| Spokane | 1 | 1 | 1 | 0.970441 | 2 | 0.966537 |

**The case for K=2**: K=2 is the largest K at which every regime corresponds to a whole station
group (purity = 1.000 on trainval **and** test). At K=3, BeaverPass is split ~50/50 and
CayusePass/Paradise are cut into two clusters (mean purity 0.833); at K=4, Quinault, BeaverPass
and Darrington are all fragmented (mean purity 0.695) — the extra "regimes" stop being physical
station types and become arbitrary within-station splits. The K=3 divergence gain (0.1133 →
0.2386) comes at the cost of interpretability and a routing target that no longer matches any
deployable station group; K=4 gains nothing on quality and fragments further.

## Per-regime interpretation (post-clustering analysis)

### Backbone54 K=2 station composition (trainval; identical for V0_Full K=2)

| station_id | n | dominant_cluster | purity | share_c0 | share_c1 | longitude | latitude |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BeaverPass_WA_990 | 2185 | 0 | 1 | 1 | 0 | -121.255 | 48.8793 |
| CayusePass_WA | 2042 | 0 | 1 | 1 | 0 | -121.534 | 46.8696 |
| Darrington | 2048 | 0 | 1 | 1 | 0 | -121.45 | 48.54 |
| Paradise_WA | 2189 | 0 | 1 | 1 | 0 | -121.748 | 46.7827 |
| Quinault | 2160 | 0 | 1 | 1 | 0 | -123.81 | 47.51 |
| SourdoughGulch_WA_985 | 2191 | 1 | 1 | 0 | 1 | -117.4 | 46.2333 |
| Spokane | 1793 | 1 | 1 | 0 | 1 | -117.53 | 47.42 |

Mean station purity (trainval): 1.000 — mean test purity: 1.000. Cluster 1 = the two eastern,
inland stations (Spokane + SourdoughGulch); cluster 0 = the five western/mountain stations.

### Station purity across strategies and K (trainval / test)

| Strategy | K | trainval purity | test purity |
| :--- | ---: | ---: | ---: |
| Clustering_Backbone54 | 2 | 1.000 | 1.000 |
| Clustering_Backbone54 | 3 | 0.833 | 0.834 |
| Clustering_Backbone54 | 4 | 0.695 | 0.704 |
| Clustering_V0_Full | 2 | 1.000 | 1.000 |
| Clustering_Static | 2 | 1.000 | 1.000 |
| Clustering_Weather | 2 | 0.984 | 0.990 |

`Clustering_Static` is perfectly station-pure by construction (features are per-station
constants), but its K=2 partition differs from the backbone's (it groups Darrington, Spokane,
SourdoughGulch and Quinault together) — the backbone's split is therefore **not** trivially
recoverable from static attributes alone; dynamic land-surface state (SMAP) matters.
`Clustering_Weather` is nearly station-pure (eastern vs western climate) but has 1.5–6.5%
temporal mixing, so weather alone cannot define crisp regimes.

### Top-15 Backbone54 features separating the K=2 regimes (trainval)

| feature | separation_index | median_cluster0 | median_cluster1 |
| :--- | ---: | ---: | ---: |
| J_bio_bio13 | 1.0 | 391 | 70 |
| J_bio_bio02 | 1.0 | 97 | 133 |
| SMAP_sm_pm_interp_rollmean30 | 0.951 | 0.4147 | 0.1945 |
| V_rollmax_E_SAR_diff_kobs30 | 0.943 | 0.1092 | 0.0589 |
| V_rollmin_SMAP_sm_interp_kobs30 | 0.941 | 0.323 | 0.1478 |
| V_rollmax_E_SAR_diff_kobs14 | 0.94 | 0.105 | 0.0567 |
| SMAP_sm_pm_interp_lag7 | 0.932 | 0.4141 | 0.1924 |
| SMAP_sm_pm_interp | 0.932 | 0.4141 | 0.1923 |
| SMAP_sm_pm_interp_lag30 | 0.93 | 0.4142 | 0.1931 |
| V_rollmin_SMAP_sm_interp_kobs14 | 0.927 | 0.3454 | 0.1622 |
| V_rollmin_s2_b12_kobs30 | 0.917 | 0.0458 | 0.137 |
| D_z_F_NDMI | 0.908 | 0.3051 | -1.6684 |
| V_rollmin_s2_b11_kobs30 | 0.817 | 0.0797 | 0.2026 |
| V_rollmax_E_SAR_ratio_kobs7 | 0.796 | 4.5249 | 5.9967 |
| V_rollmax_E_SAR_ratio_kobs30 | 0.793 | 4.7217 | 6.3671 |

As in `derived_8.4-regime-interpretation-1.0`, the regimes separate on **regional climate &
land-surface state** (annual precipitation `J_bio_bio13`, annual mean temperature `J_bio_bio02`,
long-window SMAP means): cluster 1 is the hotter, semi-arid east; cluster 0 the wet, cool west.

## Sanity checks (all pass)

- `Clustering_V0_Full` K=2 reproduces the eval-1.1 winning router counts exactly:
  trainval (10,624, 3,984) and test (4,817, 1,803).
- `Clustering_Backbone54` K=2 vs `Clustering_V0_Full` K=2 (trainval): ARI = 1.0000.
- All 15 exported JSON/Joblib parameter pairs predict identically (100%) on the val split.

## Exported clustering parameters

Every clustering strategy is exported for K = 2, 3, 4 (15 pairs), fitted on derived_8.4
trainval, in the same schema as the base experiment:

| File pattern | Format | Contents |
| :--- | :--- | :--- |
| `clustering_params_<alias>_k<K>.json` | JSON | `strategy`, `K`, `fitted_on`, `features`, `impute_means`, `scaler_mean`, `scaler_scale`, `cluster_centers` |
| `clustering_params_<alias>_k<K>.joblib` | Joblib | `features`, `impute_means`, `scaler`, `kmeans` |
| `clustering_params_combined.json` | JSON | Combined registry of all 15 parameter dicts |

Aliases: `dynamic`, `v0_full` (retained filenames from the base experiment), `backbone54`,
`static`, `weather`. Loading/prediction examples are in the notebook §14 and in the
`derived_8.3-gating-analysis-1.0` README (same schema).

## Generated artifacts

- **Notebook**: `derived_8.4_gating_analysis_1.0.ipynb`
- **CSVs**: `gating_strategies_summary.csv`, `cluster_quality_metrics.csv`,
  `regime_profile_summary_<strategy>_k<K>.csv`, `regime_station_composition_<strategy>_k<K>.csv`
- **Params**: 15 × `clustering_params_<alias>_k<K>.{json,joblib}` + `clustering_params_combined.json`
- **Figures**: strategy-level grids (`gating_target_distributions_grid.png`,
  `gating_correlation_drift_grid.png`, `gating_geographic_distribution.png`), Backbone54 K-sweep
  (`gating_target_distributions_backbone54_k_sweep.png`,
  `gating_correlation_drift_backbone54_k_sweep.png`, `gating_geographic_backbone54_k_sweep.png`),
  clustering diagnostics (`clustering_scaling_diagnostic.png`,
  `clustering_metrics_comparison.png`, `clustering_silhouette_profiles.png`,
  `clustering_tsne_projection.png`, `clustering_centroid_distances.png`,
  `clustering_backbone54_pca.png`), per-(strategy,K) interpretation
  (`regime_target_distributions_*`, `regime_seasonality_*`, `regime_geographic_distribution_*`,
  `regime_weather_drivers_*`, `regime_static_attributes_*`)

## References

- `notebooks/experiment/derived_8.3-gating-analysis-1.0/` — base K=2 gating analysis and
  parameter-export conventions
- `notebooks/experiment/derived_8.4-regime-interpretation-1.0/` — per-regime physical
  interpretation of the eval-1.1 winning router (V0-Full K=2)
- `notebooks/experiment/derived_8.4-eval-1.1/` — two-regime MoE evaluation; source of
  `shared_backbone_54` (the single global model's / experts' feature set)
