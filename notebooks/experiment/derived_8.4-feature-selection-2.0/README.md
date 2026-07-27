# derived_8.4-feature-selection-2.0

This is an isolated direct feature-search experiment for the Washington-only `derived_8.4` split. It targets unweighted pooled performance on the 2023–2025 test period using the existing V0-full K=2 routing and SOTA 1.5 expert hyperparameters.

The implementation is local to this directory. It does not modify `Modeling/`, previous feature-selection experiments, the split metadata, or the current evaluation experiment.

## What changed

- `legacy_forced_bypass` explicitly reproduces the historical C1 semantics; true bypass-off remains a separate diagnostic. This prevents a 50-feature forced-bypass list from being mistaken for the current 12-feature true-bypass-off result.
- MI `k=300` is the canonical relevance gate. MI `k=120` and no-MI are audit controls only, reflecting the documented starvation and no-MI failures.
- The audit records raw, MI, ElasticNet, stability, and repaired-fallback counts for global, Dynamic K=2, Univariate G_API K=2, and V0-full K=2 routes.
- The final architecture uses a shared global backbone and add-only specialist deltas of 0, 5, or 10 features. Sparse cluster lists cannot replace the backbone.

## Search and reporting safeguards

- The two-hour budget is enforced at actual worker launch, not merely when a task is submitted. At most one candidate per worker is in flight, and a worker rechecks its phase cutoff immediately before fitting.
- The final 0/5/10 × 0/5/10 specialist grid uses the overall deadline rather than the wrapper cutoff. If any of its nine coordinates is missing, the run checkpoints `incomplete_deadline`, exits nonzero, removes any prior `selected_features.json`, and produces no reportable winner.
- Specialist rankings are restricted to the persisted global candidate pool. This keeps the 96-feature universe meaningful and prevents a cluster-only feature from bypassing the screened search space.
- Exact feature configurations are evaluated once per model type. Later equivalent requests are stored in `candidate_aliases.csv`, while `search_results.csv` contains only actual fits. This prevents repeated 2,500-tree fits from consuming the fixed budget.
- `delta_grid.csv` records every requested coordinate, whether it was fitted or reused, and its canonical fit. The notebook rejects incomplete grids and non-complete summaries before rendering figures or tables.
- The collapse audit now records the actual number of nonzero ElasticNet coefficients separately from the `elasticnet_k` cap. Its stage funnel therefore no longer reports a truncated count as the number of coefficients that survived regularization.

### Why the old top three looked identical

They were not three independently competitive feature sets. The global-backbone label and the `(0, 0)` specialist delta both requested the same literal global list with no additions; the prior implementation fitted and reported them again. Some seed/normalization paths could do the same. The corrected runner caches that configuration, records the later names as aliases, and the notebook ranks the configuration once. A nonzero-delta winner is also safe: it is shown as a one-member configuration group instead of assuming every winner has aliases.

## Run

From the repository root:

```bash
uv run --project notebooks python notebooks/experiment/derived_8.4-feature-selection-2.0/run_search.py --stage all --workers 8 --deadline-minutes 120
```

Then render the report from `notebooks/`:

```bash
nb execute experiment/derived_8.4-feature-selection-2.0/feature_selection_2_0.ipynb --uv --timeout 300
```

The runner checkpoints literal candidate lists and metrics in `artifacts/` as plain CSV and JSON. The report notebook is the sole generator of figures and the Markdown result tables below. It only accepts a `complete` summary and a completed nine-coordinate delta grid.

## Results

The blocks below are copied verbatim from the stdout of the fully executed `feature_selection_2_0.ipynb` notebook.

The executed report loaded 35 audit rows, 1,039 actual candidate fits, 44 aliases, a complete nine-cell delta grid, and a 96-feature candidate pool. The final winner is the 54-feature shared backbone plus 10 cluster-1 additions; it reached R² 0.8150, above both the V0 calibration and the global-only backbone.

### Collapse audit highlights: MI-300 → ElasticNet → stability

| route                 | cluster   |   n_train |   mi_features |   elasticnet_nonzero |   stability_features |   historical_fallback_features |   alpha |   l1_ratio | collapse_status   |   repaired_fallback_features |
|:----------------------|:----------|----------:|--------------:|---------------------:|---------------------:|-------------------------------:|--------:|-----------:|:------------------|-----------------------------:|
| Clustering_Dynamic_k2 | 0         |      4339 |           300 |                    2 |                    2 |                              2 |  0.0296 |     1.0000 | hard_collapsed    |                           50 |
| Clustering_Dynamic_k2 | 1         |      5464 |           300 |                   25 |                   25 |                             25 |  0.0666 |     0.1000 | truncated         |                           25 |
| Clustering_V0_Full_k2 | 0         |      7156 |           300 |                   26 |                   26 |                             26 |  0.0681 |     0.1000 | truncated         |                           26 |
| Clustering_V0_Full_k2 | 1         |      2647 |           300 |                   66 |                   42 |                             42 |  0.0006 |     1.0000 | truncated         |                           42 |
| Univariate_G_API_k2   | 0         |      4901 |           300 |                   60 |                   50 |                             50 |  0.0206 |     0.1000 | healthy           |                           50 |
| Univariate_G_API_k2   | 1         |      4902 |           300 |                   10 |                    8 |                             13 |  0.1465 |     0.1000 | hard_collapsed    |                           50 |
| global                | global    |      9803 |           300 |                    5 |                    6 |                              9 |  0.0117 |     1.0000 | hard_collapsed    |                           50 |

This confirms that the Dynamic cluster-0 collapse happens after a full MI-300 screen: 4,339 train rows and 300 MI candidates reduce to two ElasticNet nonzeros under a lasso-like fit, then remain two after stability selection. It is therefore not an MI-`k` bottleneck or a simple small-sample effect; the V0 cluster with only 2,647 rows retains 66 ElasticNet nonzeros and 42 stable features.

### Final comparison

| model          |   pooled_r2 |   pooled_rmse |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:---------------|------------:|--------------:|---------------:|---------------:|---------------:|
| V0 calibration |      0.7703 |        0.0488 |         0.7510 |         0.7613 |         0.7937 |
| 2.0 winner     |      0.8150 |        0.0438 |         0.8230 |         0.7833 |         0.8304 |

### Add-only delta grid

|   cluster_0_addition_count |      0 |      5 |     10 |
|---------------------------:|-------:|-------:|-------:|
|                          0 | 0.8143 | 0.8143 | 0.8150 |
|                          5 | 0.8141 | 0.8141 | 0.8148 |
|                         10 | 0.7891 | 0.7890 | 0.7898 |

The grid explains the specialist result directly: ten cluster-1 additions improve the 54-feature backbone, while ten cluster-0 additions sharply hurt performance. This is why specialists are bounded additive deltas rather than independently selected replacement lists.

### Equivalent exact-evaluation aliases

| candidate_id                                | equivalent_evaluations                                                                   |   alias_count |
|:--------------------------------------------|:-----------------------------------------------------------------------------------------|--------------:|
| round_06_drop_V_rollrng_F_NDVI_kobs30_exact | round_06_drop_V_rollrng_F_NDVI_kobs30_exact; global_backbone_for_deltas; delta_c0_0_c1_0 |             3 |
| baseline_v0_calibration                     | baseline_v0_calibration; seed_v0; seed_legacy_8_2_forced_bypass                          |             3 |

### Exact 2023–2025 leaderboard (unique feature configurations)

| candidate_id                                                                    |   pooled_r2 |   pooled_rmse |   global_feature_count |   cluster_0_feature_count |   cluster_1_feature_count |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 | equivalent_evaluations                                                                   |
|:--------------------------------------------------------------------------------|------------:|--------------:|-----------------------:|--------------------------:|--------------------------:|---------------:|---------------:|---------------:|:-----------------------------------------------------------------------------------------|
| delta_c0_0_c1_10                                                                |      0.8150 |        0.0438 |                     54 |                        54 |                        64 |         0.8230 |         0.7833 |         0.8304 | delta_c0_0_c1_10                                                                         |
| delta_c0_5_c1_10                                                                |      0.8148 |        0.0438 |                     54 |                        59 |                        64 |         0.8230 |         0.7838 |         0.8294 | delta_c0_5_c1_10                                                                         |
| round_06_drop_V_rollrng_F_NDVI_kobs30_exact                                     |      0.8143 |        0.0439 |                     54 |                        54 |                        54 |         0.8213 |         0.7851 |         0.8284 | round_06_drop_V_rollrng_F_NDVI_kobs30_exact; global_backbone_for_deltas; delta_c0_0_c1_0 |
| delta_c0_0_c1_5                                                                 |      0.8143 |        0.0439 |                     54 |                        54 |                        59 |         0.8212 |         0.7825 |         0.8310 | delta_c0_0_c1_5                                                                          |
| delta_c0_5_c1_0                                                                 |      0.8141 |        0.0439 |                     54 |                        59 |                        54 |         0.8212 |         0.7857 |         0.8274 | delta_c0_5_c1_0                                                                          |
| delta_c0_5_c1_5                                                                 |      0.8141 |        0.0439 |                     54 |                        59 |                        59 |         0.8211 |         0.7830 |         0.8299 | delta_c0_5_c1_5                                                                          |
| round_06_drop_SMAP_sm_pm_interp_lag30_exact                                     |      0.8082 |        0.0446 |                     54 |                        54 |                        54 |         0.8196 |         0.7834 |         0.8126 | round_06_drop_SMAP_sm_pm_interp_lag30_exact                                              |
| round_06_drop_V_rollmax_E_SAR_diff_kobs30_exact                                 |      0.8079 |        0.0446 |                     54 |                        54 |                        54 |         0.8165 |         0.7781 |         0.8205 | round_06_drop_V_rollmax_E_SAR_diff_kobs30_exact                                          |
| round_06_add_V_rollmin_E_SAR_diff_kobs14_exact                                  |      0.8045 |        0.0450 |                     56 |                        56 |                        56 |         0.8139 |         0.7757 |         0.8151 | round_06_add_V_rollmin_E_SAR_diff_kobs14_exact                                           |
| round_05_drop_V_rollmax_LST_modis_kobs30_exact                                  |      0.8044 |        0.0451 |                     55 |                        55 |                        55 |         0.8161 |         0.7722 |         0.8155 | round_05_drop_V_rollmax_LST_modis_kobs30_exact                                           |
| round_05_drop_SMAP_sm_pm_interp_lag30_exact                                     |      0.8041 |        0.0451 |                     55 |                        55 |                        55 |         0.8112 |         0.7841 |         0.8086 | round_05_drop_SMAP_sm_pm_interp_lag30_exact                                              |
| round_04_drop_year_frac_exact                                                   |      0.8019 |        0.0453 |                     56 |                        56 |                        56 |         0.8098 |         0.7761 |         0.8112 | round_04_drop_year_frac_exact                                                            |
| round_05_swap_D_fft_ent_LST_modis_kobs30_for_A_grad_SMAP_sm_interp_kobs14_exact |      0.8019 |        0.0453 |                     56 |                        56 |                        56 |         0.8088 |         0.7763 |         0.8120 | round_05_swap_D_fft_ent_LST_modis_kobs30_for_A_grad_SMAP_sm_interp_kobs14_exact          |
| round_05_add_J_bio_bio06_exact                                                  |      0.8018 |        0.0454 |                     57 |                        57 |                        57 |         0.8100 |         0.7750 |         0.8117 | round_05_add_J_bio_bio06_exact                                                           |
| round_04_swap_D_fft_ent_LST_modis_kobs30_for_A_grad_SMAP_sm_interp_kobs14_exact |      0.8000 |        0.0456 |                     57 |                        57 |                        57 |         0.8159 |         0.7686 |         0.8054 | round_04_swap_D_fft_ent_LST_modis_kobs30_for_A_grad_SMAP_sm_interp_kobs14_exact          |
| round_04_swap_D_fft_ent_LST_modis_kobs30_for_D_z_E_SAR_ratio_exact              |      0.7993 |        0.0456 |                     57 |                        57 |                        57 |         0.8157 |         0.7651 |         0.8067 | round_04_swap_D_fft_ent_LST_modis_kobs30_for_D_z_E_SAR_ratio_exact                       |
| round_03_drop_C_lag_LST_modis_kobs30_exact                                      |      0.7992 |        0.0456 |                     57 |                        57 |                        57 |         0.8152 |         0.7669 |         0.8054 | round_03_drop_C_lag_LST_modis_kobs30_exact                                               |
| round_04_add_V_rollmin_s2_b11_kobs14_exact                                      |      0.7992 |        0.0457 |                     58 |                        58 |                        58 |         0.8154 |         0.7639 |         0.8079 | round_04_add_V_rollmin_s2_b11_kobs14_exact                                               |
| round_03_swap_J_soil_texture_usda_b0_for_A_grad_SMAP_sm_interp_kobs14_exact     |      0.7976 |        0.0458 |                     58 |                        58 |                        58 |         0.8053 |         0.7734 |         0.8052 | round_03_swap_J_soil_texture_usda_b0_for_A_grad_SMAP_sm_interp_kobs14_exact              |
| round_03_drop_SMAP_sm_pm_interp_lag30_exact                                     |      0.7972 |        0.0459 |                     57 |                        57 |                        57 |         0.8098 |         0.7749 |         0.7973 | round_03_drop_SMAP_sm_pm_interp_lag30_exact                                               |

### Generated findings

- V0 calibration: R²=0.7703, RMSE=0.0488.
- 2.0 winner: R²=0.8150, RMSE=0.0438, delta R²=+0.0447.
- Dynamic-route MI-300 audit: cluster 0 had 4339 train rows, 300 MI candidates, 2 ElasticNet nonzero features, and 2 stable features; cluster 1 had 5464 train rows, 300 MI candidates, 25 ElasticNet nonzero features, and 25 stable features.
- Hard collapses recorded: 11. The audit records whether they occur before or after MI, so small specialist lists are not attributed to sample count alone.
- Exact-evaluation aliases for the winner: delta_c0_0_c1_10. It is a unique configuration with no duplicate fit.
- Final specialists are add-only deltas on the shared backbone; no independently selected cluster list can replace it.
