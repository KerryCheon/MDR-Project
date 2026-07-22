# derived_8.3-eval-1.0 — Parallel MoE Evaluation on derived_8.3

This experiment evaluates **2-regime Mixture-of-Experts (MoE) models** and baseline models on the Washington-only `derived_8.3` split, utilizing the **SOTA 1.5 hyperparameters** optimized during the `derived_8.2-hyperparameters-1.5` sweep.

`derived_8.2`'s `OVERALL_SELECTED_FEATURES_V3` (47 features) is replaced by `derived_8.3`'s **`OVERALL_SELECTED_FEATURES_V0`** (50 features loaded from `data/splits/derived_8.3/dataset_metadata.py`). We evaluate 5 gating strategies ($K=2$): `Trained_Gating`, `Univariate_G_API`, `Clustering_Dynamic`, `Seasonal_Binary`, and `Clustering_V0_Full`.

## Protocol

| Item | Value |
|------|--------|
| Split | `data/splits/derived_8.3/` (train+val → trainval: 18,897 samples; test held out: 8,396 samples; 9 clean WA stations) |
| Target | `soil_moisture_5cm` |
| Primary Feature Set | **`OVERALL_SELECTED_FEATURES_V0`** (50 features from `data/splits/derived_8.3/dataset_metadata.py`) |
| Weighting | Unweighted (no temporal drift) |
| Hyperparameters | **1.5 SOTA** (`max_depth=9`, `min_child_weight=8`, `gamma=0.0`, `reg_lambda=0.75`, `reg_alpha=0.03`, `subsample=0.9`, `colsample_bytree=0.8`, `n_estimators=2500`, `learning_rate=0.005`) |
| Seed | 42 |
| Device | CUDA |
| Parallelism | `ThreadPoolExecutor` with `XGB_PARALLEL_WORKERS` (default **4**); XGB `n_jobs=1` |
| Feature selection | V6 `c1` pipeline run per regime/cluster on `derived_8.3` (`selected_features.json`) |

## Model Leaderboard

Evaluated on CUDA on `derived_8.3` test set (8,396 samples):

| Rank | Model ID | Model Name | Arm | Strategy | $R^2$ | RMSE | ubRMSE | Bias | MAE | Pearson |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **16** | **Model 16: Clustering V0 Full K=2 (Global-V0)** | `global_v0` | `Clustering_V0_Full_k2` | **0.6619** | **0.0604** | **0.0584** | **+0.0155** | **0.0435** | **0.8282** |
| **1** | **14** | **Model 14: Clustering V0 Full K=2 (Spec-old)** | `spec_old` | `Clustering_V0_Full_k2` | **0.6619** | **0.0604** | **0.0584** | **+0.0155** | **0.0435** | **0.8282** |
| 3 | 1 | Model 1: Baseline V0 | `global_v0` | Baseline | 0.6435 | 0.0620 | 0.0601 | +0.0155 | 0.0453 | 0.8172 |
| 4 | 10 | Model 10: Clustering Dynamic K=2 (Global-V0) | `global_v0` | `Clustering_Dynamic_k2` | 0.6243 | 0.0637 | 0.0613 | +0.0173 | 0.0466 | 0.8100 |
| 5 | 13 | Model 13: Seasonal Binary K=2 (Global-V0) | `global_v0` | `Seasonal_Binary_k2` | 0.6165 | 0.0644 | 0.0629 | +0.0136 | 0.0469 | 0.7984 |
| 6 | 7 | Model 7: Univariate G_API K=2 (Global-V0) | `global_v0` | `Univariate_G_API_k2` | 0.6137 | 0.0646 | 0.0625 | +0.0164 | 0.0466 | 0.8013 |
| 7 | 4 | Model 4: Trained Gating K=2 (Global-V0) | `global_v0` | `trained_gating_k2` | 0.5863 | 0.0668 | 0.0634 | +0.0213 | 0.0474 | 0.8084 |
| 8 | 11 | Model 11: Seasonal Binary K=2 (Spec-old) | `spec_old` | `Seasonal_Binary_k2` | 0.5662 | 0.0684 | 0.0660 | +0.0182 | 0.0508 | 0.7779 |
| 9 | 8 | Model 8: Clustering Dynamic K=2 (Spec-old) | `spec_old` | `Clustering_Dynamic_k2` | 0.5660 | 0.0685 | 0.0638 | +0.0248 | 0.0506 | 0.7942 |
| 10 | 6 | Model 6: Univariate G_API K=2 (Spec-new) | `spec_new` | `Univariate_G_API_k2` | 0.5605 | 0.0689 | 0.0668 | +0.0169 | 0.0497 | 0.7707 |
| 11 | 2 | Model 2: Trained Gating K=2 (Spec-old) | `spec_old` | `trained_gating_k2` | 0.5595 | 0.0690 | 0.0663 | +0.0190 | 0.0486 | 0.7878 |
| 12 | 9 | Model 9: Clustering Dynamic K=2 (Spec-new) | `spec_new` | `Clustering_Dynamic_k2` | 0.5445 | 0.0701 | 0.0677 | +0.0185 | 0.0530 | 0.7649 |
| 13 | 3 | Model 3: Trained Gating K=2 (Spec-new) | `spec_new` | `trained_gating_k2` | 0.5272 | 0.0715 | 0.0689 | +0.0190 | 0.0535 | 0.7616 |
| 14 | 12 | Model 12: Seasonal Binary K=2 (Spec-new) | `spec_new` | `Seasonal_Binary_k2` | 0.5261 | 0.0715 | 0.0690 | +0.0188 | 0.0548 | 0.7553 |
| 15 | 15 | Model 15: Clustering V0 Full K=2 (Spec-new) | `spec_new` | `Clustering_V0_Full_k2` | 0.5212 | 0.0719 | 0.0709 | +0.0120 | 0.0524 | 0.7484 |
| 16 | 5 | Model 5: Univariate G_API K=2 (Spec-old) | `spec_old` | `Univariate_G_API_k2` | 0.5086 | 0.0728 | 0.0708 | +0.0171 | 0.0525 | 0.7446 |

## Ablation: $R^2$ by Strategy $\times$ Arm

| Strategy | Global-V0 | Spec-new | Spec-old |
|----------|:--------:|:--------:|:--------:|
| **Clustering V0 Full K=2** | **0.6619** | 0.5212 | **0.6619** |
| Clustering Dynamic K=2 | 0.6243 | 0.5445 | 0.5660 |
| Seasonal Binary K=2 | 0.6165 | 0.5261 | 0.5662 |
| Univariate G_API K=2 | 0.6137 | 0.5605 | 0.5086 |
| Trained Gating K=2 | 0.5863 | 0.5272 | 0.5595 |

---

## Per-Regime Residual Analysis & Visualizations

All 15 two-regime MoE models (Models 2–16) are evaluated on their test regime partitions:
- **Regime 0**: Dry / Warm season (May–Oct) / Low antecedent moisture ($G_{API}$) / Cluster 0 / Predicted dry ($SM < 0.16$).
- **Regime 1**: Wet / Cold season (Nov–Apr) / High antecedent moisture ($G_{API}$) / Cluster 1 / Predicted wet ($SM \ge 0.16$).

### Per-Regime Performance Breakdown Table

| Model ID | Model Name | $N_{R0}$ | $R^2_{R0}$ | $\text{RMSE}_{R0}$ | $\text{Bias}_{R0}$ | $N_{R1}$ | $R^2_{R1}$ | $\text{RMSE}_{R1}$ | $\text{Bias}_{R1}$ |
|:--------:|------------|:--------:|:----------:|:-----------------:|:-----------------:|:--------:|:----------:|:-----------------:|:-----------------:|
| 2 | Model 2: Trained Gating K=2 (Spec-old) | 2919 | 0.2564 | 0.0558 | +0.0052 | 5477 | -0.0954 | 0.0751 | +0.0263 |
| 3 | Model 3: Trained Gating K=2 (Spec-new) | 2919 | 0.0790 | 0.0621 | +0.0109 | 5477 | -0.1226 | 0.0760 | +0.0233 |
| 4 | Model 4: Trained Gating K=2 (Global-V0) | 2919 | 0.2268 | 0.0569 | +0.0012 | 5477 | 0.0035 | 0.0716 | +0.0320 |
| 5 | Model 5: Univariate G_API K=2 (Spec-old) | 4442 | 0.7064 | 0.0587 | +0.0108 | 3954 | -0.0390 | 0.0860 | +0.0243 |
| 6 | Model 6: Univariate G_API K=2 (Spec-new) | 4442 | 0.7196 | 0.0573 | +0.0127 | 3954 | 0.1035 | 0.0799 | +0.0216 |
| 7 | Model 7: Univariate G_API K=2 (Global-V0) | 4442 | 0.7358 | 0.0557 | +0.0133 | 3954 | 0.2452 | 0.0733 | +0.0199 |
| 8 | Model 8: Clustering Dynamic K=2 (Spec-old) | 3739 | 0.2848 | 0.0702 | +0.0234 | 4657 | 0.6205 | 0.0670 | +0.0260 |
| 9 | Model 9: Clustering Dynamic K=2 (Spec-new) | 3739 | 0.1801 | 0.0752 | +0.0226 | 4657 | 0.6339 | 0.0658 | +0.0152 |
| 10 | Model 10: Clustering Dynamic K=2 (Global-V0) | 3739 | 0.3992 | 0.0643 | +0.0163 | 4657 | 0.6628 | 0.0632 | +0.0181 |
| 11 | Model 11: Seasonal Binary K=2 (Spec-old) | 4269 | 0.5493 | 0.0700 | +0.0225 | 4127 | 0.3025 | 0.0668 | +0.0137 |
| 12 | Model 12: Seasonal Binary K=2 (Spec-new) | 4269 | 0.5070 | 0.0732 | +0.0195 | 4127 | 0.2389 | 0.0698 | +0.0182 |
| 13 | Model 13: Seasonal Binary K=2 (Global-V0) | 4269 | 0.5568 | 0.0694 | +0.0176 | 4127 | 0.4616 | 0.0587 | +0.0095 |
| 14 | Model 14: Clustering V0 Full K=2 (Spec-old) | 6593 | 0.6114 | 0.0643 | +0.0188 | 1803 | 0.8324 | 0.0436 | +0.0034 |
| 15 | Model 15: Clustering V0 Full K=2 (Spec-new) | 6593 | 0.4288 | 0.0779 | +0.0151 | 1803 | 0.8343 | 0.0433 | +0.0009 |
| **16** | **Model 16: Clustering V0 Full K=2 (Global-V0)** | **6593** | **0.6114** | **0.0643** | **+0.0188** | **1803** | **0.8324** | **0.0436** | **+0.0034** |

---

## Key Insights

1. **Deduplicated & Streamlined Model Matrix**:
   - Removed the redundant `global_c1` arm (which was identical to `global_v0`), streamlining the experiment to 16 distinct models.
2. **Top MoE Model (`Clustering_V0_Full_k2`)**:
   - `Model 16: Clustering V0 Full K=2 (Global-V0)` achieves the highest overall rank ($R^2 = \mathbf{0.6619}$, $\text{RMSE} = \mathbf{0.0604}$), outperforming `Baseline V0` ($R^2 = 0.6435$) by **$+0.0184 \Delta R^2$**.
   - It performs remarkably well in Regime 1 ($R^2_{R1} = \mathbf{0.8324}$, $\text{RMSE}_{R1} = \mathbf{0.0436}$).
3. **Global-V0 Arm Superiority**:
   - `Global-V0` arm consistently outperforms regime-restricted `Spec-new` arms across all gating strategies, demonstrating that keeping the complete 50-feature set in specialists maintains predictive robustness.

---

## Station $\times$ Year Metrics Breakdown Charts

- `station_year_metrics_global_v0.png`: Station $\times$ Year breakdown for `Model 1: Baseline V0`.
- `station_year_metrics_clustering_dynamic_k2.png`: Station $\times$ Year breakdown for `Model 10: Clustering Dynamic K=2`.
- `station_year_metrics_clustering_v0_full_k2.png`: Station $\times$ Year breakdown for `Model 16: Clustering V0 Full K=2`.
