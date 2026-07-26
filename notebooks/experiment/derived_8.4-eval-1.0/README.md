# derived_8.4-eval-1.0 — Parallel MoE Evaluation on derived_8.4

This experiment evaluates **2-regime Mixture-of-Experts (MoE) models** and baseline models on the Washington-only `derived_8.4` split (7 stations, excluding alpine microclimates `MartenRidge_WA_999` and `RainyPass_WA_711`), utilizing the **SOTA 1.5 hyperparameters** optimized during the `derived_8.2-hyperparameters-1.5` sweep.

Feature selection is evaluated using `derived_8.4`'s **`OVERALL_SELECTED_FEATURES_V0`** (50 features loaded from `data/splits/derived_8.4/dataset_metadata.py`). We evaluate 5 gating strategies ($K=2$): `Trained_Gating`, `Univariate_G_API`, `Clustering_Dynamic`, `Seasonal_Binary`, and `Clustering_V0_Full`.

## Protocol

| Item | Value |
|------|--------|
| Split | `data/splits/derived_8.4/` (train+val → trainval: 14,608 samples; test held out: 6,620 samples; 7 clean WA stations) |
| Target | `soil_moisture_5cm` |
| Primary Feature Set | **`OVERALL_SELECTED_FEATURES_V0`** (50 features from `data/splits/derived_8.4/dataset_metadata.py`) |
| Weighting | Unweighted (no temporal drift) |
| Hyperparameters | **1.5 SOTA** (`max_depth=9`, `min_child_weight=8`, `gamma=0.0`, `reg_lambda=0.75`, `reg_alpha=0.03`, `subsample=0.9`, `colsample_bytree=0.8`, `n_estimators=2500`, `learning_rate=0.005`) |
| Seed | 42 |
| Device | CUDA |
| Parallelism | `ThreadPoolExecutor` with `XGB_PARALLEL_WORKERS` (default **4**); XGB `n_jobs=1` |
| Feature selection | V6 `c1` pipeline run per regime/cluster on `derived_8.4` (`selected_features.json`) |

## Model Leaderboard

Evaluated on CUDA on `derived_8.4` test set (6,620 samples):

| Rank | Model ID | Model Name | Arm | Strategy | $R^2$ | RMSE | ubRMSE | Bias | MAE | Pearson |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **16** | **Model 16: Clustering V0 Full K=2 (Global-V0)** | `global_v0` | `Clustering_V0_Full_k2` | **0.7703** | **0.0488** | **0.0481** | **+0.0081** | **0.0370** | **0.8816** |
| **1** | **14** | **Model 14: Clustering V0 Full K=2 (Spec-old)** | `spec_old` | `Clustering_V0_Full_k2` | **0.7703** | **0.0488** | **0.0481** | **+0.0081** | **0.0370** | **0.8816** |
| 3 | 1 | Model 1: Baseline V0 | `global_v0` | Baseline | 0.7604 | 0.0499 | 0.0489 | +0.0099 | 0.0382 | 0.8775 |
| 4 | 7 | Model 7: Univariate G_API K=2 (Global-V0) | `global_v0` | `Univariate_G_API_k2` | 0.7481 | 0.0511 | 0.0496 | +0.0124 | 0.0387 | 0.8738 |
| 5 | 10 | Model 10: Clustering Dynamic K=2 (Global-V0) | `global_v0` | `Clustering_Dynamic_k2` | 0.7467 | 0.0513 | 0.0499 | +0.0120 | 0.0391 | 0.8723 |
| 6 | 2 | Model 2: Trained Gating K=2 (Spec-old) | `spec_old` | `trained_gating_k2` | 0.7370 | 0.0522 | 0.0518 | +0.0067 | 0.0385 | 0.8658 |
| 7 | 13 | Model 13: Seasonal Binary K=2 (Global-V0) | `global_v0` | `Seasonal_Binary_k2` | 0.7318 | 0.0528 | 0.0519 | +0.0094 | 0.0391 | 0.8613 |
| 8 | 4 | Model 4: Trained Gating K=2 (Global-V0) | `global_v0` | `trained_gating_k2` | 0.7225 | 0.0537 | 0.0522 | +0.0122 | 0.0394 | 0.8645 |
| 9 | 15 | Model 15: Clustering V0 Full K=2 (Spec-new) | `spec_new` | `Clustering_V0_Full_k2` | 0.7197 | 0.0539 | 0.0519 | +0.0147 | 0.0421 | 0.8606 |
| 10 | 8 | Model 8: Clustering Dynamic K=2 (Spec-old) | `spec_old` | `Clustering_Dynamic_k2` | 0.7127 | 0.0546 | 0.0526 | +0.0145 | 0.0424 | 0.8564 |
| 11 | 9 | Model 9: Clustering Dynamic K=2 (Spec-new) | `spec_new` | `Clustering_Dynamic_k2` | 0.6918 | 0.0566 | 0.0563 | +0.0053 | 0.0434 | 0.8353 |
| 12 | 11 | Model 11: Seasonal Binary K=2 (Spec-old) | `spec_old` | `Seasonal_Binary_k2` | 0.6629 | 0.0591 | 0.0585 | +0.0086 | 0.0447 | 0.8196 |
| 13 | 3 | Model 3: Trained Gating K=2 (Spec-new) | `spec_new` | `trained_gating_k2` | 0.6597 | 0.0594 | 0.0565 | +0.0185 | 0.0458 | 0.8362 |
| 14 | 5 | Model 5: Univariate G_API K=2 (Spec-old) | `spec_old` | `Univariate_G_API_k2` | 0.6582 | 0.0596 | 0.0593 | +0.0054 | 0.0442 | 0.8156 |
| 15 | 6 | Model 6: Univariate G_API K=2 (Spec-new) | `spec_new` | `Univariate_G_API_k2` | 0.6468 | 0.0605 | 0.0598 | +0.0097 | 0.0462 | 0.8135 |
| 16 | 12 | Model 12: Seasonal Binary K=2 (Spec-new) | `spec_new` | `Seasonal_Binary_k2` | 0.6457 | 0.0606 | 0.0604 | +0.0052 | 0.0463 | 0.8075 |

## Ablation: $R^2$ by Strategy $\times$ Arm

| Strategy | Global-V0 | Spec-new | Spec-old |
|----------|:--------:|:--------:|:--------:|
| **Clustering V0 Full K=2** | **0.7703** | 0.7197 | **0.7703** |
| Univariate G_API K=2 | 0.7481 | 0.6468 | 0.6582 |
| Clustering Dynamic K=2 | 0.7467 | 0.6918 | 0.7127 |
| Trained Gating K=2 | 0.7225 | 0.6597 | 0.7370 |
| Seasonal Binary K=2 | 0.7318 | 0.6457 | 0.6629 |

---

## Per-Regime Residual Analysis & Visualizations

All 15 two-regime MoE models (Models 2–16) are evaluated on their test regime partitions:
- **Regime 0**: Dry / Warm season (May–Oct) / Low antecedent moisture ($G_{API}$) / Cluster 0 / Predicted dry ($SM < 0.16$).
- **Regime 1**: Wet / Cold season (Nov–Apr) / High antecedent moisture ($G_{API}$) / Cluster 1 / Predicted wet ($SM \ge 0.16$).

### Per-Regime Performance Breakdown Table

| Model ID | Model Name | $N_{R0}$ | $R^2_{R0}$ | $\text{RMSE}_{R0}$ | $\text{Bias}_{R0}$ | $N_{R1}$ | $R^2_{R1}$ | $\text{RMSE}_{R1}$ | $\text{Bias}_{R1}$ |
|:--------:|------------|:--------:|:----------:|:-----------------:|:-----------------:|:--------:|:----------:|:-----------------:|:-----------------:|
| 2 | Model 2: Trained Gating K=2 (Spec-old) | 1934 | 0.4684 | 0.0447 | -0.0060 | 4686 | 0.2785 | 0.0551 | +0.0119 |
| 3 | Model 3: Trained Gating K=2 (Spec-new) | 1934 | 0.1879 | 0.0552 | +0.0068 | 4686 | 0.1121 | 0.0611 | +0.0234 |
| 4 | Model 4: Trained Gating K=2 (Global-V0) | 1934 | 0.3758 | 0.0484 | -0.0065 | 4686 | 0.2622 | 0.0557 | +0.0200 |
| 5 | Model 5: Univariate G_API K=2 (Spec-old) | 3493 | 0.7640 | 0.0543 | +0.0054 | 3127 | 0.0542 | 0.0649 | +0.0054 |
| 6 | Model 6: Univariate G_API K=2 (Spec-new) | 3493 | 0.7075 | 0.0604 | +0.0128 | 3127 | 0.1743 | 0.0607 | +0.0063 |
| 7 | Model 7: Univariate G_API K=2 (Global-V0) | 3493 | 0.7890 | 0.0513 | +0.0080 | 3127 | 0.4190 | 0.0509 | +0.0173 |
| 8 | Model 8: Clustering Dynamic K=2 (Spec-old) | 2915 | 0.3188 | 0.0515 | +0.0094 | 3705 | 0.7303 | 0.0570 | +0.0185 |
| 9 | Model 9: Clustering Dynamic K=2 (Spec-new) | 2915 | 0.1202 | 0.0585 | +0.0060 | 3705 | 0.7487 | 0.0550 | +0.0048 |
| 10 | Model 10: Clustering Dynamic K=2 (Global-V0) | 2915 | 0.2922 | 0.0525 | +0.0126 | 3705 | 0.7895 | 0.0503 | +0.0115 |
| 11 | Model 11: Seasonal Binary K=2 (Spec-old) | 3415 | 0.6409 | 0.0635 | +0.0159 | 3205 | 0.0976 | 0.0542 | +0.0008 |
| 12 | Model 12: Seasonal Binary K=2 (Spec-new) | 3415 | 0.6031 | 0.0667 | +0.0099 | 3205 | 0.1226 | 0.0534 | +0.0002 |
| 13 | Model 13: Seasonal Binary K=2 (Global-V0) | 3415 | 0.7009 | 0.0579 | +0.0079 | 3205 | 0.3312 | 0.0466 | +0.0110 |
| 14 | Model 14: Clustering V0 Full K=2 (Spec-old) | 4817 | 0.7438 | 0.0506 | +0.0098 | 1803 | 0.8324 | 0.0436 | +0.0034 |
| 15 | Model 15: Clustering V0 Full K=2 (Spec-new) | 4817 | 0.6704 | 0.0574 | +0.0192 | 1803 | 0.8353 | 0.0432 | +0.0028 |
| **16** | **Model 16: Clustering V0 Full K=2 (Global-V0)** | **4817** | **0.7438** | **0.0506** | **+0.0098** | **1803** | **0.8324** | **0.0436** | **+0.0034** |

---

## Key Insights

1. **Overall Performance Improvement on `derived_8.4`**:
   - Removing the two noisy alpine snowpack stations boosted overall model performance across all models compared to `derived_8.3`. `Baseline V0` $R^2$ improved from **0.6435** to **0.7604** ($+0.1169 \Delta R^2$).
2. **Top MoE Model (`Clustering_V0_Full_k2`)**:
   - `Model 16: Clustering V0 Full K=2 (Global-V0)` and `Model 14 (Spec-old)` achieve the top rank with $R^2 = \mathbf{0.7703}$ and $\text{RMSE} = \mathbf{0.0488}$, outperforming `Baseline V0` ($R^2 = 0.7604$) by **$+0.0099 \Delta R^2$**.
   - Performs exceptionally well in Regime 1 ($R^2_{R1} = \mathbf{0.8324}$, $\text{RMSE}_{R1} = \mathbf{0.0436}$).
3. **Global-V0 Arm Superiority**:
   - The `Global-V0` feature arm consistently outperforms `Spec-new` across all gating strategies on `derived_8.4`, confirming that retaining all 50 global features within specialists prevents feature truncation degradation.

---

## Station $\times$ Year Metrics Breakdown Charts

- `station_year_metrics_global_v0.png`: Station $\times$ Year breakdown for `Model 1: Baseline V0`.
- `station_year_metrics_clustering_dynamic_k2.png`: Station $\times$ Year breakdown for `Model 10: Clustering Dynamic K=2`.
- `station_year_metrics_clustering_v0_full_k2.png`: Station $\times$ Year breakdown for `Model 16: Clustering V0 Full K=2`.
