# derived_8.2-eval-3.3 — Parallel MoE Evaluation under SOTA 1.5 Hyperparameters

This experiment evaluates the **2-regime Mixture-of-Experts (MoE) models** and baseline models on the Washington-only `derived_8.2` split, utilizing the new **SOTA 1.5 hyperparameters** optimized during the `derived_8.2-hyperparameters-1.5` sweep. 

We preserve the exact same feature arms and routing definitions as `derived_8.2-eval-3.2` for feature quality isolation, but replace the frozen 1.3-lite hyperparameters with the newly tuned SOTA configuration.

## Protocol

| Item | Value |
|------|--------|
| Split | `data/splits/derived_8.2/` (train+val → trainval; test held out) |
| Target | `soil_moisture_5cm` |
| Weighting | Unweighted (no temporal drift) |
| Hyperparameters | **1.5 SOTA** (max_depth=9, MCW=8, λ=0.75, α=0.03, sub=0.9, col=0.8, Est=2500, LR=0.005) |
| Seed | 42 |
| Device | CUDA |
| Parallelism | `XGB_PARALLEL_WORKERS` (default **4**); XGB `n_jobs=1` |
| Feature selection | Same as eval-3.2 (Spec-old from `previous_features.json`, Spec-new and Global lists from `selected_features.json`) |

## Model Leaderboard

Evaluated on CUDA. Below is the performance compared with the old **1.3-lite control** runs (eval-3.2):

| Rank | Model Name | $R^2$ (1.5) | RMSE (1.5) | $R^2$ (1.3-lite) | $\Delta R^2$ |
|------|------------|:-----------:|:----------:|:----------------:|:------------:|
| 1 | **Model 14: Clustering Dynamic K=2 (Global-c1)** | **0.6639** | **0.0610** | **0.6672** | **-0.0033** |
| 2 | Model 2: Baseline c1 | 0.6637 | 0.0611 | 0.6648 | -0.0011 |
| 3 | Model 1: Baseline V3 | 0.6589 | 0.0615 | 0.6551 | +0.0038 |
| 4 | Model 18: Seasonal Binary K=2 (Global-c1) | 0.6446 | 0.0628 | 0.6484 | -0.0038 |
| 5 | Model 10: Univariate G_API K=2 (Global-c1) | 0.6445 | 0.0628 | 0.6518 | -0.0073 |
| 6 | Model 17: Seasonal Binary K=2 (Global-V3) | 0.6435 | 0.0629 | 0.6400 | +0.0035 |
| 7 | Model 9: Univariate G_API K=2 (Global-V3) | 0.6383 | 0.0633 | 0.6388 | -0.0005 |
| 8 | Model 12: Clustering Dynamic K=2 (Spec-new) | 0.6271 | 0.0643 | 0.6258 | +0.0013 |
| 9 | Model 11: Clustering Dynamic K=2 (Spec-old) | 0.6263 | 0.0644 | 0.6273 | -0.0010 |
| 10 | Model 16: Seasonal Binary K=2 (Spec-new) | 0.6190 | 0.0650 | 0.6270 | -0.0080 |
| 11 | Model 6: Trained Gating K=2 (Global-c1) | 0.6174 | 0.0651 | 0.6133 | +0.0041 |
| 12 | Model 13: Clustering Dynamic K=2 (Global-V3) | 0.6102 | 0.0657 | 0.6029 | +0.0073 |
| 13 | Model 15: Seasonal Binary K=2 (Spec-old) | 0.5964 | 0.0669 | 0.5970 | -0.0006 |
| 14 | Model 4: Trained Gating K=2 (Spec-new) | 0.5856 | 0.0678 | 0.5791 | +0.0065 |
| 15 | Model 5: Trained Gating K=2 (Global-V3) | 0.5837 | 0.0679 | 0.5778 | +0.0059 |
| 16 | Model 3: Trained Gating K=2 (Spec-old) | 0.5782 | 0.0684 | 0.5712 | +0.0070 |
| 17 | Model 7: Univariate G_API K=2 (Spec-old) | 0.5381 | 0.0716 | 0.5395 | -0.0014 |
| 18 | Model 8: Univariate G_API K=2 (Spec-new) | 0.5363 | 0.0717 | 0.5368 | -0.0005 |

## Ablation: $R^2$ by Strategy × Arm

| Strategy | Spec-old | Spec-new | Global-V3 | Global-c1 |
|----------|:--------:|:--------:|:---------:|:---------:|
| Clustering Dynamic K=2 | 0.6263 | 0.6271 | 0.6102 | **0.6639** |
| Seasonal Binary K=2 | 0.5964 | 0.6190 | 0.6435 | **0.6446** |
| Trained Gating K=2 | 0.5782 | 0.5856 | 0.5837 | **0.6174** |
| Univariate G_API K=2 | 0.5381 | 0.5363 | 0.6383 | **0.6445** |

## Key Insights

1. **Global Hyperparameter Transfer Degradation:** Tuning hyperparameters specifically for a global single model (`Baseline V3` improved by **+0.0038** from 0.6551 to **0.6589**) does *not* automatically transfer optimally to Mixture-of-Experts. The best MoE model (`Clustering Dynamic K=2 (Global-c1)`) dropped slightly by **-0.0033** (from 0.6672 to **0.6639**). This indicates that specialists training on smaller partitioned subsets are more sensitive to high-depth (depth=9) / high-estimator (2500) parameters, likely suffering from mild overfitting compared to the more conservative 1.3-lite parameters (depth=8, estimators=1500).
2. **MoE vs. Global Baseline:** Even under tuned global hyperparameters, the **Clustering Dynamic K=2 (Global-c1)** model remains the top performer overall, edging past `Baseline c1` (0.6639 vs 0.6637).
3. **Routing Stability:** Routing accuracy for trained gating remains high (~0.874), representing a tiny increase (+0.0015) over 1.3-lite.

---

## Station × Year Metrics Breakdown

To detect weak spots, we generated a full breakdown of the best overall model (**Model 14: Clustering Dynamic K=2 (Global-c1)**) across stations and years.

![Station Year Breakdown](station_year_metrics.png)

### Weak Spots Analysis:
* **Significant Weak Spot (BurntMountain_WA):** This station exhibits extreme negative $R^2$ values across all years (2023: -1.7107, 2024: -0.4283, 2025: -2.3924), indicating the model predicts worse than the local mean. This is a severe outlier station where the model fails to capture the soil moisture dynamics.
* **Touchet_WA_824:** Also shows negative $R^2$ in 2024 (-0.1706), indicating temporal mismatches in that specific year.
* **Year 2024 Performance:** Consistent with earlier observations, 2024 remains a weak year overall ($R^2$ = 0.6292) compared to 2025 ($R^2$ = 0.6996) and 2023 ($R^2$ = 0.6470) for this MoE model.
