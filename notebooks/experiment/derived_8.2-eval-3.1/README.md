# Mixture of Experts (MoE) Evaluation Suite - derived_8.2-eval-3.1

This directory details the experimental setup and results for the Mixture of Experts (MoE) evaluation suite on the Washington-only `derived_8.2` soil moisture dataset. 

We evaluate 17 different models (including unweighted global baseline, learned gating MoE models, and unsupervised clustering-based specialist models using both specialist-selected feature sets and the global V3 feature set) using the step-shrinked SOTA XGBoost hyperparameters from `derived_8.2-hyperparameters-1.3-lite`.

---

## 1. Summary of Overall Performance

The overall test performance metrics across the held-out test split for all 17 models, sorted by overall $R^2$ score in descending order, are summarized below:

| Model ID | Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| **Model 1** | **Baseline V3 (Global Model)** | **0.6551** | **0.0618** | **0.0583** | **−0.0207** | **0.0464** | **0.0352** | **0.8373** |
| Model 17 | Univariate G_API K=3 (V3 Features) | 0.6503 | 0.0623 | 0.0596 | −0.0179 | 0.0470 | 0.0360 | 0.8278 |
| Model 13 | Clustering V3 Full K=3 (V3 Features) | 0.6466 | 0.0626 | 0.0598 | −0.0185 | 0.0458 | 0.0326 | 0.8267 |
| Model 11 | Seasonal Binary K=2 (V3 Features) | 0.6400 | 0.0632 | 0.0593 | −0.0217 | 0.0473 | 0.0355 | 0.8299 |
| Model 7 | Univariate G_API K=2 (V3 Features) | 0.6388 | 0.0633 | 0.0601 | −0.0197 | 0.0474 | 0.0359 | 0.8246 |
| Model 12 | Clustering V3 Full K=3 (Spec Features) | 0.6336 | 0.0637 | 0.0602 | −0.0209 | 0.0477 | 0.0366 | 0.8234 |
| Model 15 | Clustering Dynamic K=3 (V3 Features) | 0.6291 | 0.0641 | 0.0619 | −0.0168 | 0.0480 | 0.0358 | 0.8151 |
| Model 14 | Clustering Dynamic K=3 (Spec Features) | 0.6284 | 0.0642 | 0.0613 | −0.0190 | 0.0477 | 0.0368 | 0.8164 |
| Model 8 | Clustering Dynamic K=2 (Spec Features) | 0.6273 | 0.0643 | 0.0608 | −0.0208 | 0.0475 | 0.0355 | 0.8205 |
| Model 9 | Clustering Dynamic K=2 (V3 Features) | 0.6029 | 0.0664 | 0.0630 | −0.0207 | 0.0497 | 0.0372 | 0.8082 |
| Model 10 | Seasonal Binary K=2 (Spec Features) | 0.5970 | 0.0669 | 0.0644 | −0.0179 | 0.0497 | 0.0384 | 0.7951 |
| Model 4 | Trained Gating K=2 (V3 Features) | 0.5778 | 0.0684 | 0.0640 | −0.0242 | 0.0498 | 0.0363 | 0.8100 |
| Model 2 | Trained Gating K=2 (Spec Features) | 0.5712 | 0.0690 | 0.0669 | −0.0169 | 0.0487 | 0.0345 | 0.7916 |
| Model 5 | Trained Gating K=3 (V3 Features) | 0.5663 | 0.0693 | 0.0649 | −0.0243 | 0.0500 | 0.0356 | 0.8112 |
| Model 16 | Univariate G_API K=3 (Spec Features) | 0.5465 | 0.0709 | 0.0688 | −0.0171 | 0.0521 | 0.0381 | 0.7662 |
| Model 3 | Trained Gating K=3 (Spec Features) | 0.5419 | 0.0713 | 0.0674 | −0.0231 | 0.0501 | 0.0345 | 0.8000 |
| Model 6 | Univariate G_API K=2 (Spec Features) | 0.5395 | 0.0715 | 0.0694 | −0.0172 | 0.0511 | 0.0364 | 0.7626 |

---

## 2. Year-by-Year Performance Breakdown

The performance breakdown ($R^2$ scores) on the held-out test split for each test year (2023, 2024, and 2025):

| Model Name | Overall $R^2$ | Year 2023 $R^2$ | Year 2024 $R^2$ | Year 2025 $R^2$ |
|---|---|---|---|---|
| **Model 1: Baseline V3** | **0.6551** | **0.6581** | 0.6403 | 0.6396 |
| Model 2: Trained Gating K=2 (Spec) | 0.5712 | 0.4918 | 0.5715 | 0.6460 |
| Model 3: Trained Gating K=3 (Spec) | 0.5419 | 0.4539 | 0.5477 | 0.6200 |
| Model 4: Trained Gating K=2 (V3) | 0.5778 | 0.4997 | 0.5662 | **0.6648** |
| Model 5: Trained Gating K=3 (V3) | 0.5663 | 0.4921 | 0.5648 | 0.6357 |
| Model 6: Univariate G_API K=2 (Spec) | 0.5395 | 0.4814 | 0.5249 | 0.5991 |
| Model 7: Univariate G_API K=2 (V3) | 0.6388 | 0.6186 | 0.6147 | **0.6650** |
| Model 8: Clustering Dynamic K=2 (Spec) | 0.6273 | 0.5996 | 0.6112 | 0.6539 |
| Model 9: Clustering Dynamic K=2 (V3) | 0.6029 | 0.5902 | 0.6150 | 0.5746 |
| Model 10: Seasonal Binary K=2 (Spec) | 0.5970 | 0.5782 | 0.5787 | 0.6109 |
| Model 11: Seasonal Binary K=2 (V3) | 0.6400 | 0.6125 | 0.6412 | 0.6475 |
| Model 12: Clustering V3 Full K=3 (Spec) | 0.6336 | 0.5986 | 0.6319 | 0.6543 |
| Model 13: Clustering V3 Full K=3 (V3) | 0.6466 | 0.6033 | **0.6637** | 0.6584 |
| Model 14: Clustering Dynamic K=3 (Spec) | 0.6284 | 0.5948 | 0.6168 | 0.6581 |
| Model 15: Clustering Dynamic K=3 (V3) | 0.6291 | 0.6169 | 0.6410 | 0.6023 |
| Model 16: Univariate G_API K=3 (Spec) | 0.5465 | 0.5105 | 0.5402 | 0.5666 |
| Model 17: Univariate G_API K=3 (V3) | **0.6503** | 0.6396 | 0.6344 | 0.6549 |

---

## 3. Gating Router Classification Performance

The trained gating routers (XGBoost Classifiers) are evaluated by comparing their predictions against the true target-sliced regimes on the held-out test split:

* **Ternary threshold**: Dry ($y < 0.16$), Transition ($0.16 \le y < 0.25$), Wet ($y \ge 0.25$)
* **Binary threshold**: Dry ($y < 0.16$), Wet ($y \ge 0.16$)

| Gating Router | K | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) |
|---|---|---|---|---|---|
| Model 2 & 4 (Binary Gating) | 2 | 0.8730 | 0.8654 | 0.8636 | 0.8645 |
| Model 3 & 5 (Ternary Gating) | 3 | 0.7407 | 0.7307 | 0.7130 | 0.7013 |

---

## 4. Key Findings and Discussion

### 1. Cluster-Specific Feature Selection Degrades Performance
For almost all cluster gating strategies, using the **global V3 feature set** yields significantly higher overall $R^2$ scores and lower RMSEs compared to their cluster-specific feature selection subsets:
- **Univariate G_API K=3 (V3)** reaches **0.6503** (vs. **0.5465** for Spec).
- **Seasonal Binary K=2 (V3)** reaches **0.6400** (vs. **0.5970** for Spec).
- **Univariate G_API K=2 (V3)** reaches **0.6388** (vs. **0.5395** for Spec).
- **Clustering V3 Full K=3 (V3)** reaches **0.6466** (vs. **0.6336** for Spec).

This indicates that performing feature selection on partitioned cluster subsets discards globally useful predictive features (due to reduced sample size in stability selection), thus underperforming compared to training specialists on the globally selected feature set.

The sole exception is **Clustering Dynamic K=2**, where using the specialist feature set (Model 8) outperforms the global V3 feature set (Model 9) (**0.6273** vs. **0.6029**).

### 2. Comparison with Baseline V3
Model 1 (the single global model using baseline features `OVERALL_SELECTED_FEATURES_V3`) remains the SOTA model overall with an $R^2$ of **0.6551**. However, it is closely contested by Univariate G_API K=3 (V3) (Model 17) at **0.6503**. 

Interestingly, several MoE models using V3 features outperform the global baseline in specific test years (e.g. Model 4 reaches **0.6648** in 2025 vs. Baseline's **0.6396**, and Model 13 reaches **0.6637** in 2024 vs. Baseline's **0.6403**).

---

## 5. Visualizations and Saved Files

* **Diagnostic Plots**: Saved as `diagnostics_{model_name}.png` in this directory, representing the $2 \times 4$ grid (True vs Pred & True vs Residuals overall and yearly).
* **Consolidated Loss Curves Plot**: Saved as `loss_curves_consolidated.png` showing the overall test RMSE curves of all 17 models over 1500 boosting steps.
* **Grouped Loss Curves Comparison Plot**: Saved as `loss_curves_grouped.png` comparing Spec vs V3 feature sets side-by-side.
* **R2 Line Chart Comparison**: Saved as `r2_performance_over_years.png` showing the $R^2$ line charts of all 17 models over the test years.
* **Loss Curves Data CSV**: Saved as `all_models_loss_curves.csv` containing step-by-step test RMSE values.
* **Trained Weights**: Saved in `models/` directory for fast loading on re-runs.
