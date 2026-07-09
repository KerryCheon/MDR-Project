# derived_8.1_pos-eval-2.0 (End-to-End Gating Evaluation Report)

This directory contains the training and evaluation notebook for measuring the **end-to-end** performance of hard-gated Mixture of Experts (MoE) models on the Washington-only `derived_8.1_pos` dataset (N=8,902 test samples across 13 stations). 

In this experiment, we implement **adapted training** where the specialists are trained directly on the training subsets partitioned by the gating router's decisions (imperfectly routed data) rather than partitioned by the ground-truth soil moisture labels. This allows the specialists to co-adapt to the noise and feature distributions of the samples they will actually receive at inference time.

The gating configurations evaluated are:
1. **3-Regime models** gated by:
   - Heuristic Month-Only Gating (static seasonal rule)
   - Random Forest Gating (Month + `G_API` + `LST_modis` + `SMAP_sm_pm_interp`)
2. **2-Regime model** gated by:
   - Random Forest Gating (Month + `G_API` + `LST_modis` + `SMAP_sm_pm_interp`)

All specialist models are trained on the `OVERALL_SELECTED_FEATURES` list (skipping feature tuning) using temporal recency weights ($\beta = 0.4$) on the `trainval` split. 

---

## 1. Comparative Results Table

The performance metrics on the held-out test split (N=8,902) are summarized below:

| Model ID | Model / Gating Configuration | $R^2$ | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| **Model 1** | Single Global XGBoost (No Gating Baseline) | **0.4858** | 0.0755 | 0.0745 | −0.0120 | 0.0554 | 0.0395 | 0.7084 |
| **Model 2** | 3-Regime Heuristic Month Gating (Adapted) | **0.4690** | 0.0767 | 0.0759 | −0.0112 | 0.0559 | 0.0400 | 0.6978 |
| **Model 3** | 3-Regime Random Forest Gating (Adapted) | **0.4965** | 0.0747 | 0.0733 | −0.0147 | 0.0537 | 0.0384 | 0.7243 |
| **Model 4** | 3-Regime Oracle Gating (Overall Features - Reference) | **0.8729** | 0.0375 | 0.0361 | −0.0102 | 0.0296 | 0.0244 | 0.9393 |
| **Model 5** | 2-Regime Random Forest Gating (Adapted) | **0.5219** | 0.0728 | 0.0719 | −0.0115 | 0.0528 | 0.0385 | 0.7335 |
| **Model 6** | 2-Regime Oracle Gating (Tuned 2R - Reference) | **0.7752** | 0.0499 | 0.0483 | −0.0127 | 0.0388 | 0.0318 | 0.8887 |

### Gating Router Classification Performance Summary

The classification performance of the gating routers on the held-out test split (N=8,902) is summarized below:

| Gating Router Configuration | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|---|---|---|---|---|---|---|---|
| **3-Regime Heuristic Month Gating** | **0.4974** | 0.5342 | 0.4870 | 0.4971 | 0.5602 | 0.4974 | 0.5141 |
| **3-Regime Random Forest Gating** | **0.6136** | 0.6127 | 0.5808 | 0.5572 | 0.6235 | 0.6136 | 0.5828 |
| **2-Regime Random Forest Gating** | **0.8257** | 0.8349 | 0.7889 | 0.8022 | 0.8296 | 0.8257 | 0.8193 |

---

## 2. Key Insights and Discussion

### 1. Training on Gated Slices Resolves the "Routing Penalty"
In the previous run of this experiment, when specialists were trained strictly on the ground-truth target regimes, the end-to-end 3-Regime RF-gated model achieved an $R^2$ of only **0.0988** (a massive routing penalty compared to the oracle's **0.8729**).

By training the specialists directly on the imperfectly routed subsets predicted by the gating router:
* The 3-Regime Heuristic Month-gated model $R^2$ jumps from **0.0796** to **0.4690**.
* The 3-Regime RF-gated model $R^2$ jumps from **0.0988** to **0.4965**.
* The 2-Regime RF-gated model $R^2$ jumps from **0.3054** to **0.5219**.

This adapted training method enables the specialists to learn from the actual feature distributions they will encounter at inference time, including the noise introduced by misrouted samples. This co-adaptation successfully mitigates the penalty of hard-gating routing errors.

### 2. End-to-End Gating Beats the Single Global Model
With the adapted training framework, the hard-gated MoE models successfully **outperform** the Single Global XGBoost model baseline ($R^2 = 0.4858$):
* The **3-Regime RF Gated Model (Model 3)** improves $R^2$ to **0.4965** ($+0.0107$ absolute gain).
* The **2-Regime RF Gated Model (Model 5)** improves $R^2$ to **0.5219** ($+0.0361$ absolute gain).

This is a critical milestone for the research project, as it represents the first time that end-to-end gated specialists have successfully beaten the global baseline on the `derived_8.1_pos` Washington state splits.

### 3. Why the 2-Regime Model Wins
The 2-Regime RF Gated model ($R^2 = 0.5219$) outperforms the 3-Regime RF Gated model ($R^2 = 0.4965$) because collapsing the intermediate transition zone into a binary routing task (Dry vs. Wet/Transition at $T=0.159$) significantly improves the router's performance:
* The 2-Regime RF router achieves **82.57% accuracy** (macro F1-score of **0.8022**) on the test set, compared to **61.36% accuracy** (macro F1-score of **0.5572**) for the 3-Regime RF router.
* Because the 2R router is more accurate, the specialist models are trained on cleaner, more homogeneous data splits, and fewer samples are misrouted at test time.

---

## 3. Visualizations

The following diagnostic plots are generated and saved in this directory:
- `gating_confusion_matrices.png`: Confusion matrices comparing the 3-Regime Heuristic and Random Forest routers.
- `residuals_3r.png`: Residual plots for the Heuristic and RF 3-Regime gated models.
- `residuals_and_cm_2r.png`: Confusion matrix and residuals plot for the 2-Regime RF gated model.

---

## 4. References
- [derived_8.1_pos-eval-1.2](../derived_8.1_pos-eval-1.2/README.md) — Oracle gating results and hyperparameter tuning.
- [derived_8.1_pos-data-exploration](../derived_8.1_pos-data-exploration/README.md) — Preliminary gating separability analysis.
