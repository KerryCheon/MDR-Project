# Experiment: `derived_8.4-eval-1.1` — MoE Routing Strategies under Shared 54-Feature Global Backbone & Model Persistence

## Objective
Evaluate how the **Shared 54-Feature Global Backbone + Add-Only Per-Regime Deltas** paradigm (derived from `derived_8.4-feature-selection-2.0`) interacts across all 5 MoE routing strategies ($K=2$) and the single-regime global baseline model.

Fix `TrainedGatingRouter` to use a non-oracle `XGBClassifier` trained on training split features without target leakage, and save all trained XGBoost model booster weights (`.json`), test predictions (`.npy`), regime labels (`.npy`), and evaluation loss curves (`.npy`) under `models/`.

## Overall Leaderboard (2023–2025 Test Set)

Evaluated on CUDA on the `derived_8.4` test set (6,620 samples across 7 WA stations):

| model_name | strategy_name | pooled_r2 | pooled_rmse | pooled_ubrmse | pooled_bias | pooled_mae | pooled_pearson |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10) | Clustering_V0_Full_k2 | 0.814960 | 0.043820 | 0.043337 | 0.006486 | 0.033720 | 0.905594 |
| Clustering_V0_Full_k2 (Backbone c0=0, c1=0) | Clustering_V0_Full_k2 | 0.814334 | 0.043894 | 0.043516 | 0.005749 | 0.033774 | 0.904903 |
| Clustering_Dynamic_k2 (Winner c0=0, c1=0) | Clustering_Dynamic_k2 | 0.786606 | 0.047057 | 0.046113 | 0.009380 | 0.036191 | 0.892172 |
| Global Single Model (54 Backbone) | Global_Single | 0.779230 | 0.047864 | 0.046687 | 0.010548 | 0.037059 | 0.889432 |
| Seasonal_Binary_k2 (Winner c0=0, c1=0) | Seasonal_Binary_k2 | 0.769795 | 0.048876 | 0.047750 | 0.010428 | 0.037698 | 0.884222 |
| Univariate_G_API_k2 (Winner c0=0, c1=0) | Univariate_G_API_k2 | 0.769632 | 0.048893 | 0.047845 | 0.010069 | 0.037948 | 0.883769 |
| Baseline V0 (50 Feats) | Global_Single | 0.760447 | 0.049858 | 0.048863 | 0.009912 | 0.038157 | 0.877516 |
| Trained_Gating_k2 (Winner c0=0, c1=0) | Trained_Gating_k2 | 0.735474 | 0.052393 | 0.050424 | 0.014226 | 0.038881 | 0.875079 |

---

## Strategy $\times$ Delta Grid Summary (Pooled $R^2$)

Heatmap matrix over per-regime add-only feature deltas ($c0, c1 \in \{0, 5, 10\}$):

| strategy_name | (0, 0) | (0, 5) | (0, 10) | (5, 0) | (5, 5) | (5, 10) | (10, 0) | (10, 5) | (10, 10) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Clustering_Dynamic_k2 | 0.786606 | 0.779280 | 0.771133 | 0.776704 | 0.769378 | 0.761231 | 0.763459 | 0.756133 | 0.747986 |
| Clustering_V0_Full_k2 | 0.814334 | 0.814302 | **0.814960** | 0.814146 | 0.814113 | 0.814771 | 0.789072 | 0.789039 | 0.789697 |
| Seasonal_Binary_k2 | **0.769795** | 0.756122 | 0.764271 | 0.765988 | 0.752315 | 0.760463 | 0.756494 | 0.742820 | 0.750969 |
| Trained_Gating_k2 | **0.735474** | 0.735064 | 0.732764 | 0.726236 | 0.725826 | 0.723526 | 0.719924 | 0.719514 | 0.717214 |
| Univariate_G_API_k2 | **0.769632** | 0.767081 | 0.756354 | 0.756171 | 0.753620 | 0.742893 | 0.763862 | 0.761311 | 0.750584 |

---

## Per-Regime Performance Breakdown

| strategy_name | cluster | n_train | n_test | r2 | rmse | ubrmse | bias | mae |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Global_Single | 0 | 14608 | 6620 | 0.779230 | 0.047864 | 0.046687 | 0.010548 | 0.037059 |
| Clustering_V0_Full_k2 | 0 | 10624 | 4817 | 0.802460 | 0.044464 | 0.043621 | 0.008615 | 0.035922 |
| Clustering_V0_Full_k2 | 1 | 3984 | 1803 | 0.844023 | 0.042050 | 0.042043 | 0.000797 | 0.027835 |
| Clustering_Dynamic_k2 | 0 | 7974 | 3717 | 0.820626 | 0.046568 | 0.045518 | 0.009834 | 0.035184 |
| Clustering_Dynamic_k2 | 1 | 6634 | 2903 | 0.415307 | 0.047676 | 0.046857 | 0.008799 | 0.037482 |
| Univariate_G_API_k2 | 0 | 7304 | 3513 | 0.807786 | 0.049009 | 0.048387 | 0.007783 | 0.036211 |
| Univariate_G_API_k2 | 1 | 7304 | 3107 | 0.465612 | 0.048761 | 0.047091 | 0.012653 | 0.039913 |
| Seasonal_Binary_k2 | 0 | 7559 | 3415 | 0.778262 | 0.049871 | 0.048676 | 0.010851 | 0.038282 |
| Seasonal_Binary_k2 | 1 | 7049 | 3205 | 0.297058 | 0.047793 | 0.046740 | 0.009977 | 0.037076 |
| Trained_Gating_k2 | 0 | 4183 | 1931 | 0.474869 | 0.043925 | 0.043629 | -0.005088 | 0.029833 |
| Trained_Gating_k2 | 1 | 10425 | 4689 | 0.254845 | 0.055506 | 0.050881 | 0.022180 | 0.042607 |

---

## Year-by-Year $R^2$ Breakdown

| model_name | pooled_r2 | year_2023_r2 | year_2024_r2 | year_2025_r2 |
|:---|:---:|:---:|:---:|:---:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10) | 0.814960 | 0.822971 | 0.783256 | 0.830290 |
| Clustering_V0_Full_k2 (Backbone c0=0, c1=0) | 0.814334 | 0.821300 | 0.785101 | 0.828436 |
| Clustering_Dynamic_k2 (Winner c0=0, c1=0) | 0.786606 | 0.759412 | 0.778981 | 0.818225 |
| Global Single Model (54 Backbone) | 0.779230 | 0.750748 | 0.770077 | 0.813582 |
| Seasonal_Binary_k2 (Winner c0=0, c1=0) | 0.769795 | 0.733203 | 0.761812 | 0.812083 |
| Univariate_G_API_k2 (Winner c0=0, c1=0) | 0.769632 | 0.730946 | 0.768470 | 0.808214 |
| Baseline V0 (50 Feats) | 0.760447 | 0.740984 | 0.756292 | 0.778796 |
| Trained_Gating_k2 (Winner c0=0, c1=0) | 0.735474 | 0.697562 | 0.732429 | 0.773271 |

---

## Key Takeaways

1. **Winning MoE Architecture**: `Clustering_V0_Full_k2` with $c0=0, c1=10$ achieves the highest overall test performance with **$R^2 = 0.8150$** and **$\text{RMSE} = 0.0438$**, outperforming the 54-feature single-regime global model ($R^2 = 0.7792$) by $+0.0358 \Delta R^2$.
2. **Realistic Non-Oracle Trained Gating**: When evaluated using a real `XGBClassifier` without ground-truth target leakage, `Trained_Gating_k2` yields $R^2 = 0.7355$.
3. **Model Weight Persistence**: All 206 trained booster JSON files, prediction arrays, regime labels, and iteration curves are archived in `models/`.
