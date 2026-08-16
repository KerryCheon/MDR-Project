# Experiment: `derived_8.4-formal-eval-1.0` — Formal Statistical Evaluation of the Two-Regime Clustering Model

## Objective

Publication-oriented statistical evaluation of the claim established in `derived_8.4-eval-1.1` / `-1.3`:
**a two-regime (KMeans k=2) clustering model beats the single-regime global model and the trained-gating
model**, on the frozen temporal split (2023–2025 test) and under leave-one-station-out (LOSO) spatial
generalization.

All tables below are copied verbatim from the stdout of the executed report notebook
(`derived_8.4-formal-eval-1.0.ipynb`, executed with `nb execute --uv` from `notebooks/`).

## Configurations (20)

14 requested configurations (test-selected deltas pinned from `derived_8.4-eval-1.1`'s delta grid,
identical parsing to eval-1.3; `none` = c0=c1=0) + 6 val-selected winners (`select_deltas_val.py`,
selection on validation-period residuals with the same 2500-tree hyperparameters as the evaluation):

| config_id                            | strategy_name            | delta_source   |   cluster_0_count |   cluster_1_count |   n_global_features |   n_add0 |   n_add1 |
|:-------------------------------------|:-------------------------|:---------------|------------------:|------------------:|--------------------:|---------:|---------:|
| Clustering_V0_Full_k2_c0_0_c1_10     | Clustering_V0_Full_k2    | test           |                 0 |                10 |                  54 |        0 |       10 |
| Clustering_V0_Full_k2_c0_0_c1_0      | Clustering_V0_Full_k2    | none           |                 0 |                 0 |                  54 |        0 |        0 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Clustering_Backbone54_k2 | test           |                10 |                10 |                  54 |       10 |       10 |
| Clustering_Backbone54_k2_c0_0_c1_0   | Clustering_Backbone54_k2 | none           |                 0 |                 0 |                  54 |        0 |        0 |
| Global_Single_54                     | Global_Single            | global         |               nan |               nan |                  54 |        0 |        0 |
| Baseline_V0_50                       | Global_Single            | global         |               nan |               nan |                  50 |        0 |        0 |
| Univariate_G_API_k2_c0_10_c1_0       | Univariate_G_API_k2      | test           |                10 |                 0 |                  54 |       10 |        0 |
| Univariate_G_API_k2_c0_0_c1_0        | Univariate_G_API_k2      | none           |                 0 |                 0 |                  54 |        0 |        0 |
| Clustering_Dynamic_k2_c0_10_c1_0     | Clustering_Dynamic_k2    | test           |                10 |                 0 |                  54 |       10 |        0 |
| Clustering_Dynamic_k2_c0_0_c1_0      | Clustering_Dynamic_k2    | none           |                 0 |                 0 |                  54 |        0 |        0 |
| Seasonal_Binary_k2_c0_0_c1_5         | Seasonal_Binary_k2       | test           |                 0 |                 5 |                  54 |        0 |        5 |
| Seasonal_Binary_k2_c0_0_c1_0         | Seasonal_Binary_k2       | none           |                 0 |                 0 |                  54 |        0 |        0 |
| Trained_Gating_k2_c0_5_c1_10         | Trained_Gating_k2        | test           |                 5 |                10 |                  54 |        5 |       10 |
| Trained_Gating_k2_c0_0_c1_0          | Trained_Gating_k2        | none           |                 0 |                 0 |                  54 |        0 |        0 |
| Clustering_V0_Full_k2_val_winner     | Clustering_V0_Full_k2    | val            |                10 |                 5 |                  54 |       10 |        5 |
| Clustering_Backbone54_k2_val_winner  | Clustering_Backbone54_k2 | val            |                 5 |                10 |                  54 |        5 |       10 |
| Clustering_Dynamic_k2_val_winner     | Clustering_Dynamic_k2    | val            |                 0 |                10 |                  54 |        0 |       10 |
| Univariate_G_API_k2_val_winner       | Univariate_G_API_k2      | val            |                10 |                10 |                  54 |       10 |       10 |
| Seasonal_Binary_k2_val_winner        | Seasonal_Binary_k2       | val            |                10 |                 5 |                  54 |       10 |        5 |
| Trained_Gating_k2_val_winner         | Trained_Gating_k2        | val            |                10 |                10 |                  54 |       10 |       10 |

## Protocol

- **Temporal (primary):** experts trained on trainval (train 2017–2020 + val 2021–2022, 14,608 rows),
  evaluated on the frozen test set (2023–2025, 6,620 rows, 7 WA stations), **30 random seeds**
  (seed 42 included as exact replication anchor vs eval-1.1 / eval-1.3).
- **LOSO (secondary):** same 20 configurations × **5 seeds** × 7 held-out stations; router refitted per
  fold on the 6-station trainval (no held-out-station leakage into routing).
- **Delta-robustness:** per-regime delta features from three selection sources — *test-selected*,
  *val-selected* (re-ranked on validation-period residuals, train-only fits), *none*.
- **Seed scope:** only the XGBoost expert regressors' `random_state` varies; routers (KMeans / gating
  classifier) stay at seed 42 because the delta additions are tied to the seed-42 cluster labels.
- **Statistics:** seed-level (mean ± std, median, 95% t-CI, paired t-test, Wilcoxon signed-rank,
  % seeds A better), sample-level (paired cluster bootstrap over (station, month) blocks, percentile
  95% CI + bootstrap p), Benjamini–Hochberg FDR over the pre-specified comparison family, LOSO
  per-station win counts + two-sided sign test (n = 7; 7/7 → p ≈ 0.016, 6/7 → p = 0.125).

## Temporal results

### Seed-level summary — R² (mean ± std over seeds, [95% t-CI])

| config_label                           | delta_source   |   n_seeds | mean_std        |   median | ci               |
|:---------------------------------------|:---------------|----------:|:----------------|---------:|:-----------------|
| Clustering_V0_Full_k2  c0=0, c1=10     | test           |        30 | 0.8126 ± 0.0013 | 0.812762 | [0.8122, 0.8131] |
| Clustering_V0_Full_k2  c0=0, c1=0      | none           |        30 | 0.8118 ± 0.0014 | 0.811928 | [0.8113, 0.8124] |
| Clustering_Backbone54_k2  c0=0, c1=0   | none           |        30 | 0.8117 ± 0.0014 | 0.811832 | [0.8112, 0.8122] |
| Clustering_Backbone54_k2  c0=10, c1=10 | test           |        30 | 0.7893 ± 0.0014 | 0.789366 | [0.7888, 0.7899] |
| Clustering_Dynamic_k2  c0=0, c1=0      | none           |        30 | 0.7855 ± 0.0010 | 0.785356 | [0.7851, 0.7858] |
| Global_Single_54                       | global         |        30 | 0.7798 ± 0.0013 | 0.779674 | [0.7793, 0.7803] |
| Clustering_Dynamic_k2  c0=0, c1=10     | val            |        30 | 0.7723 ± 0.0019 | 0.772677 | [0.7716, 0.7730] |
| Seasonal_Binary_k2  c0=0, c1=0         | none           |        30 | 0.7700 ± 0.0016 | 0.770019 | [0.7694, 0.7706] |
| Univariate_G_API_k2  c0=0, c1=0        | none           |        30 | 0.7676 ± 0.0009 | 0.76754  | [0.7672, 0.7679] |
| Clustering_Dynamic_k2  c0=10, c1=0     | test           |        30 | 0.7638 ± 0.0011 | 0.763858 | [0.7634, 0.7642] |
| Univariate_G_API_k2  c0=10, c1=0       | test           |        30 | 0.7627 ± 0.0011 | 0.762707 | [0.7623, 0.7631] |
| Baseline_V0_50                         | global         |        30 | 0.7593 ± 0.0015 | 0.759359 | [0.7588, 0.7599] |
| Seasonal_Binary_k2  c0=0, c1=5         | test           |        30 | 0.7566 ± 0.0014 | 0.756557 | [0.7561, 0.7571] |
| Univariate_G_API_k2  c0=10, c1=10      | val            |        30 | 0.7517 ± 0.0012 | 0.751517 | [0.7512, 0.7521] |
| Clustering_Backbone54_k2  c0=5, c1=10  | val            |        30 | 0.7490 ± 0.0031 | 0.749295 | [0.7478, 0.7501] |
| Trained_Gating_k2  c0=0, c1=0          | none           |        30 | 0.7354 ± 0.0011 | 0.735156 | [0.7350, 0.7358] |
| Clustering_V0_Full_k2  c0=10, c1=5     | val            |        30 | 0.7351 ± 0.0025 | 0.7351   | [0.7342, 0.7361] |
| Trained_Gating_k2  c0=5, c1=10         | test           |        30 | 0.7227 ± 0.0009 | 0.722885 | [0.7224, 0.7231] |
| Trained_Gating_k2  c0=10, c1=10        | val            |        30 | 0.7220 ± 0.0014 | 0.722079 | [0.7215, 0.7225] |
| Seasonal_Binary_k2  c0=10, c1=5        | val            |        30 | 0.6909 ± 0.0020 | 0.690963 | [0.6901, 0.6916] |

### Seed-level summary — RMSE / MAE / bias (m³/m³; lower is better except bias sign)

| config_label                           | delta_source   |   n_seeds | RMSE mean ± std        |   RMSE median | MAE mean ± std       |   MAE median | BIAS mean ± std       |   BIAS median |
|:---------------------------------------|:---------------|----------:|:-----------------------|--------------:|:---------------------|-------------:|:----------------------|--------------:|
| Clustering_V0_Full_k2  c0=0, c1=10     | test           |        30 | 0.04409 ± 0.00016      |      0.0440792 | 0.03392 ± 0.00011    |     0.0339159 | 0.00660 ± 0.00024     |     0.00658228 |
| Clustering_V0_Full_k2  c0=0, c1=0      | none           |        30 | 0.04419 ± 0.00016      |      0.0441771 | 0.03398 ± 0.00011    |     0.0339656 | 0.00596 ± 0.00022     |     0.0059814 |
| Clustering_Backbone54_k2  c0=0, c1=0   | none           |        30 | 0.04420 ± 0.00016      |      0.0441885 | 0.03398 ± 0.00011    |     0.0339709 | 0.00598 ± 0.00022     |     0.00599854 |
| Clustering_Backbone54_k2  c0=10, c1=10 | test           |        30 | 0.04676 ± 0.00016      |      0.046752  | 0.03579 ± 0.00012    |     0.0358191 | 0.00707 ± 0.00024     |     0.00707422 |
| Clustering_Dynamic_k2  c0=0, c1=0      | none           |        30 | 0.04718 ± 0.00011      |      0.0471949 | 0.03631 ± 0.00010    |     0.0363103 | 0.00958 ± 0.00021     |     0.00956118 |
| Global_Single_54                       | global         |        30 | 0.04780 ± 0.00014      |      0.0478155 | 0.03694 ± 0.00012    |     0.0369541 | 0.01004 ± 0.00032     |     0.0100552 |
| Clustering_Dynamic_k2  c0=0, c1=10     | val            |        30 | 0.04861 ± 0.00021      |      0.0485688 | 0.03736 ± 0.00016    |     0.0373542 | 0.01213 ± 0.00031     |     0.0121233 |
| Seasonal_Binary_k2  c0=0, c1=0         | none           |        30 | 0.04885 ± 0.00017      |      0.048852  | 0.03767 ± 0.00013    |     0.0376961 | 0.01071 ± 0.00021     |     0.0107012 |
| Univariate_G_API_k2  c0=0, c1=0        | none           |        30 | 0.04911 ± 0.00010      |      0.0491146 | 0.03815 ± 0.00008    |     0.0381569 | 0.01063 ± 0.00024     |     0.0106203 |
| Clustering_Dynamic_k2  c0=10, c1=0     | test           |        30 | 0.04951 ± 0.00012      |      0.049502  | 0.03873 ± 0.00011    |     0.0387046 | 0.00966 ± 0.00024     |     0.00966423 |
| Univariate_G_API_k2  c0=10, c1=0       | test           |        30 | 0.04962 ± 0.00011      |      0.0496225 | 0.03852 ± 0.00011    |     0.0385211 | 0.01130 ± 0.00022     |     0.0112902 |
| Baseline_V0_50                         | global         |        30 | 0.04997 ± 0.00015      |      0.0499713 | 0.03830 ± 0.00011    |     0.038278  | 0.00958 ± 0.00026     |     0.00959518 |
| Seasonal_Binary_k2  c0=0, c1=5         | test           |        30 | 0.05026 ± 0.00014      |      0.0502614 | 0.03875 ± 0.00013    |     0.038728  | 0.01079 ± 0.00021     |     0.0107854 |
| Univariate_G_API_k2  c0=10, c1=10      | val            |        30 | 0.05076 ± 0.00012      |      0.050779  | 0.03948 ± 0.00011    |     0.0394881 | 0.01157 ± 0.00019     |     0.0115562 |
| Clustering_Backbone54_k2  c0=5, c1=10  | val            |        30 | 0.05104 ± 0.00031      |      0.0510056 | 0.03883 ± 0.00019    |     0.0388452 | 0.00893 ± 0.00030     |     0.00895874 |
| Trained_Gating_k2  c0=0, c1=0          | none           |        30 | 0.05240 ± 0.00011      |      0.0524241 | 0.03882 ± 0.00009    |     0.0388224 | 0.01453 ± 0.00018     |     0.0145035 |
| Clustering_V0_Full_k2  c0=10, c1=5     | val            |        30 | 0.05243 ± 0.00025      |      0.0524297 | 0.03943 ± 0.00014    |     0.0394353 | 0.00758 ± 0.00021     |     0.00758365 |
| Trained_Gating_k2  c0=5, c1=10         | test           |        30 | 0.05364 ± 0.00009      |      0.0536248 | 0.04035 ± 0.00007    |     0.0403555 | 0.01571 ± 0.00015     |     0.0156895 |
| Trained_Gating_k2  c0=10, c1=10        | val            |        30 | 0.05371 ± 0.00013      |      0.0537028 | 0.04112 ± 0.00011    |     0.0410982 | 0.01663 ± 0.00021     |     0.0166027 |
| Seasonal_Binary_k2  c0=10, c1=5        | val            |        30 | 0.05664 ± 0.00018      |      0.0566293 | 0.04220 ± 0.00012    |     0.0421812 | 0.01003 ± 0.00017     |     0.00999082 |

95% t-CIs are in `temporal_config_summary.csv` (all metrics).

### Focused pairwise comparisons (R²; mean diff A−B, [95% CI], paired t p, Wilcoxon p, % seeds A better, q = BH-FDR)

The pre-specified family: every model vs {Global_Single_54, Baseline_V0_50, Trained_Gating_k2_c0_5_c1_10}
plus within-strategy delta ablations. All rows for the four metrics are in `temporal_pairwise_focused.csv`
(300 rows); the all-pairwise seed-level matrix is in `temporal_pairwise_all.csv` (760 rows).

| A                                    | B                                   |   mean_A |   mean_B |   mean_diff | ci                   |     t_p |   wilcoxon_p |   pct_A_better |    q_bh |
|:-------------------------------------|:------------------------------------|---------:|---------:|------------:|:---------------------|--------:|-------------:|---------------:|--------:|
| Clustering_V0_Full_k2_c0_0_c1_10     | Trained_Gating_k2_c0_5_c1_10        |  0.81264 |  0.72274 |     0.08990 | [0.08935, 0.09046]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_V0_Full_k2_c0_0_c1_0      | Trained_Gating_k2_c0_5_c1_10        |  0.81184 |  0.72274 |     0.08910 | [0.08855, 0.08965]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Backbone54_k2_c0_0_c1_0   | Trained_Gating_k2_c0_5_c1_10        |  0.81172 |  0.72274 |     0.08898 | [0.08843, 0.08953]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Global_Single_54                    |  0.81264 |  0.77979 |     0.03285 | [0.03213, 0.03357]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_V0_Full_k2_c0_0_c1_0      | Global_Single_54                    |  0.81184 |  0.77979 |     0.03205 | [0.03131, 0.03279]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Backbone54_k2_c0_0_c1_0   | Global_Single_54                    |  0.81172 |  0.77979 |     0.03193 | [0.03119, 0.03267]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Global_Single_54                    |  0.78932 |  0.77979 |     0.00952 | [0.00882, 0.01022]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Baseline_V0_50                      |  0.78932 |  0.75934 |     0.02998 | [0.02920, 0.03076]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Baseline_V0_50                      |  0.81264 |  0.75934 |     0.05331 | [0.05255, 0.05406]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Global_Single_54                     | Baseline_V0_50                      |  0.77979 |  0.75934 |     0.02046 | [0.01975, 0.02117]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Global_Single_54                     | Trained_Gating_k2_c0_5_c1_10        |  0.77979 |  0.72274 |     0.05705 | [0.05643, 0.05767]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Clustering_V0_Full_k2_c0_0_c1_0     |  0.81264 |  0.81184 |     0.00080 | [0.00064, 0.00097]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Clustering_V0_Full_k2_val_winner    |  0.81264 |  0.73512 |     0.07752 | [0.07663, 0.07841]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Backbone54_k2_c0_0_c1_0   | Clustering_Backbone54_k2_val_winner |  0.81172 |  0.74897 |     0.06275 | [0.06156, 0.06395]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Dynamic_k2_c0_0_c1_0      | Clustering_Dynamic_k2_val_winner    |  0.78546 |  0.77228 |     0.01318 | [0.01238, 0.01398]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Seasonal_Binary_k2_c0_0_c1_0         | Seasonal_Binary_k2_val_winner       |  0.77002 |  0.69087 |     0.07914 | [0.07832, 0.07997]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Univariate_G_API_k2_c0_0_c1_0        | Univariate_G_API_k2_val_winner      |  0.76757 |  0.75166 |     0.01591 | [0.01540, 0.01642]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Trained_Gating_k2_c0_0_c1_0          | Trained_Gating_k2_val_winner        |  0.73536 |  0.72198 |     0.01338 | [0.01284, 0.01392]   | 0.00000 |      0.00000 |      100.00000 | 0.00000 |
| Clustering_Dynamic_k2_val_winner     | Global_Single_54                    |  0.77228 |  0.77979 |    -0.00751 | [-0.00840, -0.00662] | 0.00000 |      0.00000 |        0.00000 | 0.00000 |
| Clustering_Backbone54_k2_val_winner  | Global_Single_54                    |  0.74897 |  0.77979 |    -0.03083 | [-0.03205, -0.02960] | 0.00000 |      0.00000 |        0.00000 | 0.00000 |
| Clustering_V0_Full_k2_val_winner     | Global_Single_54                    |  0.73512 |  0.77979 |    -0.04467 | [-0.04565, -0.04370] | 0.00000 |      0.00000 |        0.00000 | 0.00000 |

### Sample-level bootstrap (paired cluster bootstrap over (station, month) blocks; seed-42 fits)

Percentile 95% CIs and two-sided bootstrap p for the A−B difference; full matrix in
`temporal_bootstrap.csv` (300 rows).

| A                                | B                                | metric |   diff_mean | diff CI                |   bootstrap_p |
|:---------------------------------|:---------------------------------|-------:|------------:|:------------------------|--------------:|
| Clustering_V0_Full_k2_c0_0_c1_10 | Global_Single_54                 | R2     |     0.03520 | [0.01899, 0.05488]      |        0.0005 |
| Clustering_V0_Full_k2_c0_0_c1_10 | Global_Single_54                 | RMSE   |    -0.00395 | [-0.00596, -0.00222]    |        0.0005 |
| Clustering_V0_Full_k2_c0_0_c1_10 | Global_Single_54                 | BIAS   |    -0.00403 | [-0.00587, -0.00220]    |        0.0005 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Global_Single_54              | R2     |     0.00945 | [-0.01325, 0.03154]     |        0.3670 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Global_Single_54              | RMSE   |    -0.00103 | [-0.00335, 0.00144]     |        0.3670 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Global_Single_54              | BIAS   |    -0.00331 | [-0.00555, -0.00123]    |        0.0050 |
| Clustering_V0_Full_k2_c0_0_c1_10 | Trained_Gating_k2_c0_5_c1_10     | R2     |     0.09217 | [0.05918, 0.12930]      |        0.0005 |
| Clustering_V0_Full_k2_c0_0_c1_10 | Trained_Gating_k2_c0_5_c1_10     | RMSE   |    -0.00972 | [-0.01292, -0.00660]    |        0.0005 |
| Clustering_V0_Full_k2_c0_0_c1_10 | Trained_Gating_k2_c0_5_c1_10     | BIAS   |    -0.00902 | [-0.01221, -0.00587]    |        0.0005 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Baseline_V0_50                 | R2     |     0.02897 | [0.00833, 0.04953]      |        0.0040 |

Sensitivity with (station, year) blocks (21 blocks) gives the same conclusions
(V0_Full (0,10) vs Global R² diff +0.0362 [0.0134, 0.0619] p=0.0005; Backbone54 (10,10) vs Global
R² diff +0.0104 [-0.0188, 0.0414] p=0.532 — see notebook cell 13).

## LOSO results

### Per-configuration summary (per-station median over seeds, then mean/median over the 7 stations)

RMSE / MAE / bias are in m³/m³ (lower is better except bias sign); full precision in
`loso_config_summary.csv`.

| config_label                           | delta_source   |   loso_mean_r2 |   loso_mean_rmse |   loso_mean_mae |   loso_mean_bias |   loso_median_r2 |   loso_median_rmse |   loso_median_mae |   loso_median_bias |
|:---------------------------------------|:---------------|---------------:|-----------------:|----------------:|-----------------:|-----------------:|-------------------:|------------------:|-------------------:|
| Clustering_V0_Full_k2  c0=0, c1=10     | test           |         0.6430 |          0.05566 |         0.04428 |          0.01562 |           0.6936 |            0.05121 |           0.04160 |            0.01981 |
| Clustering_V0_Full_k2  c0=0, c1=0      | none           |         0.6372 |          0.05605 |         0.04476 |          0.01528 |           0.6936 |            0.05121 |           0.04160 |            0.01981 |
| Clustering_Backbone54_k2  c0=10, c1=10 | test           |         0.6236 |          0.05687 |         0.04656 |          0.01441 |           0.6766 |            0.05597 |           0.04668 |            0.01446 |
| Clustering_Backbone54_k2  c0=0, c1=0   | none           |         0.6208 |          0.05705 |         0.04603 |          0.01519 |           0.6868 |            0.05121 |           0.04160 |            0.01981 |
| Baseline_V0_50                         | global         |         0.5907 |          0.06011 |         0.04854 |          0.01682 |           0.5709 |            0.05765 |           0.04739 |            0.01980 |
| Seasonal_Binary_k2  c0=0, c1=5         | test           |         0.5808 |          0.06145 |         0.04953 |          0.02084 |           0.5894 |            0.05716 |           0.04407 |            0.01607 |
| Global_Single_54                       | global         |         0.5795 |          0.06102 |         0.04966 |          0.02224 |           0.6269 |            0.05161 |           0.04322 |            0.02513 |
| Clustering_Dynamic_k2  c0=0, c1=10     | val            |         0.5782 |          0.06172 |         0.04980 |          0.02166 |           0.5873 |            0.05691 |           0.04492 |            0.01768 |
| Clustering_Dynamic_k2  c0=10, c1=0     | test           |         0.5765 |          0.06094 |         0.04985 |          0.02205 |           0.5183 |            0.05581 |           0.04506 |            0.02107 |
| Univariate_G_API_k2  c0=10, c1=0       | test           |         0.5751 |          0.06162 |         0.05024 |          0.02219 |           0.5619 |            0.05582 |           0.04770 |            0.01892 |
| Seasonal_Binary_k2  c0=0, c1=0         | none           |         0.5744 |          0.06189 |         0.05034 |          0.02130 |           0.5928 |            0.05762 |           0.04664 |            0.01507 |
| Univariate_G_API_k2  c0=10, c1=10      | val            |         0.5628 |          0.06265 |         0.05054 |          0.02212 |           0.5623 |            0.05912 |           0.04666 |            0.01744 |
| Clustering_Dynamic_k2  c0=0, c1=0      | none           |         0.5627 |          0.06243 |         0.05101 |          0.02187 |           0.6286 |            0.05997 |           0.04765 |            0.01848 |
| Trained_Gating_k2  c0=5, c1=10         | test           |         0.5551 |          0.06178 |         0.04688 |          0.01900 |           0.5618 |            0.05767 |           0.04070 |            0.01351 |
| Univariate_G_API_k2  c0=0, c1=0        | none           |         0.5490 |          0.06340 |         0.05133 |          0.02169 |           0.5508 |            0.06392 |           0.04803 |            0.01736 |
| Seasonal_Binary_k2  c0=10, c1=5        | val            |         0.5439 |          0.06394 |         0.05183 |          0.01953 |           0.5752 |            0.06208 |           0.04990 |            0.01407 |
| Trained_Gating_k2  c0=10, c1=10        | val            |         0.5407 |          0.06304 |         0.04839 |          0.02034 |           0.5227 |            0.05910 |           0.04285 |            0.01998 |
| Clustering_Backbone54_k2  c0=5, c1=10  | val            |         0.5161 |          0.06564 |         0.05176 |          0.02081 |           0.4980 |            0.06427 |           0.05242 |            0.00943 |
| Clustering_V0_Full_k2  c0=10, c1=5     | val            |         0.5060 |          0.06659 |         0.05233 |          0.02036 |           0.5278 |            0.06958 |           0.05542 |            0.00836 |
| Trained_Gating_k2  c0=0, c1=0          | none           |         0.4906 |          0.06669 |         0.05290 |          0.03004 |           0.4339 |            0.06019 |           0.04839 |            0.03213 |

### Focused LOSO pairwise tests (R²; wins "k of 7 stations", two-sided sign test, paired t / Wilcoxon on the 7 per-station medians)

Full matrix in `loso_pairwise_station.csv` (760 rows); focused family in `loso_pairwise_focused.csv`.

| A                                    | B                                   |   n_stations |   mean_diff |   wins |   sign_p |    t_p |   wilcoxon_p |   q_bh |
|:-------------------------------------|:------------------------------------|-------------:|------------:|-------:|---------:|-------:|-------------:|-------:|
| Clustering_V0_Full_k2_c0_0_c1_10     | Clustering_V0_Full_k2_val_winner    |            7 |      0.1369 |      7 |   0.0156 | 0.0366 |       0.0156 | 0.5849 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Trained_Gating_k2_c0_5_c1_10        |            7 |      0.0879 |      5 |   0.4531 | 0.3842 |       0.3750 | 0.9569 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Global_Single_54                    |            7 |      0.0635 |      6 |   0.1250 | 0.2648 |       0.2188 | 0.9569 |
| Clustering_V0_Full_k2_c0_0_c1_0      | Global_Single_54                    |            7 |      0.0577 |      6 |   0.1250 | 0.3177 |       0.2188 | 0.9569 |
| Clustering_V0_Full_k2_c0_0_c1_10     | Baseline_V0_50                      |            7 |      0.0522 |      5 |   0.4531 | 0.4347 |       0.5781 | 0.9569 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Global_Single_54                    |            7 |      0.0441 |      3 |   1.0000 | 0.4678 |       0.9375 | 0.9569 |
| Clustering_Backbone54_k2_c0_10_c1_10 | Baseline_V0_50                      |            7 |      0.0329 |      5 |   0.4531 | 0.6486 |       0.5781 | 0.9569 |
| Baseline_V0_50                       | Global_Single_54                    |            7 |      0.0112 |      4 |   1.0000 | 0.8740 |       0.9375 | 0.9693 |

Per-station pair plots (`loso_pair_r2_*.png`, `loso_pair_rmse_*.png`) and per-station bars with seed
error bars (`loso_station_bars_r2_*.png`) support the descriptive "A beats B on k of 7 stations" claims.

## Delta-robustness (does the two-regime conclusion survive the delta-selection source?)

| strategy                 | test_config                          | test_temporal_r2   |   test_loso_r2 | val_config                          | val_temporal_r2   |   val_loso_r2 | none_config                        | none_temporal_r2   |   none_loso_r2 |
|:-------------------------|:-------------------------------------|:-------------------|---------------:|:------------------------------------|:------------------|--------------:|:-----------------------------------|:-------------------|---------------:|
| Clustering_V0_Full_k2    | Clustering_V0_Full_k2_c0_0_c1_10     | 0.8126 ± 0.0013    |         0.643  | Clustering_V0_Full_k2_val_winner    | 0.7351 ± 0.0025   |        0.506  | Clustering_V0_Full_k2_c0_0_c1_0    | 0.8118 ± 0.0014    |         0.6372 |
| Clustering_Backbone54_k2 | Clustering_Backbone54_k2_c0_10_c1_10 | 0.7893 ± 0.0014    |         0.6236 | Clustering_Backbone54_k2_val_winner | 0.7490 ± 0.0031   |        0.5161 | Clustering_Backbone54_k2_c0_0_c1_0 | 0.8117 ± 0.0014    |         0.6208 |
| Univariate_G_API_k2      | Univariate_G_API_k2_c0_10_c1_0       | 0.7627 ± 0.0011    |         0.5751 | Univariate_G_API_k2_val_winner      | 0.7517 ± 0.0012   |        0.5628 | Univariate_G_API_k2_c0_0_c1_0      | 0.7676 ± 0.0009    |         0.549  |
| Clustering_Dynamic_k2    | Clustering_Dynamic_k2_c0_10_c1_0     | 0.7638 ± 0.0011    |         0.5765 | Clustering_Dynamic_k2_val_winner    | 0.7723 ± 0.0019   |        0.5782 | Clustering_Dynamic_k2_c0_0_c1_0    | 0.7855 ± 0.0010    |         0.5627 |
| Seasonal_Binary_k2       | Seasonal_Binary_k2_c0_0_c1_5         | 0.7566 ± 0.0014    |         0.5808 | Seasonal_Binary_k2_val_winner       | 0.6909 ± 0.0020   |        0.5439 | Seasonal_Binary_k2_c0_0_c1_0       | 0.7700 ± 0.0016    |         0.5744 |
| Trained_Gating_k2        | Trained_Gating_k2_c0_5_c1_10         | 0.7227 ± 0.0009    |         0.5551 | Trained_Gating_k2_val_winner        | 0.7220 ± 0.0014   |        0.5407 | Trained_Gating_k2_c0_0_c1_0        | 0.7354 ± 0.0011    |         0.4906 |

## Replication checks (seed 42 must reproduce the deterministic historical runs)

```
TEMPORAL (vs eval-1.1 pooled test R2):
  Clustering_V0_Full_k2_c0_0_c1_10: got=0.814960 expected=0.814960 |diff|=1.12e-07 [OK]
  Global_Single_54: got=0.779230 expected=0.779230 |diff|=1.95e-07 [OK]
  Baseline_V0_50: got=0.760447 expected=0.760447 |diff|=3.83e-07 [OK]
LOSO (vs eval-1.2/-1.3 loso mean R2):
  Clustering_Backbone54_k2_c0_10_c1_10: got=0.6243 expected=0.6243 |diff|=0.0000 [OK]
  Clustering_Backbone54_k2_c0_0_c1_0: got=0.6171 expected=0.6171 |diff|=0.0000 [OK]
  Clustering_V0_Full_k2_c0_0_c1_10: got=0.6415 expected=0.6415 |diff|=0.0000 [OK]
```

## Key takeaways (for the paper)

1. **Temporal, seed-level:** the two-regime clustering models beat the single-regime global model and
   the trained-gating model with overwhelming significance. V0_Full (0,10): R² 0.8126 ± 0.0013 vs
   Global_54 0.7798 ± 0.0013 (+0.0329, p < 1e-12, 100% of 30 seeds, q = 0); Backbone54 (0,0):
   +0.0319 (p < 1e-12); Backbone54 (10,10): +0.0095 (p < 1e-12 at seed level). RMSE, MAE and bias all
   improve consistently (V0_Full (0,10) RMSE 0.04409 vs Global 0.04780).
2. **Temporal, sample-level (cluster bootstrap):** the V0_Full (0,10) advantage over the global model
   is significant with both block schemes (ΔR² +0.0352, 95% CI [0.019, 0.055], p = 0.0005). The
   Backbone54 (10,10) temporal advantage is NOT significant at the sample level (ΔR² +0.0095,
   CI [−0.013, 0.032], p = 0.37) — its edge is primarily spatial.
3. **LOSO spatial:** V0_Full (0,10) mean-over-stations R² 0.6430 vs Global_54 0.5795 and Baseline
   0.5907; it wins on 6 of 7 stations vs the global model (sign test p = 0.125 — n = 7 lacks power;
   per-station pair plots carry the claim). The (10,10) Backbone54 point generalizes best among the
   54-backbone variants (0.6236).
4. **Delta robustness — important caveat:** with **test-selected** deltas or **no** deltas, the
   two-regime clustering advantage holds temporally (0.812/0.812 vs 0.780) and spatially (0.643/0.637
   vs 0.580). With **val-selected** deltas the temporal advantage does NOT hold (V0_Full val winner
   0.7351 < Global 0.7798; all val winners sit below the global model temporally, and below their own
   no-delta variants). Delta selection transfers poorly from the 2021–2022 validation period to the
   2023–2025 test period — a selection-instability finding that must be stated in the paper. The
   regime split itself (no-delta arm) carries the temporal gain.
5. **Replication:** seed-42 runs reproduce eval-1.1/-1.2/-1.3 to machine precision (|diff| ≤ 4e-7).

## Methods & caveats (as they appear in the paper)

**Statistical tests.** Seed-level inference (frozen split): mean ± std and 95% t-CI over the 30 seeds
quantify **fitting stochasticity only** — the paper must state this explicitly. Test-set sampling
variability is quantified by the paired cluster bootstrap over (station, month) blocks (252 blocks;
percentile 95% CI; two-sided bootstrap p; sensitivity with (station, year) blocks). Pairwise model
differences use paired t-tests and Wilcoxon signed-rank on the per-seed differences; p-values are
Benjamini–Hochberg FDR-corrected within the pre-specified comparison family, per metric. LOSO claims
use per-station win counts with the two-sided sign test (power: 7/7 → 0.016, 6/7 → 0.125) and paired
tests on the 7 per-station medians (n = 7, descriptive).

**Leakage.** (1) The per-regime delta features were historically selected on test-period residuals;
the test / val / no-delta ablation addresses this. (2) The (c0, c1) counts were historically chosen on
test; the val-selected protocol re-chooses them on validation (same 2500-tree hyperparameters). (3) The
54-feature backbone and V0-50 feature sets were selected targeting the test period
(`derived_8.4-feature-selection-2.0`) — accepted as a caveat: shared by all compared models, so
relative claims are less affected; not re-fixable under the frozen-split constraint. (4) XGBoost
hyperparameters stem from earlier test-era tuning (shared).

**Other caveats.** n = 7 stations for spatial claims (low power, one hard station drags the mean —
hence pair plots + win counts); 2025 test coverage is partial at several stations; seed variation does
not cover routing stochasticity (deliberately fixed to preserve config identity); the val-selected
winners were additionally verified to be identical under the corrected selection protocol
(`refresh_val_winners.py --check` printed unchanged after the fixed re-selection).

## Reproducibility

```bash
cd notebooks/experiment/derived_8.4-formal-eval-1.0
mkdir -p artifacts/slurm && sbatch run_slurm.sh        # full GPU run (val selection + temporal 30 seeds + LOSO 5 seeds)
# protocol-fix follow-up (if select_deltas_val.py changed): sbatch run_slurm_valrefresh.sh
# smoke (CPU, n_estimators=100, data_version=-1, never reused by the real run):
uv run python select_deltas_val.py --smoke
uv run python run_temporal.py --smoke --config-id <id> --seeds 42 7 --n-parallel 4
uv run python run_loso.py --smoke --config-id <id> --seeds 42 7 --max-stations 2 --n-parallel 4
uv run python -m eval_formal.stats          # statistical self-tests
# report (from notebooks/):
nb execute experiment/derived_8.4-formal-eval-1.0/derived_8.4-formal-eval-1.0.ipynb --uv
```

- Configurations are pinned in `pinned_configurations.json` before any run (audit trail).
- Per-seed artifacts use cache-safe naming `<config_id>__s<seed>__<station>` for weights, predictions
  and job meta; jobs resume via `artifacts/jobs/*/meta.json` (data_version + file-presence match).
- `--smoke` uses data_version=-1 so smoke artifacts are never reused by the real run.
- Seed-42 temporal rows reproduce eval-1.1 pooled test R² to |diff| < 1e-6; LOSO seed-42 rows reproduce
  eval-1.2/-1.3 loso_mean_r2 to |diff| < 1e-3 (printed by the drivers and the notebook).
- Storage note: per-seed model weights (~60–80 GB total) are gitignored and fully regenerable from the
  pinned configurations.
