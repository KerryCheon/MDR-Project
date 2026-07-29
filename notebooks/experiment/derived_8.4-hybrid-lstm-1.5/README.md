# Experiment: `derived_8.4-hybrid-lstm-1.5` — V21 LSTM + PCA + XGBoost with Accelerated SHAP

## Objective
Replace the V23-style LSTM with **V21 BiLSTM+Attn** (Jakob 38 + V9-unique union, seq_len=30,
1-seed training). After training, PCA is applied to all three frozen
representations (160-dim ctx, 80-dim head_hidden, 80-dim pre-ReLU) at three compression levels:
- **95% variance** explained (auto component count)
- **64 components** fixed
- **32 components** fixed

All 12 representation variants are compared against pure tabular baselines and the V21 LSTM-only baseline.

---

## Overall Leaderboard (2023–2025 Test Set)

| model_name                                                                |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:--------------------------------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| Global Single (54 Backbone)                                               |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.76674  |     0.049199  |       0.0451116 |    0.0196337  |    0.0384167 |         0.896609 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.758394 |     0.0500714 |       0.0457397 |    0.0203721  |    0.0391968 |         0.893844 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.756389 |     0.0502787 |       0.0460322 |    0.0202233  |    0.0390251 |         0.892335 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.756169 |     0.0503015 |       0.0463206 |    0.0196123  |    0.0395265 |         0.890928 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.754639 |     0.050459  |       0.0470608 |    0.0182042  |    0.0398345 |         0.887114 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.752316 |     0.0506973 |       0.0460347 |    0.0212373  |    0.0392756 |         0.892107 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.750475 |     0.0508853 |       0.0466199 |    0.0203937  |    0.0398876 |         0.889134 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.747833 |     0.051154  |       0.0451393 |    0.0240662  |    0.0405703 |         0.896524 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.744551 |     0.0514858 |       0.0455761 |    0.0239502  |    0.0407367 |         0.894331 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.744466 |     0.0514944 |       0.0473336 |    0.0202781  |    0.0403839 |         0.885504 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.743409 |     0.0516008 |       0.045793  |    0.0237831  |    0.0408188 |         0.893389 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.742166 |     0.0517257 |       0.0455116 |    0.0245814  |    0.0409449 |         0.894656 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.740423 |     0.0519002 |       0.0475768 |    0.0207385  |    0.0408463 |         0.884246 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.73649  |     0.0522918 |       0.0461596 |    0.0245709  |    0.0413415 |         0.891447 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.733858 |     0.0525524 |       0.0466602 |    0.0241782  |    0.0417662 |         0.889144 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.731762 |     0.0527589 |       0.0469179 |    0.0241291  |    0.0417006 |         0.887636 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.7288   |     0.0530494 |       0.0458039 |    0.0267626  |    0.0427938 |         0.893377 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.728371 |     0.0530914 |       0.047132  |    0.0244392  |    0.0423743 |         0.886529 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.725432 |     0.0533778 |       0.047205  |    0.0249174  |    0.0428831 |         0.886151 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.725145 |     0.0534057 |       0.0469201 |    0.0255083  |    0.0426833 |         0.887916 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.724291 |     0.0534886 |       0.045846  |    0.0275531  |    0.0432127 |         0.893349 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.721816 |     0.0537282 |       0.0481828 |    0.0237726  |    0.0433447 |         0.881065 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.720765 |     0.0538295 |       0.0475137 |    0.0252995  |    0.0431416 |         0.884695 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.717089 |     0.0541828 |       0.0469025 |    0.0271279  |    0.0434425 |         0.888319 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.706465 |     0.0551907 |       0.0490193 |    0.0253599  |    0.0428797 |         0.878369 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.702138 |     0.055596  |       0.0513944 |    0.0212023  |    0.0440101 |         0.863939 |
| Global Single (54 Backbone + 160 CTX)                                     |    0.697691 |     0.0560095 |       0.0489892 |    0.02715    |    0.044595  |         0.876936 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.695492 |     0.0562128 |       0.0479859 |    0.0292785  |    0.0446102 |         0.882157 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.69529  |     0.0562315 |       0.0491271 |    0.0273588  |    0.0440866 |         0.876931 |
| Global Single (54 Backbone + 80 Head Hidden)                              |    0.692525 |     0.056486  |       0.0484955 |    0.028963   |    0.0448483 |         0.879807 |
| BiLSTM+Attn (LSTM-only, V21)                                              |    0.582725 |     0.0658032 |       0.0524284 |    0.0397659  |    0.0524885 |       nan        |

---

## Per-Regime Performance Breakdown



---

## Year-by-Year $R^2$ Breakdown

| model_name                                                                |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:--------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| Global Single (54 Backbone)                                               |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.76674  |       0.752535 |       0.770393 |       0.771568 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.758394 |       0.751904 |       0.772422 |       0.743832 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.756389 |       0.743007 |       0.776377 |       0.743921 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.756169 |       0.749565 |       0.774584 |       0.737367 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.754639 |       0.745476 |       0.779542 |       0.732406 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.752316 |       0.728209 |       0.765136 |       0.759158 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.750475 |       0.721389 |       0.788435 |       0.738429 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.747833 |       0.721513 |       0.75876  |       0.758941 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.744551 |       0.721238 |       0.769362 |       0.73847  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.744466 |       0.714198 |       0.777206 |       0.73869  |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.743409 |       0.724384 |       0.751696 |       0.748487 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.742166 |       0.707089 |       0.776462 |       0.740364 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.740423 |       0.715437 |       0.77093  |       0.730565 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.73649  |       0.702879 |       0.784995 |       0.718889 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.733858 |       0.697919 |       0.765602 |       0.735267 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.731762 |       0.700631 |       0.747504 |       0.743181 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.7288   |       0.700158 |       0.760266 |       0.721844 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.728371 |       0.711074 |       0.740566 |       0.727098 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.725432 |       0.705907 |       0.742299 |       0.72207  |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.725145 |       0.686981 |       0.759965 |       0.725822 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.724291 |       0.686784 |       0.7577   |       0.725558 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.721816 |       0.691353 |       0.745259 |       0.72458  |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.720765 |       0.689968 |       0.748967 |       0.719224 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.717089 |       0.68481  |       0.762185 |       0.700604 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.706465 |       0.665956 |       0.741726 |       0.708785 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.702138 |       0.651923 |       0.755298 |       0.698048 |
| Global Single (54 Backbone + 160 CTX)                                     |    0.697691 |       0.637512 |       0.740217 |       0.715412 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.695492 |       0.631856 |       0.738593 |       0.716586 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.69529  |       0.636679 |       0.746754 |       0.702363 |
| Global Single (54 Backbone + 80 Head Hidden)                              |    0.692525 |       0.640708 |       0.721456 |       0.713673 |
| BiLSTM+Attn (LSTM-only, V21)                                              |    0.582725 |     nan        |     nan        |     nan        |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                                                |   pred_contribs (s) |
|:--------------------------------------------------------------------------|--------------------:|
| Global Single (54 Backbone)                                               |              1.1721 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |              1.0292 |
| Global Single (54 Backbone + 160 CTX)                                     |              1.2555 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |              1.1229 |
| Global Single (54 Backbone + 80 Head Hidden)                              |              1.1543 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |              1.1015 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |              1.2108 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |              1.1344 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |              1.0595 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |              1.0274 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |              1.2527 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |              1.0982 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |              1.1371 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |              1.0673 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |              1.111  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |              1.0156 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |              1.0462 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |              1.0372 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |              1.2931 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |              1.1442 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |              1.2189 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |              1.0743 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |              1.1228 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |              1.0593 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |              1.0537 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |              0.9994 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |              1.2564 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |              1.0952 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |              1.1885 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |              1.0695 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |              1.1268 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |              1.0243 |

### Feature Category Contribution (Tabular vs. LSTM Representations)

| Model Name                                                                |   Tabular SHAP Sum |   Tabular Mean abs(SHAP) |   Tabular Median abs(SHAP) |   Repr SHAP Sum |   Repr Mean abs(SHAP) |   Repr Median abs(SHAP) | Tabular % Share   | Repr % Share   |
|:--------------------------------------------------------------------------|-------------------:|-------------------------:|---------------------------:|----------------:|----------------------:|------------------------:|:------------------|:---------------|
| Global Single (54 Backbone)                                               |             0.1726 |                   0.0032 |                     0.0011 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |             0.1713 |                   0.0027 |                     0.0009 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Global Single (54 Backbone + 160 CTX)                                     |             0.0193 |                   0.0004 |                     0.0002 |          0.1331 |                0.0008 |                  0.0002 | 12.64%            | 87.36%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |             0.0247 |                   0.0004 |                     0.0002 |          0.127  |                0.0008 |                  0.0002 | 16.28%            | 83.72%         |
| Global Single (54 Backbone + 80 Head Hidden)                              |             0.0371 |                   0.0007 |                     0.0005 |          0.1097 |                0.0014 |                  0.0001 | 25.30%            | 74.70%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |             0.0401 |                   0.0006 |                     0.0004 |          0.1078 |                0.0013 |                  0.0002 | 27.14%            | 72.86%         |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |             0.0276 |                   0.0005 |                     0.0003 |          0.1196 |                0.0015 |                  0.0004 | 18.76%            | 81.24%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |             0.0313 |                   0.0005 |                     0.0003 |          0.1159 |                0.0014 |                  0.0004 | 21.28%            | 78.72%         |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |             0.0597 |                   0.0011 |                     0.0006 |          0.0741 |                0.0074 |                  0.0014 | 44.62%            | 55.38%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |             0.0656 |                   0.001  |                     0.0005 |          0.0715 |                0.0071 |                  0.0013 | 47.86%            | 52.14%         |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |             0.0461 |                   0.0009 |                     0.0004 |          0.0949 |                0.0015 |                  0.0004 | 32.70%            | 67.30%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |             0.0515 |                   0.0008 |                     0.0004 |          0.0928 |                0.0014 |                  0.0004 | 35.68%            | 64.32%         |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |             0.0515 |                   0.001  |                     0.0005 |          0.0861 |                0.0027 |                  0.0007 | 37.43%            | 62.57%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |             0.0574 |                   0.0009 |                     0.0005 |          0.0837 |                0.0026 |                  0.0006 | 40.67%            | 59.33%         |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |             0.0547 |                   0.001  |                     0.0005 |          0.0811 |                0.0051 |                  0.0012 | 40.25%            | 59.75%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |             0.0613 |                   0.001  |                     0.0005 |          0.0784 |                0.0049 |                  0.0011 | 43.86%            | 56.14%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |             0.0653 |                   0.0012 |                     0.0007 |          0.0648 |                0.0324 |                  0.0324 | 50.20%            | 49.80%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |             0.0713 |                   0.0011 |                     0.0006 |          0.0646 |                0.0323 |                  0.0323 | 52.48%            | 47.52%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |             0.0443 |                   0.0008 |                     0.0005 |          0.0941 |                0.0015 |                  0.0003 | 32.01%            | 67.99%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |             0.0511 |                   0.0008 |                     0.0004 |          0.0943 |                0.0015 |                  0.0003 | 35.16%            | 64.84%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |             0.0483 |                   0.0009 |                     0.0005 |          0.087  |                0.0027 |                  0.0006 | 35.71%            | 64.29%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |             0.0555 |                   0.0009 |                     0.0004 |          0.0876 |                0.0027 |                  0.0006 | 38.76%            | 61.24%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |             0.0504 |                   0.0009 |                     0.0005 |          0.0825 |                0.0052 |                  0.0013 | 37.92%            | 62.08%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |             0.0581 |                   0.0009 |                     0.0005 |          0.0837 |                0.0052 |                  0.0014 | 40.99%            | 59.01%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |             0.0692 |                   0.0013 |                     0.0008 |          0.0627 |                0.0314 |                  0.0314 | 52.44%            | 47.56%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |             0.0744 |                   0.0012 |                     0.0006 |          0.0608 |                0.0304 |                  0.0304 | 55.06%            | 44.94%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |             0.0453 |                   0.0008 |                     0.0004 |          0.0985 |                0.0015 |                  0.0004 | 31.48%            | 68.52%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |             0.0493 |                   0.0008 |                     0.0003 |          0.0966 |                0.0015 |                  0.0003 | 33.77%            | 66.23%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |             0.0507 |                   0.0009 |                     0.0005 |          0.0872 |                0.0027 |                  0.0007 | 36.77%            | 63.23%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |             0.056  |                   0.0009 |                     0.0004 |          0.0835 |                0.0026 |                  0.0006 | 40.13%            | 59.87%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |             0.0539 |                   0.001  |                     0.0005 |          0.0822 |                0.0051 |                  0.0013 | 39.59%            | 60.41%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |             0.0599 |                   0.0009 |                     0.0005 |          0.0779 |                0.0049 |                  0.0011 | 43.45%            | 56.55%         |

### Top 10 Features by Mean Absolute SHAP Value (averaged across all 32 XGBoost models)

| Feature                    |   Mean abs(SHAP) |
|:---------------------------|-----------------:|
| V_ema_LST_modis_kobs30     |           0.0047 |
| D_sin_DOY                  |           0.0045 |
| V_rollmin_LST_modis_kobs30 |           0.0041 |
| hp_pca16_0                 |           0.0038 |
| ctx_pca64_0                |           0.0038 |
| ctx_pca16_0                |           0.0038 |
| hp_pca64_0                 |           0.0038 |
| ctx_pca32_0                |           0.0038 |
| hp_pca32_0                 |           0.0038 |
| hh_pca64_0                 |           0.0037 |

---

## Key Insights & Architecture Summary
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.4` (Jakob 38 + V9-unique features, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen representations extracted:
  - `ctx` (160-dim): Attention-pooled hidden state
  - `head_hidden` (80-dim): After head `Linear(160→80)→ReLU`
  - `head_pre_relu` (80-dim): After head `Linear(160→80)` BEFORE ReLU
- **Phase 3**: PCA reduces each representation at 3 levels (95% var, 64 comps, 32 comps).
- **Phase 4**: XGBoost fit on `[Tabular + Repr]` for all 12 representation variants × 2 strategies (Global + Clustering).
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
