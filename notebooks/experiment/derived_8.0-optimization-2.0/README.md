# Experiment: `derived_8.0-optimization-2.0` — V21 LSTM + PCA + XGBoost with Accelerated SHAP

## Objective
Re-run of `derived_8.4-hybrid-lstm-1.5` on the **`derived_8.0`** dataset split (5 Washington
stations) with the exact same feature sets (54-feature shared backbone, 58-feature V21
LSTM input, V0 router features, and the c1 additions from `derived_8.4-eval-1.1`).
The architecture uses **V21 BiLSTM+Attn** (Jakob 38 + V9-unique union, seq_len=30,
1-seed training). After training, PCA is applied to all three frozen
representations (160-dim ctx, 80-dim head_hidden, 80-dim pre-ReLU) at four compression levels:
- **95% variance** explained (auto component count)
- **64 components** fixed
- **32 components** fixed
- **16 components** fixed

All 15 representation variants (3 raw + 12 PCA) are compared against pure tabular baselines and the V21 LSTM-only baseline.

---

## Overall Leaderboard (2023–2025 Test Set)

| model_name                                                                |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:--------------------------------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.83409  |     0.0383548 |       0.0376882 |   -0.00711954 |    0.0275703 |         0.916474 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.833799 |     0.0383885 |       0.0377542 |   -0.00694961 |    0.0276472 |         0.916131 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.833646 |     0.0384061 |       0.037732  |   -0.00716399 |    0.0274058 |         0.916291 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.833598 |     0.0384117 |       0.0377869 |   -0.00690009 |    0.0274041 |         0.915971 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.833067 |     0.0384728 |       0.0378948 |   -0.0066439  |    0.0274573 |         0.915472 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.833053 |     0.0384745 |       0.0379066 |   -0.00658615 |    0.0272964 |         0.915539 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.833038 |     0.0384763 |       0.037908  |   -0.00658858 |    0.0274093 |         0.915515 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.83297  |     0.038484  |       0.0377733 |   -0.0073617  |    0.0276482 |         0.916125 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.832511 |     0.038537  |       0.0379223 |   -0.00685519 |    0.027515  |         0.915406 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.8323   |     0.0385612 |       0.037841  |   -0.00741799 |    0.0275681 |         0.915813 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.831019 |     0.0387081 |       0.0379832 |   -0.00745611 |    0.027547  |         0.915182 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.830686 |     0.0387463 |       0.0379658 |   -0.00773801 |    0.0278827 |         0.915118 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.828874 |     0.0389531 |       0.0382297 |   -0.0074721  |    0.0278481 |         0.913998 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.828678 |     0.0389754 |       0.0383353 |   -0.00703469 |    0.0277994 |         0.913563 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.828373 |     0.0390101 |       0.0382917 |   -0.00745209 |    0.0277895 |         0.913733 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.828231 |     0.0390262 |       0.0383015 |   -0.00748612 |    0.0278071 |         0.913687 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.820131 |     0.0399357 |       0.0391362 |   -0.00795105 |    0.0291141 |         0.909555 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.819911 |     0.0399602 |       0.0391949 |   -0.00778301 |    0.0291127 |         0.909293 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.819411 |     0.0400156 |       0.0392466 |   -0.00780698 |    0.0292286 |         0.909033 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.819324 |     0.0400252 |       0.0393199 |   -0.00748111 |    0.0290357 |         0.908759 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.819201 |     0.0400389 |       0.0391596 |   -0.00834528 |    0.0292386 |         0.909427 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.819023 |     0.0400585 |       0.039289  |   -0.00781418 |    0.029068  |         0.908871 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.818818 |     0.0400813 |       0.0393157 |   -0.00779665 |    0.0291878 |         0.90868  |
| Global Single (54 Backbone + 160 CTX)                                     |    0.817904 |     0.0401823 |       0.0394463 |   -0.00765533 |    0.0291489 |         0.908034 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.817885 |     0.0401843 |       0.0394424 |   -0.00768581 |    0.0292697 |         0.908097 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.817787 |     0.0401951 |       0.0394645 |   -0.00762899 |    0.0291896 |         0.908057 |
| Global Single (54 Backbone + 80 Head Hidden)                              |    0.816229 |     0.0403666 |       0.0396719 |   -0.00745693 |    0.0293345 |         0.906978 |
| Global Single (54 Backbone)                                               |    0.815646 |     0.0404306 |       0.0396221 |   -0.00804493 |    0.0294862 |         0.907223 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.815578 |     0.040438  |       0.0396934 |   -0.00772418 |    0.0291968 |         0.906866 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.815143 |     0.0404857 |       0.0396963 |   -0.00795612 |    0.0294253 |         0.906864 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.815041 |     0.0404969 |       0.0397559 |   -0.00771161 |    0.0294281 |         0.906581 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.814576 |     0.0405478 |       0.0397419 |   -0.00804375 |    0.0295106 |         0.906631 |
| BiLSTM+Attn (LSTM-only, V21)                                              |    0.721683 |     0.0496768 |       0.0468745 |    0.0164488  |    0.0372314 |       nan        |

---

## Per-Regime Performance Breakdown



---

## Year-by-Year $R^2$ Breakdown

| model_name                                                                |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:--------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.83409  |       0.832038 |       0.819329 |       0.847502 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.833799 |       0.830694 |       0.82044  |       0.846812 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.833646 |       0.831198 |       0.821332 |       0.845016 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.833598 |       0.830307 |       0.816659 |       0.850298 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.833067 |       0.832341 |       0.823167 |       0.840443 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.833053 |       0.828796 |       0.821373 |       0.845482 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.833038 |       0.831386 |       0.818222 |       0.846108 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.83297  |       0.829821 |       0.81733  |       0.848252 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.832511 |       0.830245 |       0.819326 |       0.844544 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.8323   |       0.829938 |       0.815576 |       0.847896 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.831019 |       0.826506 |       0.811797 |       0.851062 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.830686 |       0.831405 |       0.817956 |       0.839443 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.828874 |       0.827077 |       0.812978 |       0.843065 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.828678 |       0.828168 |       0.810284 |       0.844115 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.828373 |       0.828003 |       0.811576 |       0.842103 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.828231 |       0.827432 |       0.813469 |       0.840362 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.820131 |       0.817571 |       0.800704 |       0.838348 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.819911 |       0.818873 |       0.799916 |       0.837254 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.819411 |       0.818088 |       0.800274 |       0.836172 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.819324 |       0.812028 |       0.808    |       0.834001 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.819201 |       0.817071 |       0.80328  |       0.833552 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.819023 |       0.810069 |       0.809326 |       0.83365  |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.818818 |       0.816943 |       0.802467 |       0.833345 |
| Global Single (54 Backbone + 160 CTX)                                     |    0.817904 |       0.812864 |       0.802586 |       0.834365 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.817885 |       0.818532 |       0.794882 |       0.836571 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.817787 |       0.809931 |       0.808054 |       0.831397 |
| Global Single (54 Backbone + 80 Head Hidden)                              |    0.816229 |       0.820858 |       0.791696 |       0.832657 |
| Global Single (54 Backbone)                                               |    0.815646 |       0.821875 |       0.790157 |       0.831505 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.815578 |       0.813217 |       0.795263 |       0.834399 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.815143 |       0.820715 |       0.789899 |       0.831367 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.815041 |       0.810708 |       0.809174 |       0.821504 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.814576 |       0.820923 |       0.788732 |       0.830653 |
| BiLSTM+Attn (LSTM-only, V21)                                              |    0.721683 |     nan        |     nan        |     nan        |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                                                |   pred_contribs (s) |
|:--------------------------------------------------------------------------|--------------------:|
| Global Single (54 Backbone)                                               |              0.6938 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |              0.6419 |
| Global Single (54 Backbone + 160 CTX)                                     |              0.7947 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |              0.7038 |
| Global Single (54 Backbone + 80 Head Hidden)                              |              0.7324 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |              0.6603 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |              0.7917 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |              0.6901 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |              0.7429 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |              0.6668 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |              0.8633 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |              0.7418 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |              0.8191 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |              0.7047 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |              0.7871 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |              0.7021 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |              0.7506 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |              0.6672 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |              0.8812 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |              0.7473 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |              0.8246 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |              0.7152 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |              0.813  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |              0.7086 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |              0.7198 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |              0.6557 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |              0.8915 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |              0.7577 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |              0.8272 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |              0.735  |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |              0.8117 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |              0.7061 |

### Feature Category Contribution (Tabular vs. LSTM Representations)

| Model Name                                                                |   Tabular SHAP Sum |   Tabular Mean abs(SHAP) |   Tabular Median abs(SHAP) |   Repr SHAP Sum |   Repr Mean abs(SHAP) |   Repr Median abs(SHAP) | Tabular % Share   | Repr % Share   |
|:--------------------------------------------------------------------------|-------------------:|-------------------------:|---------------------------:|----------------:|----------------------:|------------------------:|:------------------|:---------------|
| Global Single (54 Backbone)                                               |             0.1503 |                   0.0028 |                     0.0011 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |             0.1448 |                   0.0023 |                     0.0008 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Global Single (54 Backbone + 160 CTX)                                     |             0.1469 |                   0.0027 |                     0.0011 |          0.0099 |                0.0001 |                  0      | 93.68%            | 6.32%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |             0.14   |                   0.0022 |                     0.0007 |          0.0096 |                0.0001 |                  0      | 93.59%            | 6.41%          |
| Global Single (54 Backbone + 80 Head Hidden)                              |             0.1501 |                   0.0028 |                     0.0011 |          0.0028 |                0      |                  0      | 98.17%            | 1.83%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |             0.1441 |                   0.0023 |                     0.0008 |          0.0029 |                0      |                  0      | 98.03%            | 1.97%          |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |             0.1493 |                   0.0028 |                     0.0011 |          0.0049 |                0.0001 |                  0      | 96.81%            | 3.19%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |             0.1428 |                   0.0022 |                     0.0008 |          0.0051 |                0.0001 |                  0      | 96.54%            | 3.46%          |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |             0.1488 |                   0.0028 |                     0.0011 |          0.0037 |                0.0004 |                  0.0003 | 97.56%            | 2.44%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |             0.1426 |                   0.0022 |                     0.0008 |          0.0038 |                0.0004 |                  0.0004 | 97.38%            | 2.62%          |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |             0.1464 |                   0.0027 |                     0.0011 |          0.0098 |                0.0002 |                  0.0001 | 93.71%            | 6.29%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |             0.1393 |                   0.0022 |                     0.0007 |          0.01   |                0.0002 |                  0.0001 | 93.30%            | 6.70%          |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |             0.1474 |                   0.0027 |                     0.0011 |          0.0075 |                0.0002 |                  0.0002 | 95.13%            | 4.87%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |             0.1403 |                   0.0022 |                     0.0007 |          0.0077 |                0.0002 |                  0.0002 | 94.77%            | 5.23%          |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |             0.1484 |                   0.0027 |                     0.0011 |          0.0055 |                0.0003 |                  0.0002 | 96.42%            | 3.58%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |             0.1414 |                   0.0022 |                     0.0007 |          0.0056 |                0.0003 |                  0.0002 | 96.21%            | 3.79%          |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |             0.15   |                   0.0028 |                     0.0011 |          0.0005 |                0.0002 |                  0.0002 | 99.69%            | 0.31%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |             0.1444 |                   0.0023 |                     0.0008 |          0.0005 |                0.0003 |                  0.0003 | 99.63%            | 0.37%          |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |             0.1482 |                   0.0027 |                     0.0011 |          0.007  |                0.0001 |                  0.0001 | 95.49%            | 4.51%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |             0.1411 |                   0.0022 |                     0.0008 |          0.0075 |                0.0001 |                  0.0001 | 94.96%            | 5.04%          |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |             0.1487 |                   0.0028 |                     0.0011 |          0.0051 |                0.0002 |                  0.0001 | 96.70%            | 3.30%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |             0.1418 |                   0.0022 |                     0.0008 |          0.0055 |                0.0002 |                  0.0002 | 96.26%            | 3.74%          |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |             0.1499 |                   0.0028 |                     0.0011 |          0.0036 |                0.0002 |                  0.0002 | 97.64%            | 2.36%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |             0.1428 |                   0.0022 |                     0.0008 |          0.0039 |                0.0002 |                  0.0002 | 97.36%            | 2.64%          |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |             0.1502 |                   0.0028 |                     0.0011 |          0.0003 |                0.0003 |                  0.0003 | 99.79%            | 0.21%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |             0.1448 |                   0.0023 |                     0.0008 |          0.0004 |                0.0004 |                  0.0004 | 99.71%            | 0.29%          |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |             0.1461 |                   0.0027 |                     0.0011 |          0.0091 |                0.0001 |                  0.0001 | 94.11%            | 5.89%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |             0.1392 |                   0.0022 |                     0.0007 |          0.0091 |                0.0001 |                  0.0001 | 93.88%            | 6.12%          |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |             0.1474 |                   0.0027 |                     0.0011 |          0.0066 |                0.0002 |                  0.0002 | 95.72%            | 4.28%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |             0.1411 |                   0.0022 |                     0.0008 |          0.0064 |                0.0002 |                  0.0002 | 95.64%            | 4.36%          |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |             0.1485 |                   0.0028 |                     0.0011 |          0.0047 |                0.0003 |                  0.0002 | 96.91%            | 3.09%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |             0.142  |                   0.0022 |                     0.0008 |          0.0046 |                0.0003 |                  0.0003 | 96.87%            | 3.13%          |

### Top 10 Features by Mean Absolute SHAP Value (averaged across all 32 XGBoost models)

| Feature                         |   Mean abs(SHAP) |
|:--------------------------------|-----------------:|
| V_rollmin_LST_modis_kobs30      |           0.02   |
| SMAP_sm_pm_interp_rollmean30    |           0.0165 |
| V_ema_LST_modis_kobs30          |           0.0158 |
| G_API                           |           0.0144 |
| SMAP_x_year                     |           0.0082 |
| D_sin_DOY                       |           0.0066 |
| SMAP_sm_pm_interp_lag7          |           0.0058 |
| G_rain_sum_3d                   |           0.0037 |
| V_rollmin_SMAP_sm_interp_kobs30 |           0.0036 |
| SMAP_sm_pm_interp               |           0.0035 |

---

## Key Insights & Architecture Summary
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.0` (Jakob 38 + V9-unique features, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen representations extracted:
  - `ctx` (160-dim): Attention-pooled hidden state
  - `head_hidden` (80-dim): After head `Linear(160→80)→ReLU`
  - `head_pre_relu` (80-dim): After head `Linear(160→80)` BEFORE ReLU
- **Phase 3**: PCA reduces each representation at 3 levels (95% var, 64 comps, 32 comps).
- **Phase 4**: XGBoost fit on `[Tabular + Repr]` for all 12 representation variants × 2 strategies (Global + Clustering).
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
