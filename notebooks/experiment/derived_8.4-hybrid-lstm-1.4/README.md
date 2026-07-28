# Experiment: `derived_8.4-hybrid-lstm-1.4` — V23 LSTM + PCA + XGBoost with Accelerated SHAP

## Objective
Replace the V9-style LSTM with **V23 BiLSTM+Attn** (top25 curated features, seq_len=30, ReduceLROnPlateau,
5-seed training with best-seed selection). After training, PCA is applied to all three frozen
representations (160-dim ctx, 80-dim head_hidden, 80-dim pre-ReLU) at three compression levels:
- **95% variance** explained (auto component count)
- **64 components** fixed
- **32 components** fixed

All 12 representation variants are compared against pure tabular baselines and the V23 LSTM-only baseline.

---

## Overall Leaderboard (2023–2025 Test Set)

| model_name                                                                |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:--------------------------------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| Global Single (54 Backbone)                                               |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.741767 |     0.0517656 |       0.0511204 |    0.00814759 |    0.0393047 |         0.870067 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.725419 |     0.053379  |       0.0520726 |    0.0117372  |    0.040849  |         0.863756 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.724975 |     0.0534222 |       0.0527432 |    0.00849068 |    0.0402359 |         0.862282 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.722743 |     0.0536385 |       0.0528343 |    0.00925335 |    0.0414425 |         0.859031 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.721489 |     0.0537598 |       0.0527375 |    0.0104341  |    0.0414311 |         0.85849  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.721304 |     0.0537776 |       0.0529247 |    0.0095396  |    0.041027  |         0.860811 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.716809 |     0.0542095 |       0.0526796 |    0.0127882  |    0.0410511 |         0.861363 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.71568  |     0.0543175 |       0.0534154 |    0.00985835 |    0.0411808 |         0.858505 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.715044 |     0.0543782 |       0.0536328 |    0.00897248 |    0.0416052 |         0.857035 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.713035 |     0.0545695 |       0.053423  |    0.0111275  |    0.0420904 |         0.855215 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.711516 |     0.0547138 |       0.0536009 |    0.010979   |    0.0424074 |         0.854282 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.706817 |     0.0551576 |       0.0541771 |    0.0103536  |    0.0418552 |         0.853603 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.705791 |     0.055254  |       0.0538163 |    0.0125225  |    0.0430952 |         0.852679 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.703438 |     0.0554745 |       0.0545899 |    0.00986743 |    0.0421867 |         0.851307 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.703122 |     0.0555041 |       0.0537782 |    0.0137333  |    0.0424266 |         0.85459  |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.701069 |     0.0556956 |       0.0549009 |    0.00937516 |    0.0418713 |         0.850903 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.699387 |     0.0558522 |       0.0543036 |    0.0130605  |    0.0429068 |         0.850959 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.698593 |     0.0559258 |       0.055198  |    0.00899303 |    0.0417302 |         0.84949  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.697502 |     0.056027  |       0.0553419 |    0.0087348  |    0.0421864 |         0.84788  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.696997 |     0.0560737 |       0.0553647 |    0.00888897 |    0.0428702 |         0.847252 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.696687 |     0.0561024 |       0.0543894 |    0.0137573  |    0.0423931 |         0.852433 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.682194 |     0.0574271 |       0.0567722 |    0.00864843 |    0.043257  |         0.840424 |
| BiLSTM+Attn (LSTM-only, V23)                                              |    0.67849  |     0.0577608 |       0.0545177 |    0.0190822  |    0.0434376 |       nan        |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.675667 |     0.0580138 |       0.0570815 |    0.0103589  |    0.0427192 |         0.843326 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.674332 |     0.0581331 |       0.0573313 |    0.00962147 |    0.0429634 |         0.838884 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.672665 |     0.0582817 |       0.0573051 |    0.0106246  |    0.0436564 |         0.840811 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.670402 |     0.0584828 |       0.057677  |    0.00967504 |    0.0436326 |         0.836513 |
| Global Single (54 Backbone + 160 CTX)                                     |    0.669249 |     0.058585  |       0.058066  |    0.00778078 |    0.0440823 |         0.833486 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.666466 |     0.058831  |       0.0579581 |    0.0100963  |    0.0447643 |         0.834388 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.665966 |     0.0588751 |       0.0574683 |    0.0127934  |    0.0441643 |         0.84099  |
| Global Single (54 Backbone + 80 Head Hidden)                              |    0.664386 |     0.0590141 |       0.0578522 |    0.011653   |    0.0435638 |         0.839574 |

---

## Per-Regime Performance Breakdown

| model_name                                                                |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |        bias |       mae |   pearson |
|:--------------------------------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|------------:|----------:|----------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 | 0.00861491  | 0.0359221 |  0.900537 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797068 | 0.0278349 |  0.9191   |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |         0 |     10624 |     4817 | 0.669563 | 0.0575074 | 0.0569604 | 0.00791238  | 0.0437729 |  0.837841 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |         1 |      3984 |     1803 | 0.711263 | 0.0572122 | 0.0562188 | 0.0106149   | 0.0418784 |  0.850607 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |         0 |     10624 |     4817 | 0.657878 | 0.0585154 | 0.0575139 | 0.0107797   | 0.0435353 |  0.840354 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |         1 |      3984 |     1803 | 0.716889 | 0.056652  | 0.0558942 | 0.00923473  | 0.0405391 |  0.854228 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |         0 |     10624 |     4817 | 0.653964 | 0.0588491 | 0.0577732 | 0.0112013   | 0.0447809 |  0.836726 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |         1 |      3984 |     1803 | 0.716028 | 0.0567381 | 0.0560061 | 0.009084    | 0.0406521 |  0.853719 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |         0 |     10624 |     4817 | 0.708595 | 0.0540043 | 0.0530849 | 0.00992244  | 0.0415585 |  0.857456 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |         1 |      3984 |     1803 | 0.750648 | 0.0531671 | 0.0524806 | 0.00851677  | 0.0396071 |  0.871013 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |         0 |     10624 |     4817 | 0.697025 | 0.0550659 | 0.0537894 | 0.0117877   | 0.0433582 |  0.849088 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |         1 |      3984 |     1803 | 0.745039 | 0.0537618 | 0.0530336 | 0.00881849  | 0.0398671 |  0.867791 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |         0 |     10624 |     4817 | 0.712604 | 0.0536314 | 0.0527772 | 0.0095343   | 0.0420763 |  0.855337 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |         1 |      3984 |     1803 | 0.746027 | 0.0536575 | 0.0529795 | 0.00850276  | 0.0397494 |  0.868473 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |         0 |     10624 |     4817 | 0.700225 | 0.0547743 | 0.0540098 | 0.0091195   | 0.0423561 |  0.852457 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |         1 |      3984 |     1803 | 0.74935  | 0.0533053 | 0.0526103 | 0.00857968  | 0.0395989 |  0.870346 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |         0 |     10624 |     4817 | 0.718876 | 0.053043  | 0.0524024 | 0.00821866  | 0.040518  |  0.860767 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |         1 |      3984 |     1803 | 0.738736 | 0.0544223 | 0.0536361 | 0.00921744  | 0.039482  |  0.866586 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |         0 |     10624 |     4817 | 0.682439 | 0.0563759 | 0.0557219 | 0.00856224  | 0.0437501 |  0.842012 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |         1 |      3984 |     1803 | 0.730647 | 0.0552583 | 0.0543892 | 0.00976188  | 0.0405192 |  0.861519 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |         0 |     10624 |     4817 | 0.683327 | 0.056297  | 0.0556979 | 0.00819086  | 0.0428608 |  0.84266  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |         1 |      3984 |     1803 | 0.730249 | 0.0552992 | 0.0543526 | 0.010188    | 0.0403847 |  0.86208  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |         0 |     10624 |     4817 | 0.68473  | 0.0561721 | 0.0554755 | 0.00881893  | 0.0423059 |  0.845192 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |         1 |      3984 |     1803 | 0.730608 | 0.0552624 | 0.054447  | 0.00945816  | 0.0401919 |  0.861633 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |         0 |     10624 |     4817 | 0.736203 | 0.0513824 | 0.0507888 | 0.00778753  | 0.0397599 |  0.868238 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |         1 |      3984 |     1803 | 0.754305 | 0.0527758 | 0.0519837 | 0.00910958  | 0.0380883 |  0.87488  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |         0 |     10624 |     4817 | 0.712511 | 0.0536401 | 0.0523969 | 0.0114818   | 0.0416242 |  0.856974 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |         1 |      3984 |     1803 | 0.74203  | 0.0540781 | 0.0535364 | 0.00763507  | 0.040915  |  0.864462 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |         0 |     10624 |     4817 | 0.718758 | 0.0530542 | 0.0516575 | 0.0120934   | 0.0411684 |  0.863095 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |         1 |      3984 |     1803 | 0.740508 | 0.0542374 | 0.0531542 | 0.0107856   | 0.0399958 |  0.867007 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |         0 |     10624 |     4817 | 0.711099 | 0.0537717 | 0.051997  | 0.0137007   | 0.0411415 |  0.862281 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |         1 |      3984 |     1803 | 0.729633 | 0.0553623 | 0.0543862 | 0.0103503   | 0.0408096 |  0.860823 |

---

## Year-by-Year $R^2$ Breakdown

| model_name                                                                |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:--------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| Global Single (54 Backbone)                                               |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.741767 |       0.745253 |       0.703106 |       0.76662  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.725419 |       0.706374 |       0.690682 |       0.772001 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.724975 |       0.721282 |       0.687684 |       0.756233 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.722743 |       0.715416 |       0.686449 |       0.757164 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.721489 |       0.705211 |       0.692765 |       0.758838 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.721304 |       0.717475 |       0.677991 |       0.758487 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.716809 |       0.697332 |       0.674631 |       0.770874 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.71568  |       0.714526 |       0.677866 |       0.744183 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.715044 |       0.711843 |       0.67236  |       0.750666 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.713035 |       0.689628 |       0.683283 |       0.759368 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.711516 |       0.7099   |       0.675811 |       0.738347 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.706817 |       0.681572 |       0.676144 |       0.755967 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.705791 |       0.697914 |       0.668235 |       0.741497 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.703438 |       0.680626 |       0.672016 |       0.750382 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.703122 |       0.672289 |       0.683451 |       0.747859 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.701069 |       0.692585 |       0.674293 |       0.726767 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.699387 |       0.680525 |       0.681638 |       0.728228 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.698593 |       0.681246 |       0.672943 |       0.733384 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.697502 |       0.679215 |       0.676527 |       0.728769 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.696997 |       0.687404 |       0.665283 |       0.728673 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.696687 |       0.668172 |       0.676748 |       0.738776 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.682194 |       0.653377 |       0.669671 |       0.716873 |
| BiLSTM+Attn (LSTM-only, V23)                                              |    0.67849  |     nan        |     nan        |     nan        |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.675667 |       0.672526 |       0.609332 |       0.733001 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.674332 |       0.649686 |       0.649578 |       0.715872 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.672665 |       0.657177 |       0.60995  |       0.740671 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.670402 |       0.645615 |       0.654585 |       0.703223 |
| Global Single (54 Backbone + 160 CTX)                                     |    0.669249 |       0.642496 |       0.654687 |       0.703083 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.666466 |       0.646572 |       0.643243 |       0.700722 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.665966 |       0.640664 |       0.616374 |       0.732281 |
| Global Single (54 Backbone + 80 Head Hidden)                              |    0.664386 |       0.64569  |       0.617073 |       0.720755 |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                                                |   pred_contribs (s) |
|:--------------------------------------------------------------------------|--------------------:|
| Global Single (54 Backbone)                                               |              1.4276 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |              1.0642 |
| Global Single (54 Backbone + 160 CTX)                                     |              1.2104 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |              1.1004 |
| Global Single (54 Backbone + 80 Head Hidden)                              |              1.0711 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |              1.0434 |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |              1.0586 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |              1.0553 |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |              1.0106 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |              0.9624 |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |              1.1392 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |              1.0411 |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |              1.0638 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |              1.0044 |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |              1.0236 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |              0.9638 |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |              1.0105 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |              0.9586 |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |              1.1603 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |              1.0806 |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |              1.0956 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |              1.0304 |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |              1.0277 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |              1.0089 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |              1.0056 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |              0.967  |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |              1.1757 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |              1.0596 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |              1.1048 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |              1.0334 |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |              1.0325 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |              0.9893 |

### Feature Category Contribution (Tabular vs. LSTM Representations)

| Model Name                                                                |   Tabular SHAP Sum |   Tabular Mean abs(SHAP) |   Tabular Median abs(SHAP) |   Repr SHAP Sum |   Repr Mean abs(SHAP) |   Repr Median abs(SHAP) | Tabular % Share   | Repr % Share   |
|:--------------------------------------------------------------------------|-------------------:|-------------------------:|---------------------------:|----------------:|----------------------:|------------------------:|:------------------|:---------------|
| Global Single (54 Backbone)                                               |             0.1726 |                   0.0032 |                     0.0011 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |             0.1713 |                   0.0027 |                     0.0009 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Global Single (54 Backbone + 160 CTX)                                     |             0.0147 |                   0.0003 |                     0.0001 |          0.1355 |                0.0008 |                  0.0002 | 9.77%             | 90.23%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |             0.0142 |                   0.0002 |                     0.0001 |          0.1386 |                0.0009 |                  0.0003 | 9.28%             | 90.72%         |
| Global Single (54 Backbone + 80 Head Hidden)                              |             0.0289 |                   0.0005 |                     0.0003 |          0.1255 |                0.0016 |                  0.0004 | 18.74%            | 81.26%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |             0.0296 |                   0.0005 |                     0.0003 |          0.1231 |                0.0015 |                  0.0005 | 19.40%            | 80.60%         |
| Global Single (54 Backbone + 80 Pre-ReLU)                                 |             0.0239 |                   0.0004 |                     0.0003 |          0.1337 |                0.0017 |                  0.0005 | 15.15%            | 84.85%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |             0.024  |                   0.0004 |                     0.0003 |          0.1298 |                0.0016 |                  0.0005 | 15.61%            | 84.39%         |
| Global Single (54 Backbone + 160 CTX PCA-95%)                             |             0.0429 |                   0.0008 |                     0.0004 |          0.0829 |                0.0059 |                  0.0013 | 34.08%            | 65.92%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |             0.0443 |                   0.0007 |                     0.0004 |          0.0819 |                0.0058 |                  0.0015 | 35.10%            | 64.90%         |
| Global Single (54 Backbone + 160 CTX PCA-64)                              |             0.0326 |                   0.0006 |                     0.0002 |          0.098  |                0.0015 |                  0.0004 | 24.98%            | 75.02%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |             0.0334 |                   0.0005 |                     0.0002 |          0.097  |                0.0015 |                  0.0004 | 25.63%            | 74.37%         |
| Global Single (54 Backbone + 160 CTX PCA-32)                              |             0.0381 |                   0.0007 |                     0.0003 |          0.089  |                0.0028 |                  0.0007 | 29.98%            | 70.02%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |             0.04   |                   0.0006 |                     0.0003 |          0.0876 |                0.0027 |                  0.0006 | 31.34%            | 68.66%         |
| Global Single (54 Backbone + 160 CTX PCA-16)                              |             0.0422 |                   0.0008 |                     0.0004 |          0.084  |                0.0052 |                  0.001  | 33.43%            | 66.57%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |             0.0435 |                   0.0007 |                     0.0003 |          0.0834 |                0.0052 |                  0.0013 | 34.28%            | 65.72%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |             0.0556 |                   0.001  |                     0.0005 |          0.0729 |                0.0365 |                  0.0365 | 43.29%            | 56.71%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |             0.0548 |                   0.0009 |                     0.0005 |          0.0749 |                0.0375 |                  0.0375 | 42.23%            | 57.77%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |             0.0338 |                   0.0006 |                     0.0003 |          0.1011 |                0.0016 |                  0.0003 | 25.04%            | 74.96%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |             0.0349 |                   0.0005 |                     0.0003 |          0.1019 |                0.0016 |                  0.0003 | 25.50%            | 74.50%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |             0.0384 |                   0.0007 |                     0.0004 |          0.0937 |                0.0029 |                  0.0004 | 29.08%            | 70.92%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |             0.0393 |                   0.0006 |                     0.0003 |          0.0935 |                0.0029 |                  0.0004 | 29.59%            | 70.41%         |
| Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |             0.0428 |                   0.0008 |                     0.0004 |          0.0875 |                0.0055 |                  0.0006 | 32.82%            | 67.18%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |             0.0436 |                   0.0007 |                     0.0004 |          0.0872 |                0.0054 |                  0.0006 | 33.35%            | 66.65%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |             0.0627 |                   0.0012 |                     0.0006 |          0.0671 |                0.0671 |                  0.0671 | 48.32%            | 51.68%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |             0.0635 |                   0.001  |                     0.0005 |          0.0659 |                0.0659 |                  0.0659 | 49.09%            | 50.91%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |             0.036  |                   0.0007 |                     0.0002 |          0.1004 |                0.0016 |                  0.0004 | 26.41%            | 73.59%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |             0.0378 |                   0.0006 |                     0.0002 |          0.0976 |                0.0015 |                  0.0003 | 27.94%            | 72.06%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |             0.0411 |                   0.0008 |                     0.0003 |          0.093  |                0.0029 |                  0.0006 | 30.67%            | 69.33%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |             0.043  |                   0.0007 |                     0.0003 |          0.0903 |                0.0028 |                  0.0007 | 32.29%            | 67.71%         |
| Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |             0.0448 |                   0.0008 |                     0.0004 |          0.0876 |                0.0055 |                  0.0011 | 33.82%            | 66.18%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |             0.0468 |                   0.0007 |                     0.0003 |          0.0846 |                0.0053 |                  0.001  | 35.63%            | 64.37%         |

### Top 10 Features by Mean Absolute SHAP Value (averaged across all 32 XGBoost models)

| Feature                |   Mean abs(SHAP) |
|:-----------------------|-----------------:|
| D_sin_DOY              |           0.0059 |
| V_ema_LST_modis_kobs30 |           0.0043 |
| hp_pca64_0             |           0.0042 |
| hp_pca16_0             |           0.0042 |
| hh_pca64_0             |           0.0042 |
| hh_pca16_0             |           0.0042 |
| hp_pca95_0             |           0.0042 |
| hh_pca32_0             |           0.0042 |
| hp_pca32_0             |           0.0041 |
| hh_pca95_0             |           0.0041 |

---

## Key Insights & Architecture Summary
- **Phase 1**: V23 BiLSTM+Attn trained from scratch on `derived_8.4` (top25 features, seq_len=30, ReduceLROnPlateau, 5 seeds, best by val RMSE).
- **Phase 2**: Three frozen representations extracted:
  - `ctx` (160-dim): Attention-pooled hidden state
  - `head_hidden` (80-dim): After head `Linear(160→80)→ReLU`
  - `head_pre_relu` (80-dim): After head `Linear(160→80)` BEFORE ReLU
- **Phase 3**: PCA reduces each representation at 3 levels (95% var, 64 comps, 32 comps).
- **Phase 4**: XGBoost fit on `[Tabular + Repr]` for all 12 representation variants × 2 strategies (Global + Clustering).
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
