# Experiment: `derived_8.4-hybrid-lstm-1.6` — V21 BiLSTM+Attn Hidden-Size Sweep (no PCA)

## Objective
In `derived_8.4-hybrid-lstm-1.5` (H=80), the raw representations (160-dim ctx,
80-dim head_hidden, 80-dim pre-ReLU) diluted the 54 tabular features in the hybrid
XGBoost models, and PCA was required to restore the tabular share. This experiment
tests whether a **smaller LSTM hidden size** produces a compact raw representation
that no longer needs PCA. We sweep `hidden_size ∈ {40, 20, 16, 8, 4}` (1 seed each,
V21 BiLSTM+Attn, seq_len=30) and evaluate the hybrid models on **raw (non-PCA)**
representations. Reference rows from `derived_8.4-hybrid-lstm-1.5` (H=80 ± PCA) are
included in the leaderboard (marked `[1.5]`) for direct comparison.

Per hidden size H the representations are:
- `ctx` (2H-dim): attention-pooled bidirectional hidden state
- `head_hidden` (H-dim): after head `Linear(2H→H)→ReLU`
- `head_pre_relu` (H-dim): after head `Linear(2H→H)` BEFORE ReLU

---

## Overall Leaderboard (2023–2025 Test Set)

| model_name                                                                      |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:--------------------------------------------------------------------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                                |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| [1.5] Global Single (54 Backbone)                                               |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| Global Single (54 Backbone)                                                     |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.76674  |     0.049199  |       0.0451116 |    0.0196337  |    0.0384167 |         0.896609 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.758394 |     0.0500714 |       0.0457397 |    0.0203721  |    0.0391968 |         0.893844 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.756389 |     0.0502787 |       0.0460322 |    0.0202233  |    0.0390251 |         0.892335 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.756169 |     0.0503015 |       0.0463206 |    0.0196123  |    0.0395265 |         0.890928 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])                 |    0.754996 |     0.0504222 |       0.0479503 |    0.0155939  |    0.0389788 |         0.884291 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.754639 |     0.050459  |       0.0470608 |    0.0182042  |    0.0398345 |         0.887114 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20])         |    0.752932 |     0.0506342 |       0.0478914 |    0.0164388  |    0.0389331 |         0.882679 |
| Global Single (54 Backbone + 40 CTX [H20])                                      |    0.752359 |     0.0506929 |       0.0473342 |    0.0181452  |    0.0392618 |         0.885759 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.752316 |     0.0506973 |       0.0460347 |    0.0212373  |    0.0392756 |         0.892107 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.750475 |     0.0508853 |       0.0466199 |    0.0203937  |    0.0398876 |         0.889134 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.747833 |     0.051154  |       0.0451393 |    0.0240662  |    0.0405703 |         0.896524 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])            |    0.745201 |     0.0514203 |       0.0481712 |    0.0179885  |    0.0393917 |         0.881278 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.744551 |     0.0514858 |       0.0455761 |    0.0239502  |    0.0407367 |         0.894331 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.744466 |     0.0514944 |       0.0473336 |    0.0202781  |    0.0403839 |         0.885504 |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.743409 |     0.0516008 |       0.045793  |    0.0237831  |    0.0408188 |         0.893389 |
| Global Single (54 Backbone + 20 Pre-ReLU [H20])                                 |    0.742619 |     0.0516802 |       0.0478167 |    0.0196063  |    0.0403473 |         0.883001 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])              |    0.742443 |     0.0516979 |       0.0479801 |    0.0192505  |    0.0405788 |         0.882238 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.742166 |     0.0517257 |       0.0455116 |    0.0245814  |    0.0409449 |         0.894656 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.740423 |     0.0519002 |       0.0475768 |    0.0207385  |    0.0408463 |         0.884246 |
| Global Single (54 Backbone + 20 Head Hidden [H20])                              |    0.739913 |     0.0519511 |       0.0483196 |    0.0190823  |    0.0404485 |         0.880377 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])           |    0.738107 |     0.0521312 |       0.0484172 |    0.0193246  |    0.0408708 |         0.879872 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.73649  |     0.0522918 |       0.0461596 |    0.0245709  |    0.0413415 |         0.891447 |
| Global Single (54 Backbone + 8 Head Hidden [H8])                                |    0.733909 |     0.0525473 |       0.0476188 |    0.0222186  |    0.0417019 |         0.884759 |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.733858 |     0.0525524 |       0.0466602 |    0.0241782  |    0.0417662 |         0.889144 |
| Global Single (54 Backbone + 8 Pre-ReLU [H8])                                   |    0.733817 |     0.0525565 |       0.0475882 |    0.0223057  |    0.0416023 |         0.885011 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.731762 |     0.0527589 |       0.0469179 |    0.0241291  |    0.0417006 |         0.887636 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.7288   |     0.0530494 |       0.0458039 |    0.0267626  |    0.0427938 |         0.893377 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.728371 |     0.0530914 |       0.047132  |    0.0244392  |    0.0423743 |         0.886529 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.725432 |     0.0533778 |       0.047205  |    0.0249174  |    0.0428831 |         0.886151 |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.725145 |     0.0534057 |       0.0469201 |    0.0255083  |    0.0426833 |         0.887916 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])                  |    0.724513 |     0.0534671 |       0.0475959 |    0.0243589  |    0.0426123 |         0.885432 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.724291 |     0.0534886 |       0.045846  |    0.0275531  |    0.0432127 |         0.893349 |
| Global Single (54 Backbone + 16 Pre-ReLU [H16])                                 |    0.722498 |     0.0536622 |       0.0485191 |    0.0229246  |    0.0416668 |         0.879314 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.721816 |     0.0537282 |       0.0481828 |    0.0237726  |    0.0433447 |         0.881065 |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.720765 |     0.0538295 |       0.0475137 |    0.0252995  |    0.0431416 |         0.884695 |
| Global Single (54 Backbone + 16 Head Hidden [H16])                              |    0.719788 |     0.0539236 |       0.0487581 |    0.0230306  |    0.0416721 |         0.878026 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.717089 |     0.0541828 |       0.0469025 |    0.0271279  |    0.0434425 |         0.888319 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16])         |    0.714247 |     0.0544542 |       0.050301  |    0.0208583  |    0.0415138 |         0.869637 |
| Global Single (54 Backbone + 16 CTX [H8])                                       |    0.714201 |     0.0544586 |       0.047933  |    0.0258488  |    0.043389  |         0.884963 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])            |    0.712879 |     0.0545843 |       0.0505483 |    0.020599   |    0.0418048 |         0.868255 |
| Global Single (54 Backbone + 32 CTX [H16])                                      |    0.708794 |     0.0549713 |       0.0503754 |    0.0220036  |    0.0424845 |         0.869217 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.706465 |     0.0551907 |       0.0490193 |    0.0253599  |    0.0428797 |         0.878369 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])                 |    0.703916 |     0.0554298 |       0.0516102 |    0.0202199  |    0.0423911 |         0.862772 |
| BiLSTM+Attn H16 (LSTM-only, V21)                                                |    0.703207 |     0.0554961 |       0.0540537 |    0.0125706  |    0.0433533 |       nan        |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.702138 |     0.055596  |       0.0513944 |    0.0212023  |    0.0440101 |         0.863939 |
| [1.5] Global Single (54 Backbone + 160 CTX)                                     |    0.697691 |     0.0560095 |       0.0489892 |    0.02715    |    0.044595  |         0.876936 |
| BiLSTM+Attn H8 (LSTM-only, V21)                                                 |    0.696313 |     0.056137  |       0.0553832 |    0.00916818 |    0.0455778 |       nan        |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.695492 |     0.0562128 |       0.0479859 |    0.0292785  |    0.0446102 |         0.882157 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.69529  |     0.0562315 |       0.0491271 |    0.0273588  |    0.0440866 |         0.876931 |
| BiLSTM+Attn H20 (LSTM-only, V21)                                                |    0.69309  |     0.0564341 |       0.0542288 |    0.0156219  |    0.0439822 |       nan        |
| [1.5] Global Single (54 Backbone + 80 Head Hidden)                              |    0.692525 |     0.056486  |       0.0484955 |    0.028963   |    0.0448483 |         0.879807 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])           |    0.691703 |     0.0565615 |       0.0497662 |    0.0268798  |    0.0430417 |         0.873326 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])                   |    0.684478 |     0.0572204 |       0.0498143 |    0.0281551  |    0.0440026 |         0.872963 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])              |    0.679918 |     0.0576324 |       0.0501511 |    0.0283965  |    0.0437607 |         0.871252 |
| Global Single (54 Backbone + 8 CTX [H4])                                        |    0.678224 |     0.0577847 |       0.0495127 |    0.029792   |    0.0450094 |         0.875225 |
| Global Single (54 Backbone + 4 Pre-ReLU [H4])                                   |    0.667166 |     0.0587692 |       0.0499439 |    0.0309746  |    0.045567  |         0.873094 |
| Global Single (54 Backbone + 4 Head Hidden [H4])                                |    0.66573  |     0.0588959 |       0.050457  |    0.0303779  |    0.0456513 |         0.870037 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])            |    0.643591 |     0.0608149 |       0.0504216 |    0.0340018  |    0.0470259 |         0.868914 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40])         |    0.642315 |     0.0609237 |       0.0506675 |    0.0338305  |    0.0470773 |         0.86753  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])                 |    0.632925 |     0.0617182 |       0.0512324 |    0.0344149  |    0.0484144 |         0.864818 |
| Global Single (54 Backbone + 40 Head Hidden [H40])                              |    0.628863 |     0.0620587 |       0.0494023 |    0.0375593  |    0.0493859 |         0.8756   |
| Global Single (54 Backbone + 40 Pre-ReLU [H40])                                 |    0.626165 |     0.0622839 |       0.0495051 |    0.037796   |    0.0494006 |         0.874962 |
| Global Single (54 Backbone + 80 CTX [H40])                                      |    0.610982 |     0.0635362 |       0.0499429 |    0.0392753  |    0.0509103 |         0.873116 |
| [1.5] BiLSTM+Attn (LSTM-only, V21)                                              |    0.582725 |     0.0658032 |       0.0524284 |    0.0397659  |    0.0524885 |       nan        |
| BiLSTM+Attn H40 (LSTM-only, V21)                                                |    0.552283 |     0.0681612 |       0.0587193 |    0.0346122  |    0.0545359 |       nan        |
| BiLSTM+Attn H4 (LSTM-only, V21)                                                 |    0.526266 |     0.0701137 |       0.0591161 |    0.0376992  |    0.0546092 |       nan        |

---

## Per-Regime Performance Breakdown

| model_name                                                              |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |        bias |       mae |   pearson |
|:------------------------------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|------------:|----------:|----------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                        |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 | 0.00861491  | 0.0359221 |  0.900537 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                        |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 | 0.000797068 | 0.0278349 |  0.9191   |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])         |         0 |     10624 |     4817 | 0.626127 | 0.0611704 | 0.0518969 | 0.032381    | 0.0478571 |  0.854949 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])         |         1 |      3984 |     1803 | 0.648125 | 0.0631584 | 0.0490008 | 0.0398486   | 0.0499035 |  0.891958 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40]) |         0 |     10624 |     4817 | 0.623902 | 0.0613522 | 0.0516919 | 0.0330459   | 0.0473506 |  0.856438 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40]) |         1 |      3984 |     1803 | 0.684932 | 0.059764  | 0.0477599 | 0.0359267   | 0.0463471 |  0.895065 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])    |         0 |     10624 |     4817 | 0.630408 | 0.0608192 | 0.0509874 | 0.0331551   | 0.0471219 |  0.860449 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])    |         1 |      3984 |     1803 | 0.673876 | 0.0608035 | 0.0488059 | 0.0362636   | 0.0467696 |  0.88953  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])         |         0 |     10624 |     4817 | 0.763366 | 0.0486651 | 0.0465875 | 0.0140677   | 0.0379937 |  0.889637 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])         |         1 |      3984 |     1803 | 0.734697 | 0.0548413 | 0.0511918 | 0.0196715   | 0.0416108 |  0.877214 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20]) |         0 |     10624 |     4817 | 0.765495 | 0.0484457 | 0.0462094 | 0.0145492   | 0.0376018 |  0.887427 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20]) |         1 |      3984 |     1803 | 0.72274  | 0.0560635 | 0.0517825 | 0.0214871   | 0.0424901 |  0.874405 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])    |         0 |     10624 |     4817 | 0.750143 | 0.0500064 | 0.0470027 | 0.0170699   | 0.0388893 |  0.88318  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])    |         1 |      3984 |     1803 | 0.732968 | 0.0550198 | 0.0510812 | 0.0204424   | 0.040734  |  0.877461 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])         |         0 |     10624 |     4817 | 0.707876 | 0.0540709 | 0.0513404 | 0.0169653   | 0.0417841 |  0.860132 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])         |         1 |      3984 |     1803 | 0.693905 | 0.0589068 | 0.0513218 | 0.028915    | 0.0440128 |  0.876829 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16]) |         0 |     10624 |     4817 | 0.694794 | 0.0552683 | 0.0510766 | 0.0211132   | 0.0425196 |  0.860492 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16]) |         1 |      3984 |     1803 | 0.759481 | 0.052217  | 0.0481611 | 0.0201774   | 0.0388269 |  0.893236 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])    |         0 |     10624 |     4817 | 0.698812 | 0.0549033 | 0.0510618 | 0.0201758   | 0.0423915 |  0.860772 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])    |         1 |      3984 |     1803 | 0.745409 | 0.0537228 | 0.049132  | 0.0217299   | 0.0402373 |  0.889315 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])          |         0 |     10624 |     4817 | 0.707254 | 0.0541283 | 0.0477657 | 0.025462    | 0.0439788 |  0.878957 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])          |         1 |      3984 |     1803 | 0.764594 | 0.051659  | 0.0470126 | 0.0214119   | 0.0389616 |  0.904601 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])   |         0 |     10624 |     4817 | 0.731237 | 0.0518638 | 0.0487517 | 0.0176953   | 0.0414387 |  0.873655 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])   |         1 |      3984 |     1803 | 0.753715 | 0.0528391 | 0.0472371 | 0.0236776   | 0.0393533 |  0.902465 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])      |         0 |     10624 |     4817 | 0.733037 | 0.0516898 | 0.0483947 | 0.0181602   | 0.041148  |  0.875603 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])      |         1 |      3984 |     1803 | 0.764042 | 0.0517195 | 0.0467299 | 0.0221634   | 0.0390578 |  0.906778 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])           |         0 |     10624 |     4817 | 0.695953 | 0.0551633 | 0.0499747 | 0.0233563   | 0.0422048 |  0.866629 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])           |         1 |      3984 |     1803 | 0.656696 | 0.0623845 | 0.0470404 | 0.0409759   | 0.0488058 |  0.900307 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])   |         0 |     10624 |     4817 | 0.696255 | 0.0551359 | 0.0499521 | 0.0233399   | 0.042096  |  0.866736 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])   |         1 |      3984 |     1803 | 0.680265 | 0.0602049 | 0.0480023 | 0.0363374   | 0.0455683 |  0.895956 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])      |         0 |     10624 |     4817 | 0.68314  | 0.0563136 | 0.0504478 | 0.0250247   | 0.0427844 |  0.863866 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])      |         1 |      3984 |     1803 | 0.67159  | 0.0610162 | 0.0482063 | 0.0374049   | 0.0463689 |  0.895228 |

---

## Year-by-Year $R^2$ Breakdown

| model_name                                                                      |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:--------------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                                |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                          |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| [1.5] Global Single (54 Backbone)                                               |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| Global Single (54 Backbone)                                                     |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-95%)    |    0.76674  |       0.752535 |       0.770393 |       0.771568 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-16)         |    0.758394 |       0.751904 |       0.772422 |       0.743832 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-95%)        |    0.756389 |       0.743007 |       0.776377 |       0.743921 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-32)         |    0.756169 |       0.749565 |       0.774584 |       0.737367 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])                 |    0.754996 |       0.723    |       0.783944 |       0.755301 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX PCA-64)         |    0.754639 |       0.745476 |       0.779542 |       0.732406 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20])         |    0.752932 |       0.722345 |       0.785618 |       0.747873 |
| Global Single (54 Backbone + 40 CTX [H20])                                      |    0.752359 |       0.705015 |       0.804493 |       0.747682 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-95%) |    0.752316 |       0.728209 |       0.765136 |       0.759158 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-64)  |    0.750475 |       0.721389 |       0.788435 |       0.738429 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-95%)                         |    0.747833 |       0.721513 |       0.75876  |       0.758941 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])            |    0.745201 |       0.699411 |       0.790732 |       0.744938 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-16)     |    0.744551 |       0.721238 |       0.769362 |       0.73847  |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-32)  |    0.744466 |       0.714198 |       0.777206 |       0.73869  |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-95%)                      |    0.743409 |       0.724384 |       0.751696 |       0.748487 |
| Global Single (54 Backbone + 20 Pre-ReLU [H20])                                 |    0.742619 |       0.691929 |       0.803262 |       0.733159 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])              |    0.742443 |       0.751796 |       0.738287 |       0.726747 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-32)     |    0.742166 |       0.707089 |       0.776462 |       0.740364 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden PCA-16)  |    0.740423 |       0.715437 |       0.77093  |       0.730565 |
| Global Single (54 Backbone + 20 Head Hidden [H20])                              |    0.739913 |       0.699826 |       0.797115 |       0.72143  |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])           |    0.738107 |       0.750748 |       0.730679 |       0.72165  |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU PCA-64)     |    0.73649  |       0.702879 |       0.784995 |       0.718889 |
| Global Single (54 Backbone + 8 Head Hidden [H8])                                |    0.733909 |       0.735918 |       0.734906 |       0.721394 |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-64)                       |    0.733858 |       0.697919 |       0.765602 |       0.735267 |
| Global Single (54 Backbone + 8 Pre-ReLU [H8])                                   |    0.733817 |       0.729631 |       0.741729 |       0.721717 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-95%)                             |    0.731762 |       0.700631 |       0.747504 |       0.743181 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-16)                          |    0.7288   |       0.700158 |       0.760266 |       0.721844 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-16)                              |    0.728371 |       0.711074 |       0.740566 |       0.727098 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-32)                              |    0.725432 |       0.705907 |       0.742299 |       0.72207  |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-32)                       |    0.725145 |       0.686981 |       0.759965 |       0.725822 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])                  |    0.724513 |       0.708628 |       0.73265  |       0.725439 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-32)                          |    0.724291 |       0.686784 |       0.7577   |       0.725558 |
| Global Single (54 Backbone + 16 Pre-ReLU [H16])                                 |    0.722498 |       0.671147 |       0.762436 |       0.733371 |
| [1.5] Global Single (54 Backbone + 160 CTX PCA-64)                              |    0.721816 |       0.691353 |       0.745259 |       0.72458  |
| [1.5] Global Single (54 Backbone + 80 Head Hidden PCA-16)                       |    0.720765 |       0.689968 |       0.748967 |       0.719224 |
| Global Single (54 Backbone + 16 Head Hidden [H16])                              |    0.719788 |       0.663657 |       0.762067 |       0.733821 |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU PCA-64)                          |    0.717089 |       0.68481  |       0.762185 |       0.700604 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16])         |    0.714247 |       0.668481 |       0.745048 |       0.727298 |
| Global Single (54 Backbone + 16 CTX [H8])                                       |    0.714201 |       0.702467 |       0.718691 |       0.713526 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])            |    0.712879 |       0.673307 |       0.748864 |       0.713626 |
| Global Single (54 Backbone + 32 CTX [H16])                                      |    0.708794 |       0.643775 |       0.762972 |       0.721111 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Head Hidden)         |    0.706465 |       0.665956 |       0.741726 |       0.708785 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])                 |    0.703916 |       0.664496 |       0.742016 |       0.702107 |
| BiLSTM+Attn H16 (LSTM-only, V21)                                                |    0.703207 |     nan        |     nan        |     nan        |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 160 CTX)                |    0.702138 |       0.651923 |       0.755298 |       0.698048 |
| [1.5] Global Single (54 Backbone + 160 CTX)                                     |    0.697691 |       0.637512 |       0.740217 |       0.715412 |
| BiLSTM+Attn H8 (LSTM-only, V21)                                                 |    0.696313 |     nan        |     nan        |     nan        |
| [1.5] Global Single (54 Backbone + 80 Pre-ReLU)                                 |    0.695492 |       0.631856 |       0.738593 |       0.716586 |
| [1.5] Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 Pre-ReLU)            |    0.69529  |       0.636679 |       0.746754 |       0.702363 |
| BiLSTM+Attn H20 (LSTM-only, V21)                                                |    0.69309  |     nan        |     nan        |     nan        |
| [1.5] Global Single (54 Backbone + 80 Head Hidden)                              |    0.692525 |       0.640708 |       0.721456 |       0.713673 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])           |    0.691703 |       0.721211 |       0.724064 |       0.615138 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])                   |    0.684478 |       0.718008 |       0.702525 |       0.617007 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])              |    0.679918 |       0.704727 |       0.711287 |       0.609367 |
| Global Single (54 Backbone + 8 CTX [H4])                                        |    0.678224 |       0.720122 |       0.677446 |       0.619253 |
| Global Single (54 Backbone + 4 Pre-ReLU [H4])                                   |    0.667166 |       0.713927 |       0.675228 |       0.593519 |
| Global Single (54 Backbone + 4 Head Hidden [H4])                                |    0.66573  |       0.718179 |       0.668153 |       0.590955 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])            |    0.643591 |       0.578502 |       0.687738 |       0.663549 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40])         |    0.642315 |       0.588043 |       0.688412 |       0.647773 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])                 |    0.632925 |       0.586969 |       0.656359 |       0.65059  |
| Global Single (54 Backbone + 40 Head Hidden [H40])                              |    0.628863 |       0.580307 |       0.670934 |       0.631166 |
| Global Single (54 Backbone + 40 Pre-ReLU [H40])                                 |    0.626165 |       0.573688 |       0.669796 |       0.631397 |
| Global Single (54 Backbone + 80 CTX [H40])                                      |    0.610982 |       0.566764 |       0.627398 |       0.63274  |
| [1.5] BiLSTM+Attn (LSTM-only, V21)                                              |    0.582725 |     nan        |     nan        |     nan        |
| BiLSTM+Attn H40 (LSTM-only, V21)                                                |    0.552283 |     nan        |     nan        |     nan        |
| BiLSTM+Attn H4 (LSTM-only, V21)                                                 |    0.526266 |     nan        |     nan        |     nan        |

---

## Accelerated SHAP Feature Importance Analysis

### SHAP Computation Time (C++/CUDA `pred_contribs`)

| Model Name                                                              |   pred_contribs (s) |
|:------------------------------------------------------------------------|--------------------:|
| Global Single (54 Backbone)                                             |              1.2159 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                        |              1.0533 |
| Global Single (54 Backbone + 80 CTX [H40])                              |              1.1269 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])         |              1.0514 |
| Global Single (54 Backbone + 40 Head Hidden [H40])                      |              1.0691 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40]) |              1.0367 |
| Global Single (54 Backbone + 40 Pre-ReLU [H40])                         |              1.0887 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])    |              1.0482 |
| Global Single (54 Backbone + 40 CTX [H20])                              |              1.1638 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])         |              1.0867 |
| Global Single (54 Backbone + 20 Head Hidden [H20])                      |              1.1368 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20]) |              1.0547 |
| Global Single (54 Backbone + 20 Pre-ReLU [H20])                         |              1.1501 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])    |              1.0797 |
| Global Single (54 Backbone + 32 CTX [H16])                              |              1.2494 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])         |              1.1103 |
| Global Single (54 Backbone + 16 Head Hidden [H16])                      |              1.1218 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16]) |              1.0457 |
| Global Single (54 Backbone + 16 Pre-ReLU [H16])                         |              1.1374 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])    |              1.0809 |
| Global Single (54 Backbone + 16 CTX [H8])                               |              1.1652 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])          |              1.0661 |
| Global Single (54 Backbone + 8 Head Hidden [H8])                        |              1.0993 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])   |              1.0365 |
| Global Single (54 Backbone + 8 Pre-ReLU [H8])                           |              1.1159 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])      |              1.0353 |
| Global Single (54 Backbone + 8 CTX [H4])                                |              1.0809 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])           |              1.0395 |
| Global Single (54 Backbone + 4 Head Hidden [H4])                        |              1.0669 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])   |              1.0203 |
| Global Single (54 Backbone + 4 Pre-ReLU [H4])                           |              1.0426 |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])      |              1.0119 |

### Feature Category Contribution (Tabular vs. LSTM Representations)

| Model Name                                                              |   Tabular SHAP Sum |   Tabular Mean abs(SHAP) |   Tabular Median abs(SHAP) |   Repr SHAP Sum |   Repr Mean abs(SHAP) |   Repr Median abs(SHAP) | Tabular % Share   | Repr % Share   |
|:------------------------------------------------------------------------|-------------------:|-------------------------:|---------------------------:|----------------:|----------------------:|------------------------:|:------------------|:---------------|
| Global Single (54 Backbone)                                             |             0.1726 |                   0.0032 |                     0.0011 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10)                        |             0.1713 |                   0.0027 |                     0.0009 |          0      |                0      |                  0      | 100.00%           | 0.00%          |
| Global Single (54 Backbone + 80 CTX [H40])                              |             0.0183 |                   0.0003 |                     0.0002 |          0.1181 |                0.0015 |                  0.0004 | 13.44%            | 86.56%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 80 CTX [H40])         |             0.0191 |                   0.0003 |                     0.0002 |          0.119  |                0.0015 |                  0.0005 | 13.85%            | 86.15%         |
| Global Single (54 Backbone + 40 Head Hidden [H40])                      |             0.0373 |                   0.0007 |                     0.0004 |          0.0982 |                0.0025 |                  0.0006 | 27.50%            | 72.50%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Head Hidden [H40]) |             0.0373 |                   0.0006 |                     0.0004 |          0.1004 |                0.0025 |                  0.0009 | 27.07%            | 72.93%         |
| Global Single (54 Backbone + 40 Pre-ReLU [H40])                         |             0.0344 |                   0.0006 |                     0.0004 |          0.0991 |                0.0025 |                  0.0007 | 25.77%            | 74.23%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 Pre-ReLU [H40])    |             0.0338 |                   0.0005 |                     0.0004 |          0.1027 |                0.0026 |                  0.0009 | 24.79%            | 75.21%         |
| Global Single (54 Backbone + 40 CTX [H20])                              |             0.0358 |                   0.0007 |                     0.0004 |          0.1096 |                0.0027 |                  0.0007 | 24.64%            | 75.36%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 40 CTX [H20])         |             0.0381 |                   0.0006 |                     0.0004 |          0.1101 |                0.0028 |                  0.0008 | 25.69%            | 74.31%         |
| Global Single (54 Backbone + 20 Head Hidden [H20])                      |             0.0473 |                   0.0009 |                     0.0006 |          0.0776 |                0.0039 |                  0      | 37.88%            | 62.12%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Head Hidden [H20]) |             0.0511 |                   0.0008 |                     0.0005 |          0.0794 |                0.004  |                  0      | 39.13%            | 60.87%         |
| Global Single (54 Backbone + 20 Pre-ReLU [H20])                         |             0.0433 |                   0.0008 |                     0.0006 |          0.0877 |                0.0044 |                  0.0013 | 33.08%            | 66.92%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 20 Pre-ReLU [H20])    |             0.0449 |                   0.0007 |                     0.0005 |          0.0908 |                0.0045 |                  0.0018 | 33.12%            | 66.88%         |
| Global Single (54 Backbone + 32 CTX [H16])                              |             0.0354 |                   0.0007 |                     0.0005 |          0.111  |                0.0035 |                  0.001  | 24.19%            | 75.81%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 32 CTX [H16])         |             0.0409 |                   0.0006 |                     0.0004 |          0.1093 |                0.0034 |                  0.0009 | 27.22%            | 72.78%         |
| Global Single (54 Backbone + 16 Head Hidden [H16])                      |             0.049  |                   0.0009 |                     0.0006 |          0.0838 |                0.0052 |                  0      | 36.89%            | 63.11%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Head Hidden [H16]) |             0.0538 |                   0.0008 |                     0.0005 |          0.0817 |                0.0051 |                  0      | 39.71%            | 60.29%         |
| Global Single (54 Backbone + 16 Pre-ReLU [H16])                         |             0.0454 |                   0.0008 |                     0.0006 |          0.0885 |                0.0055 |                  0.0008 | 33.91%            | 66.09%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 Pre-ReLU [H16])    |             0.0497 |                   0.0008 |                     0.0005 |          0.0859 |                0.0054 |                  0.0015 | 36.66%            | 63.34%         |
| Global Single (54 Backbone + 16 CTX [H8])                               |             0.04   |                   0.0007 |                     0.0005 |          0.0876 |                0.0055 |                  0.0014 | 31.32%            | 68.68%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 16 CTX [H8])          |             0.0432 |                   0.0007 |                     0.0004 |          0.0863 |                0.0054 |                  0.0018 | 33.33%            | 66.67%         |
| Global Single (54 Backbone + 8 Head Hidden [H8])                        |             0.0491 |                   0.0009 |                     0.0007 |          0.0748 |                0.0094 |                  0.0031 | 39.62%            | 60.38%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Head Hidden [H8])   |             0.0534 |                   0.0008 |                     0.0005 |          0.0731 |                0.0091 |                  0.0046 | 42.18%            | 57.82%         |
| Global Single (54 Backbone + 8 Pre-ReLU [H8])                           |             0.0468 |                   0.0009 |                     0.0006 |          0.0809 |                0.0101 |                  0.0058 | 36.64%            | 63.36%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 Pre-ReLU [H8])      |             0.0509 |                   0.0008 |                     0.0005 |          0.077  |                0.0096 |                  0.0054 | 39.81%            | 60.19%         |
| Global Single (54 Backbone + 8 CTX [H4])                                |             0.0436 |                   0.0008 |                     0.0006 |          0.0828 |                0.0103 |                  0.0051 | 34.47%            | 65.53%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 8 CTX [H4])           |             0.0447 |                   0.0007 |                     0.0005 |          0.0841 |                0.0105 |                  0.0071 | 34.71%            | 65.29%         |
| Global Single (54 Backbone + 4 Head Hidden [H4])                        |             0.0516 |                   0.001  |                     0.0006 |          0.0825 |                0.0206 |                  0.0126 | 38.45%            | 61.55%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Head Hidden [H4])   |             0.0532 |                   0.0008 |                     0.0005 |          0.0815 |                0.0204 |                  0.0143 | 39.52%            | 60.48%         |
| Global Single (54 Backbone + 4 Pre-ReLU [H4])                           |             0.0505 |                   0.0009 |                     0.0007 |          0.0695 |                0.0174 |                  0.007  | 42.08%            | 57.92%         |
| Clustering_V0_Full_k2 (54 Backbone, c0=0, c1=10 + 4 Pre-ReLU [H4])      |             0.0522 |                   0.0008 |                     0.0006 |          0.0688 |                0.0172 |                  0.0083 | 43.14%            | 56.86%         |

### Top 10 Features by Mean Absolute SHAP Value (averaged across all 32 XGBoost models)

| Feature       |   Mean abs(SHAP) |
|:--------------|-----------------:|
| hh_5          |           0.0047 |
| hp_5          |           0.0044 |
| hp_1          |           0.0043 |
| hh_1          |           0.0042 |
| D_sin_DOY     |           0.0034 |
| hh_11         |           0.0033 |
| ctx_5         |           0.0031 |
| SMAP_x_year   |           0.0031 |
| G_rain_sum_3d |           0.0031 |
| hp_11         |           0.0029 |

---

## Key Insights & Architecture Summary
- **Phase 1**: V21 BiLSTM+Attn trained from scratch on `derived_8.4` for each hidden size
  `H ∈ {40, 20, 16, 8, 4}` (Jakob 38 + V9-unique features, seq_len=30, ReduceLROnPlateau, 1 seed).
- **Phase 2**: Three frozen raw representations extracted per hidden size:
  `ctx` (2H-dim), `head_hidden` (H-dim), `head_pre_relu` (H-dim).
- **Phase 3**: NO PCA — raw representations are used directly in the hybrid XGBoost models.
- **Phase 4**: XGBoost fit on `[Tabular + Repr]` for all 15 representation variants (5 hidden sizes × ctx/hh/hp) × 2 strategies (Global + Clustering) + 2 tabular baselines.
- **Phase 5**: SHAP feature importance via XGBoost native `pred_contribs=True` with CUDA acceleration.
- Reference rows from `derived_8.4-hybrid-lstm-1.5` (H=80 ± PCA) are marked `[1.5]` (33 rows).
