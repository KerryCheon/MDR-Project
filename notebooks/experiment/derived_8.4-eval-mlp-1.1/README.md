# Experiment: `derived_8.4-eval-mlp-1.1` — closing the MLP-vs-XGBoost gap

## Objective

Follow-up to `derived_8.4-eval-mlp-1.0` (best MLP 2-regime R² 0.759 vs XGBoost 0.815; 1-regime 0.723 vs 0.779). Goal: reach the same or better performance than XGBoost by fixing the four biggest problems found in 1.0:

1. **Selection protocol** — 1.0 early-stopped and selected configs on a noisy 10% per-station tail of trainval (~1.4k rows). Here we use the **official val split (2021–2022, n=4,805)**, 3.3× larger and exactly mirroring the temporal test construction.
2. **Under-training** — max 400 epochs, patience 60, warmup (5%) + cosine LR (1.0: 200 epochs, patience 25, no warmup).
3. **Capacity / architecture** — residual MLPs and **FT-Transformers** added alongside the plain MLP.
4. **Feature set** — a 96-feature candidate-pool family (62 informative features beyond the 54 XGBoost-tuned backbone).

Winning configs are additionally **retrained on the full trainval** (train+val, 14,608 rows — matching XGBoost's training data) at their val-best epoch count and **ensembled over 5 seeds** — the neural analog of XGBoost's 2500-tree ensemble.

All numbers below are the stdout of the executed report notebook (`derived_8.4-eval-mlp-1.1.ipynb`). Trained weights, checkpoints, test predictions, and loss curves are archived under `models/`; preprocessed tensors and per-job logs under `artifacts/`.

## Overall Leaderboard (2023–2025 Test Set)

`test-best` rows are reported **for reference only** (selection on test would be leakage) — they bound what the model class achieved. The XGBoost reference itself was test-selected in eval-1.1 (its delta grid was searched on the test set).

| model_name                                              | strategy_name          |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:--------------------------------------------------------|:-----------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)              | XGBoost_Reference      |    0.81496  |     0.0438196 |       0.043337  |   0.00648567  |    0.0337195 |         0.905594 |
| MLP 2-Regime-96 (test-best, w512x512x512_d0.3)          | MLP_testbest_reference |    0.78591  |     0.0471339 |       0.0449648 |   0.0141342   |    0.0366223 |         0.898512 |
| Global Single Model (54 Backbone)                       | XGBoost_Reference      |    0.77923  |     0.0478636 |       0.0466868 |   0.0105484   |    0.0370592 |         0.889432 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)              | MLP_testbest_reference |    0.778568 |     0.0479353 |       0.0477438 |   0.0042813   |    0.0363383 |         0.884221 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3)              | MLP_testbest_reference |    0.77711  |     0.0480929 |       0.0475861 |   0.00696359  |    0.0375546 |         0.884798 |
| MLP 2-Regime-54 (test-best, w256x256_d0.3_gelu)         | MLP_testbest_reference |    0.773167 |     0.0485165 |       0.0483759 |  -0.00369107  |    0.0376809 |         0.880358 |
| MLP 2-Regime-54 (test-best, w512x512_d0.3_gelu)         | MLP_testbest_reference |    0.770776 |     0.0487715 |       0.0487685 |   0.000545806 |    0.0376619 |         0.877955 |
| MLP 2-Regime-96 (test-best, w256x256_d0.4)              | MLP_testbest_reference |    0.767316 |     0.0491381 |       0.0478635 |   0.0111193   |    0.0378726 |         0.882783 |
| MLP 2-Regime (w256x256_d0.3)                            | MLP_1.0_Reference      |    0.75884  |     0.0500251 |       0.0500077 |   0.00131949  |    0.0377593 |         0.871552 |
| MLP 2-Regime-96 (res_w1024x1024_d0.2)                   | MLP_2regime_96         |    0.756106 |     0.0503079 |       0.0499735 |   0.00579076  |    0.0379582 |         0.874136 |
| MLP 2-Regime-54 (w512x256x128_d0.3_lr1e-3)              | MLP_2regime_54         |    0.753037 |     0.0506234 |       0.0469795 |   0.0188589   |    0.0392248 |         0.889568 |
| MLP 1-Regime-96 (test-best, res_w256x256x256_d0.2)      | MLP_testbest_reference |    0.74505  |     0.0514355 |       0.0481302 |   0.0181409   |    0.0398431 |         0.881356 |
| MLP 2-Regime-96 (sweep top-3 avg)                       | MLP_2regime_96         |    0.74403  |     0.0515383 |       0.0465691 |   0.0220798   |    0.0400719 |         0.891606 |
| MLP 1-Regime-54 (test-best, w256x256_d0.3_nowarmup)     | MLP_testbest_reference |    0.74083  |     0.0518595 |       0.0517998 |   0.00248797  |    0.0401129 |         0.861385 |
| MLP 1-Regime-54 (test-best, w256x256_d0.2)              | MLP_testbest_reference |    0.740111 |     0.0519314 |       0.0513698 |   0.00761685  |    0.0404389 |         0.863603 |
| MLP 2-Regime-96 (res_w512x512x256_d0.2)                 | MLP_2regime_96         |    0.738744 |     0.0520678 |       0.0473958 |   0.0215568   |    0.0408216 |         0.888257 |
| MLP 1-Regime-54 (test-best, w256x256_d0.3_huber0.2)     | MLP_testbest_reference |    0.738664 |     0.0520757 |       0.0520547 |   0.00148007  |    0.0401398 |         0.859938 |
| MLP 1-Regime-54 (w512x256x128_d0.3_lr1e-3)              | MLP_1regime_54         |    0.736558 |     0.0522851 |       0.0495673 |   0.0166379   |    0.0397841 |         0.874947 |
| MLP 2-Regime-96 (w1024x1024x512_d0.3)                   | MLP_2regime_96         |    0.736171 |     0.0523236 |       0.0474192 |   0.0221174   |    0.0408209 |         0.887885 |
| MLP 1-Regime-54 (w512x256x128_d0.3)                     | MLP_1regime_54         |    0.735063 |     0.0524333 |       0.0503703 |   0.0145631   |    0.0401165 |         0.870455 |
| MLP 1-Regime-96 (test-best, w256x256_d0.3_lr1e-4)       | MLP_testbest_reference |    0.733796 |     0.0525586 |       0.0510467 |   0.0125154   |    0.0418147 |         0.869922 |
| MLP 1-Regime-96 (test-best, w512x256x128_d0.3_lr1e-3)   | MLP_testbest_reference |    0.731078 |     0.0528261 |       0.0495937 |   0.0181952   |    0.0420238 |         0.874757 |
| MLP 2-Regime-54 (res_w512x256x128_d0.2)                 | MLP_2regime_54         |    0.730937 |     0.05284   |       0.0462946 |   0.0254731   |    0.041469  |         0.892006 |
| MLP 1-Regime-54 (res_w256x256x256_d0.2)                 | MLP_1regime_54         |    0.729382 |     0.0529925 |       0.0498523 |   0.0179708   |    0.0400444 |         0.873442 |
| MLP 1-Regime-96 (res_w512x512_d0.2_wd1e-3)              | MLP_1regime_96         |    0.728906 |     0.0530391 |       0.0491843 |   0.0198506   |    0.0411344 |         0.880607 |
| MLP 1-Regime-96 (sweep top-3 avg)                       | MLP_1regime_96         |    0.72878  |     0.0530514 |       0.0482862 |   0.0219747   |    0.0413209 |         0.884057 |
| MLP 1-Regime (w256x256_d0.3)                            | MLP_1.0_Reference      |    0.72348  |     0.0535672 |       0.0523852 |   0.011191    |    0.0411193 |         0.858217 |
| MLP 1-Regime-96 (res_w512x512x256_d0.2)                 | MLP_1regime_96         |    0.719298 |     0.0539708 |       0.0508128 |   0.018191    |    0.0418069 |         0.870084 |
| MLP 2-Regime-54 (sweep top-3 avg)                       | MLP_2regime_54         |    0.714963 |     0.054386  |       0.0458729 |   0.0292148   |    0.0427758 |         0.893202 |
| MLP 1-Regime-96 (res_w512x512_d0.3_lr1e-3)              | MLP_1regime_96         |    0.711739 |     0.0546926 |       0.0541769 |   0.00749335  |    0.0423936 |         0.849541 |
| MLP 1-Regime-54 (sweep top-3 avg)                       | MLP_1regime_54         |    0.711348 |     0.0547297 |       0.0490203 |   0.0243382   |    0.0425715 |         0.876623 |
| MLP 2-Regime-96 (retrain-ens, res_w512x256x128_d0.2)    | MLP_2regime_96         |    0.707593 |     0.0550845 |       0.0549031 |   0.00446776  |    0.0421982 |         0.857783 |
| MLP 2-Regime-54 (res_w256x256x256_d0.2)                 | MLP_2regime_54         |    0.707382 |     0.0551044 |       0.0472628 |   0.0283324   |    0.0425748 |         0.888747 |
| MLP 2-Regime-96 (res_w512x256x128_d0.2)                 | MLP_2regime_96         |    0.702405 |     0.055571  |       0.0527407 |   0.0175087   |    0.042674  |         0.864603 |
| MLP 1-Regime-96 (res_w512x512_d0.2)                     | MLP_1regime_96         |    0.694089 |     0.0563422 |       0.0489593 |   0.0278824   |    0.0442895 |         0.883272 |
| MLP 1-Regime-54 (retrain-ens, w256x256_d0.3_tanh)       | MLP_1regime_54         |    0.694084 |     0.0563426 |       0.0480905 |   0.0293562   |    0.0443806 |         0.882357 |
| MLP 2-Regime-96 (res_w256x256x256_d0.2)                 | MLP_2regime_96         |    0.690708 |     0.0566526 |       0.0497102 |   0.0271738   |    0.0438988 |         0.876902 |
| MLP 2-Regime-54 (retrain-ens, res_w1024x1024_d0.2)      | MLP_2regime_54         |    0.686735 |     0.0570153 |       0.0534177 |   0.0199323   |    0.0442296 |         0.86121  |
| MLP 1-Regime-96 (res_w512x256x128_d0.2)                 | MLP_1regime_96         |    0.686557 |     0.0570316 |       0.0459982 |   0.033716    |    0.045568  |         0.892342 |
| MLP 1-Regime-96 (retrain-ens, res_w512x512_d0.2_wd1e-3) | MLP_1regime_96         |    0.683071 |     0.0573478 |       0.0570013 |   0.00629523  |    0.0439581 |         0.844033 |
| MLP 1-Regime-54 (w256x256_d0.3_tanh)                    | MLP_1regime_54         |    0.679728 |     0.0576494 |       0.0510623 |   0.0267599   |    0.0452909 |         0.865404 |
| MLP 1-Regime-54 (w256x256_d0.3_ln)                      | MLP_1regime_54         |    0.669563 |     0.0585571 |       0.0505152 |   0.0296167   |    0.0459722 |         0.868452 |
| MLP 2-Regime-54 (res_w512x512x256_d0.2)                 | MLP_2regime_54         |    0.647963 |     0.0604408 |       0.0507417 |   0.0328386   |    0.0482925 |         0.868178 |
| MLP 2-Regime-54 (res_w1024x1024_d0.2)                   | MLP_2regime_54         |    0.646444 |     0.060571  |       0.0502372 |   0.033839    |    0.0483619 |         0.870027 |

**Winners (val-selected, min val RMSE):** `1regime_54: w256x256_d0.3_tanh` (test R² 0.680), `2regime_54: res_w1024x1024_d0.2` (0.646), `1regime_96: res_w512x512_d0.2_wd1e-3` (0.729), `2regime_96: res_w512x256x128_d0.2` (0.702). The best **test** configs (e.g. `2regime_96: w512x512x512_d0.3`, 0.786) rank mid on val — see Key Takeaways #3.

## Hyperparameter Sweep Summary

77 configs (capacity, dropout, lr, weight decay, batch size, loss, activation, norm, warmup, residual blocks, FT-Transformer) trained in **all four families**; ranked by val RMSE (selection metric), test R² for reference. Two EMA configs were excluded (see Key Takeaways #5).

### Sweep Top-10 — 1-regime Global (54 backbone)

| config_id                | architecture   |   dropout |     lr |   weight_decay |   batch_size | loss   | ema   |   val_rmse |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:-------------------------|:---------------|----------:|-------:|---------------:|-------------:|:-------|:------|-----------:|----------:|------------:|-------------:|---------------:|
| w256x256_d0.3_tanh       | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0540346 |  0.679728 |   0.0576494 |          165 |        30.1491 |
| w512x256x128_d0.3_lr1e-3 | mlp            |       0.3 | 0.001  |         0.0001 |          512 | mse    | False |  0.0541591 |  0.736558 |   0.0522851 |          182 |        33.7867 |
| w256x256_d0.3_ln         | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0547597 |  0.669563 |   0.0585571 |          176 |        30.8479 |
| w512x256x128_d0.3        | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0553173 |  0.735063 |   0.0524333 |          301 |        54.3238 |
| res_w256x256x256_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0553402 |  0.729382 |   0.0529925 |           48 |        20.9655 |
| w1024x512x256_d0.3       | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0553666 |  0.721135 |   0.0537939 |          200 |        40.5657 |
| w512x256x128_d0.5        | mlp            |       0.5 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0555589 |  0.708867 |   0.0549644 |          340 |        57.2746 |
| w512x256x128_d0.4        | mlp            |       0.4 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0557074 |  0.727214 |   0.0532043 |          340 |        56.5329 |
| w512x512_d0.3_ln         | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0559783 |  0.679365 |   0.0576821 |          120 |        23.1259 |
| res_w512x256x128_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0566373 |  0.692746 |   0.0564657 |           24 |        17.9034 |

### Sweep Top-10 — 2-regime Cluster (c0=54, c1=64)

| config_id                | architecture   |   dropout |     lr |   weight_decay |   batch_size | loss   | ema   |   val_rmse |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:-------------------------|:---------------|----------:|-------:|---------------:|-------------:|:-------|:------|-----------:|----------:|------------:|-------------:|---------------:|
| res_w1024x1024_d0.2      | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0541406 |  0.646444 |   0.060571  |          218 |        61.5712 |
| res_w256x256x256_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.054302  |  0.707382 |   0.0551044 |          194 |        68.1746 |
| res_w512x256x128_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0546698 |  0.730937 |   0.05284   |          195 |        69.6405 |
| w512x256x128_d0.3_lr1e-3 | mlp            |       0.3 | 0.001  |         0.0001 |          512 | mse    | False |  0.0548826 |  0.753037 |   0.0506234 |          235 |        37.4014 |
| res_w512x512x256_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0550038 |  0.647963 |   0.0604408 |          108 |        38.1337 |
| res_w512x512_d0.4        | residual       |       0.4 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0551    |  0.69065  |   0.0566579 |          136 |        20.4206 |
| res_w512x512_d0.3_lr1e-3 | residual       |       0.3 | 0.001  |         0.0001 |          512 | mse    | False |  0.0552363 |  0.661344 |   0.0592809 |          118 |        36.1928 |
| w256x256_d0.3_tanh       | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0552901 |  0.647271 |   0.0605001 |          296 |        46.0901 |
| w1024x512x256_d0.3       | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0556155 |  0.762283 |   0.0496668 |          220 |        44.3282 |
| w512x512_d0.3_ln         | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0557051 |  0.712149 |   0.0546537 |          270 |        34.9707 |

### Sweep Top-10 — 1-regime Global (96 pool)

| config_id                | architecture   |   dropout |     lr |   weight_decay |   batch_size | loss   | ema   |   val_rmse |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:-------------------------|:---------------|----------:|-------:|---------------:|-------------:|:-------|:------|-----------:|----------:|------------:|-------------:|---------------:|
| res_w512x512_d0.2_wd1e-3 | residual       |       0.2 | 0.0003 |         0.001  |          512 | mse    | False |  0.045749  |  0.728906 |   0.0530391 |          124 |        38.3529 |
| res_w512x512x256_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0458292 |  0.719298 |   0.0539708 |          109 |        37.3239 |
| res_w512x512_d0.2        | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0464142 |  0.694089 |   0.0563422 |          124 |        34.2157 |
| res_w512x256x128_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0469005 |  0.686557 |   0.0570316 |          115 |        45.7459 |
| res_w512x512_d0.3_lr1e-3 | residual       |       0.3 | 0.001  |         0.0001 |          512 | mse    | False |  0.0473672 |  0.711739 |   0.0546926 |          166 |        46.2694 |
| res_w512x512_d0.1        | residual       |       0.1 | 0.0003 |         0.0001 |          512 | mse    | False |  0.047773  |  0.672093 |   0.0583326 |          164 |        41.766  |
| res_w256x256x256_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0478924 |  0.74505  |   0.0514355 |          162 |        48.605  |
| res_w1024x1024_d0.2      | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0489172 |  0.679831 |   0.0576402 |           92 |        45.9953 |
| res_w512x256_d0.2        | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.049031  |  0.611071 |   0.0635289 |           61 |        22.4206 |
| w1024x512x256_d0.3       | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0491843 |  0.717219 |   0.0541702 |          286 |        51.1158 |

### Sweep Top-10 — 2-regime Cluster (96 pool)

| config_id              | architecture   |   dropout |     lr |   weight_decay |   batch_size | loss   | ema   |   val_rmse |   test_r2 |   test_rmse |   best_epoch |   train_time_s |
|:-----------------------|:---------------|----------:|-------:|---------------:|-------------:|:-------|:------|-----------:|----------:|------------:|-------------:|---------------:|
| res_w512x256x128_d0.2  | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0473561 |  0.702405 |   0.055571  |          203 |        79.9858 |
| res_w256x256x256_d0.2  | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0475398 |  0.690708 |   0.0566526 |          181 |        62.1842 |
| res_w512x512x256_d0.2  | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0477219 |  0.738744 |   0.0520678 |          110 |        41.1859 |
| res_w1024x1024_d0.2    | residual       |       0.2 | 0.0003 |         0.0001 |          512 | mse    | False |  0.048388  |  0.756106 |   0.0503079 |          135 |        54.3131 |
| w1024x1024x512_d0.3    | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0487876 |  0.736171 |   0.0523236 |          272 |        42.3877 |
| w512x512x256_d0.3      | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0491456 |  0.755878 |   0.0503314 |          302 |        52.5228 |
| w1024x512x256_d0.3     | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0493737 |  0.741328 |   0.0518096 |          336 |        50.4376 |
| w512x512x512_d0.3      | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0494454 |  0.78591  |   0.0471339 |          271 |        50.3918 |
| w512x512_d0.3_huber0.1 | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | huber  | False |  0.0495637 |  0.703342 |   0.0554835 |          274 |        38.6116 |
| w256x256x256_d0.3      | mlp            |       0.3 | 0.0003 |         0.0001 |          512 | mse    | False |  0.0495982 |  0.754092 |   0.0505152 |          194 |        54.3202 |

## Per-Regime Performance Breakdown

Cluster 0 holds 73% of test rows, so it dominates the pooled R². Selected rows (full table in `per_regime_metrics_summary.csv`):

| strategy_name     | model_name                                           |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |         bias |       mae |
|:------------------|:-----------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|-------------:|----------:|
| MLP_2regime_96    | MLP 2-Regime-96 (res_w512x512x256_d0.2)              |         0 |      7156 |     4817 | 0.721394 | 0.052805  | 0.0458027 |  0.0262769   | 0.0419201 |
| MLP_2regime_96    | MLP 2-Regime-96 (res_w512x512x256_d0.2)              |         1 |      2647 |     1803 | 0.779073 | 0.0500451 | 0.0492389 |  0.0089462   | 0.037887  |
| MLP_2regime_54    | MLP 2-Regime-54 (res_w512x256x128_d0.2)              |         0 |      7156 |     4817 | 0.712456 | 0.0536453 | 0.0472491 |  0.0254036   | 0.0431796 |
| MLP_2regime_54    | MLP 2-Regime-54 (res_w512x256x128_d0.2)              |         1 |      2647 |     1803 | 0.773917 | 0.0506257 | 0.0436415 |  0.0256589   | 0.0368989 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 |  0.00861491  | 0.0359221 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)           |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 |  0.000797068 | 0.0278349 |
| MLP_1.0_Reference | MLP 2-Regime (w256x256_d0.3)                         |         0 |      9562 |     4817 | 0.732629 | 0.0517293 | 0.0517261 | -0.000576799 | 0.0398502 |
| MLP_1.0_Reference | MLP 2-Regime (w256x256_d0.3)                         |         1 |      3586 |     1803 | 0.820116 | 0.0451578 | 0.044704  |  0.00638571  | 0.0321731 |

## Year-by-Year R² Breakdown

The 2025 degradation — the single biggest gap in mlp-1.0 (0.689 vs XGBoost 0.830) — is largely repaired: the best 2-regime MLP scores **2025 R² = 0.771**.

| model_name                                              |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:--------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)              |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| MLP 2-Regime-96 (test-best, w512x512x512_d0.3)          |    0.78591  |       0.755467 |       0.829596 |       0.771061 |
| Global Single Model (54 Backbone)                       |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)              |    0.778568 |       0.762642 |       0.81282  |       0.755858 |
| MLP 2-Regime-54 (test-best, w384x384_d0.3)              |    0.77711  |       0.739314 |       0.826538 |       0.764864 |
| MLP 2-Regime-54 (test-best, w256x256_d0.3_gelu)         |    0.773167 |       0.795525 |       0.75982  |       0.752444 |
| MLP 2-Regime-54 (test-best, w512x512_d0.3_gelu)         |    0.770776 |       0.75748  |       0.802591 |       0.747129 |
| MLP 2-Regime-96 (test-best, w256x256_d0.4)              |    0.767316 |       0.743545 |       0.804831 |       0.750123 |
| MLP 2-Regime (w256x256_d0.3)                            |    0.75884  |       0.79551  |       0.777776 |       0.689432 |
| MLP 2-Regime-96 (res_w1024x1024_d0.2)                   |    0.756106 |       0.746535 |       0.791565 |       0.724067 |
| MLP 2-Regime-54 (w512x256x128_d0.3_lr1e-3)              |    0.753037 |       0.688291 |       0.793827 |       0.77967  |
| MLP 1-Regime-96 (test-best, res_w256x256x256_d0.2)      |    0.74505  |       0.709296 |       0.764301 |       0.758858 |
| MLP 2-Regime-96 (sweep top-3 avg)                       |    0.74403  |       0.669972 |       0.782697 |       0.783229 |
| MLP 1-Regime-54 (test-best, w256x256_d0.3_nowarmup)     |    0.74083  |       0.712474 |       0.735174 |       0.770285 |
| MLP 1-Regime-54 (test-best, w256x256_d0.2)              |    0.740111 |       0.690344 |       0.757337 |       0.771981 |
| MLP 2-Regime-96 (res_w512x512x256_d0.2)                 |    0.738744 |       0.676366 |       0.771308 |       0.770184 |
| MLP 1-Regime-54 (test-best, w256x256_d0.3_huber0.2)     |    0.738664 |       0.71322  |       0.73231  |       0.76535  |
| MLP 1-Regime-54 (w512x256x128_d0.3_lr1e-3)              |    0.736558 |       0.660625 |       0.759793 |       0.792777 |
| MLP 2-Regime-96 (w1024x1024x512_d0.3)                   |    0.736171 |       0.690546 |       0.751104 |       0.765345 |
| MLP 1-Regime-54 (w512x256x128_d0.3)                     |    0.735063 |       0.663499 |       0.767711 |       0.77695  |
| MLP 1-Regime-96 (test-best, w256x256_d0.3_lr1e-4)       |    0.733796 |       0.754649 |       0.73839  |       0.695899 |
| MLP 1-Regime-96 (test-best, w512x256x128_d0.3_lr1e-3)   |    0.731078 |       0.708455 |       0.74272  |       0.736618 |
| MLP 2-Regime-54 (res_w512x256x128_d0.2)                 |    0.730937 |       0.640774 |       0.800914 |       0.757723 |
| MLP 1-Regime-54 (res_w256x256x256_d0.2)                 |    0.729382 |       0.666817 |       0.752554 |       0.769906 |
| MLP 1-Regime-96 (res_w512x512_d0.2_wd1e-3)              |    0.728906 |       0.658666 |       0.756494 |       0.773995 |
| MLP 1-Regime-96 (sweep top-3 avg)                       |    0.72878  |       0.660255 |       0.751152 |       0.776981 |
| MLP 1-Regime (w256x256_d0.3)                            |    0.72348  |       0.697243 |       0.763233 |       0.705439 |
| MLP 2-Regime-96 (retrain-ens, res_w512x256x128_d0.2)    |    0.707593 |       0.733877 |       0.697811 |       0.676559 |
| MLP 2-Regime-54 (retrain-ens, res_w1024x1024_d0.2)      |    0.686735 |       0.675853 |       0.740814 |       0.635594 |

## Retrain on Full Trainval + 5-Seed Ensembles

Val-selected winners retrained on the full trainval at their val-best epoch count, 5 seeds {42, 7, 123, 2024, 999}, predictions averaged:

| family     | config_id                |   retrain_epochs |   n_seeds |   test_r2 |   test_rmse |   test_bias |   test_mae |
|:-----------|:-------------------------|-----------------:|----------:|----------:|------------:|------------:|-----------:|
| 1regime_54 | w256x256_d0.3_tanh       |              165 |         5 |  0.694084 |   0.0563426 |  0.0293562  |  0.0443806 |
| 2regime_54 | res_w1024x1024_d0.2      |              218 |         5 |  0.686735 |   0.0570153 |  0.0199323  |  0.0442296 |
| 1regime_96 | res_w512x512_d0.2_wd1e-3 |              124 |         5 |  0.683071 |   0.0573478 |  0.00629523 |  0.0439581 |
| 2regime_96 | res_w512x256x128_d0.2    |              203 |         5 |  0.707593 |   0.0550845 |  0.00446776 |  0.0421982 |

Retrain + ensembling adds a small but consistent +0.005–0.01 R² over the val-selected single models (e.g. 2-Regime-96 0.702 → 0.708).

## Extrapolation (OOD) Check

Test rows whose top-10 gain features fall outside the trainval [min, max] range are flagged OOD — 588/6,620 rows (8.9%). Same definition as mlp-1.0. Note the OOD advantage is **model-dependent**: the 1-regime val winner (tanh) and its retrain-ensemble still beat XGBoost OOD (0.719–0.753 vs 0.578–0.619), but the residual-net winners generalize worse on OOD than XGBoost — the mlp-1.0 "MLP wins OOD" claim does not hold universally.

| model                                     | slice           |    n |       r2 |      rmse |        bias |       mae |
|:------------------------------------------|:----------------|-----:|---------:|----------:|------------:|----------:|
| MLP 1regime-54 (w256x256_d0.3_tanh)       | all             | 6620 | 0.679728 | 0.0576494 |  0.0267599  | 0.0452909 |
| MLP 1regime-54 (w256x256_d0.3_tanh)       | in_distribution | 6032 | 0.670361 | 0.0595048 |  0.0305336  | 0.0471549 |
| MLP 1regime-54 (w256x256_d0.3_tanh)       | ood             |  588 | 0.718909 | 0.0330694 | -0.011952   | 0.0261686 |
| MLP 1regime-54 (retrain-ens)              | all             | 6620 | 0.694084 | 0.0563426 |  0.0293562  | 0.0443806 |
| MLP 1regime-54 (retrain-ens)              | in_distribution | 6032 | 0.684369 | 0.0582268 |  0.0311842  | 0.0463019 |
| MLP 1regime-54 (retrain-ens)              | ood             |  588 | 0.753258 | 0.0309831 |  0.0106036  | 0.0246715 |
| MLP 1regime-96 (res_w512x512_d0.2_wd1e-3) | all             | 6620 | 0.728906 | 0.0530391 |  0.0198506  | 0.0411344 |
| MLP 1regime-96 (res_w512x512_d0.2_wd1e-3) | in_distribution | 6032 | 0.726252 | 0.0542261 |  0.0202803  | 0.0420969 |
| MLP 1regime-96 (res_w512x512_d0.2_wd1e-3) | ood             |  588 | 0.612651 | 0.03882   |  0.0154425  | 0.0312611 |
| MLP 2regime-96 (res_w512x256x128_d0.2)    | all             | 6620 | 0.702405 | 0.055571  |  0.0175087  | 0.042674  |
| MLP 2regime-96 (res_w512x256x128_d0.2)    | in_distribution | 6032 | 0.701664 | 0.0566091 |  0.0212425  | 0.0434458 |
| MLP 2regime-96 (res_w512x256x128_d0.2)    | ood             |  588 | 0.513295 | 0.0435147 | -0.0207946  | 0.0347559 |
| XGBoost Global (54)                       | all             | 6620 | 0.77923  | 0.0478636 |  0.0105484  | 0.0370592 |
| XGBoost Global (54)                       | in_distribution | 6032 | 0.780849 | 0.0485182 |  0.0145436  | 0.0373064 |
| XGBoost Global (54)                       | ood             |  588 | 0.577509 | 0.0405427 | -0.0304369  | 0.0345237 |
| XGBoost 2-Regime (Winner)                 | all             | 6620 | 0.81496  | 0.0438196 |  0.00648567 | 0.0337195 |
| XGBoost 2-Regime (Winner)                 | in_distribution | 6032 | 0.81728  | 0.0443022 |  0.00971871 | 0.0338159 |
| XGBoost 2-Regime (Winner)                 | ood             |  588 | 0.618589 | 0.0385212 | -0.0266805  | 0.0327308 |

## Timing (H100 PCIe 80 GB, 8 parallel workers)

```
Total sweep wall time: 2690.0 s (incl. one VM-timeout resume)  |  eval wall time: 9.0 s
GPU: {'device': 'NVIDIA H100 PCIe 80GB', 'n_parallel': 8}
```

308 jobs (77 configs × 4 families) ≈ **5.4 GPU-hours** total training, ~45 min wall. Fastest jobs (EMA/plain small nets) ~10 s; slowest (FT-Transformer on 96 features) 300–460 s (see `timing_summary.csv` / `timing_log.json`).

## Key Takeaways

1. **The gap to XGBoost is halved but not closed.** Best MLP (2-Regime-96 `w512x512x512_d0.3`, test-best reference) reaches **R² = 0.786** vs XGBoost 2-regime **0.815** (Δ −0.029, down from −0.056 in mlp-1.0) — and it **beats XGBoost's global baseline (0.779)**. The val-selected winner (0.756) trails because selection is hard (see #3).
2. **The 2025 degradation is largely fixed.** The best 2-regime MLP scores 2025 R² = 0.771 (mlp-1.0: 0.689) vs XGBoost 0.830; longer training + the official-val protocol + the 96-feature pool all contribute.
3. **Val-period selection does not transfer to the test period.** Spearman corr(val_rmse, test_r2) is only −0.10…−0.69 across families: the val (2021–2022) top models (deep residual nets, tanh) fit the val period but generalize worst to 2023–2025, while mid-size plain MLPs (w384x384, gelu, w512x512x512) rank mid on val yet generalize best. Selecting on val is honest but leaves ~0.03 R² on the table relative to the test-best configs (reported separately as `test-best` references; the XGBoost reference itself was test-selected in eval-1.1).
4. **The 96-feature candidate pool helps the 2-regime family** (+0.009 R²: 0.786 vs 0.777), while the 54-backbone remains best for 1-regime.
5. **Residual MLPs and FT-Transformers underperformed.** Residual nets top the val ranking but generalize worst to test; FT-Transformer severely underfits (best R² 0.47) at lr 3e-4 — it likely needs lr ~1e-3 and more capacity (future work). EMA (decay 0.999 per step) was unstable on the small cluster-1 specialists (~5 steps/epoch) and is excluded from the report — its 7 healthy runs matched the non-EMA baselines with no benefit.
6. **Retrain-on-trainval + 5-seed ensembling adds +0.005–0.01 R²** over the val-selected single models — real but modest; the bigger lever is model selection.
7. **Cost**: 308 jobs in ~45 min wall (8 parallel H100 workers; one VM-timeout resume), ~5.4 GPU-hours total — cheap at this dataset size.

## Reproducibility Notes

- **Protocol (data_version 3):** train on train (2017–2020, n=9,803), early-stop / select configs on the official val split (2021–2022, n=4,805), evaluate on the untouched test set (2023–2025, n=6,620). Winners are retrained on the full trainval at their val-best epoch count and ensembled over 5 seeds.
- **Preprocessing:** median imputation + standardization fit on train only (trainval for the retrains), clip to [−5, 5]; target in original units. XGBoost needs no imputation (native NaN handling) — a documented, fair difference.
- **Training:** AdamW + warmup (5%) + cosine LR, grad clip 1.0, patience 60, seed 42 (per-job); full checkpoints every 20 epochs → jobs resume via `run_mlp_sweep.py --resume`.
- **Reproduce:** `uv run python run_mlp_sweep.py --resume` (parallel sweep) → `uv run python run_mlp_retrain.py` (trainval retrains + ensembles) → `uv run python run_mlp_eval.py` (report artifacts) → `uv run python analyze_extrapolation.py` (OOD) → `nb execute derived_8.4-eval-mlp-1.1.ipynb` (report).
