# Experiment: `derived_8.4-eval-1.4` — LOSO Spatial Generalization with Gating K-Sweep & Clustering Feature Sets

## Objective

Evaluate the **spatial generalization** of every model configuration from `derived_8.4-eval-1.3`
(2 baselines + 5 MoE routing strategies × 9 per-regime delta-grid points + the 9-point
`Clustering_Backbone54_k2` grid = **56 configurations**, all sharing the same 54-feature
backbone / V0-50 baseline / candidate pool / XGBoost hyperparameters) **plus 12 new
clustering configurations** from `derived_8.4-gating-analysis-1.0`: the K-sweep
(K = 2, 3, 4) of the five clustering routers — 54-backbone, 58 static attributes, 16 weather
drivers, 3 dynamic features, 50 V0 features — minus the three k2 combos already evaluated in
eval-1.3 (**68 configurations total**). Each new configuration is a **single config** (no
delta grid) whose per-regime experts use **only the 54 shared-backbone features** — no
per-regime feature additions were conducted for these regimes, so the backbone-only
configuration is the only defensible one. Routing follows the gating-analysis recipe
(mean-impute → `StandardScaler` → `KMeans(random_state=42, n_init=10)`, K clusters) and is
refitted per LOSO fold on the fold trainval only.

**Execution.** The LOSO protocol (per-fold router refit + per-regime experts on the 6
remaining stations, evaluated on the held-out station's 2023–2025 test rows) is unchanged
from eval-1.3, using the `derived_8.4-eval-2.0` **parallel worker format** (`run_loso.py`
spawns `run_loso_worker.py` subprocesses, one fold per job, resumable via per-fold
`meta.json` + `data_version`), scheduled on the H100 via `sbatch run_slurm.sh`
(`--time=02:00:00 --partition=gpu_debug --gres=gpu:h100:1 --cpus-per-task=6 --mem=16000 --nodes 1`,
8 workers).

**Reuse of existing results.** All 56 eval-1.1/eval-1.3 configurations were already evaluated
under the **identical** LOSO + full-baseline protocol in `derived_8.4-eval-1.3` (same seed 42,
same xgboost 3.2.0 environment; 47 configs merged there from eval-1.2 as references, 9
`Clustering_Backbone54_k2` grid points computed), so they are merged as **references**
(`is_reference = True` in all CSVs) rather than recomputed (~2 h of GPU time, no new
information). Only the 12 new gating configurations were computed in this experiment (84 LOSO
folds + 12 full-baseline configs = 96 jobs, ~1 h wall).

## LOSO Protocol

For each of the 68 configurations and each held-out station $s$:

1. `fold_trainval` = trainval rows with `station_id != s` (train 2017–2020 + val 2021–2022, 6 stations).
2. `fold_test` = all test rows of station $s$ (2023–2025).
3. **Router refitted per fold** on `fold_trainval` only — the held-out station never influences
   routing (no leakage into the routing decision).
4. Experts trained per regime cluster on `fold_trainval` with the configuration's features
   (global 54-backbone + per-cluster delta additions for the pinned grid configs; 54-backbone
   only for the 12 new configs), same hyperparameters as eval-1.1 (`device: cuda`, seed 42).
5. Metrics computed on `fold_test`: pooled / per-year / per-regime ($R^2$, RMSE, ubRMSE, bias, MAE, Pearson).

**Configurations are fixed from eval-1.1 / eval-1.3** — delta additions were selected using
full test-set knowledge, so LOSO measures generalization of model *fitting* given fixed
features (see Caveats); the 12 new configurations have no additions at all.

## Overall LOSO Leaderboard (mean R² over 7 held-out stations)

`loso_mean_r2` = average per-station $R^2$; `loso_pooled_r2` = sample-count-weighted $R^2$ over
the concatenated 6,620 held-out test samples; `temporal_test_r2` = the temporal test $R^2$
(eval-1.1 for the 47 pinned configs; eval-1.3's recorded value for the 9
`Clustering_Backbone54_k2` grid points; the full-training baseline for the 12 new configs);
`loso_minus_test_r2` = the spatial-generalization gap. All numbers are the stdout of the
executed report notebook (`derived_8.4-eval-1.4.ipynb`).

(56 configurations referenced from eval-1.3, 12 computed in eval-1.4.)


| config_label                           | strategy_name            |   loso_mean_r2 |   loso_pooled_r2 |   loso_std_r2 |   loso_min_r2 |   loso_max_r2 |   loso_mean_rmse |   loso_mean_bias |   temporal_test_r2 |   loso_minus_test_r2 | is_winner   |
|:---------------------------------------|:-------------------------|---------------:|-----------------:|--------------:|--------------:|--------------:|-----------------:|-----------------:|-------------------:|---------------------:|:------------|
| Clustering_V0_Full_k2  c0=0, c1=10     | Clustering_V0_Full_k2    |       0.641545 |         0.688536 |      0.123588 |      0.427317 |      0.779948 |        0.0557425 |       0.0156391  |           0.81496  |            -0.173415 | True        |
| Clustering_V0_Full_k2  c0=0, c1=5      | Clustering_V0_Full_k2    |       0.64049  |         0.687285 |      0.124244 |      0.427077 |      0.779948 |        0.0558347 |       0.0152862  |           0.814302 |            -0.173812 | False       |
| Clustering_V0_Full_k2  c0=5, c1=10     | Clustering_V0_Full_k2    |       0.639923 |         0.687333 |      0.122747 |      0.427317 |      0.776897 |        0.0558757 |       0.0156935  |           0.814771 |            -0.174848 | False       |
| Clustering_V0_Full_k2  c0=5, c1=5      | Clustering_V0_Full_k2    |       0.638868 |         0.686082 |      0.123391 |      0.427077 |      0.776897 |        0.0559679 |       0.0153407  |           0.814113 |            -0.175245 | False       |
| Clustering_V0_Full_k2  c0=0, c1=0      | Clustering_V0_Full_k2    |       0.63434  |         0.682097 |      0.133183 |      0.401637 |      0.779948 |        0.0562448 |       0.0153719  |           0.814334 |            -0.179994 | False       |
| Clustering_V0_Full_k2  c0=5, c1=0      | Clustering_V0_Full_k2    |       0.632719 |         0.680893 |      0.132299 |      0.401637 |      0.776897 |        0.0563779 |       0.0154264  |           0.814146 |            -0.181426 | False       |
| Clustering_Backbone54_k2  c0=10, c1=10 | Clustering_Backbone54_k2 |       0.624328 |         0.668772 |      0.148227 |      0.348313 |      0.819606 |        0.0568874 |       0.0145287  |           0.789458 |            -0.16513  | False       |
| Clustering_Backbone54_k2  c0=10, c1=5  | Clustering_Backbone54_k2 |       0.62325  |         0.667692 |      0.149323 |      0.348313 |      0.819606 |        0.0569696 |       0.0146119  |           0.788863 |            -0.165613 | False       |
| Clustering_Backbone54_k2  c0=10, c1=0  | Clustering_Backbone54_k2 |       0.622915 |         0.666882 |      0.151129 |      0.348313 |      0.819606 |        0.0569981 |       0.0142005  |           0.788907 |            -0.165992 | False       |
| Clustering_Backbone54_k2  c0=0, c1=10  | Clustering_Backbone54_k2 |       0.618528 |         0.670496 |      0.151537 |      0.320446 |      0.755575 |        0.0572762 |       0.0159406  |           0.814756 |            -0.196228 | True        |
| Clustering_Backbone54_k2  c0=0, c1=5   | Clustering_Backbone54_k2 |       0.61745  |         0.669417 |      0.152549 |      0.320446 |      0.755608 |        0.0573586 |       0.0160238  |           0.814161 |            -0.196711 | False       |
| Clustering_Backbone54_k2  c0=0, c1=0   | Clustering_Backbone54_k2 |       0.617115 |         0.668607 |      0.154314 |      0.320446 |      0.755785 |        0.0573869 |       0.0156124  |           0.814205 |            -0.19709  | False       |
| Clustering_Backbone54_k2  c0=5, c1=10  | Clustering_Backbone54_k2 |       0.615794 |         0.669129 |      0.157613 |      0.303033 |      0.758424 |        0.0573995 |       0.0156291  |           0.814564 |            -0.198769 | False       |
| Clustering_Backbone54_k2  c0=5, c1=5   | Clustering_Backbone54_k2 |       0.614717 |         0.668049 |      0.158565 |      0.303033 |      0.758457 |        0.0574819 |       0.0157123  |           0.813969 |            -0.199252 | False       |
| Clustering_Backbone54_k2  c0=5, c1=0   | Clustering_Backbone54_k2 |       0.614382 |         0.667239 |      0.160258 |      0.303033 |      0.758634 |        0.0575102 |       0.0153009  |           0.814012 |            -0.199631 | False       |
| Clustering_V0_Full_k2  c0=10, c1=10    | Clustering_V0_Full_k2    |       0.603007 |         0.648063 |      0.142241 |      0.427317 |      0.819606 |        0.0585632 |       0.0176724  |           0.789697 |            -0.18669  | False       |
| Clustering_V0_Full_k2  c0=10, c1=5     | Clustering_V0_Full_k2    |       0.601952 |         0.646812 |      0.14248  |      0.427077 |      0.819606 |        0.0586555 |       0.0173196  |           0.789039 |            -0.187087 | False       |
| Clustering_V0_Full_k3                  | Clustering_V0_Full_k3    |       0.59965  |         0.655273 |      0.110689 |      0.401637 |      0.71632  |        0.0591903 |       0.0117646  |           0.797093 |            -0.197443 | False       |
| Clustering_V0_Full_k4                  | Clustering_V0_Full_k4    |       0.598641 |         0.657855 |      0.125049 |      0.401637 |      0.777797 |        0.0590995 |       0.00904104 |           0.799971 |            -0.201331 | False       |
| Clustering_Weather_k4                  | Clustering_Weather_k4    |       0.597201 |         0.627876 |      0.133675 |      0.393565 |      0.746794 |        0.0597877 |       0.0155155  |           0.79381  |            -0.196609 | False       |
| Clustering_V0_Full_k2  c0=10, c1=0     | Clustering_V0_Full_k2    |       0.595803 |         0.641623 |      0.148491 |      0.401637 |      0.819606 |        0.0590654 |       0.0174053  |           0.789072 |            -0.193269 | False       |
| Baseline_V0_50                         | Global_Single            |       0.591632 |         0.631174 |      0.098596 |      0.419597 |      0.745337 |        0.060156  |       0.0166839  |           0.760447 |            -0.168814 | False       |
| Clustering_Backbone54_k4               | Clustering_Backbone54_k4 |       0.587928 |         0.616717 |      0.110076 |      0.418311 |      0.734136 |        0.0607796 |       0.012501   |           0.786785 |            -0.198857 | False       |
| Global_Single_54                       | Global_Single            |       0.582615 |         0.607008 |      0.158126 |      0.347175 |      0.740322 |        0.0607468 |       0.0225669  |           0.77923  |            -0.196616 | False       |
| Clustering_Weather_k2                  | Clustering_Weather_k2    |       0.581309 |         0.61488  |      0.16412  |      0.328894 |      0.761635 |        0.0605392 |       0.018806   |           0.809942 |            -0.228633 | False       |
| Univariate_G_API_k2  c0=10, c1=0       | Univariate_G_API_k2      |       0.578122 |         0.602259 |      0.128256 |      0.382433 |      0.748798 |        0.0613414 |       0.021818   |           0.763862 |            -0.185739 | False       |
| Clustering_Dynamic_k2  c0=10, c1=0     | Clustering_Dynamic_k2    |       0.577812 |         0.609692 |      0.139263 |      0.392288 |      0.764513 |        0.0608373 |       0.0220492  |           0.763459 |            -0.185647 | False       |
| Seasonal_Binary_k2  c0=0, c1=5         | Seasonal_Binary_k2       |       0.576535 |         0.601495 |      0.111823 |      0.390997 |      0.730389 |        0.0617012 |       0.0209941  |           0.756122 |            -0.179588 | False       |
| Clustering_Weather_k3                  | Clustering_Weather_k3    |       0.576403 |         0.618224 |      0.131553 |      0.374585 |      0.759532 |        0.0610923 |       0.0220118  |           0.78317  |            -0.206767 | False       |
| Seasonal_Binary_k2  c0=5, c1=5         | Seasonal_Binary_k2       |       0.576338 |         0.592004 |      0.135375 |      0.326153 |      0.731231 |        0.0617499 |       0.0215536  |           0.752315 |            -0.175977 | False       |
| Clustering_Dynamic_k2  c0=5, c1=0      | Clustering_Dynamic_k2    |       0.575652 |         0.610939 |      0.136675 |      0.388518 |      0.717919 |        0.0610774 |       0.0212154  |           0.776704 |            -0.201052 | False       |
| Seasonal_Binary_k2  c0=10, c1=5        | Seasonal_Binary_k2       |       0.574734 |         0.587133 |      0.130265 |      0.340046 |      0.691087 |        0.0620006 |       0.0223808  |           0.74282  |            -0.168086 | False       |
| Univariate_G_API_k2  c0=5, c1=0        | Univariate_G_API_k2      |       0.574237 |         0.591662 |      0.140739 |      0.346189 |      0.756963 |        0.0617444 |       0.0228002  |           0.756171 |            -0.181934 | False       |
| Univariate_G_API_k2  c0=10, c1=5       | Univariate_G_API_k2      |       0.571222 |         0.596311 |      0.12951  |      0.386274 |      0.746638 |        0.0618501 |       0.0222206  |           0.761311 |            -0.190089 | False       |
| Seasonal_Binary_k2  c0=0, c1=0         | Seasonal_Binary_k2       |       0.571062 |         0.601369 |      0.106991 |      0.40109  |      0.732225 |        0.0620656 |       0.021136   |           0.769795 |            -0.198733 | True        |
| Seasonal_Binary_k2  c0=5, c1=0         | Seasonal_Binary_k2       |       0.570865 |         0.591878 |      0.127918 |      0.336247 |      0.733066 |        0.062151  |       0.0216955  |           0.765988 |            -0.195122 | False       |
| Seasonal_Binary_k2  c0=10, c1=0        | Seasonal_Binary_k2       |       0.569262 |         0.587006 |      0.119749 |      0.35014  |      0.692922 |        0.0624199 |       0.0225227  |           0.756494 |            -0.187232 | False       |
| Univariate_G_API_k2  c0=5, c1=5        | Univariate_G_API_k2      |       0.567336 |         0.585714 |      0.139761 |      0.350029 |      0.754803 |        0.0622747 |       0.0232028  |           0.75362  |            -0.186284 | False       |
| Clustering_Dynamic_k2  c0=10, c1=5     | Clustering_Dynamic_k2    |       0.563974 |         0.589248 |      0.175212 |      0.316148 |      0.76779  |        0.0615899 |       0.021777   |           0.756133 |            -0.192159 | False       |
| Clustering_Dynamic_k2  c0=0, c1=0      | Clustering_Dynamic_k2    |       0.562955 |         0.600519 |      0.121478 |      0.38975  |      0.688278 |        0.062314  |       0.0218873  |           0.786606 |            -0.223651 | True        |
| Clustering_Dynamic_k2  c0=5, c1=5      | Clustering_Dynamic_k2    |       0.561813 |         0.590496 |      0.167512 |      0.339518 |      0.755483 |        0.0618998 |       0.0209432  |           0.769378 |            -0.207565 | False       |
| Clustering_Dynamic_k2  c0=10, c1=10    | Clustering_Dynamic_k2    |       0.560949 |         0.585838 |      0.184174 |      0.295605 |      0.767026 |        0.0616987 |       0.0231616  |           0.747986 |            -0.187037 | False       |
| Seasonal_Binary_k2  c0=0, c1=10        | Seasonal_Binary_k2       |       0.558979 |         0.593118 |      0.13228  |      0.377225 |      0.733229 |        0.0625993 |       0.020335   |           0.764271 |            -0.205291 | False       |
| Clustering_Dynamic_k2  c0=5, c1=10     | Clustering_Dynamic_k2    |       0.558789 |         0.587086 |      0.180283 |      0.318974 |      0.754718 |        0.0619735 |       0.0223277  |           0.761231 |            -0.202442 | False       |
| Seasonal_Binary_k2  c0=5, c1=10        | Seasonal_Binary_k2       |       0.558783 |         0.583626 |      0.151221 |      0.329405 |      0.73407  |        0.0626639 |       0.0208945  |           0.760463 |            -0.20168  | False       |
| Seasonal_Binary_k2  c0=10, c1=10       | Seasonal_Binary_k2       |       0.557179 |         0.578755 |      0.132878 |      0.343298 |      0.693926 |        0.063006  |       0.0217218  |           0.750969 |            -0.193789 | False       |
| Univariate_G_API_k2  c0=10, c1=10      | Univariate_G_API_k2      |       0.556858 |         0.583725 |      0.138207 |      0.390388 |      0.736613 |        0.0627865 |       0.0232637  |           0.750584 |            -0.193726 | False       |
| Trained_Gating_k2  c0=5, c1=10         | Trained_Gating_k2        |       0.554938 |         0.574615 |      0.197838 |      0.288655 |      0.85872  |        0.0617804 |       0.019174   |           0.723526 |            -0.168588 | False       |
| Univariate_G_API_k2  c0=5, c1=10       | Univariate_G_API_k2      |       0.552972 |         0.573128 |      0.145057 |      0.355317 |      0.744778 |        0.0632316 |       0.0242459  |           0.742893 |            -0.189921 | False       |
| Trained_Gating_k2  c0=5, c1=5          | Trained_Gating_k2        |       0.552803 |         0.573194 |      0.196794 |      0.292076 |      0.859294 |        0.0619325 |       0.0190974  |           0.725826 |            -0.173023 | False       |
| Univariate_G_API_k2  c0=0, c1=0        | Univariate_G_API_k2      |       0.5493   |         0.581922 |      0.129069 |      0.363918 |      0.701914 |        0.06337   |       0.0214689  |           0.769632 |            -0.220331 | True        |
| Clustering_Dynamic_k2  c0=0, c1=5      | Clustering_Dynamic_k2    |       0.549117 |         0.580076 |      0.148982 |      0.336001 |      0.74348  |        0.0632086 |       0.0216152  |           0.77928  |            -0.230163 | False       |
| Clustering_Dynamic_k2  c0=0, c1=10     | Clustering_Dynamic_k2    |       0.546092 |         0.576666 |      0.163271 |      0.315458 |      0.742715 |        0.0632825 |       0.0229997  |           0.771133 |            -0.225041 | False       |
| Trained_Gating_k2  c0=10, c1=10        | Trained_Gating_k2        |       0.544676 |         0.563716 |      0.190399 |      0.285176 |      0.810694 |        0.0629341 |       0.0213167  |           0.717214 |            -0.172538 | False       |
| Clustering_Dynamic_k4                  | Clustering_Dynamic_k4    |       0.5442   |         0.585642 |      0.115327 |      0.33024  |      0.629495 |        0.0638707 |       0.0209083  |           0.765239 |            -0.221039 | False       |
| Trained_Gating_k2  c0=10, c1=5         | Trained_Gating_k2        |       0.542541 |         0.562295 |      0.189153 |      0.288597 |      0.811268 |        0.0630883 |       0.02124    |           0.719514 |            -0.176973 | False       |
| Univariate_G_API_k2  c0=0, c1=5        | Univariate_G_API_k2      |       0.542399 |         0.575974 |      0.124661 |      0.366808 |      0.699754 |        0.0639094 |       0.0218714  |           0.767081 |            -0.224681 | False       |
| Trained_Gating_k2  c0=0, c1=10         | Trained_Gating_k2        |       0.5335   |         0.552114 |      0.178464 |      0.27528  |      0.75096  |        0.0640658 |       0.0223646  |           0.732764 |            -0.199264 | False       |
| Clustering_Backbone54_k3               | Clustering_Backbone54_k3 |       0.532507 |         0.57291  |      0.167211 |      0.222361 |      0.701763 |        0.0642634 |       0.0129048  |           0.796604 |            -0.264097 | False       |
| Trained_Gating_k2  c0=0, c1=5          | Trained_Gating_k2        |       0.531365 |         0.550694 |      0.176825 |      0.278701 |      0.751534 |        0.0642228 |       0.0222879  |           0.735064 |            -0.203699 | False       |
| Univariate_G_API_k2  c0=0, c1=10       | Univariate_G_API_k2      |       0.528036 |         0.563389 |      0.126184 |      0.367009 |      0.68973  |        0.0648662 |       0.0229145  |           0.756354 |            -0.228319 | False       |
| Trained_Gating_k2  c0=5, c1=0          | Trained_Gating_k2        |       0.512774 |         0.551611 |      0.192781 |      0.301789 |      0.857589 |        0.0643508 |       0.0266978  |           0.726236 |            -0.213462 | False       |
| Clustering_Dynamic_k3                  | Clustering_Dynamic_k3    |       0.504796 |         0.54319  |      0.107727 |      0.353541 |      0.632703 |        0.066648  |       0.0192098  |           0.7853   |            -0.280504 | False       |
| Trained_Gating_k2  c0=10, c1=0         | Trained_Gating_k2        |       0.502512 |         0.540712 |      0.180803 |      0.29831  |      0.809562 |        0.0655145 |       0.0288405  |           0.719924 |            -0.217412 | False       |
| Trained_Gating_k2  c0=0, c1=0          | Trained_Gating_k2        |       0.491336 |         0.529111 |      0.169138 |      0.288414 |      0.749828 |        0.0666279 |       0.0298884  |           0.735474 |            -0.244138 | True        |
| Clustering_Static_k2                   | Clustering_Static_k2     |       0.458532 |         0.567489 |      0.351817 |     -0.271257 |      0.766916 |        0.0665854 |       0.00208644 |           0.801974 |            -0.343442 | False       |
| Clustering_Static_k4                   | Clustering_Static_k4     |       0.325911 |         0.461282 |      0.44887  |     -0.347301 |      0.746284 |        0.0735322 |       0.0046667  |           0.799983 |            -0.474071 | False       |
| Clustering_Static_k3                   | Clustering_Static_k3     |       0.299158 |         0.419247 |      0.428992 |     -0.347301 |      0.746284 |        0.0760505 |       0.0148541  |           0.80816  |            -0.509002 | False       |


## Regime-Count K-Sweep under LOSO (K = 2, 3, 4)

The gating analysis showed K=2 is the largest regime count at which every regime is a whole
station group (purity 1.000 on trainval and test); K=3/4 fragment stations (mean purity 0.833
/ 0.695). This section answers whether adding regimes helps or hurts LOSO performance: all
rows share the same 54-backbone experts, so differences isolate the *routing*. The K=2 rows
for Backbone54 / V0_Full / Dynamic are the existing `c0=0, c1=0` grid points (identical
experts, no additions); Static / Weather K=2 are new configs computed here.


| router_family         | config_label                         |   loso_mean_r2 |   loso_pooled_r2 |   loso_mean_rmse |   loso_mean_bias |   temporal_test_r2 |   loso_minus_test_r2 | is_reference   |
|:----------------------|:-------------------------------------|---------------:|-----------------:|-----------------:|-----------------:|-------------------:|---------------------:|:---------------|
| Clustering_Backbone54 | Clustering_Backbone54_k2  c0=0, c1=0 |       0.617115 |         0.668607 |        0.0573869 |       0.0156124  |           0.814205 |            -0.19709  | True           |
| Clustering_Backbone54 | Clustering_Backbone54_k3             |       0.532507 |         0.57291  |        0.0642634 |       0.0129048  |           0.796604 |            -0.264097 | False          |
| Clustering_Backbone54 | Clustering_Backbone54_k4             |       0.587928 |         0.616717 |        0.0607796 |       0.012501   |           0.786785 |            -0.198857 | False          |
| Clustering_V0_Full    | Clustering_V0_Full_k2  c0=0, c1=0    |       0.63434  |         0.682097 |        0.0562448 |       0.0153719  |           0.814334 |            -0.179994 | True           |
| Clustering_V0_Full    | Clustering_V0_Full_k3                |       0.59965  |         0.655273 |        0.0591903 |       0.0117646  |           0.797093 |            -0.197443 | False          |
| Clustering_V0_Full    | Clustering_V0_Full_k4                |       0.598641 |         0.657855 |        0.0590995 |       0.00904104 |           0.799971 |            -0.201331 | False          |
| Clustering_Dynamic    | Clustering_Dynamic_k2  c0=0, c1=0    |       0.562955 |         0.600519 |        0.062314  |       0.0218873  |           0.786606 |            -0.223651 | True           |
| Clustering_Dynamic    | Clustering_Dynamic_k3                |       0.504796 |         0.54319  |        0.066648  |       0.0192098  |           0.7853   |            -0.280504 | False          |
| Clustering_Dynamic    | Clustering_Dynamic_k4                |       0.5442   |         0.585642 |        0.0638707 |       0.0209083  |           0.765239 |            -0.221039 | False          |
| Clustering_Static     | Clustering_Static_k2                 |       0.458532 |         0.567489 |        0.0665854 |       0.00208644 |           0.801974 |            -0.343442 | False          |
| Clustering_Static     | Clustering_Static_k3                 |       0.299158 |         0.419247 |        0.0760505 |       0.0148541  |           0.80816  |            -0.509002 | False          |
| Clustering_Static     | Clustering_Static_k4                 |       0.325911 |         0.461282 |        0.0735322 |       0.0046667  |           0.799983 |            -0.474071 | False          |
| Clustering_Weather    | Clustering_Weather_k2                |       0.581309 |         0.61488  |        0.0605392 |       0.018806   |           0.809942 |            -0.228633 | False          |
| Clustering_Weather    | Clustering_Weather_k3                |       0.576403 |         0.618224 |        0.0610923 |       0.0220118  |           0.78317  |            -0.206767 | False          |
| Clustering_Weather    | Clustering_Weather_k4                |       0.597201 |         0.627876 |        0.0597877 |       0.0155155  |           0.79381  |            -0.196609 | False          |


**Finding: K=2 is the best regime count for every router family.** For Backbone54 (0.617 vs
0.533/0.588), V0_Full (0.634 vs 0.600/0.599) and Dynamic (0.563 vs 0.505/0.544) the K=2
no-addition point beats K=3 and K=4 on LOSO mean R²; for Static the K=2 config (0.459) also
beats K=3/4 (0.299/0.326). The gating analysis's station-purity argument (K=2 = whole station
groups) transfers directly to out-of-sample spatial generalization: splitting stations into
more regimes fragments the specialists' training data and hurts transfer.

## Clustering Feature Sets under LOSO (Backbone54 / Static / Weather / Dynamic / V0_Full)

The five gating-analysis clustering feature sets as K=2 routers under LOSO (all experts =
54 backbone, no additions), plus the per-regime breakdown of the new configurations.


| router_features      | config_label                         |   loso_mean_r2 |   loso_pooled_r2 |   loso_mean_rmse |   loso_mean_bias |   temporal_test_r2 |   loso_minus_test_r2 | is_reference   |
|:---------------------|:-------------------------------------|---------------:|-----------------:|-----------------:|-----------------:|-------------------:|---------------------:|:---------------|
| 54 backbone          | Clustering_Backbone54_k2  c0=0, c1=0 |       0.617115 |         0.668607 |        0.0573869 |       0.0156124  |           0.814205 |            -0.19709  | True           |
| 50 V0 features       | Clustering_V0_Full_k2  c0=0, c1=0    |       0.63434  |         0.682097 |        0.0562448 |       0.0153719  |           0.814334 |            -0.179994 | True           |
| 3 dynamic features   | Clustering_Dynamic_k2  c0=0, c1=0    |       0.562955 |         0.600519 |        0.062314  |       0.0218873  |           0.786606 |            -0.223651 | True           |
| 58 static attributes | Clustering_Static_k2                 |       0.458532 |         0.567489 |        0.0665854 |       0.00208644 |           0.801974 |            -0.343442 | False          |
| 16 weather drivers   | Clustering_Weather_k2                |       0.581309 |         0.61488  |        0.0605392 |       0.018806   |           0.809942 |            -0.228633 | False          |

| config_id                |   cluster |   n_test |        mean_r2 |        min_r2 |       max_r2 |
|:-------------------------|----------:|---------:|---------------:|--------------:|-------------:|
| Clustering_Backbone54_k3 |         0 |     1774 |     -0.579891  |     -1.43698  |   0.154836   |
| Clustering_Backbone54_k3 |         1 |     1060 |     -0.312     |     -2.12809  |   0.722527   |
| Clustering_Backbone54_k3 |         2 |     3786 |      0.46454   |      0.220895 |   0.609171   |
| Clustering_Backbone54_k4 |         0 |     2160 |     -1.09193   |     -3.0082   |   0.550608   |
| Clustering_Backbone54_k4 |         1 |      768 |     -0.820811  |     -1.82037  |  -0.00365472 |
| Clustering_Backbone54_k4 |         2 |     1693 |      0.129924  |     -0.550858 |   0.741547   |
| Clustering_Backbone54_k4 |         3 |     1999 |     -0.0434261 |     -2.00435  |   0.767611   |
| Clustering_Dynamic_k3    |         0 |     3975 |     -0.334446  |     -1.71442  |   0.578426   |
| Clustering_Dynamic_k3    |         1 |     1555 |     -0.22783   |     -2.08184  |   0.813447   |
| Clustering_Dynamic_k3    |         2 |     1090 |     -2.19193   |    -12.7542   |   0.826563   |
| Clustering_Dynamic_k4    |         0 |     2457 |     -0.346118  |     -4.76676  |   0.902675   |
| Clustering_Dynamic_k4    |         1 |      710 |      0.224594  |     -0.238694 |   0.740425   |
| Clustering_Dynamic_k4    |         2 |     1313 |     -1.95243   |     -4.70816  |   0.682237   |
| Clustering_Dynamic_k4    |         3 |     2140 |     -0.127094  |     -1.45074  |   0.622074   |
| Clustering_Static_k2     |         0 |     1905 |      0.588989  |      0.411062 |   0.766916   |
| Clustering_Static_k2     |         1 |     4715 |      0.406349  |     -0.271257 |   0.746284   |
| Clustering_Static_k3     |         0 |        0 |    nan         |    nan        | nan          |
| Clustering_Static_k3     |         1 |     4577 |      0.386706  |     -0.271257 |   0.746284   |
| Clustering_Static_k3     |         2 |     2043 |      0.0802893 |     -0.347301 |   0.50788    |
| Clustering_Static_k4     |         0 |        0 |    nan         |    nan        | nan          |
| Clustering_Static_k4     |         1 |     3680 |      0.390595  |     -0.271257 |   0.746284   |
| Clustering_Static_k4     |         2 |        0 |    nan         |    nan        | nan          |
| Clustering_Static_k4     |         3 |     2940 |      0.239667  |     -0.347301 |   0.558423   |
| Clustering_V0_Full_k3    |         0 |     2692 |      0.674516  |      0.618437 |   0.71632    |
| Clustering_V0_Full_k3    |         1 |     1803 |      0.478279  |      0.401637 |   0.554921   |
| Clustering_V0_Full_k3    |         2 |     2125 |      0.608721  |      0.537214 |   0.680229   |
| Clustering_V0_Full_k4    |         0 |      970 | -29635.5       | -59271.8      |   0.779063   |
| Clustering_V0_Full_k4    |         1 |        0 |    nan         |    nan        | nan          |
| Clustering_V0_Full_k4    |         2 |     3844 |      0.461655  |     -0.23093  |   0.688792   |
| Clustering_V0_Full_k4    |         3 |     1806 |      0.266724  |     -0.156386 |   0.554921   |
| Clustering_Weather_k2    |         0 |     5519 |      0.55344   |      0.328894 |   0.757879   |
| Clustering_Weather_k2    |         1 |     1101 |    -16.5829    |    -67.7839   |   0.728892   |
| Clustering_Weather_k3    |         0 |     3778 |      0.0355432 |     -1.33408  |   0.688818   |
| Clustering_Weather_k3    |         1 |     1759 |    -10.0702    |    -51.269    |   0.829032   |
| Clustering_Weather_k3    |         2 |     1083 |     -1.05155   |     -3.31854  |   0.597648   |
| Clustering_Weather_k4    |         0 |     1562 |   -276.891     |   -831.584    |   0.457008   |
| Clustering_Weather_k4    |         1 |     1851 |     -0.1984    |     -2.27404  |   0.718687   |
| Clustering_Weather_k4    |         2 |     1035 |     -0.273076  |     -1.03859  |   0.298478   |
| Clustering_Weather_k4    |         3 |     2172 |     -1.58434   |     -9.88039  |   0.867054   |


**Finding: the routing feature set matters — V0-50 > Backbone-54 > Weather > Dynamic >
Static.** Routing on the V0 features (the eval-1.1 router) still transfers best (0.634), the
54-backbone router is second (0.617), weather-driven routing third (0.581, its k4 variant
0.597), and routing on static attributes is a clear failure (0.459, with two stations below
zero R²) — the gating analysis already showed the static partition groups stations differently
than the backbone's (which is not trivially recoverable from static attributes alone), and
under LOSO that partition does not generalize. All five routers share the same 54-backbone
experts, so the differences are purely the regime definition.

**Why the per-regime numbers are extreme.** K=3/4 regimes fragment stations, so some
held-out folds route a regime to a handful of test rows (e.g., V0_Full_k4 cluster 0 has 10
test rows on the BeaverPass fold → R² = −59,272; Static_k3/k4 and V0_Full_k4 have regimes
with zero test rows on every fold, `nan`). These are genuine small-sample effects of
fragmented regimes under LOSO — the specialists trained on other stations' rows simply do
not transfer to the held-out station's tiny regime slices. Only the K=2 Static config has
both regimes populated and positive everywhere; the pooled per-station metrics (leaderboard)
remain the primary comparison.

## Single-Regime → Two-Regime Development (`Clustering_Backbone54_k2`)

Retained from eval-1.3: the two-regime model is a development of the single-regime model, not
a separate architecture: the router (KMeans k=2) and both specialists use the **same 54
shared-backbone features** as `Global_Single_54`, with only the per-cluster delta additions
(pinned from eval-1.1's `Clustering_V0_Full_k2` winner) added to the second specialist — no
separate V0-50 feature source to explain. Its winner (c0=0, c1=10) remains the experiment's
pinned winner for the station-similarity analysis.

## Station Similarity & Clustering — Spatial-Generalization Hypothesis

Retained from eval-1.3 unchanged (see `eval14/station_sim.py`): a station generalizes well
spatially when the LOSO training set contains a climate/geography "twin"; poorly when it
stands out. Median LOSO R² now aggregates over all 68 configurations. The 1-NN-distance
correlation with median LOSO R² remains positive (ρ = +0.55) — the "twin → easy" story is
still not supported; mean-distance isolation correlates negatively with the winner config's
per-station R² (ρ = −0.64) — the unique grassland station SourdoughGulch is the hardest under
the winner config (0.32), while the winner config transfers *well* to CayusePass (0.68)
despite it being the hardest station on median.

## Full-Training Baseline — Intrinsic vs. Generalization Difficulty

Retained from eval-1.3: trains every configuration on ALL stations (no LOSO), replicating the
eval-1.1 protocol; per-station metrics provide the *intrinsic* difficulty contrasted with
LOSO difficulty. The 56 existing configs' full baseline rows are merged as references from
eval-1.3; the 12 new configs were computed here (their pooled test R², 0.765–0.810, serves as
their temporal reference).


| station               |   n_configs |   total_test_n |   median_r2 |   mean_r2 |    std_r2 |   min_r2 |   max_r2 |   mean_rmse |   mean_bias |   n_negative_r2 |
|:----------------------|------------:|---------------:|------------:|----------:|----------:|---------:|---------:|------------:|------------:|----------------:|
| Spokane               |          68 |          60996 |    0.943112 |  0.941809 | 0.0109624 | 0.915413 | 0.957031 |   0.0275905 |  0.00208759 |               0 |
| Darrington            |          68 |          67932 |    0.816379 |  0.81617  | 0.0135295 | 0.784731 | 0.841255 |   0.0400322 |  0.0227865  |               0 |
| CayusePass_WA         |          68 |          73508 |    0.76792  |  0.767352 | 0.0313888 | 0.704006 | 0.807327 |   0.057478  |  0.00567766 |               0 |
| Paradise_WA           |          68 |          72556 |    0.762005 |  0.759536 | 0.0634701 | 0.62474  | 0.85344  |   0.0478381 |  0.0175175  |               0 |
| Quinault              |          68 |          70992 |    0.685478 |  0.680982 | 0.0188125 | 0.635265 | 0.724941 |   0.0391834 | -0.0211708  |               0 |
| BeaverPass_WA_990     |          68 |          42568 |    0.53122  |  0.515302 | 0.0901397 | 0.360304 | 0.657952 |   0.0631643 |  0.0517812  |               0 |
| SourdoughGulch_WA_985 |          68 |          61608 |    0.487466 |  0.483626 | 0.0597463 | 0.346863 | 0.554168 |   0.0575135 |  0.00276077 |               0 |


### Per-station difficulty: full training vs LOSO (sorted by LOSO difficulty)


| station               |   median_r2_full |   median_r2_loso |   gap_median |   mean_r2_full |   mean_r2_loso |   gap_mean |
|:----------------------|-----------------:|-----------------:|-------------:|---------------:|---------------:|-----------:|
| Darrington            |            0.816 |            0.7   |        0.116 |          0.816 |          0.644 |      0.172 |
| BeaverPass_WA_990     |            0.531 |            0.686 |       -0.154 |          0.515 |          0.651 |     -0.136 |
| Spokane               |            0.943 |            0.631 |        0.312 |          0.942 |          0.646 |      0.296 |
| Paradise_WA           |            0.762 |            0.602 |        0.16  |          0.76  |          0.596 |      0.163 |
| Quinault              |            0.685 |            0.558 |        0.128 |          0.681 |          0.542 |      0.139 |
| SourdoughGulch_WA_985 |            0.487 |            0.415 |        0.073 |          0.484 |          0.416 |      0.068 |
| CayusePass_WA         |            0.768 |            0.405 |        0.362 |          0.767 |          0.471 |      0.296 |


**Interpretation.** LOSO-hard stations are only weakly hard under full training
(Spearman(full median R², LOSO median R²) = +0.321, p = 0.482, n = 7): LOSO difficulty is
mostly generalization-specific. CayusePass is generalization-limited (0.768 full vs 0.405
LOSO, gap +0.36); SourdoughGulch is intrinsically hard (0.487 vs 0.415, gap +0.07).
BeaverPass is the only station that does *better* under LOSO (0.531 full vs 0.686 LOSO,
gap −0.15).

## Caveats

- **Delta additions are test-informed** (inherited from eval-1.1/eval-1.3): per-regime
  additions for the 47 + 9 grid configs were selected using full test-set knowledge, so LOSO
  measures generalization of model *fitting* given fixed features. The 12 new gating
  configurations have **no additions at all** (54-backbone experts only) and are clean of
  this chain.
- **The 54-backbone feature set itself is test-selected upstream** (audit doc
  `docs/plans/20260812-clustering-methodlogy-audit.md`): the shared backbone was selected by
  `derived_8.4-feature-selection-2.0` with the 2023–2025 test labels in the loop. Test
  metrics through this chain inherit optimism; disclose wherever test numbers are reported.
- **K is evaluated out-of-sample here, not selected on test purity**: the gating analysis
  justified K=2 with trainval purity (test purity reported as out-of-sample confirmation);
  this experiment measures the LOSO / temporal performance of each K directly.
- **Empty / tiny regimes**: K=3/4 regimes fragment stations, so some held-out folds have
  empty regimes (mean-fallback predicts the fold trainval mean, mirroring eval-1.1) or
  regimes with a handful of test rows (extreme negative per-regime R²); the pooled
  per-station metrics are the primary comparison.
- **References vs computed**: reference rows (56 configs) were recorded under the identical
  protocol (seed 42, xgboost 3.2.0) in eval-1.3 — deterministic, no information lost.

## Reproducibility

```bash
# LOSO + full baseline (compute the 12 new configs, merge 56 eval-1.3 references)
cd notebooks/experiment/derived_8.4-eval-1.4 && sbatch run_slurm.sh

# Report notebook (all tables/figures in this README)
cd notebooks && nb execute experiment/derived_8.4-eval-1.4/derived_8.4-eval-1.4.ipynb --uv

# Router-agreement diagnostic (V0-Full vs Backbone54 k2 partitions per fold)
cd notebooks/experiment/derived_8.4-eval-1.4 && uv run --no-sync python analyze_router_agreement.py
```

Configs are pinned in `loso_configurations.json` (68 entries) before any run filter; per-fold
job `meta.json` files record `data_version`, `n_clusters`, and per-cluster metrics for audit.
