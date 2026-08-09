# Plan: `derived_8.4-eval-2.0` — MLP Leave-One-Station-Out (LOSO) Spatial Generalization

## Objective

Continuation of `derived_8.4-eval-1.2` (same output format: LOSO protocol +
full-training baseline + station-similarity analysis) that evaluates the
**MLP models** whose hyperparameters were established in
`derived_8.4-eval-mlp-1.3` (2-regime) and `derived_8.4-eval-mlp-1.1`
(1-regime) under **leave-one-station-out** across the 7 WA stations of the
`derived_8.4` split — testing the **spatial-generalization capability of MLPs**
(the mlp-1.3 finding: MLPs extrapolate smoothly in feature space, OOD R2 0.75
vs XGBoost 0.62, motivating the hypothesis that they generalize to unseen
stations better than trees).

Scope (confirmed with user): **one regime (global single MLP) AND two regimes**,
with only the **best clustering strategy** (`Clustering_V0_Full_k2`, c0=0,
c1=10). 6 pinned MLP configs = val-selected winners + mlp-1.3 findings, at
per-family seeds matching the source experiments (2-regime: {42,7} from mlp-1.3;
1-regime: {42} from mlp-1.1) so the full-baseline replication check is exact.

## Configurations (6 MLP, fixed per fold)

| family | config_id | structure | features | selection status | temporal test R2 |
|--------|-----------|-----------|----------|------------------|------------------|
| 1regime_54 | w256x256_d0.3_tanh | global | backbone_54 | mlp-1.1 val winner | 0.680 |
| 1regime_96 | res_w512x512_d0.2_wd1e-3 | global | candidate_pool_96 | mlp-1.1 val winner | 0.729 |
| 2regime_54 | w512x512x512_d0.3_huber0.1 | cluster (c0=54, c1=54+10) | backbone_54 | mlp-1.3 val winner | 0.765 |
| 2regime_54 | w448x448_d0.3_gelu | cluster | backbone_54 | mlp-1.3 finding (near-zero bias) | 0.781 |
| 2regime_54 | w384x384_d0.3_gelu | cluster | backbone_54 | mlp-1.3 test-best (reference only) | 0.789 |
| 2regime_96 | w512x512x512_d0.3_lr1e-3 | cluster (c0=96, c1=96+10) | candidate_pool_96 | mlp-1.3 val winner | 0.761 |

Plus the XGBoost LOSO references from eval-1.2 (`Global_Single_54`,
`Clustering_V0_Full_k2_c0_0_c1_10`) merged into the leaderboard (no retraining).

## Protocol

LOSO fold for held-out station s (MLP-adapted from eval-1.2):
1. `fold_train` = train rows with station != s (2017-2020, 6 stations);
   `fold_val` = val rows with station != s (2021-2022, 6 stations);
   `fold_test` = all test rows of station s (2023-2025).
2. Router refit per fold on `fold_trainval` only (GlobalSingle / V0Full KMeans
   k=2 on 50 V0 feats, seed 42) — no held-out-station leakage into routing.
3. MLP specialists trained per cluster on fold_train, early-stopped on fold_val
   (mlp-1.3 trainer, patience 60, aux2020 diagnostic), predict fold_test.
   Empty fold-train cluster -> fold-train-mean fallback (eval-1.2 behavior).
4. Metrics on fold_test: pooled / per-year / per-regime.

Full-training baseline (`run_full_baseline.py`): same configs trained on ALL 7
stations (mlp-1.3/1.1 temporal protocol) -> per-station intrinsic difficulty;
pooled test R2 validated against the mlp-1.3/1.1 metrics_summary.csv
(deterministic -> expect |diff| ~ 0).

## Deliverables (new files under `notebooks/experiment/derived_8.4-eval-2.0/`)

- `config.yaml` (data + 4 families + 6 pinned configs + family seeds + loso section)
- `eval20/` package (data.py fold tensors, evaluator.py per-fold training,
  plots.py reuse of eval12.plots with MLP colors, references.py temporal refs)
- `run_loso.py` / `run_loso_worker.py` (parallel, resumable, eval-1.2-format CSVs)
- `run_full_baseline.py` (full-training baseline + replication check)
- `derived_8.4-eval-2.0.ipynb` (report notebook mirroring eval-1.2's structure)
- `README.md` (tables from executed-notebook stdout)
- CSVs: `loso_config_summary.csv`, `loso_station_summary.csv`,
  `loso_per_config_station.csv`, `loso_per_regime_metrics.csv`,
  `loso_per_year_metrics.csv`, `full_*.csv`, `loso_configurations.json`
- Figures: `loso_r2_*.png` (4), `station_*.png` (5), `full_*.png` (3)
- `models/`, `predictions/`, `predictions_full/` (gitignored, regenerable)

## Execution status

1. Scaffold + config + eval20 + drivers: done.
2. Smoke tests (global + 2-regime cluster paths, resume, figures): passed.
3. Full LOSO run (70 jobs, 8 workers): **completed 2026-08-08** (~6 min wall).
   Leaderboard: MLP 2-Regime-96 winner LOSO pooled R² 0.668 (vs XGBoost 2-regime 0.689,
   XGBoost global 0.607); MLP 1-Regime-54 pooled 0.610 (beats XGBoost global single);
   54-family 2-regime MLPs transfer worse (0.51-0.59 pooled); CayusePass no longer
   generalization-limited under the MLPs (gap +0.05 vs +0.40 for XGBoost).
4. Full-baseline run (10 jobs): **completed** — 2-regime configs replicate mlp-1.3 pooled
   test R² bit-identically (|diff| = 0); 1-regime drift vs mlp-1.1 documented as
   torch-version environment drift (mlp-1.1 ran on an earlier torch).
5. Report notebook executed cleanly (`nb execute --uv` from `notebooks/`, 0 errors);
   README.md tables populated from executed-notebook stdout.
6. Final `nb execute` verification passed; no commit made.

## Compute estimate

- LOSO: 1regime_54 (7) + 1regime_96 (7) + 2regime_54 (3x14) + 2regime_96 (14)
  = 70 jobs ~30-120s each -> ~1.5-2.0 GPU-h, wall ~15-30 min at 8 workers.
- Full baseline: 10 jobs -> ~0.15 GPU-h.
- Smoke tests negligible (3 epochs each).
