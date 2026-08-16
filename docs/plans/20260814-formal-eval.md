# Plan: `derived_8.4-formal-eval-1.0` — Statistical Evaluation for the Two-Regime Clustering Paper

## 1. Goal

Formal, publication-ready statistical evaluation of the claim established in `derived_8.4-eval-1.1` / `-1.3`:
**a two-regime (KMeans k=2) clustering model beats the single-regime global model and the trained-gating
model** on temporal (frozen split) and LOSO spatial performance.

Deliverables:
- Per-model statistics: **mean ± std, median, 95% CI, p-values** for **R², RMSE, MAE, bias** (Temporal + LOSO spatial).
- Multi-seed runs (XGBoost is cheap) under a **frozen temporal split** (no cross-validation; feature selection is too expensive to redo per fold/year).
- **Delta-selection leakage robustness**: per-regime delta features evaluated under three selection sources — *test-selected* (current), *val-selected* (new), *none* (c0=c1=0).
- **LOSO per-station pair plots** so claims like "model A beats model B on k of 7 stations" are supported, plus win counts/tests.

## 2. Configurations (14 requested + val-selected additions)

Pinned from `derived_8.4-eval-1.1/delta_grid_summary.csv` (test-selected additions) and `eval-1.3`
`loso_configurations.json` (Backbone54 rows), identical hyperparameters (xgboost 3.2.0, n_estimators 2500,
lr 0.005, depth 9, min_child_weight 8, gamma 0, λ=0.75, α=0.03, subsample 0.9, colsample 0.8, hist/cuda).

| # | config | delta source |
|---|---|---|
| 1 | Clustering_V0_Full_k2 c0=0, c1=10 | test-selected (eval-1.1 winner) |
| 2 | Clustering_V0_Full_k2 c0=0, c1=0 | none |
| 3 | Clustering_Backbone54_k2 c0=10, c1=10 | test-selected (best LOSO point) |
| 4 | Clustering_Backbone54_k2 c0=0, c1=0 | none |
| 5 | Global_Single_54 | — (global) |
| 6 | Baseline_V0_50 | — (global baseline) |
| 7 | Univariate_G_API_k2 c0=10, c1=0 | test-selected |
| 8 | Univariate_G_API_k2 c0=0, c1=0 | none |
| 9 | Clustering_Dynamic_k2 c0=10, c1=0 | test-selected |
| 10 | Clustering_Dynamic_k2 c0=0, c1=0 | none |
| 11 | Seasonal_Binary_k2 c0=0, c1=5 | test-selected |
| 12 | Seasonal_Binary_k2 c0=0, c1=0 | none |
| 13 | Trained_Gating_k2 c0=5, c1=10 | test-selected |
| 14 | Trained_Gating_k2 c0=0, c1=0 | none |

Plus **val-selected delta variants** (see §3): per-strategy val-driven winner for all 6 MoE strategies
(6 configs). Cross variant (val additions at test counts) dropped per user decision.
→ **20 configs total**.

## 3. Val-selected delta protocol (`select_deltas_val.py`)

The eval-1.1 delta ranking (`compute_delta_rankings`) ranks candidate-pool features by
`gain_rank + |spearman corr(feature, per-cluster test residual)|` — the residual term uses **test**
(leakage). Val-selected protocol re-runs the same ranking with an honest temporal holdout:

1. **Router fit on TRAIN only** (2017–2020), labels for train + val.
2. **Backbone (0,0) experts fit on TRAIN only** → predictions on VAL → per-cluster **val residuals**.
3. **Candidate-pool evidence rebuilt on val**: `residual_association` recomputed on val residuals;
   gain scores refit on TRAIN only (500-tree proxy, `fs20` proxy_params); MI/seed evidence kept from
   `derived_8.4-feature-selection-2.0` artifacts (trainval-based prior evidence); same 96-pool formula.
4. **Per-cluster delta rankings** with val residuals (same `gain_rank + corr_rank` formula as eval-1.1).
5. **9-point delta grid evaluated on VAL** (experts fit on TRAIN only, seed 42) → **val winner per strategy**
   (tie-break: RMSE, then (c0,c1) lexicographic).
6. Final test evaluation uses the standard protocol (experts on trainval) with the val-selected additions.

Outputs: `val_selected_deltas.json` (rankings, grid results, winners), `val_grid_summary.csv`.

## 4. Temporal protocol (primary; frozen split)

- Split frozen: train 2017–2020 + val 2021–2022 = 14,608 rows; test 2023–2025 = 6,620 rows (7 stations).
- **30 seeds** (user-approved; seed 42 always included as replication anchor). Per (config, seed):
  train experts on trainval, evaluate on test; save pooled / per-station / per-year / per-cluster metrics
  + predictions `.npy` (tiny, enables later re-analysis).
- **Seed scope decision**: expert (XGBRegressor) `random_state` varies; **router/KMeans/gating stay at seed 42**
  — the delta additions are tied to seed-42 cluster labels (a KMeans label flip across seeds would apply
  cluster-1 additions to the wrong regime and silently corrupt the config). State this in the paper.
- Validation: seed-42 rows must reproduce eval-1.1/eval-1.3 exactly (e.g. V0_Full (0,10) R² = 0.814960,
  Global_54 = 0.779230, Baseline_V0_50 = 0.760447; eval-1.3's full baseline already replicated to 0.000000).

## 5. LOSO spatial protocol (secondary)

- Same 20 configs × **5 seeds** (user-approved) × 7 held-out stations; per-fold router
  refit on the 6-station trainval (identical to eval-1.3, no held-out-station leakage).
- Validation: seed-42 LOSO rows must reproduce eval-1.2/eval-1.3 (e.g. Backbone54 (10,10) loso_mean_r2 = 0.6243).
- Reporting: pooled LOSO metrics + **mean and median over stations** (with the small-n=7 caveat) + per-station
  pair plots (A vs B scatter with identity line, win counts "k of 7 stations" annotated) for the headline
  comparisons, + per-station bars with seed error bars.

## 6. Statistical methodology (`eval_formal/stats.py`)

**Seed-level (fitting stochasticity — must be labeled as such in the paper):**
- Per config × metric: mean, std, median, min/max, **95% t-CI** (df = R−1).
- Pairwise A vs B: mean diff, paired **t-test**, **Wilcoxon signed-rank**, % seeds where A > B.

**Sample-level (test-set sampling variability):**
- **Paired cluster bootstrap over (station, month) blocks** (7 × 36 = 252 blocks; resample blocks with
  replacement, recompute pooled metrics; paired across models — same blocks). Percentile 95% CI + bootstrap
  p-value for differences. Rationale: i.i.d. bootstrap is invalid on autocorrelated daily soil-moisture
  series; month blocks respect within-block dependence; sensitivity check with (station, year) blocks (21).

**Multiplicity (user-approved focused family):**
- Paper p-values for: each model vs {Global_Single_54, Baseline_V0_50, Trained_Gating winner} +
  within-strategy delta ablations (test-selected vs val-selected vs none). Benjamini–Hochberg FDR
  (q < 0.05) over this family, per metric.
- All-pairwise p-value matrices also exported as CSVs (supplementary material).

**LOSO spatial:**
- Per (A, B): wins "k of 7 stations" (median across seeds per station), **two-sided sign test**
  (7/7 → p≈0.016; **6/7 → p≈0.125 — not significant at 0.05 — must state power limitation**),
  paired t-test/Wilcoxon on the 7 per-station medians (n=7, low power, descriptive).

## 7. Known leakage & how it is handled (for the paper's caveats)

| Leakage | Status |
|---|---|
| Per-regime delta selection used test residuals | Fixed: test / val / no-delta three-way ablation |
| (c0,c1) winner counts chosen on test (eval-1.1) | Val protocol re-selects counts on val |
| **54-feature backbone & V0-50 selected targeting test period** (`feature-selection-2.0` README states it explicitly) | **Not re-fixable within the frozen-split constraint** — the expensive round the team ruled out. Paper caveat: backbone selection is shared by all compared models, so relative conclusions are less affected; delta ablation partially bounds the impact. **Flag this to the user as the biggest remaining scrutiny risk.** |
| XGBoost hyperparameters from earlier test-era tuning | Caveat (shared across all models) |
| 2025 partial test coverage at some stations | Per-year table with caution note (existing convention) |
| Seed variation only covers fitting stochasticity | Explicitly stated; sampling variability covered by cluster bootstrap |

## 8. Compute & scheduling

- `run_slurm.sh` with the user-specified params:
  `--time=14:00:00 --partition=gpu --gres=gpu:h100:1 --cpus-per-task=6 --mem=16000 --nodes 1`
  (time raised from 1h; see cost estimate).
- **n_parallel = 8** worker subprocesses (user-specified). Note: XGBoost GPU folds serialize on one H100,
  so workers buy resilience/resume, not throughput (observed in eval-1.3).
- Worker format copied from eval-1.3 (driver + `run_*_worker.py`, per-job `meta.json` + `data_version`
  resume, atomic folds, `runtime.json` overrides, `--smoke` CPU mode with reduced n_estimators).
- Estimated GPU wall (per-fold ~25–35 s as measured in eval-1.3):
  - Temporal: 20 configs × 30 seeds = 600 jobs ≈ 5 h
  - Val selection grid: 6 strategies × ~11 fits ≈ 30 min
  - LOSO: 20 × 7 × 5 seeds = 700 folds ≈ 5.8 h
  - **Total ≈ 11–12 h → `--time=14:00:00`** (resume-safe; if the wall is hit, re-submit continues
    completed folds via `meta.json` + `data_version` + weight/prediction presence)

### Thorough smoke test (user decision — the GPU run is expensive)

Before submitting the real run, a `--smoke` pass (data_version=-1, n_estimators=100, CPU) must exercise
the full pipeline end-to-end so the GPU run only re-does the heavy fits:

1. `select_deltas_val.py --smoke`: val pool rebuild + per-cluster rankings + tiny val grid → val winners
   for all 6 strategies (verifies the selection code path on real data).
2. `run_temporal.py --smoke`: at least 1 job per config family (global, clustering, gating, val-winner)
   → worker, meta.json, weights, predictions, aggregation, summary CSVs.
3. `run_loso.py --smoke`: 1 fold per strategy → worker, aggregation, pair-plot inputs.
4. `eval_formal/stats.py` self-tests: paired t / Wilcoxon / cluster bootstrap / FDR / sign test on
   synthetic data with known answers (CI coverage sanity, FDR monotonicity, p-value correctness).
5. `nb execute derived_8.4-formal-eval-1.0.ipynb --uv`: the report notebook renders on smoke artifacts
   (tables + figures exist, README tables copyable).
6. Resume logic: re-run a smoke job and confirm it is skipped as completed.
- **Model weights ARE saved** (user decision) with cache-safe per-seed naming:
  `models/<config_id>__s<seed>__<station>_{meta,spec_0,spec_1,reg,gating}.json` — the seed in the
  filename prevents cross-seed collisions, and job completion (`meta.json` status) additionally requires
  the weights to exist, so a partially-written seed never counts as done. Storage ≈ 60–80 GB total
  (~58 MB/job as measured in eval-1.3; 4.7 TB free on the filesystem — fine).
- Predictions + metrics are also saved per (config, seed[, station]).

## 9. Reproducibility (repo conventions)

- New dir `notebooks/experiment/derived_8.4-formal-eval-1.0/` with `config.yaml`, drivers, workers,
  `eval_formal/` package (data, evaluator, routers, valselect, stats, plots), `run_slurm.sh`, README.
- Report notebook `derived_8.4-formal-eval-1.0.ipynb` (created via `nb create`, executed with
  `nb execute --uv` from `notebooks/`); **README tables copied verbatim from notebook stdout**; figures
  generated only by the notebook.
- Pin `loso_configurations.json` (all 20 configs) before running, as eval-1.3 does.
- No modifications to existing versioned experiments; imports reused where possible
  (`eval13.data.load_experiment_data`, eval-1.1 delta-grid CSVs, `fs20` proxy params).

## 10. User decisions (locked)

1. **Val-selected delta scope:** val winner for all 6 MoE strategies only (cross variant dropped) → **20 configs**.
2. **Seeds:** 30 temporal / 5 LOSO; `--time=14:00:00` (~11–12 h estimated wall).
3. **Statistical battery:** full — seed stats (mean±std, median, t-CI), paired t + Wilcoxon, % seeds A>B,
   paired cluster bootstrap (station, month) CI + p, FDR, LOSO win counts + sign test + paired tests.
4. **Paper comparison family:** each model vs {Global_Single_54, Baseline_V0_50, Trained_Gating winner}
   + within-strategy delta ablations; all-pairwise matrices as supplementary CSVs.
5. **Backbone/V0 selection leakage:** accepted as a paper caveat (shared by all models; delta ablation
   bounds residual impact). No new backbone selection.
6. **Model weights saved** with cache-safe per-seed naming (resume integrity + later diagnostics).
7. **Thorough smoke test** (full pipeline on CPU, reduced n_estimators, data_version=-1, stats self-tests,
   notebook execution, resume check) before the GPU submission.
