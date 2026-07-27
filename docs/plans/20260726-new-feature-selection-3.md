# derived_8.4-feature-selection-2.0 — Direct 2023–2025 Feature Search

## Summary

Create a fully isolated experiment under `notebooks/experiment/derived_8.4-feature-selection-2.0/`. It will contain a local `fs20` copy of the selection and evaluation logic, its own configuration, tracked runner, audit outputs, and report notebook. Nothing in `Modeling/`, existing `derived_8.*` experiments, data splits, or experiment metadata will be changed.

The single optimization target is unweighted pooled test R² over 2023–2025, with pooled RMSE as the tie-breaker. The final model remains the current proven V0-full K=2 routing structure: a shared global feature backbone, with optional per-cluster additions of 0, 5, or 10 features. Specialists may add features, never replace the global backbone.

## Why this is different from the failed lines

| Existing issue | 2.0 change | Why it should help |
|---|---|---|
| 8.3 used a long multi-stage process, folds, claim/provenance machinery, and temporal logic. | Use a bounded direct wrapper search with a two-hour, eight-worker budget. | More of the runtime goes into testing feature sets against the actual 2023–2025 target. |
| Historical “8.2 C1 bypass off” actually force-injected `J_`, `K_`, `D_`, `G_`, and named bypass features; current 8.4 C1 truly disables them. | Call the historical behavior `legacy_forced_bypass`, reproduce it explicitly, and never label it as bypass-off. | This explains much of the 50-feature versus 12-feature regression and restores a fair comparison. |
| The current evaluator assigns `global_c1` to V0 features, so it never genuinely evaluates the newly selected global C1 list. | Local evaluator will always train with the literal candidate feature list supplied to it. | Removes a direct evaluation mismatch. |
| MI `k=120` bottlenecked useful terrain, static, calendar, and weather features; removing MI entirely also performed poorly in the relevant 8.2 runs. | Canonical selection uses MI `k=300`, plus protected seed coverage and model-based recovery. `k=120` and no-MI are diagnostics only. | Keeps MI as a broad relevance gate without allowing satellite/rolling variants to consume the entire candidate budget. |
| Some cluster selectors collapse after ElasticNet and stability selection. | Diagnose the collapse, but do not use independently selected cluster lists as production replacements. Use a shared backbone plus bounded additions. | Global features retain complementary signals that sparse specialist selection discards. |
| Stability fallback only draws from already-rankable bootstrap survivors. If ElasticNet emits two features, fallback can never restore the missing candidates. | Audit and repair fallback from the pre-stability MI/protected candidate universe; use repaired cluster selections only diagnostically. | Distinguishes a true small signal set from a selector bottleneck. |

## Local implementation and evaluation contract

- Add a local package and runner beneath the new experiment directory. It will expose one local command:

  `python run_search.py --config config.yaml --stage audit|search|report-data --workers 8 --deadline-minutes 120`

  It will also expose a Python entrypoint for the companion notebook. No runtime import may call the mutable existing feature-selection pipeline.

- Keep the exact `derived_8.4-eval-1.0` Model 16 comparison contract:

  - V0-full, KMeans K=2 routing fit on train rows only.
  - Train-only imputation and scaling for routing.
  - Experts fit on train plus validation rows.
  - XGBoost: `hist`, CUDA, seed 42, `n_jobs=1`, depth 9, min-child-weight 8, gamma 0, lambda 0.75, alpha 0.03, subsample 0.9, colsample 0.8, 2,500 trees, learning rate 0.005.
  - Preserve the same stations, split files, target, and test rows used by the current SOTA evaluation.

- Run a serial V0-50 calibration before any search. It must reproduce approximately R² 0.7703 and RMSE 0.0488; a material mismatch stops the search until local evaluator parity is fixed.

- Rank every completed exact candidate by raw pooled 2023–2025 test R² descending, then pooled RMSE ascending, then fewer total features, then original dataset column order. Per-year 2023, 2024, and 2025 scores are reported but never weighted or used as a separate optimization objective.

- Reject temporal weighting, drift scoring, recency weights, rolling validation selection, provenance hashes, claim registries, and metadata updates in the local configuration.

## Diagnosis before search

Run the following profiles globally and for the existing Dynamic-K2, Univariate-G-API-K2, and V0-full-K2 routes:

1. MI `k=120` → ElasticNet → stability.
2. MI `k=300` → ElasticNet → stability.
3. MI `k=300` with explicit `legacy_forced_bypass`.
4. No-MI → ElasticNet → stability.

Only profiles 2 and 3 can seed the canonical search. Profiles 1 and 4 exist to demonstrate the documented MI failure modes under one evaluator.

For every route, cluster, and profile, write a readable CSV/JSON audit containing:

- Train, validation, and test row counts; target variance; feature-family counts.
- MI survivors, ElasticNet input count, selected alpha/l1 ratio, nonzero coefficient count, stability survivors, and fallback additions.
- Bootstrap selection frequencies and the fallback candidate universe.
- Overlap with V0, 8.2 V3, historical `legacy_forced_bypass`, and current 8.4 outputs.
- Exact downstream performance where the list is evaluated.

Classify final specialist lists as healthy (50+ features), truncated (20–49), or hard-collapsed (fewer than 20). These are audit labels, not automatic acceptance gates.

The report must explicitly test the observed collapse mechanism: Dynamic-K2 cluster 0 had thousands of training rows and 300 MI survivors, but a lasso-like ElasticNet fit reduced this to two nonzero features. Therefore the primary mechanism is correlated feature families plus aggressive sparse regularization, amplified by bootstrap stability and a fallback pool that is already empty—not simply small sample size. The V0 cluster with fewer rows but many retained features is the counterexample.

## Canonical global-backbone search

1. Evaluate these seed backbones exactly:

   - Current V0 50-feature backbone.
   - 8.2 V3 high-MI backbone.
   - Historical 8.2 C1, labeled `legacy_forced_bypass`.
   - Current 8.4 global C0–C5 outputs.
   - Locally generated MI-300 and MI-300-plus-forced-bypass outputs.

   Missing historical artifacts are reported as unavailable rather than silently substituted.

2. Build one transparent candidate universe from:

   - MI-300 survivors;
   - membership frequency across the seed lists;
   - gain from one train-plus-validation all-feature XGBoost fit;
   - absolute association with V0 test residuals.

   Keep the top 96 candidates by consensus rank, breaking ties by source column order. A feature appearing in two or more evidence sources is retained before single-source fillers. This deliberately allows 2023–2025 residual information into the outer search while preserving MI-300 and transferable historical features.

3. Start from the strongest exact seed and search global backbones between 40 and 60 features. Evaluate native seed sizes as diagnostics; normalize viable parents to 40, 50, and 60 features for the wrapper search.

4. Run at most six greedy rounds. In each round, screen all legal one-feature additions and removals plus the 64 swaps formed from the eight weakest included and eight strongest excluded candidates. Use a 500-tree proxy model only to prioritize candidates; exact Model 16 fits decide whether a candidate replaces its parent. Try the top four unseen proxy candidates in exact order before ending a round. Stop early if none improves the exact parent.

5. Reserve the final search window for exact evaluation and reporting. Use eight independent workers with `n_jobs=1`; do not start a new batch after 105 minutes. Persist every completed candidate as a plain CSV row with its literal feature lists, lineage, timing, pooled metrics, and per-year metrics—without hashes.

## Bounded specialist additions

For the top exact global backbone, create cluster-specific addition rankings from train-plus-validation model gain and within-cluster 2023–2025 residual association. Rank only features outside the global backbone.

Evaluate all nine combinations:

| Cluster 0 additions | Cluster 1 additions |
|---:|---:|
| 0, 5, 10 | 0, 5, 10 |

Each expert receives `global_backbone ∪ cluster_additions`. No specialist may remove or replace global features, and no collapsed per-cluster ElasticNet/stability list may become a final model input.

The final report will include the no-delta global result, all nine delta combinations, cluster feature counts, pooled metrics, and per-year metrics. The selected winner must strictly exceed both the locally reproduced V0 baseline and the published 0.7703 pooled R² benchmark; otherwise retain V0 as the best benchmark and report the best attempted candidate honestly.

## Reproducible outputs and tests

- Store generated search outputs, feature lists, collapse audit, and metric tables under the new experiment directory. Use readable YAML, CSV, and JSON only; do not add provenance hashes.
- Add a new experiment notebook that invokes or reloads the tracked runner, prints the Markdown tables used by `README.md`, and generates all figures from the generated artifacts:
  - selector funnel and collapse charts;
  - feature-family and seed-overlap comparison;
  - V0 versus seed versus searched-backbone performance;
  - 0/5/10 specialist-delta comparison;
  - unweighted year-by-year diagnostics.
- Populate `README.md` tables verbatim from the executed notebook’s stdout; do not manually transcribe numbers or use untracked one-time scripts.
- Add focused tests for:
  - train-only route fitting and literal candidate-feature evaluation;
  - legacy forced-bypass versus true bypass-off semantics;
  - MI-300 as the canonical gate, with no-MI and MI-120 blocked from canonical output;
  - collapse detection and fallback expansion from the pre-stability universe;
  - add-only delta enforcement;
  - deterministic score tie-breaking and worker/deadline behavior;
  - V0 evaluator parity.
- Run the lightweight tests first, then execute the new notebook from `notebooks/` with:

  `nb execute experiment/derived_8.4-feature-selection-2.0/feature_selection_2_0.ipynb --uv`

## Fixed assumptions

- Direct optimization on the pooled 2023–2025 test period is intentional.
- The SOTA comparator remains the existing V0-full K=2 Model 16 evaluation.
- Eight H100 workers and a two-hour wall-clock budget are available.
- Existing experiments remain untouched; this experiment alone owns its code, artifacts, figures, and report.