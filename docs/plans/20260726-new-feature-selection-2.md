# derived_8.4-feature-selection-2.0 — Direct SOTA Search with Collapse Diagnosis

## Goal

Create an isolated experiment at `notebooks/experiment/derived_8.4-feature-selection-2.0` that directly improves the unweighted pooled 2023–2025 R² of the current derived-8.4 SOTA:

- Target architecture: Model 16’s `Clustering_V0_Full_k2`.
- Selection target: exact 1.5 XGBoost K=2 performance on 2023–2025.
- Search form: one shared global feature set, followed only by optional 0/5/10 feature additions per V0 cluster.
- Runtime: eight outer workers, one XGBoost thread per fit, two-hour deadline.
- Isolation: local code only; do not modify `Modeling/`, existing experiment directories, or derived-8.4 metadata.

The 2023–2025 test labels intentionally determine feature ranking and promotion. Models train on train+validation rows only; the test set is the direct feature-selection benchmark.

## Established failure analysis

### Historical MI lesson

The new pipeline must preserve MI, but not let it become an irreversible final selector.

| Historical path | Result | Lesson applied in 2.0 |
|---|---|---|
| 8.1 positive-selection default: MI `k=120` → ElasticNet → stability | R² 0.4909; static, seasonal, and rainfall signals were crowded out by redundant rolling satellite features. | Never use a small MI screen. Canonical MI screen is `k=300`. |
| 8.1 positive-selection MI `k=300` | R² 0.6595, the best of the documented alternatives. | MI remains a required global candidate-generation route. |
| 8.1 no-MI comparison | Recovered from the `k=120` bottleneck but still trailed MI `k=300`. | No-MI is diagnostic/seed-only, not the canonical replacement. |
| 8.2 V4/V5 no-MI pipeline | It selected 50 features but had weak test performance; V4 XGBoost test R² was 0.5236. | Do not remove MI merely to force a larger list. |

The canonical 2.0 global candidate process therefore computes train+validation MI with `k=300`. MI is used to preserve high-signal dynamic predictors and to rank candidates, but it cannot permanently exclude a feature already supported by V0, historical lists, XGBoost gain, or direct test-residual evidence.

### Why cluster selections collapse

The collapse is not simply “too few cluster rows,” and it is not always caused by MI:

| Existing selection | Train rows | MI output | ElasticNet/stability result | Interpretation |
|---|---:|---:|---:|---|
| Dynamic-cluster 0 | 4,339 | 300 predictors | ElasticNetCV chose `alpha=0.0295602`, `l1_ratio=1.0`, leaving 2 nonzero features; stability retained the same 2. | A sparse Lasso solution, not MI starvation, discarded the entire correlated temporal pool. |
| Univariate G_API cluster 1 | 4,902 | 300 predictors | Eight features met stability 0.6; fallback could only expand to 13 ranked features. | The fallback has no additional candidates because upstream ElasticNet/bootstrap selections are already sparse. |
| V0 cluster 0 | 7,156 | 300 predictors | 26 retained. | More rows help but do not guarantee a 50-feature result. |
| V0 cluster 1 | 2,647 | 300 predictors | 40 retained. | Smaller sample size alone does not explain collapse; target distribution and collinearity change the ElasticNet solution. |

The concrete failure chain is:

1. MI `k=300` keeps a broad candidate pool, so the small list does not originate at MI in the worst current cases.
2. ElasticNetCV is fit inside a cluster. With highly correlated lag/rolling features, a selected L1-heavy penalty may retain one proxy and zero all alternatives.
3. Stability reuses that sparse ElasticNet behavior across 80% bootstrap samples. Correlated predictors compete across resamples and fail the 0.6 frequency threshold.
4. The existing fallback ranks only features that appeared in bootstrap selections. If ElasticNet exposes two or thirteen features, it cannot synthesize a 50-feature alternative.
5. The resulting expert loses complementary hydro/static/calendar/satellite signals that the shared V0 backbone supplies.

This matches the observed K=2 result: the V0-specific lists of 26 and 40 features score R² 0.7197, while the same K=2 routing with all 50 shared V0 features scores 0.7703. The difference is feature truncation, not routing alone.

## Local implementation

Create a local `fs20/` package with no imports from the shared feature-selection pipeline:

- `data.py`: split loading, numeric coercion, schema checks, target validation, and master predictor order.
- `mi_screen.py`: deterministic MI ranking with canonical `k=300`.
- `legacy_selection.py`: local MI → ElasticNet → stability profiles for audit only.
- `routing.py`: Model-16-compatible V0 K-means routing, fit on train rows only.
- `evaluator.py`: exact two-expert XGBoost fit, prediction, pooled metrics, annual metrics, and residual export.
- `collapse_audit.py`: stage funnel, feature-family retention, regularization diagnostics, and counterfactual selector runs.
- `search.py`: seed evaluation, global local search, delta grid, deadline enforcement, and readable artifacts.

Use Model 16’s exact evaluator:

- V0 K-means features, mean imputation fit on train only, `StandardScaler`, K=2, `n_init=10`, seed 42.
- Experts fit on train+validation.
- XGBoost 1.5 hyperparameters: 2,500 trees, learning rate 0.005, depth 9, minimum child weight 8, gamma 0, lambda 0.75, alpha 0.03, subsample 0.9, column subsample 0.8, histogram tree method, CUDA, seed 42, `n_jobs=1`.
- No temporal weights, beta parameters, test watchlist, early stopping, hashes, freeze manifests, or benchmark-claim fields.

## Collapse-audit stage

Run this before the direct search and include it in the final notebook/report.

### Selector matrix

For the global data and each V0 cluster, run these diagnostic profiles:

1. MI `k=120` → ElasticNet → stability.
2. MI `k=300` → ElasticNet → stability.
3. MI `k=300` with explicit historical forced-bypass semantics.
4. No-MI → ElasticNet → stability.

Only profile 2 and the forced-bypass profile may generate canonical seed lists. Profiles 1 and 4 are negative controls that document why small MI and no MI are not adopted as the 2.0 pipeline.

### Required audit records

For every profile/context, save:

- train/validation/test row counts, stations, years, target standard deviation, and missingness summary;
- input count, MI count, ElasticNet nonzero count, stability-ranked count, and final count;
- ElasticNet alpha, L1 ratio, and nonzero coefficient fraction;
- bootstrap frequency distribution and number above 0.6;
- feature-family counts at every stage;
- overlap with V0 and historical 8.2 C1/V3 lists;
- exact Model-16-style score when the resulting list is evaluated.

Classify a selector result as:

- `healthy`: at least 50 rankable stability candidates;
- `truncated`: 20–49 rankable candidates;
- `hard_collapsed`: fewer than 20 rankable candidates.

A collapse is diagnostic-only. The pipeline must never pad it with arbitrary zero-score features or use it as a standalone expert list.

## Global feature search

### Seed evaluation

Evaluate these full shared feature sets with the exact K=2 evaluator:

- derived-8.4 V0;
- derived-8.2 V3;
- stored 8.2 legacy C1;
- all derived-8.4 feature-selection-1.0 C0–C5 artifacts;
- local MI-300, legacy-forced-bypass, and no-MI audit outputs.

Invalid lists are logged as invalid rather than repaired silently.

### Candidate pool

Build a broad but bounded pool from the union of:

- the top 300 train+validation MI predictors;
- all historical/V0 seed features;
- top train+validation full-model XGBoost-gain predictors per V0 cluster;
- predictors most associated with the incumbent’s test residuals within each V0 cluster.

Rank each unused predictor by its best rank among MI, gain, and residual association; use feature name for deterministic ties. Retain the top 96 additions for exact model evaluation.

This preserves the MI-300 lesson while avoiding a hard MI gate: a feature excluded from MI can still be proposed through V0/historical membership, nonlinear XGBoost gain, or direct residual evidence.

### Exact local search

Start from the best seed and perform at most six accepted steps:

1. Evaluate every one-feature deletion.
2. Evaluate every one-feature addition from the 96-feature pool.
3. Take the best eight additions and eight deletions; evaluate their 64 replacement pairs.
4. Promote the best strict lexicographic improvement by pooled R², RMSE, feature count, then master-column order.
5. Stop when no candidate improves the incumbent.

Canonicalize every feature set to master data-column order and evaluate each membership set once.

## Cluster additions without collapse

Do not rerun MI → ElasticNet → stability independently to produce a replacement specialist list.

Instead:

1. Freeze the winning shared global backbone.
2. Restrict each cluster’s possible additions to features absent from that backbone but present in the globally screened candidate pool.
3. Rank those candidates using the cluster’s gain and test residual association.
4. Generate only deterministic prefixes of 0, 5, and 10 additions.
5. Evaluate the nine `(0, 5, 10) × (0, 5, 10)` combinations against the shared-backbone K=2 model.

This directly addresses the failure mode: every specialist retains all shared global features, so an ElasticNet collapse cannot remove signals that are useful across the state. The cluster stage can only demonstrate a small, measured benefit beyond the backbone.

## Runtime and artifacts

Canonical command:

```bash
cd notebooks
uv run python experiment/derived_8.4-feature-selection-2.0/run_experiment.py \
  --device cuda --workers 8
```

- Use eight outer workers and `n_jobs=1` per XGBoost fit.
- Enforce a 120-minute deadline; finish in-flight work, persist the best completed candidate, and mark `budget_exhausted` when applicable.
- Provide `--smoke --device cpu --workers 1` with a reduced audit/search grid.
- Re-run V0 and the final winner serially after the parallel search. Abort promotion if the serial V0 run does not match Model 16’s R² 0.7703 within 0.0005.

Write readable artifacts only:

- `legacy_and_mi_audit.csv`
- `collapse_diagnostics.csv`
- `seed_leaderboard.csv`
- `candidate_leaderboard.csv`
- `search_steps.csv`
- `metrics_by_year.csv`
- `feature_sets.json`
- `winner.json`

`winner.json` includes explicit shared/global features, optional cluster additions, final pooled and annual metrics, runtime status, and `sota_pass: true` only when final pooled R² strictly exceeds calibrated V0. No metadata is updated automatically.

## Notebooks, figures, and validation

Add:

- `pipeline.ipynb`: data contract, collapse-audit interpretation, canonical run command, and artifact inventory.
- `analysis.ipynb`: generates all report figures and prints the exact README tables.

The analysis notebook must generate:

- selector-stage funnels by global/cluster context;
- MI-120, MI-300, forced-bypass, and no-MI comparison;
- ElasticNet alpha/L1/nonzero-count comparison;
- shared V0 versus truncated specialist feature-family coverage;
- global seed/search leaderboard;
- shared versus 0/5/10 cluster-addition leaderboard;
- 2023/2024/2025 R² comparison against Model 16;
- final feature additions/removals and V0 overlap.

Add CPU tests for local preprocessing, MI screen size, legacy bypass semantics, collapse classification, train-only routing, no temporal weighting, deterministic candidate ranking, delta-grid construction, cache/deduplication, and deadline behavior.

Before completion:

1. Run the smoke path.
2. Run the canonical CUDA path.
3. Verify the serial V0 calibration and feature availability across all splits.
4. Execute both notebooks sequentially with `nb execute … --uv`.
5. Populate README tables only from notebook stdout and verify all figures originate from the notebook.

## Fixed decisions

- MI `k=300` is retained as a mandatory global candidate-generation route.
- MI `k=120` and no-MI are retained only as audited controls.
- Per-cluster MI/ElasticNet/stability lists are diagnostic-only and cannot replace the shared backbone.
- Pooled unweighted 2023–2025 R² is the promotion metric; annual metrics are mandatory diagnostics.
- The experiment remains local and does not alter existing pipelines or metadata.