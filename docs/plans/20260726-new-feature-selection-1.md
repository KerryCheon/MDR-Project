# derived_8.4-feature-selection-2.0 — Direct 2023–2025 SOTA Search

## Objective and design change

Create a self-contained feature-selection experiment at `notebooks/experiment/derived_8.4-feature-selection-2.0` that directly optimizes the current derived-8.4 SOTA model on 2023–2025:

- Target model: `Clustering_V0_Full_k2`, with the exact 1.5 XGBoost parameters used by Model 16.
- Primary objective: unweighted pooled test R² across 2023–2025.
- Tie-breakers: lower pooled RMSE, fewer features, then master dataset-column order.
- Scope: one shared global feature set plus optional additions of 0, 5, or 10 features for each of the two V0 clusters.
- Isolation: do not modify `Modeling/`, split metadata, or prior experiments. Do not add hashes, freezes, claim registries, or temporal weighting.

The 2023–2025 labels intentionally influence candidate discovery, ranking, and final selection. Models still train only on train+validation rows; test rows are the direct feature-selection benchmark.

## What changes from the older experiments, and why

| Existing behavior | Problem | 2.0 change | Why it may work better |
|---|---|---|---|
| 8.3 feature-selection 2.1 optimizes rolling 2017–2022 folds, then benchmarks later years. | Its optimization target differs from the requested 2023–2025 goal; it also spends most runtime on nested folds, guards, hashes, and bookkeeping. | Optimize the exact 2023–2025 Model 16 architecture directly, with a fixed two-hour search budget. | Every candidate is judged by the metric and years that define success. |
| 8.4 feature-selection 1.0 uses generic MI/ElasticNet/stability or feature-importance pipelines. | A sparse linear selector can discard nonlinear, interacting, static, and hydrology variables that XGBoost benefits from. | Use legacy selectors only as diagnostic/seed generators; choose the winner with exact K=2 XGBoost performance. | The final learner, not a proxy selector, determines which feature changes help. |
| Historical 8.2 “C1 bypass off” selected 50 mixed-family features. | The old selector ignored the YAML bypass-off setting: default function arguments force-kept bypass columns. It is not equivalent to today’s true C1. | Preserve it as an explicitly named `legacy_forced_bypass` baseline and compare it with true bypass-off behavior. | Recovers a potentially transferable feature set without mislabeling its semantics. |
| Current true C1 on 8.4 produces 12 mostly satellite features. | ElasticNet/stability reduces its candidate pool to 12, so its fallback cannot restore missing hydro/static/calendar candidates. | Audit each stage locally and use broader historical + residual/gain candidate discovery for the direct search. | Prevents a sparse linear stage from becoming an irreversible bottleneck. |
| Current 8.4 evaluation labels `global_c1` but assigns V0 to it. | No newly selected global feature set was actually evaluated in that arm. | Evaluate every new global candidate explicitly and store its feature list with its metrics. | Separates genuine global improvements from V0 aliases. |
| Full specialist lists are selected independently per regime. | Existing results show specialist lists often underperform the shared V0 backbone; some cluster selections collapse to very small sets. | Start with the best shared list; specialists may only receive small additions. | Keeps the strong global signal while allowing limited cluster-specific correction. |
| Drift weighting is evaluated alongside unweighted scoring. | It changes the effective training objective and does not match the requested policy. | Omit beta, date weights, and weighted leaderboards entirely. | The selected set must perform under the same unweighted fit used for reporting. |

## Local implementation

Create a local `fs20/` package inside the experiment rather than importing the shared feature-selection pipeline. It will contain:

- `data.py`: split loading, numeric coercion, schema checks, target validation, and master predictor ordering.
- `legacy_selection.py`: local copies of MI, ElasticNet, bootstrap stability, and two explicit profiles:
  - `legacy_forced_bypass`: historical temporal/satellite-first MI pool followed by forced bypass reinjection.
  - `true_bypass_off`: all predictors enter MI with no bypass reinjection.
- `routing.py`: Model-16-compatible V0 K-means routing:
  - V0 features loaded from derived-8.4 metadata;
  - mean imputation fit on train only;
  - `StandardScaler`, K=2, `n_init=10`, seed 42;
  - routing applied to train, validation, and test after fitting only on train.
- `evaluator.py`: exact two-expert XGBoost training and pooled/per-year metrics.
- `search.py`: seed evaluation, candidate screening, greedy global search, bounded delta search, cache handling, and deadline control.
- `artifacts.py`: readable JSON/CSV writes only; no checksums or provenance hashes.

Use the current SOTA XGBoost configuration exactly: `reg:squarederror`, 2,500 trees, learning rate 0.005, depth 9, minimum child weight 8, gamma 0, lambda 0.75, alpha 0.03, subsample 0.9, column subsample 0.8, histogram tree method, seed 42, CUDA, and `n_jobs=1`.

Each expert trains on concatenated train+validation data. The evaluator will not supply test data as an XGBoost watchlist, since there is no early stopping and the fixed tree count makes it unnecessary.

## Diagnostic and candidate-generation flow

1. Run the local legacy audit before any search.

   - Apply both local selector profiles to derived-8.2 and derived-8.4.
   - Compare feature count, stage candidate count, family composition, overlap with the stored 8.2 C1 artifact, and exact Model-16-style score on derived-8.4.
   - Include the current 8.4 C0–C5 artifacts as external comparison rows rather than rerunning shared code.
   - This distinguishes semantic drift in the selector from transfer differences caused by removing the two alpine stations and reducing training rows.

2. Evaluate fixed seed sets with the exact K=2 evaluator.

   Seed sets include V0, derived-8.2 V3, the stored legacy 8.2 C1 list, every 8.4 C0–C5 list, and the locally generated legacy/true-bypass selections. Invalid lists are recorded and skipped rather than silently repaired.

3. Build the direct-search pool around the best seed.

   - Fit a full-predictor train+validation model per V0 cluster and collect normalized gain importance.
   - Calculate test residuals for the incumbent shared model.
   - Rank unused predictors by weighted absolute within-cluster Spearman association with test residuals and by full-model gain; use the better of the two ranks, then feature name.
   - Retain the top 96 screened predictors. This keeps search centered on concrete evidence while allowing features that historical selection never proposed.

4. Run deterministic local improvement.

   For up to six accepted rounds:

   - evaluate every single-feature removal from the incumbent;
   - evaluate every single-feature addition from the 96-feature screened pool;
   - take the eight best additions and eight best removals, then evaluate all 64 replacement pairs;
   - accept only the best lexicographic improvement in pooled R²/RMSE/size/order;
   - stop early if no candidate improves the incumbent.

   Feature membership is canonicalized to dataset-column order, so the same feature set is evaluated once even if generated by multiple paths.

5. Search small cluster additions.

   - Keep the winning shared list unchanged as the backbone.
   - For each V0 cluster, rank only features absent from the shared list using the same local residual/gain method.
   - Form deterministic prefixes of 0, 5, and 10 additions.
   - Evaluate all nine two-cluster combinations and compare them directly with the shared winner.
   - Do not permit cluster-specific deletions or replacement-only specialist lists.

This approach is intentionally a local neighborhood search rather than a fresh 496-feature global selection. V0 already has strong performance, so testing targeted additions, removals, and swaps is more likely to retain its useful structure while correcting a small number of harmful or missing features.

## Runtime, execution, and artifacts

Provide one canonical runner:

```bash
cd notebooks
uv run python experiment/derived_8.4-feature-selection-2.0/run_experiment.py \
  --device cuda --workers 8
```

- Use `ThreadPoolExecutor(max_workers=8)` and `n_jobs=1` per XGBoost fit.
- Enforce a 120-minute deadline. Do not start another search batch after the deadline; complete in-flight models, retain completed results, and write `budget_exhausted` status if necessary.
- Add `--smoke --device cpu --workers 1` with a reduced seed/search grid for wiring checks.
- Perform final serial reruns of V0 and the selected winner after the parallel search. Promotion uses these final metrics, avoiding a concurrency-only discrepancy.

Persist:

- `artifacts/legacy_audit.csv`
- `artifacts/seed_leaderboard.csv`
- `artifacts/candidate_leaderboard.csv`
- `artifacts/search_steps.csv`
- `artifacts/metrics_by_year.csv`
- `artifacts/feature_sets.json`
- `artifacts/winner.json`

`winner.json` contains the explicit shared list, optional per-cluster additions, final metrics, and `sota_pass: true` only when final pooled R² is strictly above the calibrated V0/Model-16 result. If no candidate clears it, V0 remains the reported winner within this experiment; metadata remains untouched.

Keep temporary models and prediction caches under an ignored local cache directory. All metrics, explicit feature lists, figures, configuration, and source code remain visible in the repository.

## Notebooks and report outputs

Add two new notebooks:

- `pipeline.ipynb`: documents the data contract, legacy audit, canonical runner command, and resulting artifact locations.
- `analysis.ipynb`: reads only generated artifacts and creates:
  - legacy-selector semantic comparison;
  - seed/global/delta leaderboard;
  - 2023, 2024, and 2025 R² comparison against V0;
  - feature-overlap and final add/drop summaries.

The analysis notebook prints the exact markdown tables used by the experiment README. All report figures are generated there, not by an untracked one-off command.

## Validation and acceptance criteria

- Add CPU tests covering:
  - local preprocessing and split schema validation;
  - forced-bypass versus true-bypass semantics;
  - train-only router fitting and stable cluster labeling;
  - absence of temporal weighting;
  - candidate deduplication and deterministic tie-breaking;
  - deadline/budget-exhausted handling;
  - 0/5/10 delta-grid construction.
- Run the smoke workflow before the canonical GPU search.
- Abort canonical promotion if the local V0 evaluator does not reproduce Model 16 pooled R² 0.7703 within 0.0005.
- Verify every selected feature exists in every split and each expert receives nonempty training data.
- Execute both notebooks sequentially with `nb execute … --uv` before declaring the experiment complete.
- Do not alter `data/splits/derived_8.4/dataset_metadata.py` or rerun/overwrite previous experiment artifacts.

## Fixed decisions

- The search intentionally uses 2023–2025 labels.
- Pooled unweighted R² is the primary objective; yearly results are required diagnostics, not promotion guards.
- The target is shared V0-cluster K=2 with bounded additions, not a sweep over all gating strategies.
- Canonical runtime is capped at two hours on eight workers.
- No temporal drift weighting, provenance hashes, frozen manifests, or shared-pipeline changes are included.