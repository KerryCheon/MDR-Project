# Codex Session Export

- **Session ID:** `019f88c3-f650-7640-bd3e-e3209e499317`
- **Time:** 2026-07-22T07:39:31.909Z
- **Source:** codex-tui
- **Workspace:** `/scratch/user/u.rp352032/MDR-Project`
- **Model:** openai

---

## 👤 User

A previous agent produced the plan below to accomplish the user's task. Implement the plan in a fresh context. Treat the plan as the source of user intent, re-read files as needed, and carry the work through implementation and verification.

# `derived_8.3-feature-selection-2.1` Evaluation-First Feature Selection

## Summary

Create an isolated experiment under `notebooks/experiment/derived_8.3-feature-selection-2.1/` without modifying the completed 8.2/2.2 or 8.3/2.0 experiments.

The experiment will:

- Keep all nine derived-8.3 stations and use only 2017–2022 for development.
- Use the exact 1.3-lite XGBoost learner for ranking and evaluation.
- Make V0 the reference and retain it unless a new candidate passes robust development gates.
- Include 2021–2022 both as rolling-origin validation and in the final all-development consensus ranking.
- Treat year-dependent feature utility as a stability problem rather than immediately creating year-specific models.
- Diagnose monthly and difficult-station failures from row-level OOF predictions.
- Use separate global and MoE configurations.
- Always run MoE causal diagnostics, but prohibit MoE promotion if the global gate fails.
- Keep 2023–2025 behind a separate benchmark command and never use it to choose features, beta, routing, or architecture.
- Allow a qualifying frozen winner to claim **project SOTA on the reused derived-8.3 2023–2025 benchmark**, while explicitly withholding any fresh-holdout or unbiased external-generalization claim.

No derived-8.0 models will be retrained. Its completed results remain historical context.

Treat `derived_8.2-feature-selection-2.2` as a failed selection methodology for the current dataset. Reuse its implementation ideas, provenance safeguards, and handoff notes, but do not treat its selected features as promotable. `derived_8.3-feature-selection-2.0` is the current source of truth for clean-station metrics and confirms the transfer failure.

## Interpretation Boundary and SOTA Eligibility

The 2023–2025 period is no longer an untouched holdout because `derived_8.3-feature-selection-1.0`, `derived_8.3-eval-1.0`, the error analyses, and `derived_8.3-feature-selection-2.0` have already inspected its year, month, station, and regime behavior. Keeping it physically separate in 2.1 prevents further selection leakage, but it cannot reverse those earlier adaptive observations.

Because ECE sensor data are not yet available, the experiment may nevertheless use 2023–2025 as the established project benchmark:

- Every 2023–2025 artifact records `retrospective_test: true` and `benchmark_reused: true`.
- `unbiased_sota_eligible` and `unbiased_generalization_claim_eligible` remain `false`.
- A separate `benchmark_sota_eligible` verdict determines whether the result is a new project benchmark SOTA.
- Development promotion means the configuration is frozen and eligible for benchmark comparison; benchmark results cannot change the configuration.
- `OVERALL_SELECTED_FEATURES_V0` is never overwritten automatically. Any eventual metadata promotion is a separate reviewed change.
- Future ECE observations remain the required independent external confirmation.

If the final benchmark gate passes, use wording equivalent to:

> “The model establishes a new project SOTA on the reused derived-8.3 Washington 2023–2025 benchmark, pending confirmation on the forthcoming ECE sensor deployment.”

Do not describe the result as an untouched holdout, unbiased SOTA, or external spatial-generalization result.

## Global Selection Design

### Data and fold geometry

- Load only `train.csv` and `val.csv` during development.
- Reject dates after 2022 and any development code path resolving to `test.csv`.
- Preflight must confirm the current derived-8.3 station list, split hashes, and expected development coverage from 2.0.
- Use rolling outer origins 2020, 2021, and 2022. For origin `t`, training is strictly years `< t` and validation is year `t`.
- Generate two fold families:
  - `forward_time`: all prior-year stations train; all stations in the outer year validate.
  - `station_time`: all rows from held-out stations are excluded from outer training, and those stations validate in the outer year.
- Candidate generation for every outer task receives only that task’s training frame. A held station group and outer-year labels must therefore be absent from feature ranking as well as model fitting.
- Build five deterministic row-balanced station partitions from `seed = 42 + i`, where `i ∈ [0,4]`, and reuse each mapping across origins.
- Separate partition and learner uncertainty without a full Cartesian sweep:
  - partition seeds 42–46 with learner seed 42;
  - learner seeds 42–44 with partition seed 42;
  - deduplicate the shared `(42, 42)` run, producing seven station-time repeats.
- Forward-time runs use learner seeds 42–44.
- Reject zero-observation station-year folds and persist assigned versus actually observed coverage.

For each outer origin, inner ranking uses up to the last two eligible rolling years after at least two earlier training years. The 2020 origin therefore has one eligible inner year, 2019; it is permitted only when row and usable-fold minimums remain satisfied. No future year may be substituted to create another fold.

This geometry ensures:

- 2021 is an outer validation year and contributes to the 2022 candidate-generation history.
- 2022 is an outer validation year.
- Both 2021 and 2022 participate in the final all-development consensus ranking.
- The test period remains absent from every development decision.

### Learner and pruning path

Use the exact 1.3-lite learner everywhere, including ranking fits:

```yaml
objective: reg:squarederror
max_depth: 8
min_child_weight: 10
reg_lambda: 1.5
reg_alpha: 0.03
subsample: 0.9
colsample_bytree: 0.8
n_estimators: 1500
learning_rate: 0.01
tree_method: hist
n_jobs: 1
```

Additional runtime rules:

- XGBoost native missing-value handling; no global median or mean imputation.
- CUDA for canonical runs.
- Four independent workers, with every XGBoost fit retaining `n_jobs=1`.
- Model seed supplied by the repeat configuration.
- Residual is always `truth - prediction`.
- The 160-tree ranking surrogate and the 1.5 learner are not used.

Retain the historical endpoint counts:

```text
[150, 125, 100, 80, 65, 50, 40]
```

Progressive bridge counts are generated from the starting 496 predictors using:

```text
next_size = max(target_size, current_size - target_size)
```

For example, the 496-to-150 reduction inserts 346 and 196 before reaching 150. Bridge sizes are ranking steps, not additional promotion endpoints.

Run a base-seed direct-versus-progressive screen using partition seed 42, learner seed 42, both fold families, and all three outer origins. Use one deterministic permutation repeat during this screen.

Select direct elimination only if its paired 95% upper confidence bound for primary-risk difference versus progressive is below zero. Otherwise freeze progressive elimination. The full repeated stability run then executes only the chosen method.

Full feature ranking will:

- remain feature-name and feature-family agnostic;
- use three deterministic permutation repeats;
- score the change in station-year macro RMSE rather than normalized row-pooled fold RMSE;
- rank by importance lower confidence bound, mean importance, then original column position;
- keep beta out of ranking so beta arms cannot inflate the effective fold count;
- refit after each reduction so correlated substitutes can gain importance after stronger correlates are removed.

A post-selection correlation diagnostic will form training-only Spearman correlation components at `|ρ| ≥ 0.95` and jointly permute each component. This explains correlated substitutes and pruning instability but cannot change the selected list.

### Candidate matrix

For each path source and endpoint, evaluate:

- `selected_k`: the selector’s ordered feature list.
- `v0_union_selected_k`: the exact ordered V0 list followed by selected features not already present.

Fixed controls are:

- exact derived-8.3 V0;
- all 496 numeric predictors;
- the completed 2.0 original, crossed, and nested lists.

The all-predictor and 2.0 arms are diagnostic only and cannot be promoted. Their purpose is to distinguish selection failure from input insufficiency and to confirm the historical transfer failure under the new OOF geometry.

### Prediction ledger and repeat handling

Persist raw OOF predictions as compressed CSV with:

- model, candidate, path source, endpoint, actual count, and ordered-feature hash;
- fold family, outer origin, fold ID, station partition seed, and learner seed;
- station, date, year, and month;
- truth, prediction, residual, absolute error, and squared error;
- beta and complete model-configuration ID;
- router regime and route distance when applicable.

Repeated predictions must not act as independent observations:

- For primary risk, average squared error across repeats for the same candidate, fold family, origin, station, and date before forming station-year RMSE.
- For secondary prediction-based metrics, average prediction across those repeats and calculate residuals once.
- Persist `repeat_count` and reject candidates whose repeat coverage differs from V0.
- Never concatenate seed repeats as additional validation rows.

### Primary and secondary risk

Primary risk is station-year macro RMSE:

1. Compute RMSE for each observed station × outer-origin block.
2. Average blocks equally within each fold family.
3. Combine `forward_time` and `station_time` risks with equal 50/50 weight.

Use 2,000 paired hierarchical bootstrap replicates with seed 42:

- sample stations with replacement;
- sample outer years with replacement within each sampled station;
- retain candidate/V0 pairing;
- calculate percentile 95% confidence intervals and bootstrap standard errors.

Secondary diagnostics include:

- pooled RMSE and R²;
- MAE and Pearson correlation;
- bias;
- station-macro RMSE;
- 90th-percentile station-year RMSE;
- worst-station RMSE;
- monthly RMSE, MAE, and bias;
- target count, range, standard deviation, and interquartile range alongside every subgroup R².

If a subgroup has zero target variance, report R² as `NaN` with an explicit reason rather than substituting zero or an arbitrary sentinel.

### Candidate and count decision

A new global candidate is eligible only when:

- the upper endpoint of its paired 95% bootstrap interval for combined `ΔRMSE = candidate - V0` is below zero;
- its point estimate does not regress in either fold family;
- its 90th-percentile station-year RMSE does not exceed V0 by more than one paired bootstrap standard error;
- its worst-station RMSE does not exceed V0 by more than one paired bootstrap standard error;
- its 90th-percentile monthly RMSE does not exceed V0 by more than one paired bootstrap standard error;
- its OOF row and repeat coverage exactly matches V0.

Among eligible candidates:

1. Find the minimum combined primary risk.
2. Apply the one-standard-error rule and choose the smallest actual feature count within one bootstrap standard error of that minimum.
3. Break remaining ties by higher cross-origin selection stability.
4. Then prefer a pure selected list over a V0 union of identical actual size.
5. Then use fixed path-source order `station_time`, `forward_time`.
6. Finally use lexical candidate ID.

If no candidate passes:

- retain V0 as the global winner;
- record the lowest-risk candidate as `best_failed_candidate`;
- allow that candidate to appear in development and station/MoE diagnostics;
- prohibit its promotion or project-SOTA eligibility.

### Beta decision

After selection method, list form, and count are frozen:

- Compare beta 0.0 and beta 0.2 as separate paired arms.
- For beta 0.2, calculate training weights as:

```text
w = exp(0.2 × (year - latest_training_year))
w = w / mean(w)
```

- Select beta 0.2 only if its combined paired 95% upper confidence bound versus beta 0.0 is below zero and neither fold family regresses in point estimate.
- Otherwise select beta 0.0.
- Beta arms are never pooled as extra folds or used to alter feature identity.

The canonical final model uses learner seed 42. Seeds 43–44 provide robustness evidence and are not benchmark-time seed candidates.

### Final all-development list

After method, path source, list form, count, and beta are frozen:

- Rerun the frozen selector over all 2017–2022 development data.
- Build internal rolling importance tasks ending in 2020, 2021, and 2022.
- Give years equal weight so row-rich years do not dominate.
- Build the exact-count consensus list by sorting features by:
  1. selection frequency;
  2. median percentile rank;
  3. mean percentile rank;
  4. original column position.
- For a V0 union, preserve V0’s canonical order and append consensus features not already present.
- Save the exact ordered list, source rankings, hashes, and feature-count interpretation.

Report:

- origin-specific lists;
- rank correlations;
- Jaccard overlap;
- V0 overlap;
- direct/progressive overlap;
- feature support by year;
- features gaining or losing support in 2021–2022;
- feature-family summaries applied only after selection.

Do not deploy separate year-specific models in this experiment.

## Station, Month, and Temporal Diagnostics

Keep Marten Ridge, Rainy Pass, and every other current station in scope.

For each station and month, compare under identical OOF geometry:

- V0;
- the selected 2.1 candidate, or `best_failed_candidate` when no candidate qualifies;
- the V0-union candidate;
- the all-predictor diagnostic.

Classify station behavior as:

- `global_features_sufficient`: V0 or the promoted compact candidate performs adequately;
- `selection_failure`: all predictors materially help while compact lists do not;
- `current_input_limitation`: V0, selected, union, and all predictors fail similarly;
- `low_target_variance_artifact`: poor R² occurs without corresponding RMSE/MAE degradation;
- `uncertain`: paired intervals overlap.

For each station persist:

- target count, range, variance, and seasonal climatology;
- feature missingness and out-of-range rates;
- monthly RMSE, MAE, bias, and residual distribution;
- feature-distribution distance from the development population;
- selected-versus-all-predictor paired intervals.

Report fixed-2017–2019 versus expanding-window fits for V0 and the chosen candidate as a diagnostic only. This tests whether difficult stations require adaptation to recent conditions without making the fit-window choice part of promotion.

Produce tables for all months rather than special-casing October, while highlighting transition-month failures when supported by the results.

No stations will be pruned and no snowpack, soil-temperature, freeze/thaw, or new sensor inputs will be added in 2.1. The report will identify stations for which those inputs are the appropriate next experiment.

## Multi-Regime MoE Diagnostics

### Separate MoE configuration

Use `moe_config.yaml`, distinct from `global_config.yaml`, while retaining the frozen `Clustering_V0_Full_k2` router:

- exact ordered 50-feature V0 router inputs;
- numeric coercion and infinity-to-missing conversion;
- fit-frame mean imputation;
- `StandardScaler`;
- K-means with `K=2`, `n_init=10`, seed 42;
- router preprocessing fitted only on each outer training frame;
- target-free cluster IDs aligned to a 2017–2019 reference router through centroid-distance matching.

Reference alignment is for consistent reporting only and must not alter routing predictions.

Save regime populations by year, month, and station, centroid drift, route distances, and router-feature missingness.

### Causal model matrix

Run these arms under the same OOF ledger:

| Arm | Question answered |
|---|---|
| V0 single global | Reference |
| 2.1 single global | Did selection improve the global model? |
| 2.1 ∪ V0 single global | Did V0 retain globally useful inputs omitted by selection? |
| V0 shared hard experts | Does routing help or fragment a known feature set? |
| 2.1 shared hard experts | Does sample fragmentation explain loss versus one global fit? |
| 2.1 ∪ V0 shared hard experts | Does full V0/router-input access change that conclusion? |
| Strongest shared control plus regime deltas | Is there incremental regime-specific feature signal? |
| Saved eval-1.0 specialist lists | Does the historical small-list/47-feature collapse reproduce? |

“Shared hard experts” means two independently fitted expert models use the same complete feature backbone, with the router selecting the prediction. This separates expert specialization from feature-set differences.

If the global gate fails, “2.1” refers to the saved `best_failed_candidate` and every corresponding MoE result remains diagnostic.

### Regime-delta selection

Never create a standalone independently selected expert list. Every expert begins with the complete frozen shared backbone and may only add unused predictors.

Retain delta counts:

```text
[0, 5, 10, 15]
```

For each regime:

- rank unused predictors against that regime’s training-frame residuals;
- use the same causal outer/inner boundaries and 1.3-lite learner;
- require the regime to occur in every outer origin and at least `ceil(9/2) = 5` stations;
- require the paired 95% improvement interval versus shared-only to exclude zero in the improving direction;
- require no point worsening in that regime’s worst-station RMSE;
- choose the smallest qualifying delta within one standard error of the best;
- otherwise assign zero delta features.

Always report:

- regime population and target dispersion;
- station, year, and month composition;
- route-distance distribution;
- global versus expert performance;
- shared-only versus delta performance;
- feature rank stability within each regime;
- whether zero delta reflects insufficient coverage, unstable rankings, or no measured benefit.

These comparisons must explicitly distinguish:

- missing V0/global information;
- hard-routing and sample-fragmentation loss;
- per-regime feature-selection collapse;
- genuine incremental regime-specific signal.

### MoE promotion rule

MoE diagnostics always run.

If the global feature-selection gate fails:

- no MoE can be promoted;
- no MoE can receive `benchmark_sota_eligible: true`;
- MoE results remain causal diagnostics only.

If the global gate passes, a MoE may become the frozen benchmark challenger only when:

- it significantly beats the strongest corresponding single-global model with a paired 95% upper `ΔRMSE` below zero;
- neither fold family regresses in point estimate;
- it satisfies the same station-year, worst-station, and monthly robustness guards.

Otherwise the single-global candidate remains the frozen challenger.

## Benchmark Evaluation and Project-SOTA Verdict

### Benchmark isolation

Create a separate `run_benchmark.py --confirm-benchmark` command. It must:

- verify `development_freeze.json` before importing or reading test data;
- reject any mismatch in code, config, split, feature list, beta, learner, router, or model decision;
- train on all 2017–2022 development rows;
- use learner seed 42;
- evaluate on 2023–2025 exactly once per frozen model ID;
- write only benchmark artifacts and reports;
- never rewrite development, selection, or freeze artifacts.

Benchmark models are:

- V0 through the exact 1.3-lite harness;
- one predeclared 2.1 challenger:
  - the promoted global model;
  - or the promoted MoE if it passed both development gates;
  - or `best_failed_candidate` as diagnostic-only when the global gate failed.

A diagnostic-only challenger cannot receive a SOTA claim even if its retrospective score is high, because it failed the predeclared development gate.

### Historical benchmark registry

Add `benchmark_registry.yaml`. Resolve scores and predictions from saved source artifacts rather than manually copying them into report code.

The current project-SOTA reference is:

- experiment: `derived_8.3-eval-1.0`;
- model: Model 16, `Clustering_V0_Full_k2 (Global-V0)`;
- learner: historical 1.5 configuration;
- R²: `0.6618718115185884`;
- RMSE: `0.06042772002760553`;
- CUDA, seed 42.

The 2.1 experiment does not retrain or tune 1.5. It uses the saved Model 16 predictions as the strongest historical comparator.

Before paired comparison:

- verify the historical label vector length is 8,396;
- verify it exactly matches the current ordered test target;
- reconstruct station/date keys from the verified test row order;
- reject SOTA eligibility if prediction-to-row alignment cannot be proven.

### Benchmark-SOTA gate

Set `benchmark_sota_eligible: true` only when all conditions hold:

- the global development gate passed;
- the frozen challenger, including an eligible promoted MoE, passed every applicable development gate;
- benchmark R² is at least `historical_best_r2 + 0.003`;
- with the current registry, this threshold is `0.6648718115185884`;
- benchmark RMSE is below `0.06042772002760553`;
- the paired 95% upper interval for station-year macro `ΔRMSE` versus historical Model 16 is below zero;
- worst-station RMSE and 90th-percentile monthly RMSE do not regress by more than one paired bootstrap standard error;
- historical-label alignment and all provenance checks pass.

The 0.003 R² margin follows the project’s documented environment/hardware reproducibility band. A score above 0.6618718 but below the margin is a nominal leaderboard improvement within reproducibility noise, not a new SOTA.

Every benchmark verdict must include:

```yaml
claim_scope: project_derived_8.3_2023_2025_benchmark
retrospective_test: true
benchmark_reused: true
benchmark_sota_eligible: true_or_false
unbiased_sota_eligible: false
unbiased_generalization_claim_eligible: false
ece_external_confirmation_pending: true
```

Benchmark results cannot feed back into 2.1. Any change motivated by them must become a new version such as `derived_8.3-feature-selection-2.2`.

## Implementation, Interfaces, and Reproducibility

### Experiment-local implementation

Create versioned scripts and artifacts inside `notebooks/experiment/derived_8.3-feature-selection-2.1/`.

Use the completed 2.0 experiment as a structural reference for:

- split provenance;
- atomic artifact replacement;
- completion markers;
- resumable stage journaling;
- report generation and `--check`;
- V0 source resolution;
- runtime device/worker handling.

Keep completed experiments and the shared selector unchanged. New repeated-fold, ledger, bootstrap, consensus, MoE-ablation, and SOTA-decision logic remains version-local.

Recommended tracked files:

- `global_config.yaml`;
- `moe_config.yaml`;
- `benchmark_registry.yaml`;
- `preflight.py`;
- `run_global_selection.py`;
- `run_station_diagnostics.py`;
- `run_moe_diagnostics.py`;
- `run_benchmark.py`;
- `run_all.py`;
- `generate_results.py`;
- version-local fold, ledger, metrics, bootstrap, selection-decision, router, and artifact-state modules;
- `README.md`, `PLAN.md`, `PROTOCOL.md`, `RESULTS.md`, `BENCHMARK_RESULTS.md`, and `CONTINUATION.md`.

No project-wide public API changes are required.

### Command surface

From `notebooks/`:

```text
uv run python experiment/derived_8.3-feature-selection-2.1/preflight.py --device cuda --workers 4

uv run python experiment/derived_8.3-feature-selection-2.1/run_all.py \
  --device cuda --workers 4 [--restart]

uv run python experiment/derived_8.3-feature-selection-2.1/run_benchmark.py \
  --confirm-benchmark --device cuda --workers 4

uv run python experiment/derived_8.3-feature-selection-2.1/generate_results.py

uv run python experiment/derived_8.3-feature-selection-2.1/generate_results.py --check
```

`run_all.py` is development-only and must not invoke benchmark evaluation implicitly.

### Resumable stage journal

Checkpoint after every origin/family/seed/candidate unit and journal:

1. data, environment, feature-universe, and coverage preflight;
2. deterministic fold manifests;
3. V0, all-feature, and 2.0 diagnostic control ledgers;
4. base-seed direct/progressive screen;
5. repeated robust candidate generation;
6. repeated candidate OOF evaluation;
7. method/count decision;
8. beta evaluation and decision;
9. all-development consensus list;
10. year/month/station and correlation diagnostics;
11. causal MoE matrix;
12. regime-delta and MoE decision;
13. development freeze;
14. generated development evidence and reports.

Every stage writes outputs atomically and creates `completion.json` last with hashes of required outputs.

`artifacts/run_state.json` records:

- command and arguments;
- Git revision;
- Python/package/device information;
- device and worker count;
- start/end timestamps;
- status and failure details;
- hashes of every runtime Python/YAML input.

Resume is allowed only when code, configuration, split, V0 source, device, and worker hashes match. `--restart` clears only the requested 2.1 stage and its downstream dependents.

### Development freeze

`development_freeze.json` must hash:

- train and validation files;
- station and fold manifests;
- predictor universe;
- global and MoE configs;
- selected ordered features;
- list form and actual count;
- beta;
- learner parameters and seed;
- router and expert decisions;
- benchmark challenger identity and eligibility;
- every script capable of affecting predictions;
- report generator.

The benchmark runner verifies this freeze before reading `test.csv`.

### Notebook deliverables

Create and manipulate all notebooks through the `nb` CLI in the managed uv environment:

- `pipeline.ipynb`: canonical development runner and complete result-table display.
- `analysis.ipynb`: feature stability and station/year/month diagnostics.
- `moe_analysis.ipynb`: causal MoE and regime-delta comparisons.
- `benchmark_eval.ipynb`: explicit post-freeze 2023–2025 benchmark evaluation.

Notebook cells may only invoke saved scripts or display generated artifacts. Every Markdown cell must explain the purpose, method, result, and interpretation. No result may exist only in notebook memory or untracked inline code.

### Primary outputs

Persist at minimum:

- `oof_predictions.csv.gz`;
- benchmark prediction ledger;
- fold and coverage manifests;
- direct/progressive candidate paths;
- feature ranks and stability tables;
- correlation-component diagnostics;
- `candidate_features.json`;
- `global_promotion_decision.json`;
- `moe_promotion_decision.json`;
- `development_freeze.json`;
- `benchmark_claim.json`;
- overall/year/month/station/station-year/regime metrics;
- paired bootstrap intervals;
- station input-sufficiency classifications;
- MoE causal-ablation tables;
- normalized generated evidence;
- `RESULTS.md`;
- `BENCHMARK_RESULTS.md`;
- `CONTINUATION.md`.

`CONTINUATION.md` must record completed stages, commands, artifact locations, hashes, failures, interpretation boundaries, and the recommended next experiment.

## Test and Acceptance Plan

### Unit tests

Add tests covering:

- strict train-before-origin enforcement;
- complete held-station exclusion from station-time training and candidate generation;
- deterministic, balanced, nonidentical station partitions;
- zero-observation fold rejection and complete current-station coverage;
- no `test.csv` access from development modules;
- exact 1.3-lite parameters;
- native XGBoost missing handling;
- router-only mean imputation;
- normalized beta 0.2 weights and absence of weights for beta 0.0;
- residual sign `truth - prediction`;
- raw-ledger row alignment;
- primary squared-error collapse across repeats;
- secondary mean-prediction collapse across repeats;
- repeat counts not inflating effective validation size;
- stable ordered-feature hashes;
- station-year macro RMSE calculation;
- deterministic hierarchical bootstrap;
- beta arms never pooled as independent folds;
- progressive bridge generation;
- feature-name-invariant ranking and original-position tie-breaking;
- one-standard-error count selection;
- deterministic path-source tie-breaking;
- automatic V0 fallback;
- consensus frequency/rank ordering;
- router train-only fitting;
- target-free cluster alignment;
- regime coverage rules;
- zero-delta fallback;
- MoE promotion prohibition after a failed global gate;
- freeze verification before benchmark evaluation;
- historical prediction/label alignment;
- 0.003 SOTA-margin logic;
- mandatory `unbiased_sota_eligible: false`;
- correct independent `benchmark_sota_eligible` calculation;
- completion-marker corruption detection;
- interrupted-run resume.

### Integration and execution verification

1. Run preflight and reduced CPU smoke stages without reading test data.
2. Run targeted pytest suites through the notebooks uv environment.
3. Verify the smoke run can be interrupted and resumed with identical completion hashes.
4. Execute the full development pipeline from `notebooks/` using CUDA, four workers, and a long timeout.
5. Execute the three development presentation notebooks through `nb ... --uv`.
6. Verify `nb search --with-errors` finds no notebook errors.
7. Run `generate_results.py --check`.
8. Verify the benchmark command refuses to run without `--confirm-benchmark`.
9. Verify it refuses to run before a valid development freeze.
10. Execute the benchmark notebook explicitly after freeze verification.
11. Confirm the report can issue a project-benchmark SOTA verdict but never an unbiased-SOTA verdict.
12. Run `generate_results.py --check` again after benchmark generation.
13. Audit that the completed 2.0 experiment, dataset metadata, prior versioned notebooks, and the user’s unrelated dirty changelog remain unchanged.

### Completion criteria

The experiment is complete when:

- all nine current stations and all required 2017–2022 development years are represented;
- every promoted comparison uses paired identical OOF rows;
- 2021 and 2022 participate in final feature identity and are reported separately;
- direct versus progressive pruning has a saved deterministic verdict;
- global, count, beta, station-sufficiency, regime-delta, and MoE decisions are machine-readable;
- per-year, per-month, per-station, per-station-year, and per-regime evidence comes from saved prediction ledgers;
- no development module reads test data;
- exactly one predeclared 2.1 challenger is exposed to the benchmark alongside V0;
- benchmark feedback does not alter any 2.1 selection artifact;
- `RESULTS.md` contains development evidence without test-driven selection;
- `BENCHMARK_RESULTS.md` contains the disclosed project-SOTA verdict;
- all tests and both report-integrity checks pass;
- `CONTINUATION.md` is sufficient for a fresh-context implementation or follow-up session.

## Assumptions

- All nine derived-8.3 stations remain in scope; performance alone is not a removal criterion.
- Only 2017–2022 data influence any 2.1 configuration choice.
- V0 remains the active development reference and automatic fallback.
- A qualifying 2.1 winner may become the new project benchmark leader, but dataset metadata is not changed automatically.
- The exact 1.3-lite learner is frozen; hyperparameter tuning and new 1.5 training are out of scope.
- The saved historical 1.5 Model 16 result remains a valid benchmark comparator.
- Feature engineering and acquisition of snowpack, soil-temperature, freeze/thaw, or ECE sensor data are follow-up experiments, not silent additions to feature selection.
- MoE work is diagnostic-first and cannot rescue a global-gate failure.
- The 2023–2025 benchmark may support a disclosed project-SOTA claim, but not an unbiased external-generalization claim.
- Any change motivated by benchmark results requires a new versioned experiment.

## 🤖 Codex

I’ll implement this as a new, isolated 2.1 experiment and preserve every existing versioned experiment. I’m using the `notebook-cli` skill because the deliverables include four `.ipynb` files; it requires all notebook work to go through the managed `nb` interface. I’ll first inventory the 2.0 structure, current split/provenance artifacts, tests, and worktree state, then build and verify the version-local pipeline.

## 🤖 Codex

The worktree has one pre-existing unrelated edit (`docs/changelogs/2026-07-21-Pan.md`), which I’ll leave untouched. The derived-8.3 inputs are large enough that the canonical run will be a substantial CUDA job, so I’m separating two goals: fully implement the production pipeline and its safety gates, then run deterministic CPU smoke/resume checks that exercise the same orchestration without consuming the retrospective benchmark.

## 👤 User

- Write the entire plan to PLAN.md
- Due to VM session timeout, just finish the code and I will do the full run them later

## 🤖 Codex

Understood. I’ll finish the implementation, tests, and lightweight verification only; I will not launch the full CUDA development or benchmark runs in this VM session. I’ll also place the complete protocol plan—not a shortened summary—into the new experiment’s `PLAN.md`, so the later execution session has the full decision contract beside the code.

## 🤖 Codex

The core contracts are now in place: exact learner enforcement, development-only loaders, deterministic fold geometry, repeat-safe ledgers, hierarchical bootstrap, pruning/consensus logic, router alignment, atomic completion markers, and freeze verification. I’m now wiring those pieces into the resumable stages; this is where the implementation will enforce that every selected feature list is generated inside its own outer training boundary rather than reused across held-out tasks.

## 👤 User

continue

## 🤖 Codex

Continuing from the completed development-stage wiring. The remaining implementation work is the post-freeze benchmark runner, documentation/notebooks, and the acceptance-focused test suite; I’ll then run syntax and targeted CPU checks only, leaving the full CUDA run to your later session as requested.

## 🤖 Codex

The full user-supplied plan is now preserved in `PLAN.md`, and the benchmark path is implemented with a dynamic post-freeze import, one-time frozen-model registry, exact historical-label alignment, paired station-year bootstrap, and permanently false unbiased-claim fields. I’m moving into tests now; these will exercise the leakage and decision invariants without training the full experiment.

## 🤖 Codex

The targeted contract suite is green: 34 tests passed, covering the fold leakage boundary, repeat collapsing, exact learner and missing-value behavior, bootstrap determinism, fallback/tie rules, router alignment, MoE gating, historical alignment, SOTA margin, completion corruption, and resume fingerprints. The notebook skill now requires me to create the four notebooks through `nb`, add cells in small logical batches, and execute each batch safely; their runner cells will be opt-in so this verification does not start the full CUDA job.

## 🤖 Codex

I’m continuing with the `notebook-cli` skill for the four required notebooks, then I’ll finish the remaining code review and lightweight CPU/static verification. I’ll avoid the full CUDA development and benchmark runs as requested.

## 🤖 Codex

The first notebook batch is valid, but execution hit a host-level Jupyter runtime-directory permission issue before any cell ran. I’m resolving that with a task-scoped runtime directory, then I’ll execute each notebook batch normally.

## 🤖 Codex

All four notebooks now execute cleanly in the managed uv environment with the expensive commands safely opt-in. I’m moving through a final implementation audit now—especially freeze isolation, MoE backbone selection, resume semantics, and report integrity—before rerunning the focused tests.

## 🤖 Codex

The core tests still pass after the stricter consensus and MoE changes. I’ve also corrected two run-time-only defects the original unit tests would not catch: full-universe consensus ordering after progressive pruning, and MoE freezing of the wrong shared backbone. Next I’m exercising the real data preflight and then tightening generated evidence/report checks.

## 🤖 Codex

The reduced screen caught a genuine smoke-only geometry issue: after enforcing the protocol’s inner-fold row minimum, the deliberately tiny smoke sample had 96 rows against the canonical 100-row floor. I’m giving smoke its own explicitly configured lower row floors; canonical thresholds remain unchanged.

## 👤 User

Continue. The session died out again so need to figure out where it was left

## 🤖 Codex

I recovered the current workspace state. The implementation and reduced CPU smoke pipeline had completed successfully; I was in the final provenance/test audit, with the full CUDA and benchmark runs intentionally left for you. I’ll resume from that exact point, clean the smoke artifacts afterward, and leave the experiment code-only.

## 🤖 Codex

I recovered the prior session state. The implementation and reduced CPU smoke run are complete; I’m resuming at the final test/provenance audit, then I’ll remove smoke-only artifacts and leave the experiment ready for your CUDA run. I’m also using the notebook CLI skill for the final notebook checks because this task includes tracked `.ipynb` deliverables.

## 👤 User

Since I'm in a fresh session so execute the full run

## 🤖 Codex

Understood—I’ll run the full canonical CUDA workflow now, including development, notebook execution, integrity checks, and the explicitly gated benchmark after the development freeze. I’ll first finish the small provenance-test fix left by the interrupted session and verify CUDA/preflight so we don’t launch a long run on a known-bad state.

## 🤖 Codex

The interrupted test fixture is repaired, and I added focused coverage for the two late fixes: cross-beta paired bootstrap and complete consensus-order reconstruction. CUDA is available on an H100 with about 81 GB free, so the canonical device requirement is satisfied.

## 🤖 Codex

All 36 focused contract tests pass, and canonical preflight passed with the expected split hashes, station coverage, exact learner, and CUDA probe. I’m launching the full development pipeline now; it will resume from the valid preflight checkpoint and use four independent workers with each XGBoost fit constrained to one job.

## 🤖 Codex

The full run is active. It has moved past the preflight without errors; the only message is XGBoost’s expected CPU-DataFrame-to-GPU prediction warning, which affects transfer overhead but not model semantics. I’m monitoring stage completions and GPU utilization while it runs.

## 🤖 Codex

Stage 2 (fold manifests) is complete, and stage 3 is building the fixed-control OOF ledgers. It has checkpointed 47 units so far and the GPU is sustaining roughly 90% utilization; resumability is working as intended.

## 🤖 Codex

The canonical fold manifest contains 114 OOF tasks (9 forward-time and 105 station-time); stage 3 therefore has 570 checkpointed control-model units. It is at 121/570 with stable GPU load. This is the first deliberately exhaustive stage, so the runtime is materially larger than the earlier smoke run.

## 🤖 Codex

Stage 3 is at 186/570 after about seven minutes, with no failed units. The run writes each fold/candidate ledger atomically and validates completion markers before moving on, so even an external VM interruption would resume from the last completed unit rather than restart the stage.

## 🤖 Codex

The fixed-control stage is now 256/570, still error-free and at full four-worker utilization. It is evaluating V0, all predictors, and the three frozen 2.0 controls under identical rolling-origin geometry; none of these diagnostic controls can be promoted.

## 🤖 Codex

Stage 3 has crossed the midpoint at 321/570. Checkpoint throughput remains steady, and the main process is healthy (roughly 300% CPU across the four workers plus active GPU execution).

## 🤖 Codex

The control ledger has reached 384/570. No warnings beyond the initial XGBoost device-transfer notice, and no completion markers have failed validation.

## 👤 User

good, check back after 10 minutes

## 🤖 Codex

Sounds good. I’ll leave the run undisturbed and check its stage status again in about ten minutes.

## 🤖 Codex

Ten-minute check: stage 3 finished all 570 control units and wrote its aggregate completion marker. Stage 4—the direct-versus-progressive pruning screen—is now active; 2 ranking units plus the earlier directory-level count are visible, with the GPU still around 91% utilization and no errors.

## 🤖 Codex

Second ten-minute check: stage 4 has completed 14/72 pruning-rank units. This stage is slower by design because every unit repeatedly refits the exact 1.3-lite learner while eliminating features. The process remains healthy and GPU-active; at the current rate, this screen should need roughly another 45–55 minutes before its candidate evaluations and verdict.

## 🤖 Codex

Stage 4 is at 24/72 after the next ten-minute interval, still with no failures. The rate is stable at about ten rank units per ten minutes, so its remaining ranking work is approximately 45–50 minutes; the run remains fully active.

## 🤖 Codex

Stage 4 is now 28/72. Progress slowed because the queue has reached progressive pruning units, which refit after each bridge and endpoint reduction (direct units require fewer refits); that change in rate is expected, not a stall. GPU/process health checks remain clean.

## 🤖 Codex

Stage 4 is at 34/72 after 70 minutes total pipeline time. Six additional progressive units completed in this interval; all four workers are still busy and checkpointing normally.

## 🤖 Codex

Stage 4 is at 40/72. The run is progressing at the same progressive-pruning rate and remains healthy; no intervention is needed.

## 🤖 Codex

Stage 4 is at 45/72 after the latest check. The longer 2021/2022 progressive fits are completing normally; process and GPU telemetry remain stable.

## 🤖 Codex

Stage 4 is at 49/72. The current units are the largest-history progressive folds, so only four completed in this ten-minute window; the run is still actively using all workers and has not stalled.

## 🤖 Codex

Stage 4 is at 54/72, leaving 18 ranking units. The process remains error-free and actively computing.

## 🤖 Codex

Stage 4 is at 60/72, with 12 ranking units remaining. GPU utilization is back above 90%; no errors or retries have appeared.

## 🤖 Codex

Stage 4 is at 65/72; only seven ranking units remain. The screen will then evaluate both pruning paths on their paired outer folds before freezing the method verdict.

## 🤖 Codex

Stage 4 ranking is at 71/72—one final rank unit remains. After that, the stage will produce roughly 1,000 comparatively quick OOF evaluation units and calculate the paired direct-versus-progressive verdict.

## 🤖 Codex

All 72 pruning rankings are complete. Stage 4 has entered paired OOF evaluation and checkpointed 77/1,008 candidate/fold units in its first few minutes, with healthy GPU utilization.

## 🤖 Codex

Stage 4 OOF evaluation is at 425/1,008, substantially faster than the ranking phase. The paired screen remains healthy and should finish this evaluation pass in roughly another 15–20 minutes at the current rate.

## 🤖 Codex

Stage 4 OOF evaluation is at 757/1,008. About 251 units remain; current throughput puts the method verdict roughly one more ten-minute interval away.

## 🤖 Codex

The method screen completed and correctly froze **progressive elimination** (direct did not satisfy the predeclared paired upper-CI rule). Stage 5 is now running the full seven-repeat, three-permutation stability ranking using only that frozen method. This is the longest phase of the experiment.

## 🤖 Codex

Stage 5 has completed 5/228 full stability-ranking units in its first ten-minute interval. The observed rate implies this canonical phase alone will take several hours (roughly 7–8 hours if it stays constant); the run is checkpointed and healthy, and I’ll keep it running.

## 🤖 Codex

Stage 5 is at 11/228 after the second interval. Throughput is steady at roughly six full progressive-ranking units per ten minutes; all four workers remain active and no unit has failed.

## 🤖 Codex

Stage 5 is at 17/228, again adding six units in ten minutes. This confirms the several-hour estimate; the run remains healthy and resumable.

## 👤 User

Heads up: me VM session going to timeout in 10 minutes, so pause the work and I'm going to continue on the new session

## 🤖 Codex

Understood. I’m stopping the runner cleanly now. Completed units will remain valid; only the currently active, unmarked rank units will be recomputed when you resume.

## 🤖 Codex

The runner is fully stopped: no Python process or GPU job remains. It preserved 20/228 completed stage-5 rankings; two interrupted unit directories lack completion markers and will be safely recomputed. The journal records the intentional `KeyboardInterrupt`, which the normal resume path accepts without `--restart`.

## 🤖 Codex

Paused cleanly.

- Canonical stages 1–4 are complete.
- Stage 5 has 20/228 valid ranking checkpoints.
- Two interrupted units will recompute automatically.
- Progressive elimination was selected.
- No benchmark data were read.
- No runner or GPU process remains active.

Resume from `notebooks/` without `--restart`:

```bash
uv run python experiment/derived_8.3-feature-selection-2.1/run_all.py \
  --device cuda --workers 4
```

Full handoff details are recorded in [CONTINUATION.md](/scratch/user/u.rp352032/MDR-Project/notebooks/experiment/derived_8.3-feature-selection-2.1/CONTINUATION.md).

## 👤 User

Continue the work and increase the worker count to 6

## 🤖 Codex

I’ll resume with six workers without discarding the valid checkpoints. Because worker count is part of the provenance fingerprint, I’ll make this an explicit audited 4→6 concurrency migration; model seeds, folds, learner settings, and completed unit hashes remain unchanged. I’m also using the notebook CLI skill to update the canonical notebook commands from four to six workers before relaunching.

## 👤 User

no need to update the default one, just change the parameter of this run

## 🤖 Codex

Got it—I’ll leave every default, config, document, and notebook command at four workers. I’ll apply six only to this resumed execution and record the one-run concurrency override in the run journal; feature/model settings and existing checkpoints will remain unchanged.

## 🤖 Codex

The override is now narrowly scoped: the default remains four, but this resumed invocation may use six only after verifying unchanged splits, device, configuration, external inputs, all completed stage markers, and every reusable stage-5 checkpoint. The final freeze will include the immutable override record and the actual worker count.

## 🤖 Codex

All 37 focused tests pass, including the new checkpoint-preserving worker-migration test. I’m relaunching the interrupted development run with `--workers 6 --allow-worker-resume`; the default configuration remains four.

## 🤖 Codex

The audited migration succeeded exactly as intended: stages 1–4 and all 20 completed stage-5 units were reused, the journal now records six workers, and six rank units are active. XGBoost still uses `n_jobs=1` per fit; GPU utilization is healthy.

## 🤖 Codex

Ten-minute check: stage 5 is at 23/228, with six more units currently active. Three finished in this first interval because the resumed queue began with the two interrupted units and the current progressive folds are long-running; the GPU remains around 90% utilization with no failures.

## 🤖 Codex

Stage 5 is at 30/228, adding seven completed rankings in the latest interval; six additional units are in flight. The six-worker run is stable and has improved aggregate throughput over the original four-worker pace.

## 🤖 Codex

Stage 5 is at 37/228, again completing seven units in ten minutes. Process health and checkpoint validation remain normal.

## 🤖 Codex

Stage 5 is at 44/228, maintaining seven completed rankings per ten-minute interval with six units in flight.

## 👤 User

check after 20 minutes interval

## 🤖 Codex

Will do—I’ll switch to 20-minute progress checks while leaving the six-worker run undisturbed.

## 🤖 Codex

Twenty-minute check: stage 5 is at 58/228, up 14 units, with six more active. Throughput and GPU health remain steady; no failures have occurred.

## 🤖 Codex

Twenty-minute check: stage 5 is at 71/228, up 13 units, with six active. The run remains stable and error-free.

## 🤖 Codex

Twenty-minute check: stage 5 is at 79/228, with six active. This interval completed eight units because the queue has entered longer-history folds; the process remains healthy at about 90% GPU utilization.

## 🤖 Codex

Twenty-minute check: stage 5 is at 85/228. The six active units are now 2021-origin station folds, whose larger training histories make them slower; all remain actively computing with no failures.

## 🤖 Codex

Twenty-minute check: stage 5 is at 91/228, with six 2021-origin units active. The slower rate is consistent with the larger causal training windows; health checks remain clean.

## 🤖 Codex

Twenty-minute check: stage 5 is at 98/228, up seven, with six units active. No errors or checkpoint corruption are present.

## 🤖 Codex

Twenty-minute check: stage 5 is at 104/228, with six active. It remains healthy and is approaching the midpoint of the full stability-ranking matrix.

## 🤖 Codex

Twenty-minute check: stage 5 is at 110/228, with six active and no errors. The full ranking matrix is nearly halfway complete.