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
