# derived_8.3-feature-selection-2.1 Evaluation-First Feature Selection

  ## Summary

  Create an isolated experiment under notebooks/experiment/derived_8.3-feature-selection-2.1/ without modifying the completed 8.2/2.2 or 8.3/2.0 experiments.

  The experiment will:

  - Keep all nine derived_8.3 stations and use only 2017–2022 for development.
  - Use the exact 1.3-lite XGBoost learner for ranking and evaluation.
  - Make V0 the reference and retain it unless a new candidate passes robust development gates.
  - Include 2021–2022 both as rolling-origin validation and in the final all-development consensus ranking.
  - Treat year-dependent feature utility as a stability problem rather than immediately creating year-specific models.
  - Diagnose monthly and difficult-station failures from row-level OOF predictions.
  - Use separate global and MoE configurations.
  - Always run MoE causal diagnostics, but prohibit MoE promotion if the global gate fails.
  - Keep 2023–2025 behind a separate retrospective command and never use it to choose features, beta, or architecture.

  No derived_8.0 models will be retrained. Its completed results remain historical context.

  ## Interpretation Boundary and SOTA Eligibility

  The 2023–2025 results cannot support a new unbiased SOTA claim because derived_8.3-feature-selection-1.0, derived_8.3-eval-1.0, the error analyses, and derived_8.3-feature-selection-2.0 have already inspected those labels and used their year, month, station, and
  regime behavior to motivate this experiment. Keeping the files physically separate now prevents further leakage, but it cannot undo the adaptive choices already made.

  Accordingly:

  - 2023–2025 results will be labeled retrospective_test: true and unbiased_sota_eligible: false.
  - Development promotion means only “candidate for a future holdout,” not a new active baseline or SOTA.
  - OVERALL_SELECTED_FEATURES_V0 will not be overwritten automatically.
  - A new unbiased claim requires untouched future observations and/or newly deployed ECE stations evaluated after the full 2.1 contract is frozen.

  ## Global Selection Design

  ### Data and fold geometry

  - Load only train.csv and val.csv for the canonical pipeline; reject dates after 2022 and any development path resolving to test.csv.
  - Use rolling outer origins 2020, 2021, and 2022. For origin (t), training is strictly years < t and validation is year t.
  - Generate two fold families:
      - forward_time: all prior-year stations train; all stations in the outer year validate.
      - station_time: prior-year rows from held-out stations are excluded, and those stations validate in the outer year.

  - Build five deterministic row-balanced station partitions from seed = 42 + i, i ∈ [0,4], and reuse each mapping across origins.
  - Separate partition and learner uncertainty without a full Cartesian sweep:
      - partition seeds 42–46 with learner seed 42;
      - learner seeds 42–44 with partition seed 42;
      - deduplicate the shared (42, 42) run, producing seven station-time repeats.

  - Forward-time runs use learner seeds 42–44.
  - Reject zero-observation station-year folds and persist assigned versus actually observed coverage.

  For each outer origin, feature candidates must be generated only from earlier years. Inner ranking uses up to the last two eligible rolling years after at least two earlier training years. Thus 2021 contributes to the 2022 candidate ranking, while both 2021 and 2022
  contribute to outer selection risk and the final all-development ranking.

  ### Learner and pruning path

  Use the exact 1.3-lite learner everywhere:

  - reg:squarederror, depth 8, minimum child weight 10;
  - 1,500 estimators, learning rate 0.01;
  - L2 1.5, L1 0.03, subsample 0.9, column sample 0.8;
  - histogram trees, native missing-value handling, n_jobs=1;
  - CUDA by default, four independent workers, seed supplied by the repeat configuration.

  Retain the historical endpoint counts [150, 125, 100, 80, 65, 50, 40]. Progressive bridge counts are generated from the starting 496 predictors with the existing rule next_size = max(target_size, current_size - target_size).

  Run a base-seed direct-versus-progressive screen across both fold families and all three outer origins. Select direct elimination only if its paired 95% upper confidence bound for primary-risk difference versus progressive is below zero; otherwise freeze progressive
  elimination.

  Feature ranking will:

  - remain feature-name and family agnostic;
  - use three deterministic permutation repeats in the full stability stage;
  - score the change in station-year macro RMSE rather than normalized fold RMSE;
  - rank by importance lower confidence bound, mean importance, then original column position;
  - keep beta out of ranking so correlated beta evaluations cannot inflate the effective fold count.

  A post-selection correlation diagnostic will form training-only Spearman correlation components at the historical |ρ| ≥ 0.95 threshold and jointly permute each component. This explains correlated substitutes and path instability but cannot change the selected list.

  ### Prediction ledger and risk

  Persist raw OOF predictions as compressed CSV with:

  - candidate/source/count and ordered-feature hash;
  - fold family, outer origin, fold ID, station partition and learner seeds;
  - station, date, year, month;
  - truth, prediction, residual = truth - prediction, and squared error;
  - beta and model configuration;
  - router regime and distance when applicable.

  Repeated predictions for the same candidate, fold family, origin, station, and date are collapsed by averaging squared error before computing selection metrics, preventing seed repetitions from acting as independent observations.

  Primary risk is station-year macro RMSE:

  1. Compute RMSE for each observed station × outer-origin block.
  2. Average blocks equally within each fold family.
  3. Combine forward-time and station-time risks with equal 50/50 weight.

  Use 2,000 paired hierarchical bootstrap replicates, seed 42, resampling stations and then outer years while retaining paired candidate/V0 observations.

  Secondary diagnostics include pooled RMSE/R², MAE, Pearson correlation, bias, station-macro RMSE, 90th-percentile station-year RMSE, worst-station RMSE, monthly RMSE/bias, and target dispersion alongside every subgroup R².

  ### Candidate, count, beta, and final-list decisions

  Evaluate candidates from both the station-time and forward-time progressive paths, plus fixed controls for V0, all numeric predictors, and the completed 2.0 selected lists.

  A new global candidate is eligible only when:

  - the upper endpoint of its paired 95% bootstrap interval for combined RMSE difference versus V0 is below zero;
  - its point estimate does not regress in either fold family;
  - its 90th-percentile station-year RMSE, worst-station RMSE, and 90th-percentile monthly RMSE do not exceed V0 by more than one paired bootstrap standard error.

  Among eligible candidates, apply the one-standard-error rule and choose the smallest count within one standard error of the best combined risk. Break remaining ties by higher selection stability, then the fixed path-source order station_time, forward_time.

  If no candidate passes, record the lowest-risk candidate as diagnostic only and keep V0 as the reference.

  After method and count are frozen:

  - Compare beta 0.0 and 0.2 as separate paired arms.
  - Select beta 0.2 only if its combined 95% upper confidence bound versus beta 0.0 is below zero and neither fold family regresses in point estimate; otherwise select beta 0.0.
  - Rerun the frozen selector over all 2017–2022 development data.
  - Build the exact-count consensus list by sorting features by selection frequency, median percentile rank, mean percentile rank, then original column position.
  - Report origin-specific lists, rank correlations, Jaccard overlap, V0 overlap, and which features gain or lose support in 2021–2022. Do not deploy separate year-specific models in this experiment.

  ## Station, Month, and MoE Diagnostics

  ### Station and monthly sufficiency

  Keep Marten Ridge, Rainy Pass, and every other current station in scope.

  For each station and month, compare V0, the 2.1 candidate, and the all-predictor diagnostic under identical OOF geometry. This distinguishes:

  - feature-selection failure, when the all-predictor control helps but the selected list does not;
  - current-input limitation, when V0, selected, and all-predictor models all fail similarly;
  - low-target-variance R² artifacts versus genuinely high RMSE or bias.

  Report fixed-2017–2019 versus expanding-window fits for V0 and the chosen candidate as a diagnostic only. Produce month tables for all months rather than special-casing October, while highlighting transition-month failures in the interpretation.

  No stations will be pruned and no snowpack, soil-temperature, or freeze/thaw inputs will be added in 2.1. The report will identify stations for which those new data sources are the appropriate next experiment.

  ### Separate MoE configuration

  Use moe_config.yaml, distinct from the global selection configuration, while retaining the frozen Clustering_V0_Full_k2 router:

  - exact ordered V0 router inputs;
  - mean imputation, StandardScaler, K-means K=2, n_init=10, seed 42;
  - router preprocessing fitted only on each outer training frame;
  - cluster IDs aligned without target labels to a 2017–2019 reference router using centroid-distance matching.

  Run this causal matrix under the same OOF ledger:

   Arm                                         Question answered
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   V0 single global                            Reference
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   2.1 single global                           Did selection improve the global model?
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   2.1 ∪ V0 single global                      Were router-only/V0 inputs useful to a global model?
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   V0 shared hard experts                      Does routing help or fragment a known feature set?
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   2.1 shared hard experts                     Does sample fragmentation explain loss versus one global fit?
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   2.1 ∪ V0 shared hard experts                Does full router-input access change that conclusion?
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   Strongest shared control + regime deltas    Is there incremental regime-specific feature signal?
  ──────────────────────────────────────────  ───────────────────────────────────────────────────────────────
   Saved eval-1.0 specialist lists             Does the historical 4/47-feature collapse reproduce?

  Regime deltas retain additions [0, 5, 10, 15]; no new count search is introduced. Select a delta separately for each regime only when:

  - the regime occurs in every outer origin and at least ceil(9/2) = 5 stations;
  - its paired 95% improvement interval versus shared-only excludes zero;
  - it does not worsen that regime’s worst-station RMSE.

  Always produce per-regime population, target dispersion, station/year/month composition, route distance, and performance tables. These comparisons explicitly distinguish missing router inputs, hard-routing/sample-fragmentation loss, and per-regime feature-selection
  loss.

  MoE remains diagnostic if the global gate fails. If the global gate passes, an MoE may become a future-holdout candidate only if it significantly beats the strongest single-global control and passes the same station/month robustness guards.

  ## Implementation, Interfaces, and Reproducibility

  Create versioned scripts, global_config.yaml, moe_config.yaml, and resumable artifacts inside notebooks/experiment/derived_8.3-feature-selection-2.1/. Keep the existing shared selector and completed experiment directories unchanged; new repeated-fold, ledger,
  bootstrap, consensus, and causal-ablation logic will be version-local.

  The public command surface will be:

  - run_all.py --device {cpu,cuda} --workers N [--restart] for development-only stages;
  - run_retrospective.py --confirm-retrospective for the separately authorized 2023–2025 report;
  - generate_results.py [--check] for read-only report verification.

  The development runner will checkpoint after every origin/family/seed/candidate unit and journal these stages:

  1. data and coverage preflight;
  2. deterministic fold manifests;
  3. V0, all-feature, and 2.0 control ledgers;
  4. direct/progressive screen;
  5. repeated robust candidate generation and evaluation;
  6. method/count and beta decisions;
  7. all-development consensus list;
  8. station/month/year and correlation diagnostics;

  development_freeze.json will hash the train/validation files, configs, feature list, beta, learner, router/MoE decision, and report code. The retrospective runner must verify this freeze before reading test.csv and cannot rewrite any selection artifact.

  Create notebooks with nb in the managed uv environment:

  - pipeline.ipynb: canonical development runner and complete result-table display;
  - analysis.ipynb: feature stability and station/year/month diagnostics;
  - moe_analysis.ipynb: causal MoE comparisons;
  - retrospective_eval.ipynb: explicit post-freeze 2023–2025 evaluation.

  Notebook cells will only invoke saved scripts or display generated artifacts. Every Markdown cell will include explanatory prose, and every reported number will be owned by tracked Python code. RESULTS.md, CONTINUATION.md, and the separate RETROSPECTIVE.md will be
  generated or protected by generated evidence blocks.

  Key outputs include:

  - oof_predictions.csv.gz;
  - fold and coverage manifests;
  - candidate paths and feature-stability tables;
  - candidate_features.json, promotion_decision.json, and development_freeze.json;
  - overall/year/month/station/regime metrics and paired confidence intervals;
  - MoE causal-ablation tables;
  - generated development and retrospective reports.

  ## Test and Acceptance Plan

  Add tests covering:

  - strict train-before-origin and held-station exclusion;
  - deterministic, balanced, nonidentical station partitions;
  - zero-observation fold rejection and complete current-station coverage;
  - no test.csv access from development modules;
  - exact 1.3-lite parameters, native missing handling, normalized temporal weights, and residual sign;
  - ledger row alignment, repeat collapsing, and feature-hash stability;
  - station-year macro RMSE and deterministic hierarchical bootstrap calculations;
  - beta arms never pooled as independent folds;
  - progressive bridge generation and name-invariant tie-breaking;
  - one-standard-error count selection and automatic V0 fallback;
  - consensus selection frequency/rank ordering;
  - router train-only fitting, target-free cluster alignment, and regime coverage rules;
  - MoE promotion prohibition after a failed global gate;
  - freeze verification before retrospective evaluation;
  - mandatory unbiased_sota_eligible: false on every 2023–2025 artifact;
  - completion-marker corruption detection and interrupted-run resume.

  Verification sequence:

  1. Run preflight and reduced CPU smoke stages without reading test data.
  2. Run targeted pytest suites through the notebooks uv environment.
  3. Execute the development pipeline from notebooks/ with nb execute --uv, CUDA, four workers, and a long timeout.
  4. Execute the three development presentation notebooks and verify nb search --with-errors returns no errors.
  5. Run generate_results.py --check.
  6. Only after freeze verification, execute the retrospective notebook explicitly and verify its report carries no promotion or SOTA verdict.
  7. Audit that the completed 2.0 experiment, dataset metadata, and the user’s unrelated dirty changelog remain unchanged.

  ## Assumptions

  - All nine derived_8.3 stations remain in scope; performance alone is not a removal criterion.
  - V0 remains the active reference regardless of retrospective 2.1 performance.
  - Only train and validation data influence any 2.1 choice.
  - The exact 1.3-lite learner is frozen; hyperparameter tuning and the 1.5 recipe are out of scope.
  - Feature engineering and acquisition of snowpack/freeze-thaw data are follow-up experiments, not silent additions to feature selection.
  - MoE work is diagnostic-first and cannot rescue a global-gate failure.
