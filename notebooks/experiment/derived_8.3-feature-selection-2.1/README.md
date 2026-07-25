# derived_8.3-feature-selection-2.1

This isolated experiment implements evaluation-first feature selection for all nine derived-8.3 Washington stations. Only 2017–2022 train and validation rows may influence feature identity, count, beta, routing, expert structure, or promotion. The 2023–2025 split is a separately confirmed, reused retrospective project benchmark.

The production learner is the exact 1.3-lite XGBoost configuration in `global_config.yaml`. It uses native XGBoost missing-value handling, seed-controlled rolling-origin folds, five deterministic station partitions, station-year macro RMSE, paired hierarchical bootstrap intervals, automatic V0 fallback, and a separate causal MoE configuration. `OVERALL_SELECTED_FEATURES_V0` is loaded and hashed but never overwritten.

## Commands

Run these commands from `notebooks/`:

```text
uv run python experiment/derived_8.3-feature-selection-2.1/preflight.py --device cuda --workers 4

uv run python experiment/derived_8.3-feature-selection-2.1/run_all.py \
  --device cuda --workers 4

uv run python experiment/derived_8.3-feature-selection-2.1/run_benchmark.py \
  --confirm-benchmark --device cuda --workers 4

uv run python experiment/derived_8.3-feature-selection-2.1/generate_results.py
uv run python experiment/derived_8.3-feature-selection-2.1/generate_results.py --check
```

`run_all.py` is development-only and cannot invoke the benchmark. `--restart` restarts all 2.1 stages; `--restart <stage-name>` removes only that stage and downstream 2.1 artifacts. A lightweight, explicitly noncanonical CPU path is available with `--smoke --device cpu --workers 1` for orchestration checks.

## Outputs and resume behavior

Each stage writes under `artifacts/development/stages/` and creates `completion.json` only after hashing every required output. Per-fold ranking and prediction units have their own completion markers. `artifacts/development/run_state.json` records the command, Git revision, package/runtime information, device, workers, failures, and hashes of runtime Python/YAML inputs and split files. Resume is rejected if that fingerprint changes.

The canonical development handoff is `development_freeze.json`. The benchmark runner verifies every frozen code, configuration, split, fold, feature, beta, router, expert, and report-generator hash before its benchmark module can read `test.csv`. A completed frozen model ID cannot be benchmarked twice.

## Interpretation boundary

The benchmark is reused and retrospective because previous derived-8.3 work already inspected 2023–2025 behavior. A qualifying challenger may establish project SOTA on that disclosed benchmark, but `unbiased_sota_eligible` and `unbiased_generalization_claim_eligible` always remain false. Independent confirmation must come from the forthcoming ECE sensor deployment. Any change motivated by benchmark results belongs in a new experiment version, such as 2.2.

