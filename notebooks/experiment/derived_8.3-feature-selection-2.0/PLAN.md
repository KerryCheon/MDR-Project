# Reproduction plan for `derived_8.3-feature-selection-2.0`

## Objective and change boundary

This experiment is an isolated rerun of the complete
`derived_8.2-feature-selection-2.2` protocol. It keeps `derived_8.0` as the
control and replaces only the current-data arm with `derived_8.3`.

The allowed substitutions are:

- experiment directory, configuration identity, artifact version, and model
  labels become `derived_8.3-feature-selection-2.0` and `2.0_*`;
- the current dataset becomes `derived_8.3`;
- the current-data historical comparison becomes the single 50-feature
  `OVERALL_SELECTED_FEATURES_V0` baseline;
- the current-data router becomes `Clustering_V0_Full_k2` over that exact
  ordered V0 list.

Candidate sizes, folds, confidence rules, temporal betas, XGBoost parameters,
seeds, weighting, selector logic, diagnostics, and the derived 8.0 arm must not
change. `preflight.py` compares the relevant target configuration sections with
the 2.2 source configuration before a run.

## Data and interpretation contract

The source roles remain:

| Split | Years | Original arm | Nested/crossed arms |
|---|---:|---|---|
| train | 2017–2020 | combined development pool | inner candidate generation |
| validation | 2021–2022 | combined development pool | disjoint outer candidate choice |
| test | 2023–2025 | retrospective evaluation | retrospective evaluation |

`derived_8.3-eval-1.0` already consumed the derived 8.3 test split. Consequently,
`--confirm-final` is disabled, every test manifest is marked
`unbiased_sota_eligible: false`, and no result from this experiment is a new
unbiased SOTA claim.

## Baseline and router contract

`data_loading.py` resolves
`data/splits/derived_8.3/dataset_metadata.py::OVERALL_SELECTED_FEATURES_V0`
without executing the metadata module. It requires exactly 50 unique feature
names and preserves their source order.

The V0-full router contract is fixed:

```text
kind: clustering_v0_full_k2
imputation: mean
scaler: StandardScaler
n_clusters: 2
n_init: 10
random_state: 42
columns: exact ordered OVERALL_SELECTED_FEATURES_V0 list
```

The original arm fits preprocessing and K-means on its combined development
pool. It writes `router_provenance.json` but does not freeze that fit for later
evaluation; evaluation refits on its own fit frame, matching 2.2 behavior.

The nested arm fits preprocessing and K-means on inner training data only,
applies the router to the outer frame, and saves the frozen means, scaler, and
centers in `router.json`. Both router artifacts include the ordered columns,
feature source, and source-file SHA-256.

## Saved implementation

The experiment contains normal tracked Python entry points for every
calculation. The four notebooks are orchestration or read-only presentation
layers and contain no independent result-selection logic.

The master runner executes exactly 13 resumable stages:

1. original global and regime selection;
2. original-arm development validation;
3. original-arm retrospective evaluation;
4. nested global and regime selection;
5. locked outer re-score;
6. crossed candidate selection;
7. derived 8.0 progressive crossed selection;
8. nested candidate diagnostics;
9. crossed candidate diagnostics;
10. progressive candidate diagnostics;
11. nested retrospective evaluation;
12. crossed retrospective evaluation;
13. report generation.

Every stage writes atomically, records hashes in completion markers, and is
journaled in `artifacts/run_state.json`. Stage fingerprints cover all experiment
Python and YAML runtime inputs. A running or failed journal resumes with the
same device and worker count; a completed journal starts a new clean rebuild.

## Preflight and reduced verification

From `notebooks/`:

```bash
uv run python experiment/derived_8.3-feature-selection-2.0/preflight.py

uv run python experiment/derived_8.3-feature-selection-2.0/run_selection.py \
  --dataset derived_8.3 --smoke --device cuda --workers 4

uv run python experiment/derived_8.3-feature-selection-2.0/run_nested_selection.py \
  --dataset derived_8.3 --smoke --device cuda --workers 4
```

The preflight must prove that V0 contains 50 unique features, every derived 8.3
split contains all 50, both router configurations resolve to the exact same
ordered list, the non-router protocol matches 2.2, and no runtime source retains
V3, C1, derived 8.2 data, or the old router scope.

Every runner must also support `--help` without performing work. Selector,
artifact-state, diagnostic, and report tests run before the long pipeline.

## Canonical execution

From `notebooks/`:

```bash
nb execute experiment/derived_8.3-feature-selection-2.0/pipeline.ipynb \
  --uv --timeout 86400
```

The notebook invokes `run_all.py --device cuda --workers 4`. If execution is
interrupted, rerun the same command until all 13 journal stages are complete.
Do not invoke `--restart` unless discarding the incomplete experiment-local
artifact tree is intentional.

After the master run, execute the read-only notebooks:

```bash
nb execute experiment/derived_8.3-feature-selection-2.0/analysis.ipynb \
  --uv --timeout 600
nb execute experiment/derived_8.3-feature-selection-2.0/eval.ipynb \
  --uv --timeout 600
nb execute experiment/derived_8.3-feature-selection-2.0/nested_analysis.ipynb \
  --uv --timeout 600
```

Use `nb search <notebook> --with-errors` for all four notebooks. Expected
evaluation rows include the derived 8.0 MDR-v25 baseline and derived 8.3 V0,
`2.0_global`, and V0-full MoE rows.

## Reporting and completion

`generate_results.py` owns every reported calculation. It writes normalized
evidence CSVs, generates `RESULTS.md`, replaces only the protected generated
block in `CONTINUATION.md`, records report and source hashes, and writes the
report completion marker last.

After inspecting the completed evidence, revise only the human-authored text
outside the protected continuation block, then rerun:

```bash
uv run python experiment/derived_8.3-feature-selection-2.0/generate_results.py
uv run python experiment/derived_8.3-feature-selection-2.0/generate_results.py --check
```

Completion requires:

- all 13 journal stages are complete;
- every required completion marker verifies;
- split hashes agree across selection, evaluation, diagnostics, and report
  artifacts;
- both router artifacts match the ordered V0 source and source hash;
- all four notebooks execute without errors;
- the report generator passes `--check`;
- a final diff audit finds no selector or model-setting drift beyond the
  allowed dataset, baseline, router, identity, and label substitutions.
