# Simplified Reproduction Plan for Feature Selection 2.2

## Summary

- Save every computation used to produce reported numbers in normal Python scripts inside the experiment directory.
- Make `pipeline.ipynb` the single entry point that runs the complete experiment.
- Delete the existing 2.2 artifacts and regenerate them in place; do not preserve separate dirty copies.
- Generate `RESULTS.md` from the produced CSV and JSON artifacts instead of manually copying numbers.
- Remove Git commit hashes, dirty-tree hashes, source-tree hashes, and the two-commit workflow mentioned in the previous plan.
- Retain only lightweight completion markers needed to resume after a VM timeout.

## 1. Saved experiment code

### Master runner

Add `run_all.py` without a shebang. It becomes the canonical entry point and calls the existing saved scripts in this order:

1. `run_selection.py` for the global and regime selections.
2. `run_eval.py` for development validation.
3. `run_eval.py` for retrospective 2023-2025 evaluation of the final selection.
4. `run_nested_selection.py`.
5. `run_locked_outer_selection.py`.
6. `run_crossed_candidate_selection.py`.
7. `run_crossed_candidate_selection.py --progressive --dataset derived_8.0`.
8. `run_candidate_diagnostics.py` for nested candidates.
9. `run_candidate_diagnostics.py` for crossed candidates.
10. `run_candidate_diagnostics.py` for progressive candidates.
11. `run_eval.py` for nested retrospective evaluation.
12. `run_eval.py` for crossed retrospective evaluation.
13. `generate_results.py`.

Default runtime:

```text
device: cpu
workers: 16
XGBoost n_jobs per fit: 1
```

CLI:

```text
--device {cpu,cuda}   default: cpu
--workers INTEGER     default: 16
--restart             discard an incomplete run and begin again
```

The runner must use `sys.executable` and saved script paths. It must not use inline Python, heredocs, temporary `/scratch` scripts, or notebook-only calculations.

### Evaluation interface

Extend `run_eval.py` so the existing `final` artifact set can be evaluated retrospectively without claiming a new unbiased final result.

The clean rerun must label every 2023-2025 result:

```json
{
  "retrospective_test": true,
  "unbiased_sota_eligible": false
}
```

The previous one-shot-final interpretation is removed because the test period has already been consumed.

### Reporting code

Add `generate_results.py`. It owns all calculations used in `RESULTS.md`, including:

- selecting rows from evaluation summaries;
- selecting the best diagnostic candidate;
- calculating differences between global and MoE models;
- feature counts;
- pass/fail comparisons;
- rounding display values;
- rendering Markdown tables and numeric narrative statements.

Expose reusable functions such as:

```python
load_selection_summary()
build_validation_table()
build_retrospective_table()
build_candidate_ceiling_table()
build_moe_table()
render_results_markdown()
```

The notebooks import these functions instead of independently recreating the calculations.

## 2. Clean artifact rebuild and restart behavior

### Clean start

On a new completed run, `run_all.py` deletes the entire existing experiment `artifacts/` content before starting.

This removes and replaces:

- `final`;
- `smoke`;
- `nested_smoke`;
- `nested`;
- `nested_locked_outer`;
- `crossed_candidates_locked_outer`;
- `progressive_crossed_locked_outer`;
- all evaluation and candidate-diagnostic outputs.

No backup or additional artifact version is created. Git history remains the only historical copy.

The clean run recreates only the full artifact sets needed by the final experiment. Smoke artifacts are not regenerated unless their smoke runners are invoked separately.

### Automatic resume

Store a small journal at:

```text
artifacts/run_state.json
```

It contains:

```json
{
  "status": "running",
  "device": "cpu",
  "workers": 16,
  "stages": [
    {
      "name": "nested_selection",
      "command": ["..."],
      "status": "complete"
    }
  ]
}
```

Behavior:

- If no journal exists, start clean.
- If the journal says `complete`, running `pipeline.ipynb` again starts a new clean rebuild.
- If the journal says `running` or `failed`, resume from the first incomplete stage.
- `--restart` forces a clean rebuild even if an incomplete journal exists.
- Mark a stage complete only after all its required output files and completion marker are valid.
- If a stage fails, stop immediately and retain previous completed stages for the next VM session.
- Write `run_state.json` atomically.

Completion markers continue to store hashes of required output files so partial artifacts cannot be reused. They do not store Git or source hashes.

## 3. Notebook workflow

### `pipeline.ipynb`

Make this the only notebook required to reproduce the complete experiment.

Running all cells must:

1. locate the repository;
2. display the exact master-runner command;
3. invoke `run_all.py` unconditionally;
4. resume automatically if the prior VM stopped mid-run;
5. load the newly generated result tables through `generate_results.py`;
6. display every numeric table used in `RESULTS.md`;
7. print the path to each underlying CSV or JSON artifact.

Remove `RUN_SMOKE = False` and `RUN_FULL = False`. A new user must not need to discover or change hidden flags before anything runs.

Include a prominent note that the full run is long-running, defaults to 16 CPU workers, and can be resumed by running the notebook again.

### Analysis notebooks

Keep the remaining notebooks as read-only detailed views:

- `analysis.ipynb` displays final selection and fold diagnostics.
- `eval.ipynb` displays validation and retrospective evaluation tables.
- `nested_analysis.ipynb` displays nested, crossed, progressive, and MoE diagnostics.

Move numerical filtering, grouping, best-row selection, deltas, and rounding out of notebook cells and into `generate_results.py`. Notebook cells should only call saved functions and display returned DataFrames.

Every number referenced in `RESULTS.md` must appear in at least one executed notebook output. The master `pipeline.ipynb` must display all of them even if the detailed notebooks also display subsets.

All notebook edits and execution must use `nb`.

## 4. Traceable `RESULTS.md`

`RESULTS.md` becomes generated output. Its first lines must state:

```markdown
<!-- Generated by generate_results.py. Do not edit numeric tables manually. -->
```

For each section, include:

- the script responsible for producing the source artifact;
- the artifact file used;
- the relevant model, dataset, beta, or candidate filter;
- the resulting generated table.

Example:

```markdown
Source:
- Produced by `run_eval.py`
- Artifact: `artifacts/nested/retrospective_test_eval/metrics_summary.csv`
- Filter: dataset=`derived_8.2`, model=`2.2_global`, beta=`0.0`
```

All displayed values, including values embedded in prose such as architecture-performance differences, must be inserted by `generate_results.py`.

Do not maintain a separate metric ledger, Markdown template, source-tree manifest, or commit-based provenance system. The saved generator and referenced artifact rows provide the traceback.

The report generator must fail if:

- an expected artifact is missing;
- an expected row is absent or duplicated;
- a metric is non-finite;
- a required experiment stage is incomplete;
- a retrospective result is incorrectly marked SOTA-eligible.

## 5. Manifests and metadata

Remove these fields and related code from active artifacts:

- `git_commit`;
- `git_dirty`;
- `dirty_entries`;
- `git_status_sha256`;
- `tracked_diff_sha256`;
- `source_tree_sha256`;
- `source_files`;
- `source_file_count`.

Retain useful experiment metadata:

- creation time;
- exact saved runner and arguments;
- configuration values;
- random seed;
- dataset and split hashes;
- device;
- worker count;
- selected feature count;
- stopping reason;
- retrospective and unbiased status;
- required-output hashes in completion markers.

Simplify `artifact_state.py` to atomic JSON and CSV writing plus completion-marker verification. Remove Git inspection entirely.

## 6. Tests and final rerun

### Unit tests

Update tests to cover:

1. completion markers reject missing outputs;
2. completion markers reject modified outputs;
3. interrupted stages resume without rerunning completed stages;
4. a completed run starts clean when executed again;
5. `--restart` clears an incomplete run;
6. only the known experiment artifact directory can be deleted;
7. default device is CPU;
8. default worker count is 16;
9. each XGBoost fit uses one internal worker;
10. every runner supports `--help` without performing work;
11. master-runner stages execute in the required order;
12. report-table row selection is deterministic;
13. report rounding is deterministic;
14. the report generator rejects missing or duplicate rows;
15. every reported retrospective result is marked ineligible for an unbiased SOTA claim.

Run:

```bash
uv run --project notebooks --with pytest pytest \
  tests/selectors_test.py \
  tests/artifact_state_test.py \
  tests/feature_selection_diagnostics_test.py \
  -q
```

### Full execution

Run the master notebook:

```bash
nb execute \
  notebooks/experiment/derived_8.2-feature-selection-2.2/pipeline.ipynb \
  --uv --timeout 86400
```

Then execute the detailed notebooks:

```bash
nb execute \
  notebooks/experiment/derived_8.2-feature-selection-2.2/analysis.ipynb \
  --uv --timeout 600

nb execute \
  notebooks/experiment/derived_8.2-feature-selection-2.2/eval.ipynb \
  --uv --timeout 600

nb execute \
  notebooks/experiment/derived_8.2-feature-selection-2.2/nested_analysis.ipynb \
  --uv --timeout 600
```

Verify no notebook errors:

```bash
nb search \
  notebooks/experiment/derived_8.2-feature-selection-2.2/pipeline.ipynb \
  --with-errors

nb search \
  notebooks/experiment/derived_8.2-feature-selection-2.2/analysis.ipynb \
  --with-errors

nb search \
  notebooks/experiment/derived_8.2-feature-selection-2.2/eval.ipynb \
  --with-errors

nb search \
  notebooks/experiment/derived_8.2-feature-selection-2.2/nested_analysis.ipynb \
  --with-errors
```

### Completion criteria

The cleanup is complete when:

- running `pipeline.ipynb` alone rebuilds the complete experiment;
- rerunning it after an interrupted VM resumes automatically;
- all previous artifact trees have been replaced in place;
- all numerical calculations exist in committed Python scripts;
- `RESULTS.md` is generated from artifacts;
- every reported number appears in notebook output;
- no Git commit or source-state provenance remains;
- no Python file contains `#!/usr/bin/env python3`;
- all tests and notebook executions pass.
