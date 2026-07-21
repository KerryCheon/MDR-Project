# Continuation handoff

## Review-fix completion state

The position-stable tie-break, dirty-source provenance, atomic writes, and hash-verified completion markers are implemented. 
The following active trees were regenerated successfully and contain valid per-scope completion markers:

- `artifacts/nested`;
- `artifacts/nested_locked_outer`;
- `artifacts/crossed_candidates_locked_outer`;
- `artifacts/progressive_crossed_locked_outer`;
- the candidate-diagnostic folders under nested, crossed, and progressive;
- nested and crossed retrospective evaluation folders.

The first retrospective evaluation attempt stalled because 16 concurrent fits contended for one GPU.
Evaluation now supports an explicit device override; use CPU to run the 16 independent workers:

```bash
XGB_DEVICE=cpu PYTHONPATH=. uv run --project notebooks python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_eval.py \
  --artifact-set nested --retrospective-test
```

Current tie-safe selections are nested 40/65 features, both regime deltas empty, crossed 40/50, and progressive derived_8.0 at 80.

## Saved state

Everything needed to resume is under this experiment directory. The original one-shot result remains frozen under `artifacts/final`; later work never writes there.

Completed revision artifacts:

- `artifacts/nested`: station/time inner ranking and nested outer selection;
- `artifacts/nested_locked_outer`: the same candidates rescored with the exact 1,500-tree evaluation learner;
- `artifacts/crossed_candidates_locked_outer`: forward-time and station/time candidate paths unioned before locked outer selection;
- `artifacts/progressive_crossed_locked_outer`: resumable progressive-pruning diagnostic;
- each revision's `candidate_diagnostics` and `retrospective_test_eval` folders are labeled ineligible for an unbiased SOTA claim.

The long runners checkpoint each completed dataset and reuse saved inner paths,
so rerunning after a VM restart does not redo completed work.

## Current conclusions

1. Position-stable tie handling supersedes the earlier name-dependent nested results. The derived_8.2 global model reaches only 0.6352, below V3 at 0.6537 and 2.1 c1 at 0.6605.
2. derived_8.0 reaches 0.7644 versus 0.8246 for the hand set.
3. The complete crossed candidate ceilings are 0.8002 for derived_8.0 and 0.6351 for derived_8.2 at beta 0.0.
4. Progressive elimination raises the derived_8.0 ceiling to 0.8209 at 80 features, still below the hand benchmark.
5. Crossed derived_8.2 selects 50 forward-time features and reaches only 0.6090.
6. Both regime deltas are empty. Shared-only K=2 falls from 0.6352 to 0.6279, and router refitting lowers it to 0.6252.
7. No selection arm uses hard-coded feature bypasses, hand seeds, or domain family quotas.

## Runtime and verification

Parallel work defaults to 16 independent workers and each XGBoost fit uses`n_jobs: 1`. 
Override script concurrency with `XGB_PARALLEL_WORKERS` or selector concurrency with `parallel_workers` in YAML.

```bash
uv run --project notebooks --with pytest pytest tests/selectors_test.py -q

PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_crossed_candidate_selection.py

PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_candidate_diagnostics.py \
  --artifact-set crossed_candidates_locked_outer

PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_eval.py \
  --artifact-set nested --retrospective-test
```

The next scientifically valid SOTA decision needs a new untouched temporal evaluation source. 
Within the consumed data, further choices are architecture diagnostics only.
