# derived_8.2-feature-selection-2.2

This experiment replaces pooled-row selector importance with predictive utility measured on future-year, held-out-station folds. It uses no feature-name bypass, hand-feature seed, or domain-family quota.

> **Frozen original plus diagnostic revision:** all four original final gates failed. The requested nested and crossed-fold architecture work continues in separate artifact trees in this directory. 
> Since 2023–2025 was consumed by the original run, every revision test metric is retrospective and cannot be a new unbiased SOTA claim. See [`RESULTS.md`](RESULTS.md).

## Architectural diagnosis

The 2.1 pipeline computes MI, ElasticNet coefficients, correlation, and tree gain on pooled training rows. 
Its stability stage then bootstraps rows, so a feature is called stable when it repeatedly explains the same station/year mixture.
Neither station IDs nor timestamps reach the importance scorer. Sequential filters can also discard interaction-only or correlated substitute features before the final model evaluates them. Repeating that process inside regimes reduces sample and target variance, causing unstable 20/1-feature specialists.

2.2 instead measures paired validation loss changes. Each fold trains on earlier years while excluding a station group, then validates on that group in a future year. Feature utility is the lower confidence bound of the increase in normalized RMSE after permutation. Iterative refitting makes substitute features visible after stronger correlates are removed. Exact importance ties are resolved by original column position, so renaming a feature cannot change selection.

## Selection contract

- One algorithm and configuration for derived_8.0 and derived_8.2.
- The 2017–2022 train/validation period is used for grouped selection.
- The 2023–2025 test period is untouched until `run_eval.py --confirm-final`.
- The global feature set is a shared backbone for both K=2 experts.
- A regime-specific delta is admitted only when its paired improvement over the shared backbone has a positive one-standard-error lower confidence bound.
- Feature families are diagnostic only and never affect selection.

The configured candidate sizes reproduce a transparent search around the historical 38–65-feature operating range, with wider 80/100-feature controls.
The selector chooses the size minimizing the upper confidence bound of grouped OOF normalized RMSE; it does not use the test set.

The nested revision uses 2017–2020 for importance/candidate generation and 2021–2022 only for locked-model outer selection. 
A crossed arm creates two candidate paths—pure forward time and future held-out stations—before outer labels are seen. This tests the fundamental assumption that one joint-shift permutation score can represent both temporal and spatial generalization.

All independent XGBoost fits default to 16 parallel workers, with `n_jobs: 1` inside each fit to avoid nested oversubscription. Selection configs can lower `parallel_workers`; evaluation scripts can override the default with `XGB_PARALLEL_WORKERS`. 
Set `XGB_DEVICE=cpu` for multi-core evaluation; running 16 concurrent fits against one GPU can stall because they contend for the same device.

## Reproduction

From the repository root:

```bash
# Fast wiring check; writes only under artifacts/smoke
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_selection.py --smoke

# Full frozen selection on both datasets
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_selection.py

# Development evaluation on 2021-2022 only
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_eval.py

# One-shot final evaluation after the design and lists are frozen
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_eval.py \
  --confirm-final

# Nested inner/outer revision (does not read test.csv)
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_nested_selection.py

# Re-score saved candidates with the locked final learner
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_locked_outer_selection.py

# Union forward-time and station/time candidate paths; fully resumable
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_crossed_candidate_selection.py

# Diagnose one-shot batch pruning without hard-coded features
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.2/run_crossed_candidate_selection.py \
  --progressive --dataset derived_8.0
```

The notebooks are thin, narrative front ends around these scripts. 
JSON and CSV artifacts contain the exact config, split hashes, fold membership, importance uncertainty, selection path, and stopping reason. 
Each active artifact also records whether Git was dirty, hashes the tracked diff and all runtime source files (including untracked experiment code), and writes a hash-verified `completion.json` last. 
Resumable runners reject missing, altered, or source-incompatible checkpoints.

## Final gates

| Dataset/model | Required held-out test result |
|---|---:|
| derived_8.0 global, drift | R² > 0.8253479 |
| derived_8.0 global, no drift | R² > 0.8222 |
| derived_8.2 global | R² > 0.6648 |
| derived_8.2 Clustering Dynamic K=2 | R² > 0.6672 |

`RESULTS.md` records executed results. A failed gate remains a failed research result; the test split is not reused for another 2.2 tuning cycle.
