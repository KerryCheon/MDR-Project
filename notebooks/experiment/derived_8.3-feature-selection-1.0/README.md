# derived_8.3-feature-selection-1.0

Feature selection experiment on `derived_8.3` splits based on `derived_8.2-feature-selection-2.1` tooling.

## Overview

This experiment runs feature selection variants C0–C5 on `derived_8.3` and evaluates performance on the test set using the locked XGBoost protocol in `eval.ipynb`.

## Layout

| Path | Purpose |
|------|---------|
| `pipeline.ipynb` | Run selection variants C0–C5 on derived_8.3 |
| `eval.ipynb` | Train XGBoost (1.3-lite) and score test metrics (sole evaluation driver) |
| `analysis.ipynb` | Family composition and overlap analysis |
| `run_selection.py` | CLI selection runner |
| `configs/` | Versioned YAML configs for variants C0–C5 |
| `artifacts/` | `selected_features.json`, reports, metrics, gates |
| `PROTOCOL.md` | Locked eval protocol |
| `RESULTS.md` | Leaderboards and summary |

## Variants

| ID | Stages | Intent |
|----|--------|--------|
| **c0** | MI→EN→stab, bypass ON | Legacy baseline |
| **c1** | MI→EN→stab, bypass OFF | Measure bypass dependence |
| **c2** | corr(0.95)→xgb→coverage→stab(xgb) | Primary proposal |
| **c2b** | corr(0.99)→xgb→coverage→stab | Soft-corr generalist |
| **c2c** | xgb→coverage→stab (no corr) | Ablate correlation |
| **c2d** | softcorr + top_k=65 | Larger set |
| **c3** | corr→xgb→stab (no coverage) | Ablate coverage |
| **c4** | corr→MI→xgb→coverage→stab | Hybrid |
| **c5** | corr→rf→coverage→stab(rf) | RF vs XGB ranker |

## Execution

```bash
# Feature Selection CLI:
uv run python notebooks/experiment/derived_8.3-feature-selection-1.0/run_selection.py --dataset derived_8.3

# Evaluation & Metrics:
nb execute notebooks/experiment/derived_8.3-feature-selection-1.0/eval.ipynb
```
