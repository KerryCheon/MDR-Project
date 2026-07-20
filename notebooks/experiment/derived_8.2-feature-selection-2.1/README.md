# derived_8.2-feature-selection-2.1

Self-contained V6 feature-selection experiment (copy of 2.0 tooling + **locked eval protocol**).

Copy this whole directory for further experiments; paths resolve via `Path(__file__)` / `EXP_DIR` under **this** folder.

## What changed vs 2.0

| Area | Change |
|------|--------|
| Eval protocol | Mean-normalized drift weights, no median impute, Bias/ubRMSE match MDR-v25/opt-1.0, **CUDA** when available |
| Selection | Same V6 configs + artifacts (seeded from 2.0; re-runnable) |
| Packaging | Full tree: configs, pipeline, analysis, selection runner, eval, artifacts |

See [`PROTOCOL.md`](PROTOCOL.md) for the locked eval checklist. Results after re-eval: [`RESULTS.md`](RESULTS.md).

## Layout

| Path | Purpose |
|------|---------|
| `analysis.ipynb` | Autopsy of past feature sets (hand, V0–V5, opt-1.0) |
| `pipeline.ipynb` | Run selection variants C0–C5 on 8.0 and 8.2 |
| `eval.ipynb` | Train XGBoost (1.3-lite) and score test metrics |
| `run_selection.py` | CLI selection runner |
| `run_eval.py` | CLI eval runner (preferred for batch re-score) |
| `configs/` | Versioned YAML configs for each ablation |
| `artifacts/<dataset>/<variant>/` | `selected_features.json`, reports, run logs |
| `artifacts/eval/` | Metrics CSVs, gates, R² figures |
| `PROTOCOL.md` | Locked eval protocol |
| `RESULTS.md` | Leaderboards + success gates |

## Variants

| ID | Stages | Intent |
|----|--------|--------|
| **c0** | MI→EN→stab, bypass ON | Legacy baseline |
| **c1** | MI→EN→stab, bypass OFF | Measure bypass dependence; often best on 8.2 |
| **c2** | corr(0.95)→xgb→coverage→stab(xgb) | Primary proposal |
| **c2b** | corr(0.99)→xgb→coverage→stab | Soft-corr generalist |
| **c2c** | xgb→coverage→stab (no corr) | Ablate correlation |
| **c2d** | softcorr + top_k=65 | Larger set |
| **c3** | corr→xgb→stab (no coverage) | Ablate coverage |
| **c4** | corr→MI→xgb→coverage→stab | Hybrid |
| **c5** | corr→rf→coverage→stab(rf) | RF vs XGB ranker |

## Fixed eval hyperparameters (1.3-lite)

```python
XGB_PARAMS_LITE = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "min_child_weight": 10,
    "reg_lambda": 1.5,
    "reg_alpha": 0.03,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "n_estimators": 1500,
    "learning_rate": 0.01,
    "tree_method": "hist",
    "device": "cuda",  # if probe succeeds
    "random_state": 42,
}
```

| Protocol | Sample weights |
|----------|----------------|
| With drift | `w = exp(β·(year−t_max)) / mean(w)`, β=0.2 |
| No drift | unweighted |

## Quick start

From repo root, notebooks venv (GPU):

```bash
# Re-score feature sets (does not re-run selection)
PYTHONPATH=. notebooks/.venv/bin/python \
  notebooks/experiment/derived_8.2-feature-selection-2.1/run_eval.py

# Optional: re-run selection (long)
PYTHONPATH=. python notebooks/experiment/derived_8.2-feature-selection-2.1/run_selection.py

# Smoke test selection
PYTHONPATH=. python notebooks/experiment/derived_8.2-feature-selection-2.1/run_selection.py \
  --variants c2_xgb --dataset derived_8.2 --n-boot 10
```

## Success gates (executed)

| Protocol | Dataset | Target | Best V6 | Status |
|----------|---------|--------|---------|--------|
| With drift | 8.0 | hand 0.8253 | c2d **0.8152** (Δ −0.0101) | FAIL (miss by ~0.0001) |
| With drift | 8.2 | V3 0.6376 | c1 **0.6615** (Δ +0.024) | **PASS** |
| No drift | 8.0 | hand 0.8222 | c2d **0.8148** (Δ −0.0075) | **PASS** |
| No drift | 8.2 | V3 0.6551 | c1 **0.6648** (Δ +0.010) | **PASS** |

Full tables: **[RESULTS.md](RESULTS.md)**.

## Sanity

8.0 hand + drift should land near opt-1.0 Model 5 (**R² ≈ 0.8253** on GPU). Residual ΔR² ≲ 0.003 is normal env noise.

## Library changes (from 2.0; still apply)

- `Modeling/Src/soilmoist_fl/Selectors/xgb_importance.py`
- `Modeling/Src/soilmoist_fl/Selectors/family_coverage.py`
- Stability `base="xgb"`; CLI stages `xgb_importance`, `family_coverage`
- Name-based MI bypass is **opt-in** via `selection.bypass.enabled`
