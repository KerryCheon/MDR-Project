# derived_8.2-feature-selection-2.0

Generalizable, XGBoost-aligned feature selection (V6) for soil moisture models.

## Goals

1. Match hand-selected MDR-v25 features on **`derived_8.0`** (test R² within ~0.01 of ~0.825 under 1.3-lite).
2. Do **not** degrade **`derived_8.2`** vs SOTA **V3** (R² ≥ ~0.645; prefer ≥ 0.655).
3. No hard-coded feature-name bypass; soft structural family coverage only.
4. Fully versioned / traceable artifacts.

## Layout

| Path | Purpose |
|------|---------|
| `analysis.ipynb` | Autopsy of past feature sets (hand, V0–V5, opt-1.0) |
| `pipeline.ipynb` | Run selection variants C0–C5 on 8.0 and 8.2 |
| `eval.ipynb` | Train XGBoost (1.3-lite hparams) and score test R² |
| `configs/` | Versioned YAML configs for each ablation |
| `run_selection.py` | CLI runner used by `pipeline.ipynb` |
| `artifacts/<dataset>/<variant>/` | `selected_features.json`, reports, run logs |

## Variants

| ID | Stages | Intent |
|----|--------|--------|
| **c0** | MI→EN→stab, bypass ON | Legacy baseline |
| **c1** | MI→EN→stab, bypass OFF | Measure bypass dependence; **best on 8.2** |
| **c2** | corr(0.95)→xgb→coverage→stab(xgb) | Primary proposal (too aggressive corr) |
| **c2b** | corr(0.99)→xgb→coverage→stab | **Best general / best on 8.0 auto** |
| **c2c** | xgb→coverage→stab (no corr) | Ablate correlation |
| **c2d** | softcorr + top_k=65 | Larger set experiment |
| **c3** | corr→xgb→stab (no coverage) | Ablate coverage |
| **c4** | corr→MI→xgb→coverage→stab | Hybrid |
| **c5** | corr→rf→coverage→stab(rf) | RF vs XGB ranker |

See **[RESULTS.md](RESULTS.md)** for the full leaderboard and gate status.

## Fixed eval hyperparameters (from `derived_8.2-hyperparameters-1.3-lite`)

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
}
```

Eval trains **both** protocols for every feature set:

| Protocol | Weights |
|----------|---------|
| With drift | temporal β=0.2 |
| No drift | unweighted |

## Quick start

From repo root:

```bash
# Analysis (overlap tables, family histograms)
cd notebooks && uv run --with pytest jupyter execute ../notebooks/experiment/derived_8.2-feature-selection-2.0/analysis.ipynb

# Selection (all variants, both datasets) — long running
PYTHONPATH=. python notebooks/experiment/derived_8.2-feature-selection-2.0/run_selection.py

# Smoke test (fewer bootstraps)
PYTHONPATH=. python notebooks/experiment/derived_8.2-feature-selection-2.0/run_selection.py \
  --variants c2_xgb --dataset derived_8.2 --n-boot 10
```

## Success gates (executed)

| Protocol | Dataset | Target | Best V6 | Status |
|----------|---------|--------|---------|--------|
| With drift | 8.0 | hand 0.818 | c2b **0.805** (Δ −0.013) | near-miss; **+0.026 vs old pipeline** |
| With drift | 8.2 | V3 0.638 | c1 **0.658** (Δ +0.020) | **PASS** |
| No drift | 8.0 | hand 0.824 | c2b **0.807** (Δ −0.017) | FAIL vs hand |
| No drift | 8.2 | V3 0.653 | c1 **0.660** (Δ +0.007) | **PASS** |

Recommended default config: `configs/config_c2b_xgb_softcorr.yaml` (no feature-name bypass). Full dual-protocol leaderboards: **[RESULTS.md](RESULTS.md)**.

## Library changes (this experiment)

- `Modeling/Src/soilmoist_fl/Selectors/xgb_importance.py`
- `Modeling/Src/soilmoist_fl/Selectors/family_coverage.py`
- Stability `base="xgb"`; CLI stages `xgb_importance`, `family_coverage`
- Name-based MI bypass is now **opt-in** via `selection.bypass.enabled`
