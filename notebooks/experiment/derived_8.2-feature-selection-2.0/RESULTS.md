# Results: derived_8.2-feature-selection-2.0 (V6)

**Date:** 2026-07-17 (eval re-run with dual protocols)  
**Eval protocol:** XGBoost 1.3-lite (`n_estimators=1500`, `lr=0.01`, `max_depth=8`, `min_child_weight=10`); train on train+val; score on test.

Two training weight regimes:

| Protocol | Sample weights |
|----------|----------------|
| **With drift** | `w = exp(-0.2 * (t_max − year))` |
| **No drift** | uniform (unweighted) |

## Success gates

| Protocol | Dataset | Baseline | Best V6 | Δ R² | Gate (±0.01) |
|----------|---------|----------|---------|------|--------------|
| **With drift** | 8.0 | hand **0.8178** | c2b **0.8050** | −0.0127 | **FAIL** (miss by 0.0027) |
| **With drift** | 8.2 | V3 **0.6376** | c1 **0.6581** | **+0.0205** | **PASS** |
| **No drift** | 8.0 | hand **0.8240** | c2b **0.8069** | −0.0171 | **FAIL** |
| **No drift** | 8.2 | V3 **0.6534** | c1 **0.6605** | **+0.0071** | **PASS** |

## Leaderboards (test R²)

### derived_8.0 — with drift

| Feature set | n | R² |
|-------------|--:|---:|
| hand_mdr_v25 | 38 | **0.8178** |
| **v6_c2b_xgb_softcorr** | 55 | **0.8050** |
| v6_c4_hybrid | 50 | 0.8043 |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.8035 |
| v6_c5_rf | 50 | 0.8032 |
| v6_c2_xgb / c3 | 50 | 0.8016 |
| v6_c1_baseline_bypass_off | 41 | 0.7820 |
| opt1.0_pipeline | 50 | 0.7790 |
| v6_c2c_xgb_nocorr | 55 | 0.7765 |
| v6_c0_baseline_bypass_on | 44 | 0.7577 |

### derived_8.0 — no drift

| Feature set | n | R² |
|-------------|--:|---:|
| hand_mdr_v25 | 38 | **0.8240** |
| **v6_c2b_xgb_softcorr** | 55 | **0.8069** |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.8065 |
| v6_c4_hybrid | 50 | 0.8040 |
| v6_c2_xgb / c3 | 50 | 0.8025 |
| v6_c5_rf | 50 | 0.7996 |
| v6_c1_baseline_bypass_off | 41 | 0.7866 |
| opt1.0_pipeline | 50 | 0.7812 |
| v6_c2c_xgb_nocorr | 55 | 0.7793 |
| v6_c0_baseline_bypass_on | 44 | 0.7614 |

### derived_8.2 — with drift

| Feature set | n | R² |
|-------------|--:|---:|
| **v6_c1_baseline_bypass_off** | 50 | **0.6581** |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.6524 |
| v6_c2c_xgb_nocorr | 55 | 0.6485 |
| v6_c2b_xgb_softcorr | 55 | 0.6449 |
| v6_c0_baseline_bypass_on | 50 | 0.6420 |
| v6_c5_rf | 50 | 0.6415 |
| v6_c2_xgb / c3 | 50 | 0.6403 |
| V3_sota | 47 | 0.6376 |
| v6_c4_hybrid | 50 | 0.6369 |
| V5_bad | 32 | 0.5994 |

### derived_8.2 — no drift

| Feature set | n | R² |
|-------------|--:|---:|
| **v6_c1_baseline_bypass_off** | 50 | **0.6605** |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.6603 |
| v6_c0_baseline_bypass_on | 50 | 0.6588 |
| v6_c2b_xgb_softcorr | 55 | 0.6552 |
| V3_sota | 47 | 0.6534 |
| v6_c2_xgb / c3 | 50 | 0.6502 |
| v6_c2c_xgb_nocorr | 55 | 0.6498 |
| v6_c4_hybrid / c5_rf | 50 | 0.6489 |
| V5_bad | 32 | 0.5974 |

## Drift vs no-drift observations

1. **Hand on 8.0 is stronger without drift** (0.824 vs 0.818); the auto gap to hand is slightly larger unweighted (−0.017 vs −0.013).
2. **V3 on 8.2 is stronger without drift** (0.653 vs 0.638) — closer to the 0.655 figure from hyperparams-1.3-lite / eval-3.1 (which often used unweighted or different weight setups).
3. **Rank order is largely stable:** c2b remains best auto on 8.0 under both protocols; c1 remains best on 8.2 under both.
4. **c2b on 8.2 no-drift (0.655)** still beats V3 (0.653) and is a good single-config compromise across datasets.

## Recommended production configs

| Use case | Config | Artifact |
|----------|--------|----------|
| Default general FS | `configs/config_c2b_xgb_softcorr.yaml` | `artifacts/<ds>/c2b_xgb_softcorr/` |
| Peak 8.2 global | `configs/config_c1_baseline_bypass_off.yaml` | `artifacts/derived_8.2/c1_baseline_bypass_off/` |

## Artifacts

- `artifacts/eval/metrics_summary.csv` — 44 rows (22 sets × 2 protocols)
- `artifacts/eval/metrics_by_year.csv`
- `artifacts/eval/success_gates.json` — `{with_drift, no_drift}`
- `artifacts/eval/r2_comparison.png` (2×2), `r2_comparison_weighted.png`, `r2_comparison_unweighted.png`

## Library changes

- `Selectors/xgb_importance.py`, `family_coverage.py`, stability `base=xgb`, opt-in bypass  
- Tests: `tests/selectors_test.py` (10 passed)
