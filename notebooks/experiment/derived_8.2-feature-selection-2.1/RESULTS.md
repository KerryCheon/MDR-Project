# Results: derived_8.2-feature-selection-2.1 (V6, locked eval)

**Date:** 2026-07-20  
**Eval protocol:** 1.3-lite XGBoost; train on train+val; score on test; **CUDA**; mean-normalized drift weights (β=0.2); no median impute; Bias/ubRMSE match MDR-v25 / opt-1.0.  
See [`PROTOCOL.md`](PROTOCOL.md).

| Protocol | Sample weights |
|----------|----------------|
| **With drift** | `w = exp(0.2 · (year − t_max)) / mean(w)` |
| **No drift** | uniform (unweighted) |

**Sanity:** 8.0 hand + drift R² = **0.8253** matches opt-1.0 Model 5 exactly (Δ = 0.0000).

## Success gates (±0.01 R² of baseline)

| Protocol | Dataset | Baseline | Best V6 | Δ R² | Gate |
|----------|---------|----------|---------|------|------|
| With drift | 8.0 | hand **0.8253** | v6_c2d_xgb_softcorr_k65 **0.8152** | -0.0101 | **FAIL** |
| With drift | 8.2 | V3 **0.6376** | v6_c1_baseline_bypass_off **0.6615** | +0.0239 | **PASS** |
| No drift | 8.0 | hand **0.8222** | v6_c2d_xgb_softcorr_k65 **0.8148** | -0.0075 | **PASS** |
| No drift | 8.2 | V3 **0.6551** | v6_c1_baseline_bypass_off **0.6648** | +0.0097 | **PASS** |

### Gate summary vs 2.0

| Protocol | Dataset | 2.0 | 2.1 |
|----------|---------|-----|-----|
| With drift | 8.0 | FAIL (hand 0.8178, best 0.8050) | **FAIL** (hand 0.8253, best c2d **0.8152**, miss by ~0.0001) |
| With drift | 8.2 | PASS | **PASS** (c1 +0.024 vs V3) |
| No drift | 8.0 | FAIL | **PASS** (c2d −0.0075 vs hand) |
| No drift | 8.2 | PASS | **PASS** |

## Leaderboards (test)

### derived_8.0 — with drift

| Feature set | n | R² | RMSE | MAE | Pearson |
|-------------|--:|---:|-----:|----:|--------:|
| hand_mdr_v25 | 38 | 0.8253 | 0.0394 | 0.0281 | 0.9090 |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.8152 | 0.0405 | 0.0287 | 0.9033 |
| v6_c2b_xgb_softcorr | 55 | 0.8101 | 0.0410 | 0.0294 | 0.9006 |
| v6_c5_rf | 50 | 0.8097 | 0.0411 | 0.0294 | 0.9002 |
| v6_c4_hybrid | 50 | 0.8074 | 0.0413 | 0.0299 | 0.8988 |
| v6_c2_xgb | 50 | 0.8066 | 0.0414 | 0.0296 | 0.8986 |
| v6_c3_xgb_no_coverage | 50 | 0.8066 | 0.0414 | 0.0296 | 0.8986 |
| opt1.0_pipeline | 50 | 0.7836 | 0.0438 | 0.0312 | 0.8895 |
| v6_c2c_xgb_nocorr | 55 | 0.7830 | 0.0439 | 0.0320 | 0.8919 |
| v6_c1_baseline_bypass_off | 41 | 0.7788 | 0.0443 | 0.0333 | 0.8884 |
| v6_c0_baseline_bypass_on | 44 | 0.7517 | 0.0469 | 0.0343 | 0.8715 |

### derived_8.0 — no drift

| Feature set | n | R² | RMSE | MAE | Pearson |
|-------------|--:|---:|-----:|----:|--------:|
| hand_mdr_v25 | 38 | 0.8222 | 0.0397 | 0.0286 | 0.9071 |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.8148 | 0.0405 | 0.0291 | 0.9029 |
| v6_c2b_xgb_softcorr | 55 | 0.8096 | 0.0411 | 0.0297 | 0.9002 |
| v6_c4_hybrid | 50 | 0.8066 | 0.0414 | 0.0298 | 0.8985 |
| v6_c5_rf | 50 | 0.8047 | 0.0416 | 0.0301 | 0.8972 |
| v6_c2_xgb | 50 | 0.8014 | 0.0420 | 0.0301 | 0.8958 |
| v6_c3_xgb_no_coverage | 50 | 0.8014 | 0.0420 | 0.0301 | 0.8958 |
| v6_c1_baseline_bypass_off | 41 | 0.7890 | 0.0432 | 0.0324 | 0.8933 |
| v6_c2c_xgb_nocorr | 55 | 0.7852 | 0.0436 | 0.0320 | 0.8929 |
| opt1.0_pipeline | 50 | 0.7849 | 0.0437 | 0.0314 | 0.8896 |
| v6_c0_baseline_bypass_on | 44 | 0.7561 | 0.0465 | 0.0342 | 0.8727 |

### derived_8.2 — with drift

| Feature set | n | R² | RMSE | MAE | Pearson |
|-------------|--:|---:|-----:|----:|--------:|
| v6_c1_baseline_bypass_off | 50 | 0.6615 | 0.0613 | 0.0449 | 0.8269 |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.6524 | 0.0621 | 0.0467 | 0.8251 |
| v6_c2b_xgb_softcorr | 55 | 0.6518 | 0.0621 | 0.0469 | 0.8268 |
| v6_c0_baseline_bypass_on | 50 | 0.6460 | 0.0627 | 0.0467 | 0.8318 |
| v6_c2c_xgb_nocorr | 55 | 0.6440 | 0.0628 | 0.0471 | 0.8227 |
| v6_c2_xgb | 50 | 0.6400 | 0.0632 | 0.0478 | 0.8220 |
| v6_c3_xgb_no_coverage | 50 | 0.6400 | 0.0632 | 0.0478 | 0.8220 |
| v6_c4_hybrid | 50 | 0.6378 | 0.0634 | 0.0482 | 0.8280 |
| V3_sota | 47 | 0.6376 | 0.0634 | 0.0475 | 0.8281 |
| v6_c5_rf | 50 | 0.6370 | 0.0634 | 0.0476 | 0.8229 |
| V5_bad | 32 | 0.5968 | 0.0669 | 0.0496 | 0.7889 |

### derived_8.2 — no drift

| Feature set | n | R² | RMSE | MAE | Pearson |
|-------------|--:|---:|-----:|----:|--------:|
| v6_c1_baseline_bypass_off | 50 | 0.6648 | 0.0610 | 0.0450 | 0.8266 |
| v6_c2d_xgb_softcorr_k65 | 65 | 0.6620 | 0.0612 | 0.0462 | 0.8289 |
| v6_c0_baseline_bypass_on | 50 | 0.6597 | 0.0614 | 0.0460 | 0.8386 |
| v6_c2b_xgb_softcorr | 55 | 0.6581 | 0.0616 | 0.0466 | 0.8284 |
| V3_sota | 47 | 0.6551 | 0.0618 | 0.0464 | 0.8373 |
| v6_c3_xgb_no_coverage | 50 | 0.6522 | 0.0621 | 0.0473 | 0.8267 |
| v6_c2_xgb | 50 | 0.6522 | 0.0621 | 0.0473 | 0.8267 |
| v6_c2c_xgb_nocorr | 55 | 0.6488 | 0.0624 | 0.0470 | 0.8248 |
| v6_c5_rf | 50 | 0.6477 | 0.0625 | 0.0471 | 0.8279 |
| v6_c4_hybrid | 50 | 0.6452 | 0.0627 | 0.0478 | 0.8293 |
| V5_bad | 32 | 0.5975 | 0.0668 | 0.0499 | 0.7894 |

## vs 2.0 (protocol-only re-score; same feature lists)

| Dataset | Protocol | Feature set | 2.0 R² | 2.1 R² | Δ |
|---------|----------|-------------|-------:|-------:|--:|
| derived_8.0 | drift | hand_mdr_v25 | 0.8178 | 0.8253 | +0.0075 |
| derived_8.0 | drift | v6_c2b_xgb_softcorr | 0.8050 | 0.8101 | +0.0051 |
| derived_8.0 | no-drift | hand_mdr_v25 | 0.8240 | 0.8222 | -0.0018 |
| derived_8.0 | no-drift | v6_c2b_xgb_softcorr | 0.8069 | 0.8096 | +0.0027 |
| derived_8.2 | drift | V3_sota | 0.6376 | 0.6376 | -0.0000 |
| derived_8.2 | drift | v6_c1_baseline_bypass_off | 0.6581 | 0.6615 | +0.0034 |
| derived_8.2 | no-drift | V3_sota | 0.6534 | 0.6551 | +0.0017 |
| derived_8.2 | no-drift | v6_c1_baseline_bypass_off | 0.6605 | 0.6648 | +0.0043 |

### Observations

1. **Hand on 8.0 with drift rose 0.8178 → 0.8253** after mean-normalizing weights (+ no impute) — closes the opt-1.0 gap.
2. **Best auto on 8.0 is now c2d** (65 feats, R² 0.8152 drift / 0.8148 no-drift), not c2b. Rank order shifted under the fixed protocol.
3. **c2b still strong** on 8.0 (~0.810) but no longer the peak auto set.
4. **c1 remains best on 8.2** under both protocols and still beats V3.
5. Other metrics (RMSE, MAE, Pearson) move with R²; Pearson stays very stable (~0.90 on 8.0 hand, ~0.83 on 8.2 V3).

## Recommended production configs

| Use case | Config | Artifact |
|----------|--------|----------|
| Peak 8.0 auto (fixed protocol) | `configs/config_c2d_xgb_softcorr_k65.yaml` | `artifacts/derived_8.0/c2d_xgb_softcorr_k65/` |
| Peak 8.2 global | `configs/config_c1_baseline_bypass_off.yaml` | `artifacts/derived_8.2/c1_baseline_bypass_off/` |
| Compact soft-corr compromise | `configs/config_c2b_xgb_softcorr.yaml` | `artifacts/<ds>/c2b_xgb_softcorr/` |

## Artifacts

- `artifacts/eval/metrics_summary.csv` — full metric bundle
- `artifacts/eval/metrics_by_year.csv`
- `artifacts/eval/success_gates.json`
- `artifacts/eval/r2_comparison.png` (and weighted/unweighted)
- `artifacts/eval/run_eval.log`
