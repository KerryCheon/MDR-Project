# Results: derived_8.4-feature-selection-1.0

**Date:** 2026-07-26  
**Eval protocol:** 1.3-lite XGBoost trained on train+val, scored on test (`derived_8.4`); CUDA accelerated; dual protocol (drift-weighted $\beta=0.2$ vs unweighted).

---

## Leaderboard (`derived_8.4` test set)

### Unweighted (No drift) — Primary Selection Protocol

| Rank | Feature Set | n | Test R² | RMSE | MAE | Pearson | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **`8.4_V0`** | **50** | **0.7595** | **0.0500** | **0.0382** | **0.8767** | **Reference baseline from derived_8.4 dataset_metadata** |
| 2 | `8.3_V0` | 50 | 0.7595 | 0.0500 | 0.0382 | 0.8767 | Identical feature set |
| 3 | `V3_sota` | 47 | 0.7504 | 0.0509 | 0.0390 | 0.8784 | Historical 8.2 baseline |
| 4 | `v6_c0_baseline_bypass_on` | 46 | 0.7322 | 0.0527 | 0.0405 | 0.8655 | Legacy MI/EN baseline |
| 5 | `v6_c2d_xgb_softcorr_k65` | 65 | 0.7319 | 0.0527 | 0.0394 | 0.8655 | Larger soft-corr set |
| 6 | `v6_c2b_xgb_softcorr` | 55 | 0.7231 | 0.0536 | 0.0400 | 0.8619 | Soft-corr generalist |
| 7 | `v6_c5_rf` | 50 | 0.7191 | 0.0540 | 0.0413 | 0.8594 | Random Forest ranker |
| 8 | `v6_c2_xgb` | 50 | 0.7135 | 0.0545 | 0.0416 | 0.8583 | Primary XGB selection |
| 9 | `v6_c2c_xgb_nocorr` | 55 | 0.7128 | 0.0546 | 0.0408 | 0.8559 | No correlation filter |
| 10 | `v6_c4_hybrid` | 50 | 0.7121 | 0.0547 | 0.0416 | 0.8554 | Hybrid MI + XGB |
| 11 | `v6_c3_xgb_no_coverage` | 50 | 0.7106 | 0.0548 | 0.0417 | 0.8566 | No family coverage |
| 12 | `hand_mdr_v25` | 38 | 0.6969 | 0.0561 | 0.0430 | 0.8531 | Hand-curated baseline |
| 13 | `v6_c1_baseline_bypass_off` | 12 | 0.6152 | 0.0632 | 0.0482 | 0.7904 | Un-bypassed MI/EN |

---

### Drift-weighted ($\beta=0.2$)

| Rank | Feature Set | n | Test R² | RMSE | MAE | Pearson | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **`8.4_V0`** | **50** | **0.7437** | **0.0516** | **0.0390** | **0.8671** | **Reference baseline from derived_8.4 dataset_metadata** |
| 2 | `8.3_V0` | 50 | 0.7437 | 0.0516 | 0.0390 | 0.8671 | Identical feature set |
| 3 | `V3_sota` | 47 | 0.7335 | 0.0526 | 0.0404 | 0.8693 | Historical 8.2 baseline |
| 4 | `v6_c2d_xgb_softcorr_k65` | 65 | 0.7237 | 0.0535 | 0.0400 | 0.8620 | Larger soft-corr set |
| 5 | `v6_c2c_xgb_nocorr` | 55 | 0.7188 | 0.0540 | 0.0405 | 0.8595 | No correlation filter |
| 6 | `v6_c0_baseline_bypass_on` | 46 | 0.7156 | 0.0543 | 0.0415 | 0.8566 | Legacy MI/EN baseline |
| 7 | `v6_c2_xgb` | 50 | 0.7152 | 0.0544 | 0.0417 | 0.8604 | Primary XGB selection |
| 8 | `v6_c2b_xgb_softcorr` | 55 | 0.7131 | 0.0546 | 0.0408 | 0.8573 | Soft-corr generalist |
| 9 | `v6_c3_xgb_no_coverage` | 50 | 0.7127 | 0.0546 | 0.0415 | 0.8591 | No family coverage |
| 10 | `v6_c5_rf` | 50 | 0.7118 | 0.0547 | 0.0418 | 0.8575 | Random Forest ranker |
| 11 | `v6_c4_hybrid` | 50 | 0.7089 | 0.0550 | 0.0419 | 0.8573 | Hybrid MI + XGB |
| 12 | `hand_mdr_v25` | 38 | 0.6893 | 0.0568 | 0.0432 | 0.8535 | Hand-curated baseline |
| 13 | `v6_c1_baseline_bypass_off` | 12 | 0.6168 | 0.0631 | 0.0478 | 0.7915 | Un-bypassed MI/EN |

---

## Conclusion & Active Baseline

- **`OVERALL_SELECTED_FEATURES_V0`** (50 features, unweighted test $R^2 = \mathbf{0.7595}$, drift-weighted test $R^2 = \mathbf{0.7437}$) remains undefeated as the top-performing feature set on `derived_8.4`.
- None of the candidate feature selection variants C0–C5 outperformed `OVERALL_SELECTED_FEATURES_V0` on the single global model.
- Per protocol, `OVERALL_SELECTED_FEATURES_V0` is retained as the active baseline in [`data/splits/derived_8.4/dataset_metadata.py`](../../../data/splits/derived_8.4/dataset_metadata.py), and `OVERALL_SELECTED_FEATURES_V1` was not created.
