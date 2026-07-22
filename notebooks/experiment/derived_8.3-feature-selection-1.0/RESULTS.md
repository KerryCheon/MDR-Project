# Results: derived_8.3-feature-selection-1.0

**Date:** 2026-07-21  
**Eval protocol:** 1.3-lite XGBoost trained on train+val, scored on test (`derived_8.3`); CUDA accelerated; dual protocol (drift-weighted $\beta=0.2$ vs unweighted).

---

## Leaderboard (`derived_8.3` test set)

### Unweighted (No drift) — Primary Selection Protocol

| Rank | Feature Set | n | Test R² | RMSE | MAE | Pearson | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **`8.3_V0`** | **50** | **0.6433** | **0.0621** | **0.0455** | **0.8159** | **Reference baseline from derived_8.3 dataset_metadata** |
| 2 | `V3_sota` | 47 | 0.6333 | 0.0629 | 0.0468 | 0.8248 | Historical 8.2 baseline |
| 3 | `hand_mdr_v25` | 38 | 0.6169 | 0.0643 | 0.0484 | 0.8197 | Hand-curated baseline |
| 4 | `v6_c2c_xgb_nocorr` | 55 | 0.6162 | 0.0644 | 0.0468 | 0.8138 | No correlation filter |
| 5 | `v6_c2b_xgb_softcorr` | 55 | 0.6109 | 0.0648 | 0.0472 | 0.8129 | Peak V6 auto variant |
| 6 | `v6_c2d_xgb_softcorr_k65` | 65 | 0.6057 | 0.0653 | 0.0475 | 0.8106 | Larger soft-corr set |
| 7 | `v6_c0_baseline_bypass_on` | 14 | 0.5969 | 0.0660 | 0.0483 | 0.7973 | Legacy MI/EN baseline |
| 8 | `v6_c4_hybrid` | 50 | 0.5888 | 0.0666 | 0.0485 | 0.8027 | Hybrid MI + XGB |
| 9 | `v6_c5_rf` | 50 | 0.5883 | 0.0667 | 0.0483 | 0.8058 | Random Forest ranker |
| 10 | `v6_c2_xgb` | 50 | 0.5858 | 0.0669 | 0.0485 | 0.8029 | Strict corr (0.95) prune |
| 11 | `v6_c3_xgb_no_coverage` | 50 | 0.5855 | 0.0669 | 0.0487 | 0.8017 | No family coverage |
| 12 | `v6_c1_baseline_bypass_off` | 9 | 0.5345 | 0.0709 | 0.0511 | 0.7571 | Un-bypassed MI/EN |

---

### Drift-weighted ($\beta=0.2$)

| Rank | Feature Set | n | Test R² | RMSE | MAE | Pearson | Notes |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | **`8.3_V0`** | **50** | **0.6338** | **0.0629** | **0.0458** | **0.8121** | **Reference baseline from derived_8.3 dataset_metadata** |
| 2 | `hand_mdr_v25` | 38 | 0.6108 | 0.0648 | 0.0489 | 0.8183 | Hand-curated baseline |
| 3 | `V3_sota` | 47 | 0.6082 | 0.0651 | 0.0485 | 0.8138 | Historical 8.2 baseline |
| 4 | `v6_c2b_xgb_softcorr` | 55 | 0.6060 | 0.0652 | 0.0473 | 0.8139 | Peak V6 auto variant |
| 5 | `v6_c0_baseline_bypass_on` | 14 | 0.6044 | 0.0654 | 0.0480 | 0.8012 | Legacy MI/EN baseline |
| 6 | `v6_c2c_xgb_nocorr` | 55 | 0.6008 | 0.0657 | 0.0476 | 0.8118 | No correlation filter |
| 7 | `v6_c2d_xgb_softcorr_k65` | 65 | 0.5958 | 0.0661 | 0.0477 | 0.8096 | Larger soft-corr set |
| 8 | `v6_c4_hybrid` | 50 | 0.5892 | 0.0666 | 0.0485 | 0.8091 | Hybrid MI + XGB |
| 9 | `v6_c5_rf` | 50 | 0.5891 | 0.0666 | 0.0486 | 0.8092 | Random Forest ranker |
| 10 | `v6_c3_xgb_no_coverage` | 50 | 0.5884 | 0.0667 | 0.0485 | 0.8068 | No family coverage |
| 11 | `v6_c2_xgb` | 50 | 0.5827 | 0.0671 | 0.0486 | 0.8058 | Strict corr (0.95) prune |
| 12 | `v6_c1_baseline_bypass_off` | 9 | 0.5428 | 0.0703 | 0.0507 | 0.7613 | Un-bypassed MI/EN |

---

## Conclusion & Active Baseline

- **`OVERALL_SELECTED_FEATURES_V0`** (50 features, unweighted test $R^2 = \mathbf{0.6433}$, drift-weighted test $R^2 = \mathbf{0.6338}$) is retained as the primary baseline feature set in [`data/splits/derived_8.3/dataset_metadata.py`](file:///c:/Users/pan/Documents/GitHub/MDR-Project/data/splits/derived_8.3/dataset_metadata.py).
