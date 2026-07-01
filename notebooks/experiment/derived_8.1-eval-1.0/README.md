# derived_8.1-eval-1.0

Evaluation of XGBoost oracle hard gating models on the derived_8.1 test set (N=10,599, 13 WA stations, 2017–2025). Regime thresholds: T1=0.16, T2=0.25 (bimodal valleys). Test regime distribution: Dry (SM < 0.16) N=5,054 | Transition (0.16 ≤ SM < 0.25) N=2,471 | Wet (SM ≥ 0.25) N=3,074.

## Combined Model Performance

| Model | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **#1** Single Global XGBoost (no gating) | 0.4160 | 0.0904 | 0.0859 | −0.0282 | 0.06756 | 0.04943 | 0.688 |
| **#3** 3-Regime Oracle (regime-specific features) | **0.8527** | 0.0454 | 0.0417 | −0.0179 | 0.03552 | 0.02776 | 0.938 |
| **#5** 3-Regime Oracle (overall features, ablation) | 0.8452 | 0.0466 | 0.0424 | −0.0193 | 0.03637 | 0.02855 | 0.937 |
| **#6** 2-Regime Oracle (T=0.16) | 0.7803 | 0.0554 | 0.0519 | −0.0195 | 0.04425 | 0.03679 | 0.900 |

## Individual Specialist Performance

### Model 2: 3-Regime Specialists (Regime-Specific Features)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.16) | 5,054 | −0.2534 | 0.0560 | 0.0457 | −0.0323 | 0.04560 | 0.03850 | 0.446 |
| Transition (0.16–0.25) | 2,471 | 0.0183 | 0.0256 | 0.0256 | +0.0004 | 0.02160 | 0.02022 | 0.260 |
| Wet (SM ≥ 0.25) | 3,074 | −0.2005 | 0.0378 | 0.0367 | −0.0090 | 0.03014 | 0.02462 | 0.320 |

### Model 4: 3-Regime Specialists (Overall Features, Ablation)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.16) | 5,054 | −0.3327 | 0.0577 | 0.0462 | −0.0346 | 0.04664 | 0.03862 | 0.437 |
| Transition (0.16–0.25) | 2,471 | −0.0760 | 0.0268 | 0.0268 | +0.0003 | 0.02261 | 0.02047 | 0.190 |
| Wet (SM ≥ 0.25) | 3,074 | −0.1895 | 0.0377 | 0.0363 | −0.0100 | 0.03055 | 0.02650 | 0.304 |

### Model 6: 2-Regime Specialists (T=0.16)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.16) | 5,054 | −0.2534 | 0.0560 | 0.0457 | −0.0323 | 0.04560 | 0.03850 | 0.446 |
| Wet (SM ≥ 0.16) | 5,545 | 0.0307 | 0.0550 | 0.0544 | −0.0078 | 0.04302 | 0.03545 | 0.417 |

## Key Findings

- **Single global model** (R²=0.416) underfits across the entire soil moisture range — roughly half the points underpredict, half overpredict.
- **3-regime oracle gating** (R²=0.853) nearly doubles the global model's R², validating the bimodal valley-based regime boundaries.
- **Regime-specific feature selection** provides a marginal gain: 0.8527 (regime-specific) vs 0.8452 (overall features, ablation), suggesting most of the improvement comes from regime separation itself.
- **2-regime gating** (R²=0.780) still far outperforms the single global model, indicating even a coarse dry/wet split is beneficial.
- All individual specialists exhibit near-zero or negative R² on their own slices, confirming they are underfitting. Future work should focus on improving the specialists (e.g., temporal drifting as in v24/v25).

## References

- [Dataset derived_8.1](../../data/splits/derived_8.1/README.md) — dataset compilation and feature selection details.
- [Changelog 2026-06-26](../../docs/changelogs/2026-06-26-Pan.md) — initial experiment results and proposal for derived_8.1_pos.
- [Changelog 2026-06-30](../../docs/changelogs/2026-06-30-Pan.md) — 2-regime and ablation experiments, specialist underfitting analysis.
