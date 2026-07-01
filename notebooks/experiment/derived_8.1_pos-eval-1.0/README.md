# derived_8.1_pos-eval-1.0

Evaluation of XGBoost oracle hard gating models on the derived_8.1_pos test set (N=8,902, 13 WA stations, SM > 0.0 only). Derived_8.1_pos filters out soil moisture ≤ 0.0 from derived_8.1 to eliminate residual concentration at zero. Bimodal valley thresholds: T1=0.159, T2=0.248. Test regime distribution: Dry (SM < 0.159) N=3,333 | Transition (0.159 ≤ SM < 0.248) N=2,441 | Wet (SM ≥ 0.248) N=3,128.

## Combined Model Performance

| Model | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **#1** Single Global XGBoost (no gating) | 0.5023 | 0.0743 | 0.0733 | −0.0123 | 0.05483 | 0.03985 | 0.720 |
| **#3** 3-Regime Oracle (regime-specific features) | **0.8662** | 0.0385 | 0.0372 | −0.0101 | 0.03015 | 0.02499 | 0.936 |
| **#5** 3-Regime Oracle (overall features, ablation) | 0.8637 | 0.0389 | 0.0374 | −0.0106 | 0.03091 | 0.02561 | 0.935 |
| **#6** 2-Regime Oracle (T=0.159) | 0.7453 | 0.0531 | 0.0517 | −0.0122 | 0.04131 | 0.03328 | 0.872 |

## Individual Specialist Performance

### Model 2: 3-Regime Specialists (Regime-Specific Features)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | −0.0191 | 0.0464 | 0.0429 | −0.0176 | 0.03646 | 0.02977 | 0.417 |
| Transition (0.159–0.248) | 2,441 | −0.0145 | 0.0258 | 0.0258 | +0.0004 | 0.02153 | 0.01986 | 0.238 |
| Wet (SM ≥ 0.248) | 3,128 | −0.1637 | 0.0376 | 0.0362 | −0.0103 | 0.03014 | 0.02564 | 0.314 |

### Model 4: 3-Regime Specialists (Overall Features, Ablation)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | −0.0097 | 0.0461 | 0.0413 | −0.0206 | 0.03668 | 0.03042 | 0.464 |
| Transition (0.159–0.248) | 2,441 | −0.1291 | 0.0272 | 0.0272 | +0.0001 | 0.02307 | 0.02164 | 0.125 |
| Wet (SM ≥ 0.248) | 3,128 | −0.1975 | 0.0381 | 0.0372 | −0.0083 | 0.03089 | 0.02648 | 0.313 |

### Model 6: 2-Regime Specialists (T=0.159)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | −0.0191 | 0.0464 | 0.0429 | −0.0176 | 0.03646 | 0.02977 | 0.417 |
| Wet (SM ≥ 0.159) | 5,569 | −0.0259 | 0.0568 | 0.0561 | −0.0089 | 0.04421 | 0.03637 | 0.403 |

## Comparison with derived_8.1

| Model | derived_8.1 R² | derived_8.1_pos R² | Change |
|---|---|---|---|
| Single Global XGBoost | 0.4160 | **0.5023** | **+0.086** |
| 3-Regime Oracle (regime-specific features) | 0.8527 | **0.8662** | **+0.014** |
| 3-Regime Oracle (overall features, ablation) | 0.8452 | **0.8637** | **+0.019** |
| 2-Regime Oracle (T=0.159) | 0.7803 | **0.7453** | **−0.035** |

## Key Findings

- **Zero-SM removal benefits the global model substantially**: R² jumps from 0.416 to 0.502 (+0.086), as the global model no longer wastes capacity fitting the SM=0.0 spike.
- **Oracle gating sees a smaller improvement**: 0.853 → 0.866 (+0.014), suggesting the hard gating already handled the zero-SM regime via the dry specialist.
- **Regime-specific feature selection still marginal**: 0.8662 vs 0.8637 (+0.003), consistent with the derived_8.1 findings.
- **All individual specialists still underfit** (near-zero or negative R² on their slices). The dry specialist improves notably (from −0.253 to −0.019) after removing SM=0.0, but remains well below satisfactory. Specialist quality remains the key bottleneck.
- **2-regime oracle (0.745) underperforms 3-regime oracle (0.866)** by ~0.12 R², confirming the transition regime provides meaningful separation beyond a simple dry/wet split.
- **Zero-SM removal hurts 2-regime more than 3-regime**: derived_8.1_pos 2-regime drops to 0.745 vs 0.780 on derived_8.1 (−0.035), whereas the 3-regime improves. The dry specialist's improvement after removing zero-SM points is not enough to offset the wet specialist losing the SM=0.0 anchor.
- **Dataset quality improved**: derived_8.1_pos (N=8,902 test) is 2.35× larger than derived_8.0, with recalibrated thresholds from bimodal valley analysis.

## References

- [Dataset derived_8.1_pos](../../data/splits/derived_8.1_pos/README.md) — dataset compilation and feature selection.
- [EDA derived_8.1_pos](../derived_8.1_pos-data-exploration/README.md) — valley calibration and regime distribution analysis.
- [Changelog 2026-06-26](../../docs/changelogs/2026-06-26-Pan.md) — initial experiment and derived_8.1_pos proposal.
- [Changelog 2026-06-30](../../docs/changelogs/2026-06-30-Pan.md) — derived_8.1_pos dataset creation.
- [derived_8.1-eval-1.0 README](../derived_8.1-eval-1.0/README.md) — counterpart experiment on derived_8.1 (with SM=0.0).
