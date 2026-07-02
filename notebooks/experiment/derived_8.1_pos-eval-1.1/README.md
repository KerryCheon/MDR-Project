# derived_8.1_pos-eval-1.1

Evaluation of XGBoost oracle hard gating models on the derived_8.1_pos test set (N=8,902, 13 WA stations, SM > 0.0 only) with **Temporal Recency Weighting (Drift, beta=0.2)** applied during training. 

This experiment evaluates the effect of applying exponential temporal decay weights centered on the latest year ($2022$):
$$w_{\text{temporal}} = e^{\beta(Y - 2022)}$$
normalized to have a mean of $1.0$ over the training set.

## Combined Model Performance

| Model | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|
| **#1** Single Global XGBoost (no gating) | 0.4901 | 0.0752 | 0.0743 | −0.0113 | 0.05538 | 0.04040 | 0.710 |
| **#3** 3-Regime Oracle (regime-specific features) | **0.8653** | 0.0386 | 0.0373 | −0.0103 | 0.03017 | 0.02503 | 0.935 |
| **#5** 3-Regime Oracle (overall features, ablation) | **0.8713** | 0.0378 | 0.0364 | −0.0100 | 0.02993 | 0.02487 | 0.938 |
| **#6** 2-Regime Oracle (T=0.159) | 0.7443 | 0.0532 | 0.0519 | −0.0119 | 0.04108 | 0.03210 | 0.871 |

## Individual Specialist Performance

### Model 2: 3-Regime Specialists (Regime-Specific Features)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | **+0.0198** | 0.0455 | 0.0419 | −0.0176 | 0.03535 | 0.02791 | 0.448 |
| Transition (0.159–0.248) | 2,441 | −0.0155 | 0.0258 | 0.0258 | +0.0001 | 0.02170 | 0.02005 | 0.228 |
| Wet (SM ≥ 0.248) | 3,128 | −0.2562 | 0.0391 | 0.0376 | −0.0106 | 0.03126 | 0.02653 | 0.269 |

### Model 4: 3-Regime Specialists (Overall Features, Ablation)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | **+0.0873** | 0.0439 | 0.0399 | −0.0182 | 0.03449 | 0.02825 | 0.507 |
| Transition (0.159–0.248) | 2,441 | −0.0634 | 0.0264 | 0.0264 | +0.0004 | 0.02234 | 0.02083 | 0.200 |
| Wet (SM ≥ 0.248) | 3,128 | −0.2066 | 0.0383 | 0.0371 | −0.0096 | 0.03100 | 0.02680 | 0.306 |

### Model 6: 2-Regime Specialists (T=0.159)

| Specialist | N | R² | RMSE | ubRMSE | Bias | MAE | Med\|Err\| | Pearson |
|---|---|---|---|---|---|---|---|---|
| Dry (SM < 0.159) | 3,333 | **+0.0198** | 0.0455 | 0.0419 | −0.0176 | 0.03535 | 0.02791 | 0.448 |
| Wet (SM ≥ 0.159) | 5,569 | −0.0471 | 0.0574 | 0.0568 | −0.0085 | 0.04452 | 0.03568 | 0.380 |

## Comparison with derived_8.1_pos-eval-1.0 (No Weighting)

| Model | 1.0 R² (No Weighting) | 1.1 R² (With Weighting) | Change |
|---|---|---|---|
| Single Global XGBoost | **0.5023** | 0.4901 | −0.0122 |
| 3-Regime Oracle (regime-specific features) | **0.8662** | 0.8653 | −0.0009 |
| 3-Regime Oracle (overall features, ablation) | 0.8637 | **0.8713** | **+0.0076** |
| 2-Regime Oracle (T=0.159) | **0.7453** | 0.7443 | −0.0010 |

### Specialist-Specific R² Comparison

| Specialist Slice | 1.0 R² (No Weighting) | 1.1 R² (With Weighting) | Change |
|---|---|---|---|
| Dry Specialist (Model 2, Specific Features) | −0.0191 | **+0.0198** | **+0.0389** |
| Dry Specialist (Model 4, Overall Features) | −0.0097 | **+0.0873** | **+0.0970** |
| Transition Specialist (Model 4, Overall Features) | −0.1291 | **−0.0634** | **+0.0657** |
| Wet Specialist (Model 4, Overall Features) | **−0.1975** | −0.2066 | −0.0091 |

## Key Findings

- **Positive Dry Specialist R² achieved**: Removing SM=0.0 in `derived_8.1_pos` got dry specialist R² close to zero ($-0.019$). Introducing **Temporal Recency Weighting** finally pushes it into positive territory ($+0.0198$ for regime-specific and $+0.0873$ for overall features). This indicates that prioritizing recent years helps specialists learn the low-variance dry regime structure more effectively.
- **Ablation Oracle Model (#5) Outperforms regime-specific model (#3)**: Model 5 achieves the highest combined $R^2 = 0.8713$ ($+0.0076$ improvement over 1.0). This highlights that the overall selected features provide a more robust signal under temporal weighting than isolated subset features, preventing specialist overfitting.
- **Slight drop in Global and 2-Regime model R²**: The single global model dropped slightly from $0.5023$ to $0.4901$ ($-0.0122$). This suggests that when all data points are forced into a single global mapping, temporal decay weighting reduces the effective sample size and can lead to marginal loss of generalization on the test set.
- **Wet Specialists remain underfit**: While dry and transition specialists improved, the wet specialist metrics degraded slightly (e.g. Wet overall-feature expert dropped from $-0.1975$ to $-0.2066$). This shows that wet regime dynamics are highly transient and likely require different weighting or feature engineering strategies than simple exponential temporal decay.

## References

- [Dataset derived_8.1_pos](../../data/splits/derived_8.1_pos/README.md) — dataset compilation and feature selection.
- [derived_8.1_pos-eval-1.0 README](../derived_8.1_pos-eval-1.0/README.md) — baseline evaluation without temporal recency weights.
- [Changelog 2026-06-30](../../docs/changelogs/2026-06-30-Pan.md) — derived_8.1_pos dataset creation.
