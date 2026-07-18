# Feature Selection V6 (`derived_8.2-feature-selection-2.0`)

## Motivation

Past pipelines (MI→ElasticNet→stability) systematically:

1. Starved static/seasonal features when `mi_k` was tight.
2. Collapsed to pure temporal sets when MI was skipped (V4/V5).
3. Missed hand-selected hydro memory on `derived_8.0` (opt-1.0 R² 0.78 vs hand 0.82).
4. Relied on hard-coded name bypasses that do not generalize to MoE subsets.

## Design

Stages: **soft correlation (0.99) → XGBoost gain → family coverage → stability (xgb)**.  
No hard-coded feature-name whitelist. Soft structural families only (satellite / hydro / static / calendar).

## Outcome (1.3-lite protocol; both weight regimes)

| Protocol | Dataset | Baseline | Best auto | Result |
|----------|---------|----------|-----------|--------|
| With drift | 8.0 | hand 0.818 | c2b 0.805 | +0.026 vs old pipeline; −0.013 vs hand |
| With drift | 8.2 | V3 0.638 | c1 0.658 | **beats V3** by +0.020 |
| No drift | 8.0 | hand 0.824 | c2b 0.807 | −0.017 vs hand |
| No drift | 8.2 | V3 0.653 | c1 0.660 | **beats V3** by +0.007 |

Rank order is stable across drift / no-drift. Full writeup: `notebooks/experiment/derived_8.2-feature-selection-2.0/RESULTS.md`.

## Recommendation

- Prefer **c2b** as the default automatic FS for new datasets/subsets.
- Optionally use **c1** when optimizing a global 8.2-like multi-station WA model.
- Do **not** promote name-based bypass as the long-term solution.
- Next: group-aware multi-scale hydro selection to close the residual hand gap on 8.0; then re-test regime FS for MoE.
