# Hybrid Temporal-Tabular 1.0

This experiment executes the four workstreams in
`Models/Temporal/lstm/HYBRID_TEMPORAL_TABULAR_PLAN.md`: feature experiments,
Touchet error analysis, temporal-representation tracing, and a frozen LSTM
encoder whose embedding is concatenated with the MDR-v25-aligned tabular
features for XGBoost.

The experiment uses the canonical `derived_9.0` split. Historical versioned
notebooks are not modified. Generated artifacts are written beneath
`artifacts/` by the individual scripts.

## Entry points

```bash
MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
  .venv/bin/python notebooks/experiment/hybrid-temporal-tabular-1.0/run_feature_experiments.py

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
  .venv/bin/python notebooks/experiment/hybrid-temporal-tabular-1.0/run_hybrid.py

MPLCONFIGDIR=/private/tmp/mpl XDG_CACHE_HOME=/private/tmp/cache \
  .venv/bin/python notebooks/experiment/hybrid-temporal-tabular-1.0/run_touchet_analysis.py
```

