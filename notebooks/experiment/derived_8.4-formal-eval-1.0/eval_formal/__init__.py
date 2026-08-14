"""eval_formal — evaluation core for derived_8.4-formal-eval-1.0.

Seed-aware evaluation machinery for the formal statistical evaluation:
data loading (adapted from derived_8.4-eval-1.3/eval13), routers (identical to eval13),
a seed-aware evaluator that trains XGBoost experts with a per-job random_state while
keeping the router/gating at the fixed config seed (see config.yaml), and per-(config,
seed[, station]) metric/weight/prediction persistence with cache-safe naming.
"""
