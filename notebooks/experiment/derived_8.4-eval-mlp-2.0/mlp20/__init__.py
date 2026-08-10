"""mlp20 package for derived_8.4-eval-mlp-2.0 neural tabular experiments.

Follow-up to mlp13 (derived_8.4-eval-mlp-1.3): an optimized MLP architecture
to break the confirmed plain-2-regime-MLP ceiling (0.761/0.765 honest,
0.789 test-best, XGBoost 2-regime 0.815). New in 2.0:

  - FeatureGroupedMLP (architecture "fg"): per-semantic-group towers + fusion
    MLP (grouping in mlp20.feature_groups, validated: every feature in exactly
    one group). Targets 1.2's "capacity spent on period-specific interactions"
    overfitting kind.
  - PLRRegressor (architecture "plr"): piecewise-linear encoding (Gorishniy
    et al. 2022) + plain-MLP body.
  - SWA in the trainer (swa=True): Stochastic Weight Averaging once per epoch
    with BN recalibration — replaces the 1.3 per-step EMA documented failure.
  - A third family, 2regime_mixed (c0 = 96-pool, c1 = 54+10 delta), that
    allocates each cluster to the feature set that per-cluster evidence says
    is stronger (c0-96 0.754 vs c0-54 0.737; c1-54 0.831 vs c1-96 0.776).

Data loading and the K=2 V0 clustering router are reused verbatim from eval11
to guarantee identical splits/regimes across experiments.
"""

from .model import (
    FeatureGroupedMLP,
    MLPRegressor,
    PLRRegressor,
    build_model,
)

__all__ = ["MLPRegressor", "FeatureGroupedMLP", "PLRRegressor", "build_model"]
