"""mlp10 package for derived_8.4-eval-mlp-1.0 MLP experiments.

Mirrors the structure of eval11 (derived_8.4-eval-1.1) but swaps the XGBoost
experts for PyTorch MLP regressors. Data loading and the K=2 V0 clustering
router are reused verbatim from eval11 to guarantee identical splits/regimes.
"""

from .model import MLPRegressor

__all__ = ["MLPRegressor"]
