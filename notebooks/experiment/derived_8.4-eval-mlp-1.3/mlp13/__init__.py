"""mlp13 package for derived_8.4-eval-mlp-1.3 neural tabular experiments.

Follow-up to mlp12 (derived_8.4-eval-mlp-1.2): 2-regime MLP one more shot —
trainer knobs for EMA, mixup, target centering, and alternative early-stopping
rules (defaults = 1.2 behavior), phase-2 selection on val RMSE (aux2020 demoted
to diagnostic), and a curated config list (1.2 anchors + 1-seed completion +
gap-targeting configs). Data loading and the K=2 V0 clustering router are
reused verbatim from eval11 to guarantee identical splits/regimes.
"""

from .model import MLPRegressor, build_model

__all__ = ["MLPRegressor", "build_model"]
