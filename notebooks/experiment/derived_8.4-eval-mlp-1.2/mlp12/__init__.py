"""mlp11 package for derived_8.4-eval-mlp-1.1 neural tabular experiments.

Follow-up to mlp10 (derived_8.4-eval-mlp-1.0): official-val-split protocol,
warmup+cosine LR, EMA, residual MLPs and FT-Transformers, and a 96-feature
candidate-pool family. Data loading and the K=2 V0 clustering router are
reused verbatim from eval11 to guarantee identical splits/regimes.
"""

from .model import FTTransformer, MLPRegressor, ResidualMLP, build_model

__all__ = ["MLPRegressor", "ResidualMLP", "FTTransformer", "build_model"]
