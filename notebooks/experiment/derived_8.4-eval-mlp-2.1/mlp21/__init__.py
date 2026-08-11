"""mlp21 package for derived_8.4-eval-mlp-2.1 neural tabular experiments.

Follow-up to mlp20 (derived_8.4-eval-mlp-2.0): an optimization + further
parameter sweep of the 2.0 winners. 2.0 broke the plain-2-regime-MLP ceiling
with the 2regime_mixed family (c0 = 96-pool, c1 = 54+10; val top-5 ensemble
test R2 0.8003, 2-seed honest single 0.7903) and documented `fg` (grouped
towers, best 0.782) and `plr` (PLR encoding, best 0.720) as negatives vs the
plain MLP (0.790) — so 2.1 runs ONLY plain-MLP configs (the classes stay
importable so the winner-pool filter and grouping table remain valid).

New in 2.1 (the two documented SWA fixes from the 2.0 README):
  - RNG guard: the SWA BN-recalibration pass runs the train loader in train
    mode, whose dropout consumed the shared RNG in 2.0 and made a swa job's
    live trajectory diverge from its anchor. In 2.1 `_recalibrate_bn` runs
    inside `_rng_guard()`, so the live trajectory is bit-identical to the
    anchor and any `_swa*` gain is attributable to SWA (not RNG drift).
  - `swa_start_frac` is a swept knob {0.7, 0.75, 0.8, 0.85} (2.0 hard-coded
    0.6, which never beat the live best on val).

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
