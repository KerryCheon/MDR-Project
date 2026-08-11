"""mlp23 package for derived_8.4-eval-mlp-2.3 neural tabular experiments.

Follow-up to mlp22 (derived_8.4-eval-mlp-2.2): an optimization + further
parameter sweep of the 2.2 frontiers (54 320^2-hubergelu/lr6e-4 cell +
small-net region, mixed gelu 3-layer at low lr, the 96 lr3e-4 debiased
pool). 2.2 closed the 54-family 3-layer cell as a negative with evidence
(the val-overfit trap: the 2.2 54 val top-10 was 3-layer-dominated, test
0.7596-0.7790 vs the 2-layer 320^2 frontier 0.79+), so 2.3 runs only two
3-layer 54 re-check probes + the bit-identity anchors; SWA (2.1 negative,
0/152 deployments) and `fg`/`plr` (2.0 negatives) remain documented
negatives — no configs use them (the machinery stays importable for parity;
the RNG guard stays in the trainer so the code paths are unchanged).

New in mlp-2.3: NOTHING in the training path — the mlp23 trainer is
byte-identical to mlp22, so the anchors' val curves stay bit-identical
across versions (stack check via compare_anchor_vs_2.2.py). The 2.3
differences live entirely in the sweep config: the FULL 3-seed pool
(phase-2/3 top-Ns = family sizes) and the new grids.

Kept from mlp22 (mlp22/trainer.py):
  - best-val predictions are saved to `val_preds.npy` (post-training, eval-
    mode forward with the SAME deployed weights as `preds.npy`, no RNG
    consumption) so the offline val-year (2021 vs 2022) selection-reliability
    diagnostic (`analyze_val_years.py`) can be computed without retraining.

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
