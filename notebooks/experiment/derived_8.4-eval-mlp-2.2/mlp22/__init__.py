"""mlp22 package for derived_8.4-eval-mlp-2.2 neural tabular experiments.

Follow-up to mlp21 (derived_8.4-eval-mlp-2.1): an optimization + further
parameter sweep of the 2.1 winners' neighborhoods. 2.1 closed SWA as a
negative with proof (RNG guard -> 136/136 swa-live-vs-anchor val-curve pairs
bit-identical; 0/152 deployments across starts {0.6, 0.7, 0.75, 0.8, 0.85}),
so 2.2 runs NO SWA configs (the machinery stays importable for parity; the
RNG guard stays in the trainer so the code paths are unchanged). `fg`/`plr`
remain documented negatives (2.0) — 2.2 runs only plain-MLP configs.

New in 2.2 (mlp22/trainer.py):
  - best-val predictions are saved to `val_preds.npy` (post-training, eval-
    mode forward with the SAME deployed weights as `preds.npy`, no RNG
    consumption) so the offline val-year (2021 vs 2022) selection-reliability
    diagnostic (`analyze_val_years.py`) can be computed without retraining.
    The training path is byte-identical to mlp21 — anchors' val curves stay
    bit-identical across versions (stack check via compare_anchor_vs_2.1.py).

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
