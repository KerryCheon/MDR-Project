# Jakob Balkovec
# Stability Selector

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Selectors.base import _top_k, log_top
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet


def stability_from_feature_lists(feature_lists, min_freq=0.6, top_k=None):
    log = get_logger("selectors.stability")

    if not feature_lists:
        raise ValueError("stability_from_feature_lists: feature_lists is empty")

    n = len(feature_lists)
    counts = {}

    for lst in feature_lists:
        for f in set(lst):
            counts[f] = counts.get(f, 0) + 1

    freqs = {f: counts[f] / float(n) for f in counts}
    ranked = sorted(freqs.keys(), key=lambda f: (-freqs[f], -counts[f], f))

    selected = [f for f in ranked if freqs[f] >= float(min_freq)]
    if top_k is not None:
        selected = _top_k(selected, int(top_k))

    log.info(
        "stability_from_feature_lists: lists=%d min_freq=%.3f kept=%d",
        n, float(min_freq), len(selected)
    )

    # preview
    score_map = {f: float(freqs[f]) for f in ranked}
    log_top(log, "Stability|freq", ranked, score_map=score_map, n=15)

    return {
        "kind": "stability",
        "ranked": ranked,
        "scores": score_map,
        "selected": selected,
        "min_freq": float(min_freq),
        "n_lists": int(n),
    }


def stability_bootstrap_elasticnet(
    X,
    y,
    n_boot=30,
    sample_frac=0.8,
    min_freq=0.6,
    top_k=None,
    random_state=42,
    enet_k=60,
    enet_kwargs=None,
):
    log = get_logger("selectors.stability")

    if n_boot <= 1:
        raise ValueError("stability_bootstrap_elasticnet: n_boot must be >= 2")

    enet_kwargs = enet_kwargs or {}

    n = X.shape[0]
    m = int(round(float(sample_frac) * n))
    if m <= 0:
        raise ValueError("stability_bootstrap_elasticnet: sample_frac produced empty sample")

    rng = np.random.default_rng(int(random_state))
    selections = []

    log.info(
        "stability_bootstrap_elasticnet: n_boot=%d sample_frac=%.3f enet_k=%d min_freq=%.3f",
        int(n_boot), float(sample_frac), int(enet_k), float(min_freq)
    )

    # Pre-generate indices to ensure reproducibility with the parallel rng usage
    boot_indices = [rng.choice(n, size=m, replace=True) for _ in range(int(n_boot))]

    def _run_bootstrap(b, idx):
        Xb = X.iloc[idx]
        yb = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        
        kwargs = dict(enet_kwargs)
        kwargs["n_jobs"] = 1
        
        out = select_elasticnet(Xb, yb, k=enet_k, random_state=int(random_state) + b, **kwargs)
        return out["selected"]

    selections = Parallel(n_jobs=-1)(
        delayed(_run_bootstrap)(b, idx) for b, idx in enumerate(boot_indices)
    )

    out_stab = stability_from_feature_lists(selections, min_freq=min_freq, top_k=top_k)
    out_stab["bootstrap"] = {
        "n_boot": int(n_boot),
        "sample_frac": float(sample_frac),
        "base": "elasticnet",
        "enet_k": int(enet_k),
    }
    return out_stab
