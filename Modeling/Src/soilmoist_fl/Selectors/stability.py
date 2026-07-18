# Jakob Balkovec
# Stability Selector

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Selectors.base import _top_k, log_top
from Modeling.Src.soilmoist_fl.Selectors.elasticnet import select_elasticnet
from Modeling.Src.soilmoist_fl.Selectors.rf_importance import select_rf_importance
from Modeling.Src.soilmoist_fl.Selectors.xgb_importance import select_xgb_importance


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


def stability_bootstrap(
    X,
    y,
    base="elasticnet",
    n_boot=100,
    sample_frac=0.8,
    min_freq=0.6,
    top_k=None,
    random_state=42,
    base_k=60,
    base_kwargs=None,
):
    log = get_logger("selectors.stability")

    if n_boot <= 1:
        raise ValueError("stability_bootstrap: n_boot must be >= 2")

    base_kwargs = base_kwargs or {}

    n = X.shape[0]
    m = int(round(float(sample_frac) * n))
    if m <= 0:
        raise ValueError("stability_bootstrap: sample_frac produced empty sample")

    rng = np.random.default_rng(int(random_state))

    log.info(
        "stability_bootstrap: base=%s n_boot=%d sample_frac=%.3f base_k=%d min_freq=%.3f",
        base, int(n_boot), float(sample_frac), int(base_k), float(min_freq)
    )

    # Pre-generate indices to ensure reproducibility with the parallel rng usage
    boot_indices = [rng.choice(n, size=m, replace=True) for _ in range(int(n_boot))]

    def _run_bootstrap(b, idx):
        Xb = X.iloc[idx]
        yb = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

        kwargs = dict(base_kwargs)
        if base == "elasticnet":
            # For the bootstrap step, we always force standard ElasticNet by using single thread
            kwargs["n_jobs"] = 1
            out = select_elasticnet(Xb, yb, k=base_k, random_state=int(random_state) + b, **kwargs)
        elif base == "rf":
            # RandomForest bootstrap step uses 1 thread inside to prevent joblib collision
            kwargs["n_jobs"] = 1
            out = select_rf_importance(Xb, yb, k=base_k, random_state=int(random_state) + b, **kwargs)
        elif base == "xgb":
            kwargs["n_jobs"] = 1
            out = select_xgb_importance(Xb, yb, k=base_k, random_state=int(random_state) + b, **kwargs)
        else:
            raise ValueError(f"stability_bootstrap: unsupported base estimator: {base}")

        return out["selected"]

    selections = Parallel(n_jobs=-1)(
        delayed(_run_bootstrap)(b, idx) for b, idx in enumerate(boot_indices)
    )

    out_stab = stability_from_feature_lists(selections, min_freq=min_freq, top_k=top_k)
    out_stab["bootstrap"] = {
        "n_boot": int(n_boot),
        "sample_frac": float(sample_frac),
        "base": base,
        "base_k": int(base_k),
    }
    return out_stab


def stability_bootstrap_elasticnet(
    X,
    y,
    n_boot=100,
    sample_frac=0.8,
    min_freq=0.6,
    top_k=None,
    random_state=42,
    enet_k=60,
    enet_kwargs=None,
):
    # Backward compatible wrapper calling the unified stability_bootstrap
    return stability_bootstrap(
        X=X,
        y=y,
        base="elasticnet",
        n_boot=n_boot,
        sample_frac=sample_frac,
        min_freq=min_freq,
        top_k=top_k,
        random_state=random_state,
        base_k=enet_k,
        base_kwargs=enet_kwargs,
    )
