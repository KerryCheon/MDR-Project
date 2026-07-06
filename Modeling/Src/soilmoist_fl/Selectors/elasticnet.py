# Jakob Balkovec
# ElasticNet Selector

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV

from Modeling.Utils.logging import get_logger
from Modeling.Src.soilmoist_fl.Selectors.base import _basic_xy_checks, _get_feature_cols, _top_k, log_top


def select_elasticnet(
    X,
    y,
    k=60,
    l1_ratio=(0.1, 0.5, 0.9, 0.95, 1.0),
    n_alphas=100,
    cv=5,
    max_iter=20000,
    random_state=42,
    n_jobs=-1,
):
    log = get_logger("selectors.elasticnet")

    y = _basic_xy_checks(X, y)
    feature_cols = _get_feature_cols(X)

    y_num = pd.to_numeric(y, errors="coerce").to_numpy()
    if np.isnan(y_num).any():
        raise ValueError("select_elasticnet: y contains NaNs after numeric coercion")

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("enet", ElasticNetCV(
            l1_ratio=list(l1_ratio) if isinstance(l1_ratio, (list, tuple)) else l1_ratio,
            alphas=int(n_alphas),
            cv=int(cv),
            max_iter=int(max_iter),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        ))
    ])

    model.fit(X, y_num)
    enet = model.named_steps["enet"]
    coefs = enet.coef_

    abs_coef = np.abs(coefs)
    pairs = list(zip(feature_cols, abs_coef))
    pairs.sort(key=lambda t: -t[1])

    ranked = [p[0] for p in pairs]
    score_map = {p[0]: float(p[1]) for p in pairs}

    # non-zero mask
    nz = [f for f, a in pairs if a > 0]
    if len(nz) == 0:
        log.warning("select_elasticnet: all coefficients are zero; falling back to top-k by abs(coef) anyway")
        nz = ranked

    # apply k
    selected = _top_k(nz, k)

    log.info(
        "select_elasticnet: alpha=%.6g l1_ratio=%s features=%d nonzero=%d selected=%d",
        float(enet.alpha_),
        str(enet.l1_ratio_),
        len(feature_cols),
        len(nz),
        len(selected),
    )
    log_top(log, "ElasticNet|abs(coef)", ranked, score_map=score_map, n=15)

    return {
        "kind": "elasticnet",
        "ranked": ranked,
        "scores": score_map,
        "selected": selected,
        "k": int(k),
        "alpha": float(enet.alpha_),
        "l1_ratio": float(enet.l1_ratio_) if hasattr(enet, "l1_ratio_") else None,
    }
