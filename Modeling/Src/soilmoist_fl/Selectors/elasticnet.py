# Jakob Balkovec
# ElasticNet Selector

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV

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
    alpha=None,
):
    log = get_logger("selectors.elasticnet")

    y = _basic_xy_checks(X, y)
    feature_cols = _get_feature_cols(X)

    y_num = pd.to_numeric(y, errors="coerce").to_numpy()
    if np.isnan(y_num).any():
        raise ValueError("select_elasticnet: y contains NaNs after numeric coercion")

    if alpha is not None:
        # Standard ElasticNet with fixed parameters (no cross-validation)
        l1_val = l1_ratio[0] if isinstance(l1_ratio, (list, tuple)) else l1_ratio
        enet_estimator = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_val),
            max_iter=int(max_iter),
            random_state=int(random_state),
        )
    else:
        # Cross-validated ElasticNetCV
        enet_estimator = ElasticNetCV(
            l1_ratio=list(l1_ratio) if isinstance(l1_ratio, (list, tuple)) else l1_ratio,
            alphas=int(n_alphas),
            cv=int(cv),
            max_iter=int(max_iter),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
        )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("enet", enet_estimator)
    ])

    model.fit(X, y_num)
    enet = model.named_steps["enet"]
    coefs = enet.coef_
    abs_coef = np.abs(coefs)

    # Robust mapping to handle dropped (all-NaN) features
    kept_features = model[:-1].get_feature_names_out(feature_cols)
    score_map = {f: float(c) for f, c in zip(kept_features, abs_coef)}

    # Fill dropped features with 0.0
    for f in feature_cols:
        if f not in score_map:
            score_map[f] = 0.0

    # Sort and rank all features
    pairs = list(score_map.items())
    pairs.sort(key=lambda t: -t[1])
    ranked = [p[0] for p in pairs]

    # non-zero mask
    nz = [f for f, a in pairs if a > 0]
    if len(nz) == 0:
        log.warning("select_elasticnet: all coefficients are zero; falling back to top-k by abs(coef) anyway")
        nz = ranked

    # apply k
    selected = _top_k(nz, k)

    if alpha is not None:
        alpha_val = float(enet.alpha)
        l1_ratio_val = float(enet.l1_ratio)
    else:
        alpha_val = float(enet.alpha_)
        l1_ratio_val = float(enet.l1_ratio_)

    log.info(
        "select_elasticnet: alpha=%.6g l1_ratio=%s features=%d nonzero=%d selected=%d",
        alpha_val,
        str(l1_ratio_val),
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
        "alpha": alpha_val,
        "l1_ratio": l1_ratio_val,
    }

