"""Local MI, ElasticNet, and stability selection with collapse diagnostics."""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .families import family_counts


DEFAULT_BYPASS_PREFIXES = ("J_", "K_", "D_", "G_")
DEFAULT_BYPASS_EXACT = {
    "longitude",
    "latitude",
    "elev",
    "slope",
    "aspect",
    "DOY",
    "precip_mm",
    "sin_year",
    "cos_year",
}


@dataclass(frozen=True)
class SelectionProfile:
    """A documented selector variant used only inside this experiment."""

    name: str
    mi_k: int | None
    bypass_mode: str
    legacy_mi_excludes_bypass: bool
    fallback_mode: str


@dataclass
class SelectionResult:
    """Selection result including every stage needed by the collapse audit."""

    profile: str
    selected: list[str]
    stable_selected: list[str]
    repaired_selected: list[str]
    mi_ranked: list[str]
    enet_ranked: list[str]
    stability_ranked: list[str]
    stability_scores: dict[str, float]
    bypass_features: list[str]
    candidate_features: list[str]
    alpha: float | None
    l1_ratio: float | None
    enet_nonzero: int
    stage_counts: dict[str, int]
    family_counts: dict[str, int]
    fallback_applied: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES = {
    "mi120": SelectionProfile(
        name="mi120",
        mi_k=120,
        bypass_mode="none",
        legacy_mi_excludes_bypass=False,
        fallback_mode="legacy_ranked",
    ),
    "mi300": SelectionProfile(
        name="mi300",
        mi_k=300,
        bypass_mode="none",
        legacy_mi_excludes_bypass=False,
        fallback_mode="legacy_ranked",
    ),
    "legacy_forced_bypass": SelectionProfile(
        name="legacy_forced_bypass",
        mi_k=300,
        bypass_mode="legacy",
        legacy_mi_excludes_bypass=True,
        fallback_mode="legacy_ranked",
    ),
    "no_mi": SelectionProfile(
        name="no_mi",
        mi_k=None,
        bypass_mode="none",
        legacy_mi_excludes_bypass=False,
        fallback_mode="legacy_ranked",
    ),
    "mi300_repaired": SelectionProfile(
        name="mi300_repaired",
        mi_k=300,
        bypass_mode="none",
        legacy_mi_excludes_bypass=False,
        fallback_mode="pre_stability_repair",
    ),
}


def _ordered_unique(features: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(features))


def get_bypass_features(columns: Iterable[str]) -> list[str]:
    """Return the historical legacy force-include columns in source order."""
    return [
        column
        for column in columns
        if column.startswith(DEFAULT_BYPASS_PREFIXES)
        or "year" in column
        or column in DEFAULT_BYPASS_EXACT
    ]


def _rank_mi(X: pd.DataFrame, y: pd.Series, k: int, random_state: int) -> tuple[list[str], dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    kept = list(imputer.get_feature_names_out(X.columns))
    scores = mutual_info_regression(X_imp, np.asarray(y, dtype=float), random_state=random_state)
    score_map = {feature: float(score) for feature, score in zip(kept, scores)}
    score_map.update({feature: 0.0 for feature in X.columns if feature not in score_map})
    ranked = sorted(X.columns, key=lambda feature: (-score_map[feature], feature))
    return ranked[: min(k, len(ranked))], score_map


def _fit_elasticnet(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
    random_state: int,
    alpha: float | None = None,
    l1_ratio: float | None = None,
) -> tuple[list[str], list[str], dict[str, float], float, float]:
    if alpha is None:
        estimator: ElasticNet | ElasticNetCV = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.9, 0.95, 1.0],
            alphas=100,
            cv=5,
            max_iter=20_000,
            random_state=random_state,
            n_jobs=1,
        )
    else:
        estimator = ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            max_iter=20_000,
            random_state=random_state,
        )
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("enet", estimator),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, np.asarray(y, dtype=float))
    fitted = model.named_steps["enet"]
    kept = list(model[:-1].get_feature_names_out(X.columns))
    score_map = {feature: float(abs(coef)) for feature, coef in zip(kept, fitted.coef_)}
    score_map.update({feature: 0.0 for feature in X.columns if feature not in score_map})
    ranked = sorted(X.columns, key=lambda feature: (-score_map[feature], feature))
    nonzero = [feature for feature in ranked if score_map[feature] > 0.0]
    if not nonzero:
        nonzero = ranked
    selected = nonzero[: min(k, len(nonzero))]
    selected_alpha = float(fitted.alpha_ if alpha is None else fitted.alpha)
    selected_l1 = float(fitted.l1_ratio_ if alpha is None else fitted.l1_ratio)
    return selected, ranked, score_map, selected_alpha, selected_l1


def _stability(
    X: pd.DataFrame,
    y: pd.Series,
    alpha: float,
    l1_ratio: float,
    n_boot: int,
    sample_fraction: float,
    base_k: int,
    min_freq: float,
    random_state: int,
) -> tuple[list[str], dict[str, float]]:
    if n_boot < 2:
        raise ValueError("stability bootstrap requires at least two resamples")
    rng = np.random.default_rng(random_state)
    counts: dict[str, int] = {}
    sample_size = max(2, int(round(sample_fraction * len(X))))
    for bootstrap_index in range(n_boot):
        indices = rng.choice(len(X), size=sample_size, replace=True)
        selected, _, _, _, _ = _fit_elasticnet(
            X.iloc[indices],
            y.iloc[indices],
            k=base_k,
            random_state=random_state + bootstrap_index,
            alpha=alpha,
            l1_ratio=l1_ratio,
        )
        for feature in set(selected):
            counts[feature] = counts.get(feature, 0) + 1
    scores = {feature: count / float(n_boot) for feature, count in counts.items()}
    ranked = sorted(scores, key=lambda feature: (-scores[feature], feature))
    stable = [feature for feature in ranked if scores[feature] >= min_freq]
    return stable, scores


def select_features(
    X: pd.DataFrame,
    y: pd.Series,
    profile_name: str,
    *,
    top_k: int,
    elasticnet_k: int,
    bootstrap_k: int,
    n_boot: int,
    sample_fraction: float,
    min_freq: float,
    min_keep: int,
    random_state: int,
) -> SelectionResult:
    """Run a local selector and retain enough detail to explain a collapse."""
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}")
    profile = PROFILES[profile_name]
    if X.empty or len(y) != len(X):
        raise ValueError("Feature matrix and target must be non-empty and aligned.")

    all_features = list(X.columns)
    bypass = get_bypass_features(all_features) if profile.bypass_mode == "legacy" else []
    mi_input = [feature for feature in all_features if feature not in bypass]
    if not profile.legacy_mi_excludes_bypass:
        mi_input = all_features

    if profile.mi_k is None:
        mi_ranked = list(all_features)
        candidate = list(all_features)
    else:
        mi_ranked, _ = _rank_mi(X[mi_input], y, profile.mi_k, random_state)
        candidate = _ordered_unique([*mi_ranked, *bypass])
    if not candidate:
        raise ValueError("The selector candidate universe is empty.")

    enet_selected, enet_ranked, enet_scores, alpha, l1_ratio = _fit_elasticnet(
        X[candidate], y, elasticnet_k, random_state
    )
    stable, stability_scores = _stability(
        X[candidate],
        y,
        alpha,
        l1_ratio,
        n_boot,
        sample_fraction,
        bootstrap_k,
        min_freq,
        random_state,
    )
    stability_ranked = sorted(stability_scores, key=lambda feature: (-stability_scores[feature], feature))
    stable_selected = stable[:top_k]
    fallback_applied = len(stable_selected) < min_keep
    if fallback_applied:
        if profile.fallback_mode == "legacy_ranked":
            repair_source = stability_ranked
        else:
            repair_source = _ordered_unique([*stability_ranked, *enet_ranked, *mi_ranked, *bypass])
        repaired = _ordered_unique([*stable_selected, *repair_source])[:top_k]
    else:
        repaired = stable_selected[:top_k]

    return SelectionResult(
        profile=profile.name,
        selected=repaired,
        stable_selected=stable_selected,
        repaired_selected=repaired,
        mi_ranked=mi_ranked,
        enet_ranked=enet_ranked,
        stability_ranked=stability_ranked,
        stability_scores=stability_scores,
        bypass_features=bypass,
        candidate_features=candidate,
        alpha=alpha,
        l1_ratio=l1_ratio,
        enet_nonzero=sum(score > 0.0 for score in enet_scores.values()),
        stage_counts={
            "raw": len(all_features),
            "mi": len(mi_ranked) if profile.mi_k is not None else len(all_features),
            "candidate": len(candidate),
            "elasticnet_nonzero": len(enet_selected),
            "stability": len(stable_selected),
            "repaired": len(repaired),
        },
        family_counts=family_counts(repaired),
        fallback_applied=fallback_applied,
    )
