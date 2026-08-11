"""Semantic feature groups for the FeatureGroupedMLP (derived_8.4-eval-mlp-2.3).

Motivation (see experiment README): the 54-backbone / 96-pool / 64-feature
specialist sets are heterogeneous — SMAP soil-moisture series, Sentinel-2
optical bands, NDVI/NDMI vegetation indices, SAR backscatter ratios,
LST/thermal, meteorology/API, static geo/BioClim/soil, and temporal harmonics.
The plain MLP (mlp-1.x) mixes all of them in one dense stack, which is the
setup that 1.2's overfitting analysis identified as "spending capacity on
period-specific interactions". The grouped architecture gives each semantic
family its own small tower and fuses the group embeddings, forcing the model
to build per-sensor representations before mixing.

This module is the single source of truth for the grouping. It is explicit
(substring/prefix rules applied in a fixed order, first match wins) and it
VALIDATES that every feature of a family lands in exactly one group — the
builders raise instead of silently dropping or double-assigning a feature, so
a bad rule is a fixed config, not a mystery.

The rule table below is deliberately simple substring/prefix matching on the
feature names produced by the repo pipeline (src/pipeline + derived_8.4
feature-selection-2.0). Order matters: more specific rules come first
(e.g. `SMAP_x_year` before `SMAP`, `D_z_E_SAR_ratio` is SAR not temporal).
"""

from __future__ import annotations

from dataclasses import dataclass

# Canonical group order (only non-empty groups appear in a model's tower list).
GROUP_ORDER = ["smap", "optical", "vegetation", "sar", "thermal", "meteo", "static", "temporal"]

# (group, matcher) rules, applied in order; matcher is a callable on the
# feature name. First match wins. The final entry must raise on unmatched
# names so the table is audited rather than silently extended.
_RULES: list[tuple[str, object]] = [
    # --- temporal interactions that would otherwise match a sensor group ---
    ("temporal", lambda n: "SMAP_x_year" in n),          # SMAP x year trend
    # --- sensor families (substring, first-match-wins) ---
    ("thermal", lambda n: "LST_modis" in n),             # LST (incl. D_z/D_sa/FFT variants)
    ("sar", lambda n: "SAR" in n),                       # E_SAR_ratio / E_SAR_diff / D_z_E_SAR_*
    ("optical", lambda n: "s2_b" in n),                  # Sentinel-2 bands (incl. V_*/A_* variants)
    ("sar", lambda n: "rough" in n),                     # E_rough_s1_vh_*
    ("vegetation", lambda n: ("NDVI" in n) or ("NDMI" in n) or n.startswith("F_")),  # F_* = veg indices (NDVI/NDMI/MSI...)
    ("smap", lambda n: "SMAP" in n),                     # SMAP sm series (lags, rolls, ampm diff)
    # --- meteorology / API (incl. V_*/A_* variants of G_API) ---
    ("meteo", lambda n: n.startswith("G_") or ("G_API" in n) or ("rain" in n) or ("DSLR" in n) or n.startswith("precip")),
    # --- temporal harmonics ---
    ("temporal", lambda n: "DOY" in n),                  # DOY, D_sin_DOY, D_cos_DOY
    ("temporal", lambda n: n.startswith("sin_") or n.startswith("cos_")
                           or n in {"year_frac", "doy_frac", "year", "month"}),
    # --- static geo / BioClim / soil ---
    ("static", lambda n: n.startswith("J_") or n.startswith("lia_") or ("soil_texture" in n)
                         or n in {"latitude", "longitude", "elev", "elevation", "slope", "aspect", "lc_code", "landcover"}),
    # --- remaining D_* (FFT, gradients, z-scores of non-caught sources) ---
    ("temporal", lambda n: n.startswith("D_")),
]


@dataclass(frozen=True)
class FeatureGroups:
    """Resolved grouping: feature index -> group id, plus per-group index lists."""

    names: tuple[str, ...]
    group_of: tuple[int, ...]          # len == n_features, group id per feature
    groups: tuple[tuple[int, ...], ...]  # per group id: sorted feature indices

    @property
    def n_features(self) -> int:
        return len(self.names)

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    def describe(self) -> str:
        parts = []
        for gid, idxs in enumerate(self.groups):
            gname = GROUP_ORDER[gid]
            feats = ", ".join(self.names[i] for i in idxs)
            parts.append(f"  [{gid}] {gname} ({len(idxs)}): {feats}")
        return "\n".join(parts)


def group_features(feature_names: list[str]) -> FeatureGroups:
    """Assign every feature to exactly one semantic group (raises on gaps)."""
    names = [str(n) for n in feature_names]
    seen: set[str] = set()
    group_of: list[int] = []
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate feature name in input: {name!r}")
        seen.add(name)
        matched = None
        for gname, matcher in _RULES:
            if matcher(name):  # type: ignore[operator]
                matched = gname
                break
        if matched is None:
            raise ValueError(
                f"Feature {name!r} matched no semantic group. Add an explicit rule "
                f"to mlp21/feature_groups.py (rules are first-match-wins)."
            )
        group_of.append(GROUP_ORDER.index(matched))

    groups: list[list[int]] = [[] for _ in GROUP_ORDER]
    for i, gid in enumerate(group_of):
        groups[gid].append(i)
    # drop empty groups but keep canonical ordering; group ids are renumbered 0..G-1
    nonempty: list[tuple[int, ...]] = [tuple(idxs) for idxs in groups if idxs]
    gid_map = {old_gid: new_gid for new_gid, old_gid in
               enumerate(g for g, idxs in enumerate(groups) if idxs)}
    group_of = [gid_map[g] for g in group_of]
    return FeatureGroups(names=tuple(names), group_of=tuple(group_of), groups=tuple(nonempty))


def summary_table(feature_names: list[str]) -> str:
    """Markdown table of the grouping (used by the report notebook)."""
    fg = group_features(feature_names)
    rows = []
    for gid, idxs in enumerate(fg.groups):
        rows.append(f"| {gid} | {GROUP_ORDER[gid]} | {len(idxs)} | {', '.join(fg.names[i] for i in idxs)} |")
    header = "| group_id | group | n_features | features |\n|---|---|---|---|"
    return "\n".join([header, *rows])
