# Soft family-coverage repair for feature selection (no hard-coded feature names)

from Modeling.Utils.logging import get_logger

# Structural coverage families used for soft quotas.
# Presence is inferred from prefix / naming conventions in docs/features.md —
# not from a whitelist of specific columns.
DEFAULT_COVERAGE_FAMILIES = (
    "satellite",
    "hydro",
    "static",
    "calendar",
    "temporal",
)


def infer_coverage_family(feature_name: str) -> str:
    """Map a feature name to a coverage family for soft quotas.

    Returns one of: satellite, hydro, static, calendar, temporal, other.
    """
    s = str(feature_name)

    # Calendar / time coordinates
    if s in {
        "DOY",
        "D_sin_DOY",
        "D_cos_DOY",
        "year_frac",
        "sin_year",
        "cos_year",
    }:
        return "calendar"
    if s.endswith("_x_year") or s.startswith("D_sa_") or s.startswith("D_z_") or s.startswith("D_fft_"):
        return "calendar"
    if s in {"sin_year", "cos_year"} or "year" in s and s.startswith(("sin_", "cos_", "API_", "SMAP_")):
        return "calendar"

    # Hydro / precipitation memory
    if s == "precip_mm" or s.startswith("G_") or "G_API" in s or "G_DSLR" in s or "G_rain" in s:
        # Pure G_ raw and G_ transforms; rolling of G_API still counts as hydro signal
        if s.startswith("G_") or s == "precip_mm":
            return "hydro"
        # Engineered ops on G_API (V_roll*_G_API, C_lag_G_API, A_*_G_API) → hydro
        if "G_API" in s or "G_DSLR" in s or "G_rain" in s or "precip" in s.lower():
            return "hydro"

    # Static GIS / soil / terrain
    if s in {
        "elev",
        "slope",
        "aspect",
        "latitude",
        "longitude",
    }:
        return "static"
    if s.startswith("J_") or s.startswith("K_"):
        return "static"

    # Satellite / remote sensing (raw, indices, SMAP, LIA, roughness)
    if (
        s.startswith("SMAP")
        or s.startswith("s1_")
        or s.startswith("s2_")
        or s.startswith("LST")
        or s.startswith("E_")
        or s.startswith("F_")
        or s.startswith("lia_")
        or s.startswith("I_ts_")
    ):
        return "satellite"
    # Temporal operators on satellite bases still count as satellite coverage
    # if the base is a known RS series (avoid double-counting pure G_ above)
    satellite_bases = (
        "SMAP",
        "LST_modis",
        "E_SAR",
        "F_NDVI",
        "F_NDMI",
        "F_MSI",
        "s2_b",
        "s1_",
    )
    if any(b in s for b in satellite_bases):
        return "satellite"

    # Generic temporal operators without a stronger family tag
    if s.startswith(("A_d_", "A_grad_", "A_pct_", "C_lag_", "C_smm_", "V_")):
        return "temporal"

    return "other"


def group_by_coverage_family(feature_cols):
    groups = {}
    for f in feature_cols:
        fam = infer_coverage_family(f)
        groups.setdefault(fam, []).append(f)
    return groups


def enforce_min_family_coverage(
    selected,
    ranked_scores,
    available,
    min_per_family=1,
    families=None,
    only_if_present=True,
):
    """Ensure each coverage family has at least min_per_family members in selected.

    Does not hard-code feature names. Promotes highest-scoring available members
    of under-represented families. Families with no candidates in `available`
    are skipped when only_if_present is True.

    Parameters
    ----------
    selected : list[str]
        Current selection (order preserved; promotions appended).
    ranked_scores : dict[str, float]
        Score map (higher is better) used to pick promotions.
    available : list[str] | set[str]
        Candidate pool (e.g. pre-stability or full train columns).
    min_per_family : int
        Minimum members per family (default 1).
    families : sequence[str] | None
        Families to enforce; defaults to DEFAULT_COVERAGE_FAMILIES.
    only_if_present : bool
        If True, skip families with zero candidates in available.

    Returns
    -------
    dict with keys: selected, promoted, family_counts_before, family_counts_after
    """
    log = get_logger("selectors.family_coverage")

    families = list(families) if families is not None else list(DEFAULT_COVERAGE_FAMILIES)
    available_set = set(available)
    selected_list = list(selected)
    selected_set = set(selected_list)

    def _counts(feats):
        c = {fam: 0 for fam in families}
        for f in feats:
            fam = infer_coverage_family(f)
            if fam in c:
                c[fam] += 1
        return c

    counts_before = _counts(selected_list)
    promoted = []

    # Pre-index available candidates by family, sorted by score desc
    avail_by_fam = {fam: [] for fam in families}
    for f in available_set:
        fam = infer_coverage_family(f)
        if fam in avail_by_fam:
            avail_by_fam[fam].append(f)
    for fam in families:
        avail_by_fam[fam].sort(key=lambda x: -float(ranked_scores.get(x, 0.0)))

    for fam in families:
        candidates = avail_by_fam[fam]
        if only_if_present and not candidates:
            continue
        have = sum(1 for f in selected_list if infer_coverage_family(f) == fam)
        need = int(min_per_family) - have
        if need <= 0:
            continue
        for f in candidates:
            if need <= 0:
                break
            if f in selected_set:
                continue
            # Prefer non-zero score features; still allow zero if nothing else
            selected_list.append(f)
            selected_set.add(f)
            promoted.append({"feature": f, "family": fam, "score": float(ranked_scores.get(f, 0.0))})
            need -= 1

    counts_after = _counts(selected_list)
    log.info(
        "family_coverage: promoted=%d counts_before=%s counts_after=%s",
        len(promoted),
        counts_before,
        counts_after,
    )
    if promoted:
        log.info("family_coverage promotions: %s", promoted)

    return {
        "kind": "family_coverage",
        "selected": selected_list,
        "promoted": promoted,
        "family_counts_before": counts_before,
        "family_counts_after": counts_after,
        "min_per_family": int(min_per_family),
        "families": families,
    }
