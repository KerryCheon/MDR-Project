# Jakob Balkovec
# Feature Groups

from Modeling.Utils.logging import get_logger

def infer_family(feature_name):
    # desc: returns a family tag like: RAW, A, C, V, F, E, G, H, D, I, etc.

    s = str(feature_name)

    # base features: no prefix or known raw names
    raw_names = {
        "precip_mm", "s1_vv", "s1_vh", "s2_b4", "s2_b8", "s2_b11", "s2_b12",
        "LST_modis", "aspect", "DOY"
    }

    if s in raw_names:
        return "RAW"

    if "_" not in s:
        return "RAW"

    prefix = s.split("_", 1)[0]

    # common families in your naming scheme
    known = {"A", "C", "D", "E", "F", "G", "H", "I", "V"}
    if prefix in known:
        return prefix

    # sometimes you have things like "s2_b11" which are raw
    if s.startswith("s1_") or s.startswith("s2_") or s.startswith("LST_"):
        return "RAW"

    return "OTHER"


def group_features(feature_cols):
    log = get_logger("features.groups")

    groups = {}
    for f in feature_cols:
        fam = infer_family(f)
        if fam not in groups:
            groups[fam] = []
        groups[fam].append(f)

    # log a compact summary
    summary = {k: len(v) for k, v in groups.items()}
    log.info("group_features: family counts=%s", summary)

    return groups


def drop_metadata_cols(feature_cols, metadata_cols=None):
    # note: Removes columns like station_id/date if they accidentally appear in feature list
    # @see preprocess (coerce_numeric) for more info

    metadata_cols = metadata_cols or ["station_id", "date"]

    meta = set(metadata_cols)
    kept = [c for c in feature_cols if c not in meta]
    dropped = [c for c in feature_cols if c in meta]

    log = get_logger("features.groups")
    if dropped:
        log.info("drop_metadata_cols: dropped=%s", dropped)

    return kept, dropped
