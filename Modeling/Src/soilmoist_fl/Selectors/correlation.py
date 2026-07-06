# Jakob Balkovec
# Correlation Selector

import numpy as np
import pandas as pd
from Modeling.Utils.logging import get_logger


def select_correlation(X, y, threshold=0.95, random_state=42):
    log = get_logger("selectors.correlation")

    feature_cols = list(X.columns)
    
    # Coerce y to pandas Series if not already one
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    y_num = pd.to_numeric(y_series, errors="coerce").to_numpy()

    # Compute correlation with target y
    corrs_with_y = {}
    for c in feature_cols:
        col_data = pd.to_numeric(X[c], errors="coerce").to_numpy()
        mask = np.isfinite(col_data) & np.isfinite(y_num)
        if mask.sum() > 1:
            # Absolute pearson correlation coefficient
            corrs_with_y[c] = abs(np.corrcoef(col_data[mask], y_num[mask])[0, 1])
        else:
            corrs_with_y[c] = 0.0

    # Compute pairwise correlation matrix of X (pandas corr handles missing values pairwise)
    corr_matrix = X.corr(method="pearson").abs()

    to_drop = set()
    for i in range(len(feature_cols)):
        col_i = feature_cols[i]
        if col_i in to_drop:
            continue
        for j in range(i + 1, len(feature_cols)):
            col_j = feature_cols[j]
            if col_j in to_drop:
                continue

            val = corr_matrix.loc[col_i, col_j]
            if pd.notna(val) and val > threshold:
                # Keep feature with higher absolute correlation to target
                if corrs_with_y.get(col_i, 0.0) >= corrs_with_y.get(col_j, 0.0):
                    to_drop.add(col_j)
                else:
                    to_drop.add(col_i)
                    break  # col_i is dropped, no need to check other features against it

    selected = [c for c in feature_cols if c not in to_drop]
    log.info(
        "select_correlation: features=%d threshold=%.3f dropped=%d kept=%d",
        len(feature_cols),
        threshold,
        len(to_drop),
        len(selected),
    )

    score_map = {c: float(corrs_with_y.get(c, 0.0)) for c in feature_cols}
    ranked = sorted(feature_cols, key=lambda c: -score_map[c])

    return {
        "kind": "correlation",
        "ranked": ranked,
        "scores": score_map,
        "selected": selected,
        "dropped": list(to_drop),
    }
