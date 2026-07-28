from __future__ import annotations

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/cache")

import pandas as pd

from common import ARTIFACT_DIR, feature_variants, save_json
from run_hybrid import (
    load_exported_embedding_frames,
    prepare_source_splits,
    run_tree_suite,
)


def main() -> None:
    splits, _, texture_columns = prepare_source_splits()
    source = pd.concat(splits.values(), ignore_index=True)
    feature_summary = pd.read_csv(
        ARTIFACT_DIR / "feature_experiments" / "legacy_five" / "summary.csv"
    )
    selected_tabular_name = (
        feature_summary[feature_summary["split"] == "val"]
        .sort_values("rmse")
        .iloc[0]["variant"]
    )
    selected_features = feature_variants(texture_columns)[selected_tabular_name]

    combined_path = ARTIFACT_DIR / "hybrid" / "metrics.json"
    combined = json.loads(combined_path.read_text(encoding="utf-8"))
    for protocol_name in ("conventional_full_train", "strict_frozen"):
        frames, embedding_columns = load_exported_embedding_frames(
            protocol_name, source
        )
        combined[protocol_name] = run_tree_suite(
            protocol_name,
            frames,
            embedding_columns,
            selected_features,
        )
    save_json(combined, combined_path)

    for protocol_name in ("conventional_full_train", "strict_frozen"):
        summary = pd.read_csv(
            ARTIFACT_DIR / "hybrid" / protocol_name / "summary.csv"
        )
        print(f"\n{protocol_name} test")
        print(
            summary[summary["split"] == "test"]
            [["model", "r2", "rmse", "mae", "macro_station_r2"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

