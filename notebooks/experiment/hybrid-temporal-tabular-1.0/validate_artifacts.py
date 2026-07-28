from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import ARTIFACT_DIR, ID_COLUMNS, regression_metrics


def validate_embedding_protocol(protocol: str) -> None:
    root = ARTIFACT_DIR / "hybrid" / protocol
    for split_name in ("train", "val", "test"):
        frame = pd.read_csv(root / "embeddings" / f"{split_name}.csv")
        embedding_columns = [
            column for column in frame.columns if column.startswith("lstm_z_")
        ]
        assert len(embedding_columns) == 64
        assert not frame.duplicated(ID_COLUMNS).any()
        assert np.isfinite(frame[embedding_columns].to_numpy()).all()
        assert frame["window_span_days"].max() <= 20

        prediction = pd.read_csv(
            root
            / "predictions"
            / f"T3_selected_tabular_plus_embedding_{split_name}.csv"
        )
        assert not prediction.duplicated(ID_COLUMNS).any()
        assert len(prediction) == len(frame)
        merged = frame[ID_COLUMNS].merge(
            prediction[ID_COLUMNS], on=ID_COLUMNS, validate="one_to_one"
        )
        assert len(merged) == len(frame)

    summary = pd.read_csv(root / "summary.csv")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    for model_name in summary["model"].unique():
        for split_name in summary.loc[summary["model"] == model_name, "split"]:
            if model_name == "T1_temporal_only":
                continue
            prediction = pd.read_csv(
                root / "predictions" / f"{model_name}_{split_name}.csv"
            )
            recomputed = regression_metrics(prediction["y_true"], prediction["y_pred"])
            stored = metrics[model_name][split_name]["overall"]
            assert abs(recomputed["rmse"] - stored["rmse"]) < 1e-12
            assert abs(recomputed["r2"] - stored["r2"]) < 1e-12


def main() -> None:
    validate_embedding_protocol("conventional_full_train")
    validate_embedding_protocol("strict_frozen")
    touchet = pd.read_csv(ARTIFACT_DIR / "touchet_analysis" / "touchet_metrics.csv")
    assert "strict_hybrid" in touchet["model"].to_list()
    assert "persistence" in touchet["model"].to_list()
    shift = pd.read_csv(ARTIFACT_DIR / "touchet_analysis" / "touchet_feature_shift.csv")
    smap = shift[shift["feature"] == "SMAP_sm_interp"].iloc[0]
    assert smap["train_missing_fraction"] == 1.0
    assert smap["test_missing_fraction"] == 1.0
    print("All hybrid-temporal-tabular artifacts validated.")


if __name__ == "__main__":
    main()

