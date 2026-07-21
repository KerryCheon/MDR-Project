"""Rerun only global outer selection with the locked evaluation architecture.

Inner rankings and candidate lists remain frozen under ``artifacts/nested``.
This isolates model-architecture mismatch without rerunning or adapting the
importance search. Outputs go to ``artifacts/nested_locked_outer``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from artifact_state import (
    artifact_is_complete,
    atomic_write_json,
)
from runtime import add_runtime_arguments, validate_workers


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from Modeling.Src.soilmoist_fl.Selectors.grouped_oof import (  # noqa: E402
    evaluate_forward_station_time_candidates,
)
from run_nested_selection import (  # noqa: E402
    NESTED_REQUIRED_FILES,
    _load_inner_outer,
    _prepare_xy,
    _write_nested_artifact,
)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    workers = validate_workers(args.workers)
    with open(EXP_DIR / "nested_config.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    device = args.device
    outer_config = json.loads(json.dumps(config["outer_selection"]))
    outer_config["model_params"]["device"] = device
    outer_config["parallel_workers"] = workers
    output_root = EXP_DIR / "artifacts/nested_locked_outer"
    summaries = []

    for dataset in ("derived_8.0", "derived_8.2"):
        scope_dir = output_root / dataset / "global"
        selected_path = scope_dir / "selected_features.json"
        if artifact_is_complete(
            scope_dir,
            NESTED_REQUIRED_FILES,
        ):
            with open(selected_path, encoding="utf-8") as stream:
                selected_payload = json.load(stream)
            summaries.append(
                {
                    "dataset": dataset,
                    "n_features": int(selected_payload["n_features"]),
                    "outer_stopping_reason": selected_payload[
                        "outer_stopping_reason"
                    ],
                    "source_inner_selection": (
                        "notebooks/experiment/derived_8.2-feature-selection-2.2/"
                        f"artifacts/nested/{dataset}/global/inner_selection.json"
                    ),
                }
            )
            print(f"Reusing completed locked outer result for {dataset}", flush=True)
            continue
        train, outer, hashes = _load_inner_outer(dataset)
        station_col = config["data"]["station_col"]
        time_col = config["data"]["time_col"]
        X_inner, y_inner, context_inner = _prepare_xy(
            train,
            station_col,
            time_col,
        )
        X_outer, y_outer, context_outer = _prepare_xy(
            outer,
            station_col,
            time_col,
        )
        source_path = (
            EXP_DIR
            / "artifacts/nested"
            / dataset
            / "global/inner_selection.json"
        )
        if not artifact_is_complete(
            source_path.parent,
            NESTED_REQUIRED_FILES,
        ):
            raise FileNotFoundError(
                "Run nested selection first; source artifact is incomplete: "
                f"{source_path.parent}"
            )
        with open(source_path, encoding="utf-8") as stream:
            inner_result = json.load(stream)
        candidates = [
            candidate["features"] for candidate in inner_result["selection_path"]
        ]
        outer_result = evaluate_forward_station_time_candidates(
            X_inner,
            y_inner,
            context_inner,
            X_outer,
            y_outer,
            context_outer,
            candidates,
            config=outer_config,
        )
        _write_nested_artifact(
            scope_dir,
            dataset=dataset,
            scope="global_locked_outer",
            inner_result=inner_result,
            outer_result=outer_result,
            split_hashes=hashes,
            device=device,
        )
        summaries.append(
            {
                "dataset": dataset,
                "n_features": len(outer_result["selected"]),
                "outer_stopping_reason": outer_result["stopping_reason"],
                "source_inner_selection": str(source_path.relative_to(PROJECT_ROOT)),
            }
        )
        print(json.dumps(summaries[-1], indent=2), flush=True)

    atomic_write_json(
        output_root / "selection_summary.json",
        {
            "device": device,
            "parallel_workers": outer_config["parallel_workers"],
            "model_params": outer_config["model_params"],
            "datasets": summaries,
        },
    )


if __name__ == "__main__":
    main()
