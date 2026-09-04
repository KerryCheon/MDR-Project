"""Configuration pinning for derived_8.4-formal-eval-2.1-ece-v3.

Builds the 20 pinned configurations:
  - 14 requested configs with test-selected additions (parsed from
    derived_8.4-eval-1.1/delta_grid_summary.csv) or none;
  - 6 val-selected winners (read from val_selected_deltas.json, reused from formal-eval-1.0/2.0).

Each configuration carries ``delta_source`` in {test, val, none, global} for the
robustness reporting, plus the eval-1.1 test R2 replication anchor where available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
EXP_DIR = Path(__file__).resolve().parent.parent


def parse_additions(value: object) -> list[str]:
    """Parse semicolon-joined cluster additions from eval-1.1 CSV cells."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [f for f in text.split(";") if f]


def _load_eval11_grid(eval11_dir: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    grid = pd.read_csv(eval11_dir / "delta_grid_summary.csv")
    leaderboard = pd.read_csv(eval11_dir / "metrics_summary.csv")
    eval11_r2: dict[str, float] = {}
    for _, row in grid.iterrows():
        eval11_r2[str(row["candidate_id"])] = float(row["pooled_r2"])
    for _, row in leaderboard.iterrows():
        eval11_r2[str(row["candidate_id"])] = float(row["pooled_r2"])
    return grid, eval11_r2


def _grid_additions(grid: pd.DataFrame, strategy: str, c0: int, c1: int) -> dict[str, list[str]]:
    sub = grid[grid["strategy_name"] == strategy]
    row = sub[(sub["cluster_0_count"] == c0) & (sub["cluster_1_count"] == c1)]
    if row.empty:
        raise ValueError(f"No eval-1.1 grid row for {strategy} c0={c0} c1={c1}")
    row = row.iloc[0]
    return {
        "0": parse_additions(row["cluster_0_additions"]),
        "1": parse_additions(row["cluster_1_additions"]),
    }


def load_pinned_configs(data, config: dict, require_val: bool = True) -> list[dict]:
    """Pin the 20 configurations (14 test-selected/none + 6 val-selected winners)."""
    eval11_dir = PROJECT_ROOT / Path(config["val_selection"]["eval11_dir"])
    grid, eval11_r2 = _load_eval11_grid(eval11_dir)

    configs: list[dict] = []
    for config_id, spec in config["pinned_configs"].items():
        strategy = str(spec["strategy"])
        source = str(spec["additions_source"])
        if source == "global":
            global_features = data.shared_backbone_54
            additions: dict[str, list[str]] = {"0": [], "1": []}
            delta_source = "global"
        elif source == "global_v0":
            global_features = data.v0_features
            additions = {"0": [], "1": []}
            delta_source = "global"
        elif source == "none":
            global_features = data.shared_backbone_54
            additions = {"0": [], "1": []}
            delta_source = "none"
        elif source == "eval11_grid":
            global_features = data.shared_backbone_54
            additions = _grid_additions(grid, strategy, int(spec["c0"]), int(spec["c1"]))
            delta_source = "test"
        elif source == "eval11_v0full_grid":
            global_features = data.shared_backbone_54
            additions = _grid_additions(grid, "Clustering_V0_Full_k2",
                                        int(spec["c0"]), int(spec["c1"]))
            delta_source = "test"
        else:
            raise ValueError(f"Unknown additions_source {source!r} for {config_id}")
        configs.append({
            "config_id": config_id,
            "strategy_name": strategy,
            "global_features": list(global_features),
            "cluster_additions": {str(k): list(v) for k, v in additions.items()},
            "cluster_0_count": spec.get("c0"),
            "cluster_1_count": spec.get("c1"),
            "delta_source": delta_source,
            "is_baseline": source in ("global", "global_v0"),
            "eval11_test_r2": eval11_r2.get(config_id, float("nan")),
        })

    # Val-selected winners (6): one per MoE strategy, from val_selected_deltas.json.
    val_path = EXP_DIR / Path(config["val_selection"]["output_json"])
    if not val_path.exists():
        if require_val:
            raise FileNotFoundError(
                f"{val_path} not found — ensure val_selected_deltas.json is present.")
        return configs
    val_selected = json.loads(val_path.read_text(encoding="utf-8"))
    for strategy in config["val_selection"]["strategies"]:
        entry = val_selected.get("strategies", {}).get(strategy)
        if entry is None:
            raise ValueError(f"val_selected_deltas.json missing strategy {strategy}")
        config_id = f"{strategy}_val_winner"
        winner = entry["winner"]
        configs.append({
            "config_id": config_id,
            "strategy_name": strategy,
            "global_features": list(data.shared_backbone_54),
            "cluster_additions": {
                "0": list(entry["additions"]["0"][: int(winner[0])]),
                "1": list(entry["additions"]["1"][: int(winner[1])]),
            },
            "cluster_0_count": int(winner[0]),
            "cluster_1_count": int(winner[1]),
            "delta_source": "val",
            "is_baseline": False,
            "eval11_test_r2": float("nan"),
        })
    return configs


def config_frame(configurations: list[dict]) -> pd.DataFrame:
    """DataFrame describing each configuration (reporting / plotting metadata)."""
    rows = []
    for cfg in configurations:
        label = cfg["config_id"]
        if not cfg["is_baseline"] and cfg["cluster_0_count"] is not None:
            label = f"{cfg['strategy_name']}  c0={cfg['cluster_0_count']}, c1={cfg['cluster_1_count']}"
        rows.append({
            "config_id": cfg["config_id"],
            "config_label": label,
            "strategy_name": cfg["strategy_name"],
            "delta_source": cfg["delta_source"],
            "is_baseline": cfg["is_baseline"],
            "cluster_0_count": cfg["cluster_0_count"],
            "cluster_1_count": cfg["cluster_1_count"],
            "eval11_test_r2": cfg["eval11_test_r2"],
            "n_global_features": len(cfg["global_features"]),
            "n_add0": len(cfg["cluster_additions"].get("0", [])),
            "n_add1": len(cfg["cluster_additions"].get("1", [])),
        })
    return pd.DataFrame(rows)
