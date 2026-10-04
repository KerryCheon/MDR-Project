"""Configuration pinning for derived_8.4-routing-optimize-1.0 training.

Builds the pinned configurations from config.yaml ``pinned_configs``. All
entries are no-delta (``additions_source: none``): experts use the shared
54-feature backbone only. Same resolved schema as formal-eval-1.0
``eval_formal.configs`` (config_id, strategy_name, global_features,
cluster_additions, counts, delta_source, is_baseline) so drivers, workers,
and aggregation stay compatible.
"""

from __future__ import annotations


def load_pinned_configs(data, config: dict) -> list[dict]:
    """Pin the no-delta configurations declared in config.yaml."""
    configs: list[dict] = []
    for config_id, spec in config["pinned_configs"].items():
        strategy = str(spec["strategy"])
        source = str(spec["additions_source"])
        if source != "none":
            raise ValueError(
                f"Only additions_source 'none' is supported here; got {source!r} for {config_id}")
        configs.append({
            "config_id": config_id,
            "strategy_name": strategy,
            "global_features": list(data.shared_backbone_54),
            "cluster_additions": {"0": [], "1": []},
            "cluster_0_count": int(spec.get("c0", 0)),
            "cluster_1_count": int(spec.get("c1", 0)),
            "delta_source": "none",
            "is_baseline": False,
            "eval11_test_r2": float("nan"),
        })
    return configs


def config_frame(configurations: list[dict]):
    """DataFrame describing each configuration (reporting metadata)."""
    import pandas as pd

    rows = []
    for cfg in configurations:
        rows.append({
            "config_id": cfg["config_id"],
            "config_label": f"{cfg['strategy_name']}  c0={cfg['cluster_0_count']}, c1={cfg['cluster_1_count']}",
            "strategy_name": cfg["strategy_name"],
            "delta_source": cfg["delta_source"],
            "is_baseline": cfg["is_baseline"],
            "cluster_0_count": cfg["cluster_0_count"],
            "cluster_1_count": cfg["cluster_1_count"],
            "n_global_features": len(cfg["global_features"]),
            "n_add0": 0,
            "n_add1": 0,
        })
    return pd.DataFrame(rows)
