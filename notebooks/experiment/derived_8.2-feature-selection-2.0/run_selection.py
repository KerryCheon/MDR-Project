#!/usr/bin/env python3
"""Run versioned feature-selection variants for derived_8.2-feature-selection-2.0.

Usage (from repo root):
  PYTHONPATH=. python notebooks/experiment/derived_8.2-feature-selection-2.0/run_selection.py \\
      --dataset derived_8.2 --variants c2_xgb c0_baseline_bypass_on

Artifacts land under:
  notebooks/experiment/derived_8.2-feature-selection-2.0/artifacts/<dataset>/<variant>/
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = EXP_DIR / "configs"
ARTIFACTS_DIR = EXP_DIR / "artifacts"

VARIANT_CONFIGS = {
    "c0_baseline_bypass_on": "config_c0_baseline_bypass_on.yaml",
    "c1_baseline_bypass_off": "config_c1_baseline_bypass_off.yaml",
    "c2_xgb": "config_c2_xgb.yaml",
    "c2b_xgb_softcorr": "config_c2b_xgb_softcorr.yaml",
    "c2c_xgb_nocorr": "config_c2c_xgb_nocorr.yaml",
    "c2d_xgb_softcorr_k65": "config_c2d_xgb_softcorr_k65.yaml",
    "c3_xgb_no_coverage": "config_c3_xgb_no_coverage.yaml",
    "c4_hybrid": "config_c4_hybrid.yaml",
    "c5_rf": "config_c5_rf.yaml",
}

DATASETS = {
    "derived_8.0": {
        "train": "data/splits/derived_8.0/train.csv",
        "val": "data/splits/derived_8.0/val.csv",
        "test": "data/splits/derived_8.0/test.csv",
    },
    "derived_8.2": {
        "train": "data/splits/derived_8.2/train.csv",
        "val": "data/splits/derived_8.2/val.csv",
        "test": "data/splits/derived_8.2/test.csv",
    },
}


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_variant(dataset: str, variant: str, n_boot: int | None = None) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}")

    sys.path.insert(0, str(PROJECT_ROOT))
    from Modeling.Src.soilmoist_fl.cli import select_features
    from Modeling.Src.soilmoist_fl.Features.preprocess import preprocess_split
    from Modeling.Src.soilmoist_fl.Selectors.family_coverage import (
        group_by_coverage_family,
        infer_coverage_family,
    )
    import pandas as pd

    cfg_path = CONFIGS_DIR / VARIANT_CONFIGS[variant]
    cfg = _load_config(cfg_path)

    # Point splits at the requested dataset
    splits = DATASETS[dataset]
    cfg["data"]["splits"] = dict(splits)

    if n_boot is not None:
        cfg["selection"]["stability_n_boot"] = int(n_boot)
        for st in cfg["selection"].get("stages", []):
            if st.get("kind") == "stability":
                st["stability_n_boot"] = int(n_boot)

    out_dir = ARTIFACTS_DIR / dataset / variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact config used
    used_cfg_path = out_dir / "config_used.yaml"
    with open(used_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Load data
    train = pd.read_csv(PROJECT_ROOT / splits["train"])
    val = pd.read_csv(PROJECT_ROOT / splits["val"])
    test = pd.read_csv(PROJECT_ROOT / splits["test"])

    target = cfg["data"]["target"]
    drop_cols = list(cfg["data"].get("id_cols", [])) + [cfg["data"].get("time_col", "date")]
    X_tr, y_tr, _, _ = preprocess_split(train, target, drop_cols=drop_cols)
    X_va, y_va, _, _ = preprocess_split(val, target, drop_cols=drop_cols)
    X_te, y_te, _, _ = preprocess_split(test, target, drop_cols=drop_cols)

    run_dir = out_dir / "run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    res = select_features(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        config=cfg,
        run_dir=run_dir,
        run_id=f"{dataset}_{variant}",
        verbose=True,
    )

    feats = list(res["selected_features"])
    fam_groups = group_by_coverage_family(feats)
    fam_counts = {k: len(v) for k, v in sorted(fam_groups.items())}

    payload = {
        "version": "V6",
        "variant": variant,
        "dataset": dataset,
        "created": datetime.now(timezone.utc).isoformat(),
        "config_path": str(cfg_path.relative_to(PROJECT_ROOT)),
        "config_used": str(used_cfg_path.relative_to(PROJECT_ROOT)),
        "pipeline_stages": [s.get("kind") for s in cfg["selection"].get("stages", [])],
        "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
        "n_features": len(feats),
        "features": feats,
        "family_counts": fam_counts,
        "family_coverage": res.get("family_coverage"),
        "git_commit": _git_commit(),
    }

    feat_path = out_dir / "selected_features.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # Human-readable report
    lines = [
        f"# Selection report: {dataset} / {variant}",
        "",
        f"- Created: {payload['created']}",
        f"- Git commit: {payload['git_commit']}",
        f"- n_features: {len(feats)}",
        f"- stages: {payload['pipeline_stages']}",
        f"- family_counts: {fam_counts}",
        "",
        "## Features",
        "",
    ]
    for i, f in enumerate(feats, 1):
        lines.append(f"{i}. `{f}` ({infer_coverage_family(f)})")
    report_path = out_dir / "selection_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[{dataset}/{variant}] selected {len(feats)} features → {feat_path}")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        choices=list(DATASETS.keys()),
        help="Dataset(s) to run (repeatable). Default: both.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=list(VARIANT_CONFIGS.keys()),
        default=list(VARIANT_CONFIGS.keys()),
        help="Variants to run (default: all).",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=None,
        help="Override stability_n_boot for faster smoke tests.",
    )
    args = parser.parse_args(argv)

    datasets = args.dataset or list(DATASETS.keys())
    results = []
    for ds in datasets:
        for var in args.variants:
            results.append(run_variant(ds, var, n_boot=args.n_boot))

    summary_path = ARTIFACTS_DIR / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"dataset": r["dataset"], "variant": r["variant"], "n_features": r["n_features"],
              "family_counts": r["family_counts"]} for r in results],
            f,
            indent=2,
        )
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
