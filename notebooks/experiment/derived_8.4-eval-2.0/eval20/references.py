"""Temporal-test reference lookup for derived_8.4-eval-2.0.

The LOSO leaderboard compares each MLP config's spatial (LOSO) R2 against its
*temporal* test R2 from the experiments that established the configs:

  - 2-regime families  -> derived_8.4-eval-mlp-1.3 (metrics_summary.csv)
  - 1-regime families  -> derived_8.4-eval-mlp-1.1 (metrics_summary.csv)

The same map drives the full-baseline replication check: the full-training
baseline re-trains the exact mlp-1.3 / mlp-1.1 protocol on all 7 stations, so
its pooled test R2 must match these references (deterministic, |diff| ~ 0).

XGBoost LOSO reference rows (for the MLP-vs-XGBoost spatial comparison) are
loaded from derived_8.4-eval-1.2's loso_config_summary.csv /
loso_per_config_station.csv (no retraining).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

EXP_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = EXP_DIR.parents[2]

XGB_LOSO_REF_CONFIGS = [
    "Global_Single_54",
    "Clustering_V0_Full_k2_c0_0_c1_10",
]


def load_mlp_reference_map(config: dict) -> dict[tuple[str, str], float]:
    """{(family, config_id): temporal pooled test R2} from mlp-1.3 + mlp-1.1.

    Later rows win (mlp-1.3 loaded second — its rows take precedence for
    (family, config_id) collisions, which in practice do not occur).
    """
    out: dict[tuple[str, str], float] = {}
    for key in ("mlp11_reference_file", "mlp13_reference_file"):
        path = PROJECT_ROOT / Path(config[key])
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            fam = str(row.get("family", ""))
            cid = str(row.get("config_id", ""))
            r2 = row.get("pooled_r2")
            if not fam or not cid or r2 is None or pd.isna(r2):
                continue
            out[(fam, cid)] = float(r2)
    return out


def load_eval12_loso_reference_summary(config: dict) -> pd.DataFrame:
    """eval-1.2 LOSO summary rows for the XGBoost references (leaderboard merge)."""
    src = PROJECT_ROOT / Path(config["eval12_reference_dir"]) / "loso_config_summary.csv"
    df = pd.read_csv(src)
    return df[df["config_id"].isin(XGB_LOSO_REF_CONFIGS)].copy()


def load_eval12_loso_reference_folds(config: dict) -> pd.DataFrame:
    """eval-1.2 LOSO per-config x station rows for the XGBoost references."""
    src = PROJECT_ROOT / Path(config["eval12_reference_dir"]) / "loso_per_config_station.csv"
    df = pd.read_csv(src)
    return df[df["config_id"].isin(XGB_LOSO_REF_CONFIGS)].copy()
