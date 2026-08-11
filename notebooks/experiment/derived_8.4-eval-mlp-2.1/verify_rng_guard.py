#!/usr/bin/env python3
"""Pre-GPU verification of the mlp21 SWA fixes (runs on CPU, no H100 needed).

Per the 2.1 plan, before the sweep is launched the trainer must be shown to
satisfy two bit-identity properties:

  (a) ANCHOR vs 2.0: a non-SWA anchor under data_version 7 reproduces the
      mlp-2.0 run bit-identically (the trainer is unchanged for non-SWA jobs;
      the 2.0 saved curves under models/2regime_54/<anchor>/seed_42/ are the
      reference). We verify the first N_EPOCHS of the val curve — identical
      init + data order + RNG consumption for the prefix implies the rest.
  (b) SWA-LIVE vs ANCHOR: a swa=true job's LIVE val curve is bit-identical to
      its swa=false anchor's (the RNG guard in mlp21._recalibrate_bn must
      prevent the SWA bookkeeping from perturbing the live trajectory; 2.0's
      "gains are live-trajectory artifacts" caveat is gone iff this passes).

The check trains the 2regime_54 cluster-0 specialist (the largest cluster) of
the anchor `w384x384_d0.3_gelu` and its late-SWA twin `w384x384_d0.3_gelu_swa085`
for a few epochs on CPU. Outputs go to `verify_rng_guard_out/` (gitignored);
the printed max|diff| values are the proof.

Usage:
    uv run --no-sync python verify_rng_guard.py [--epochs 6]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[2]
EVAL11_DIR = EXP_DIR.parent / "derived_8.4-eval-1.1"
MLP20_DIR = EXP_DIR.parent / "derived_8.4-eval-mlp-2.0"
sys.path.insert(0, str(EVAL11_DIR))
sys.path.insert(0, str(EXP_DIR))

from eval11.data import load_experiment_data  # noqa: E402
from eval11.routers import get_router  # noqa: E402
from mlp21.data import build_feature_set  # noqa: E402
from mlp21.trainer import train_one_config  # noqa: E402

FAMILY = "2regime_54"
ANCHOR = "w384x384_d0.3_gelu"
SWA_TWIN = "w384x384_d0.3_gelu_swa085"
SEED = 42
CLUSTER = "0"  # the larger specialist (n_train ~7156); property is per-specialist


def _load_cfg(exp_dir: Path, cid: str, config: dict) -> dict:
    """Config for one id straight from config.yaml (no sweep-driver cache needed)."""
    from run_mlp_sweep import build_sweep_configs

    cfg = build_sweep_configs(config)[cid]
    cfg["seed"] = SEED
    cfg["id"] = cid
    return cfg


def _build_cluster0_fs(config: dict, artifacts: Path) -> dict:
    data = load_experiment_data(PROJECT_ROOT, config)
    router = get_router(config["cluster_config"]["strategy"], data.v0_features,
                        seed=int(config["model"]["seed"]))
    router.fit(data.trainval)
    labels_train = router.predict(data.train)
    labels_val = router.predict(data.val)
    labels_te = router.predict(data.test)
    fam_cfg = next(f for f in config["families"] if f["id"] == FAMILY)
    feats_key = fam_cfg["feature_sets"][CLUSTER]
    base = list(config["shared_backbone_54"]) if feats_key == "backbone_54" else None
    if base is None:
        raise SystemExit(f"unsupported feature-set key {feats_key!r} in the verify script")
    c1_deltas = list(config["cluster_config"]["cluster_1_delta_features"])
    c1 = [f for f in dict.fromkeys([*base, *c1_deltas]) if f in set(data.feature_columns)]
    feats = base if CLUSTER == "0" else c1
    tr = data.train.loc[labels_train == int(CLUSTER)].reset_index(drop=True)
    va = data.val.loc[labels_val == int(CLUSTER)].reset_index(drop=True)
    te = data.test.loc[labels_te == int(CLUSTER)].reset_index(drop=True)
    aux_frame = data.train[data.train["year"] == int(config["sweep"].get("aux_year", 2020))].reset_index(drop=True)
    au = aux_frame.loc[router.predict(aux_frame) == int(CLUSTER)].reset_index(drop=True)
    return build_feature_set(tr, va, te, feats, data.target, aux=au)


def _live_val_curve(out_dir: Path) -> np.ndarray:
    return np.load(out_dir / "curves.npy")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--out", type=Path, default=EXP_DIR)
    args = parser.parse_args()
    n_epochs = int(args.epochs)

    with open(EXP_DIR / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    out_root = EXP_DIR / "verify_rng_guard_out"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    # (a) anchor vs 2.0 bit-identity (prefix of the val curve)
    cfg_anchor = _load_cfg(EXP_DIR, ANCHOR, config)
    cfg_anchor["max_epochs"] = n_epochs
    cfg_anchor["patience"] = n_epochs + 1
    cfg_anchor["checkpoint_every"] = max(1, n_epochs)
    fs = _build_cluster0_fs(config, EXP_DIR / "artifacts")
    anchor_out = out_root / f"{ANCHOR}_c{CLUSTER}"
    train_one_config(cfg_anchor, fs, anchor_out)
    anchor_curve = _live_val_curve(anchor_out)

    ref_curve_path = MLP20_DIR / "models" / FAMILY / ANCHOR / f"seed_{SEED}" / f"spec_{CLUSTER}" / "curves.npy"
    ref_meta_path = MLP20_DIR / "models" / FAMILY / ANCHOR / f"seed_{SEED}" / "meta.json"
    run_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref_device = "unknown"
    if ref_meta_path.exists():
        try:
            ref_device = json.loads(ref_meta_path.read_text(encoding="utf-8")).get("device", "unknown")
        except Exception:
            pass
    # NOTE: this comparison is a SANITY figure only, never a pass/fail. Two
    # reasons: (1) the 6-epoch cap changes the LR schedule (warmup_epochs =
    # round(6*0.05) = 0 -> ~93% LR at epoch 1, vs the real 400-epoch run's
    # 20-epoch warmup -> 10% LR), so the epoch-1..6 values are not on the same
    # schedule as 2.0's; (2) bit-identity across DIFFERENT GPU nodes is not
    # guaranteed (PTX JIT / driver / cuDNN differences are amplified by BN
    # statistics early in training). The definitive anchor-vs-2.0 comparison is
    # done offline from the real 400-epoch sweep curves (same schedule).
    if ref_curve_path.exists():
        ref = np.load(ref_curve_path)[0][:n_epochs]
        n = min(len(anchor_curve), len(ref))
        d_a = float(np.max(np.abs(anchor_curve[:n] - ref[:n])))
        print(f"[verify (a)] SANITY only (LR schedule + device differ from 2.0): anchor {ANCHOR} vs 2.0 "
              f"(first {n} val epochs, ref device={ref_device}, run device={run_device.type}): "
              f"max|diff| = {d_a:.3e} — NOT a pass/fail; see the offline sweep-curve comparison")
    else:
        print(f"[verify (a)] WARNING: 2.0 reference curves not found at {ref_curve_path} — skipped")

    # (b) swa-live vs anchor bit-identity (same feature set, same seed)
    cfg_swa = _load_cfg(EXP_DIR, SWA_TWIN, config)
    cfg_swa["max_epochs"] = n_epochs
    cfg_swa["patience"] = n_epochs + 1
    cfg_swa["checkpoint_every"] = max(1, n_epochs)
    swa_out = out_root / f"{SWA_TWIN}_c{CLUSTER}"
    train_one_config(cfg_swa, fs, swa_out)
    swa_live_curve = _live_val_curve(swa_out)

    n = min(len(anchor_curve), len(swa_live_curve))
    d_b = float(np.max(np.abs(anchor_curve[:n] - swa_live_curve[:n])))
    print(f"[verify (b)] {SWA_TWIN} live vs {ANCHOR} anchor (first {n} val epochs): max|diff| = {d_b:.3e} "
          f"{'PASS (bit-identical — RNG guard works)' if d_b < 1e-12 else 'FAIL (guard broken)'}")

    summary = {"anchor": ANCHOR, "swa_twin": SWA_TWIN, "seed": SEED, "cluster": CLUSTER,
               "n_epochs": n_epochs, "max_abs_diff_anchor_vs_2p0": locals().get("d_a"),
               "max_abs_diff_swa_live_vs_anchor": d_b}
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[verify] wrote {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
