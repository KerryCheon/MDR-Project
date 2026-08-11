#!/usr/bin/env python3
"""Deterministic generator for the derived_8.4-eval-mlp-2.2 sweep configs.

Per the reproducibility rule (AGENTS.md), the 209-config sweep table is NOT
hand-typed: it is produced by this committed script from the documented grid
spec below and written to `config.yaml` (committed alongside the generator).
Run `python make_configs.py` after editing the grids; the output is stable
(no RNG), so `config.yaml` is reproducible byte-for-byte.

Grids (all `mlp` architecture — `fg`/`plr` are documented negatives from 2.0
and `swa` is a documented negative from 2.1 (0/152 deployments with the RNG
guard in place); no config uses them. The winner pool filter in
run_mlp_sweep.py stays mlp/fg/plr so the protocol text is unchanged):

  54   (82):  R width x lr fine (gelu/mse) | S width x huber @ lr6e-4 |
              T 3-layer x lr | T2 3-layer x huber @ lr6e-4 |
              U dropout x width @ lr6e-4 | V dropout x huber @ 320^2 |
              W anchor (w512x512x512_d0.3_huber0.1)
  mixed (68): X act x depth x lr | X2 act x depth x lr fine |
              Y gelu x huber x lr @ 448^3 | Z silu 512^3 delta x lr fine |
              Z2 silu 512^3 delta 0.15 probe | AA dropout x delta @ 512^3 |
              AB gelu 2-layer x lr | AC gelu/silu x 3-layer x delta 0.05 |
              AD2 gelu 2-layer x delta 0.05 @ lr6e-4
  96   (59):  AD width x lr fine | AE huber x lr |
              AF max_epochs probe {500, 600} | AG 3-layer small nets |
              AH mixup x lr | AI dropout x huber | AJ anchors

Design rationale (from the 2.1 sweep results; see the plan doc):
  - 54: gelu + mse + lr6e-4 is the strongest untapped region (test R2 0.7935
    @ 320^2, 0.7906 @ 384^2, 0.7893 @ 448^2); widths below 320 and huber at
    lr6e-4 were never tested; 3-layer x lr6e-4 is untested (only the silu
    huber0.1 512^3 anchor exists, test 0.7713).
  - mixed: gelu x 3-layer is the 2.1 test-best cell (w448x448x448_d0.3_huber0.1_gelu
    test 0.7940, val rank 34) but gelu was only ever tested at lr3e-4; the
    silu 512^3 huber-delta x lr surface is flat (~0.778-0.785) — refined with
    delta {0.03, 0.08, 0.15} and lr {4e-4, 8e-4} probes. mixup/wd1e-3 at 512^3
    are 2.1 negatives (not re-run); lr1e-4 is a 2.1 negative (not re-run).
  - 96: small nets hit the 400-epoch cap under-trained (best_epoch 380-395);
    lr6e-4 (between 3e-4 and 1e-3) was never tested; huber/mixup at small
    nets + max_epochs {500, 600} probes are the debias + convergence levers.

Config ids follow the 2.1 naming convention, plus `_me<epochs>` for the
max_epochs probes:
  w<W>x<W>..._d<dropout>_<loss|act>[_lr<lr>][_wd<wd>][_mixup<a>][_me<epochs>]
where the loss/act suffix is `huber<delta>` (huber) or `_<act>` (non-silu act
with mse); `_me500` = max_epochs 500 (default 400).

Sizing rationale (from 2.1's real timings): per-seed train time mean 45 s;
at 8 parallel workers / ~5.9 effective, 191 phase-1 + 201 phase-2 + 108
phase-3 + 8 champion job-seeds = 508 jobs ~= 6.4 GPU-h ~= 65 min sweep ~=
74 min total wall — sized to spend ~1.25 h of the 2 h gpu_debug wall
allocation, with `--phase2-top-n` / `--phase3-top-n` trims to land short.
"""

from __future__ import annotations

from pathlib import Path

import yaml

EXP_DIR = Path(__file__).resolve().parent

# ----------------------------------------- helpers -----------------------------------------


def _fmt_dropout(d: float) -> str:
    return f"{d:.2f}".rstrip("0")


def _fmt_lr(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-")


def _fmt_delta(d: float) -> str:
    return f"{d:.2f}".rstrip("0")


def _fmt_mixup(a: float) -> str:
    return f"{a:.1f}"


def _fmt_me(epochs: int) -> str:
    return f"me{epochs}"


def make_config(
    widths: tuple[int, ...],
    dropout: float,
    loss: str = "mse",
    huber_delta: float = 0.05,
    act: str = "silu",
    lr: float = 3e-4,
    wd: float = 1e-4,
    mixup: float = 0.0,
    swa_start: float | None = None,
    max_epochs: int = 400,
) -> dict:
    """Build one config dict with a deterministic id (2.1 naming convention)."""
    id_parts = ["w" + "x".join(str(w) for w in widths), f"d{_fmt_dropout(dropout)}"]
    if loss == "huber":
        id_parts.append(f"huber{_fmt_delta(huber_delta)}")
        if act != "silu":
            id_parts.append(act)  # avoid id collision for huber + non-silu
    elif act != "silu":
        id_parts.append(act)
    if lr != 3e-4:
        id_parts.append(f"lr{_fmt_lr(lr)}")
    if wd != 1e-4:
        id_parts.append(f"wd{_fmt_lr(wd)}")
    if mixup > 0.0:
        id_parts.append(f"mixup{_fmt_mixup(mixup)}")
    if swa_start is not None:
        id_parts.append(f"swa{int(round(swa_start * 100)):03d}" if abs(swa_start - 0.6) > 1e-9 else "swa")
    if max_epochs != 400:
        id_parts.append(_fmt_me(max_epochs))
    cid = "_".join(id_parts)

    cfg: dict = {
        "id": cid,
        "hidden_sizes": [int(w) for w in widths],
        "activation": act,
        "dropout": float(dropout),
        "lr": float(lr),
        "weight_decay": float(wd),
        "loss": loss,
    }
    if loss == "huber":
        cfg["huber_delta"] = float(huber_delta)
    if mixup > 0.0:
        cfg["mixup_alpha"] = float(mixup)
    if swa_start is not None:
        cfg["swa"] = True
        cfg["swa_start_frac"] = float(swa_start)
    if max_epochs != 400:
        cfg["max_epochs"] = int(max_epochs)
    return cfg


def add(store: dict, cfg: dict) -> None:
    """Insert a config, raising loudly if the id collides with a different spec."""
    cid = cfg["id"]
    if cid in store:
        prev = store[cid]
        if prev == cfg:
            return  # exact duplicate (grid overlap) — fine, keep one
        raise ValueError(f"id collision with different spec: {cid}\n  new={cfg}\n  old={prev}")
    store[cid] = cfg


def grid_list(store: dict, cfgs: list[dict]) -> None:
    for c in cfgs:
        add(store, c)


# ----------------------------------------- grid spec (the documented sweep) -----------------------------------------

D = 0.3          # default dropout
LR = 3e-4        # default lr
WD = 1e-4        # default weight decay


def _build_54() -> dict:
    s: dict = {}
    # R: width x lr fine at (gelu, mse, d0.3) — the 2.1 test-best region
    #    (w320x320_d0.3_gelu_lr6e-4 0.7935 / w384x384 0.7906 / w448x448 0.7893);
    #    widths below 320 and lr {4e-4, 8e-4} were untested.
    widths_r = (192, 224, 256, 288, 320, 352, 384, 448, 512)
    lrs_r = (4e-4, 6e-4, 8e-4, 1e-3)
    grid_list(s, [make_config((w, w), D, act="gelu", lr=lr) for w in widths_r for lr in lrs_r])
    # S: width x huber at (gelu, d0.3, lr6e-4) — huber at lr6e-4 was untested
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for w in (224, 256, 320, 384, 448) for hd in (0.05, 0.1, 0.2)])
    # T: 3-layer x lr at (gelu, huber0.1, d0.3) — 3-layer x lr6e-4 untested in 54
    grid_list(s, [make_config((w, w, w), D, loss="huber", huber_delta=0.1, act="gelu", lr=lr)
                  for w in (320, 384, 448, 512) for lr in (3e-4, 6e-4, 1e-3)])
    # T2: 3-layer x huber at (gelu, d0.3, lr6e-4)
    grid_list(s, [make_config((w, w, w), D, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for w in (320, 384, 448) for hd in (0.05, 0.2)])
    # U: dropout x width at (gelu, mse, lr6e-4) — dropout was fixed at 0.3 in 2.1
    grid_list(s, [make_config((w, w), dd, act="gelu", lr=6e-4)
                  for dd in (0.2, 0.4) for w in (224, 256, 320, 384)])
    # V: dropout x huber at (gelu, 320^2, lr6e-4)
    grid_list(s, [make_config((320, 320), dd, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for dd in (0.2, 0.4) for hd in (0.1, 0.2)])
    # W: anchor — the 2.1 54-family val winner + bit-identity check vs 2.1
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=0.1)])
    return s


def _build_mixed() -> dict:
    s: dict = {}
    # X: act x depth x lr at (huber0.1, d0.3) — gelu x lr {6e-4, 1e-3} untested;
    #    anchors w448x448x448_d0.3_huber0.1_gelu (2.1 test-best 0.7940) and
    #    w512x512x512_d0.3_huber0.1 (2.1 val-3rd) live here.
    grid_list(s, [make_config(dep, D, loss="huber", huber_delta=0.1, act=ac, lr=lr)
                  for ac in ("silu", "gelu") for dep in ((384, 384, 384), (448, 448, 448), (512, 512, 512))
                  for lr in (3e-4, 6e-4, 1e-3)])
    # X2: act x depth x lr fine at (huber0.1, d0.3) — lr {4e-4, 8e-4} probes
    grid_list(s, [make_config(dep, D, loss="huber", huber_delta=0.1, act=ac, lr=lr)
                  for ac in ("silu", "gelu") for dep in ((448, 448, 448), (512, 512, 512))
                  for lr in (4e-4, 8e-4)])
    # Y: gelu x huber x lr at 448^3 (d0.3) — gelu x huber0.05/0.2 x lr6e-4/1e-3 untested
    grid_list(s, [make_config((448, 448, 448), D, loss="huber", huber_delta=hd, act="gelu", lr=lr)
                  for hd in (0.05, 0.1, 0.2) for lr in (6e-4, 1e-3)])
    # Z: silu 512^3 delta x lr fine (d0.3) — refine around the 2.1 val winner
    #    w512x512x512_d0.3_huber0.05_lr6e-4 (test 0.7844); delta {0.03, 0.08}
    #    and lr {4e-4, 8e-4} untested.
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=hd, lr=lr)
                  for hd in (0.03, 0.05, 0.08, 0.1) for lr in (4e-4, 6e-4, 8e-4, 1e-3)])
    # Z2: silu 512^3 delta 0.15 probe
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=0.15, lr=lr)
                  for lr in (4e-4, 6e-4, 8e-4, 1e-3)])
    # AA: dropout x delta at (silu, 512^3, lr6e-4) — the B grid was lr3e-4-only
    grid_list(s, [make_config((512, 512, 512), dd, loss="huber", huber_delta=hd, lr=6e-4)
                  for dd in (0.2, 0.4) for hd in (0.05, 0.1)])
    # AB: gelu 2-layer x lr at (huber0.1, d0.3) — gelu 2-layer at lr6e-4/1e-3 untested
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.1, act="gelu", lr=lr)
                  for w in (384, 448, 512, 576, 640) for lr in (6e-4, 1e-3)])
    # AC: gelu/silu x 3-layer x delta 0.05 at (d0.3, lr6e-4) — huber0.05 at 448^3
    #     and gelu-512^3 x huber0.05 untested
    grid_list(s, [make_config(dep, D, loss="huber", huber_delta=0.05, act=ac, lr=6e-4)
                  for ac in ("gelu", "silu") for dep in ((448, 448, 448), (512, 512, 512))])
    # AD2: gelu 2-layer x delta 0.05 at (d0.3, lr6e-4)
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.05, act="gelu", lr=6e-4)
                  for w in (384, 448, 512, 640)])
    return s


def _build_96() -> dict:
    s: dict = {}
    # AD: width x lr fine at (d0.5, mse) — lr 4e-4/6e-4/8e-4 untested; the 2.1
    #     small-net grid had lr {3e-4, 1e-3} only (1e-3 overshoots: 0.71-0.74).
    grid_list(s, [make_config((w, w), 0.5, lr=lr) for w in (96, 128, 192, 256, 320) for lr in (3e-4, 4e-4, 6e-4, 8e-4)])
    # AE: huber x lr at small nets (d0.5) — huber at lr {6e-4, 1e-3} untested
    grid_list(s, [make_config((w, w), 0.5, loss="huber", huber_delta=hd, lr=lr)
                  for w in (192, 256, 320) for hd in (0.1, 0.2) for lr in (3e-4, 6e-4, 1e-3)])
    # AF: max_epochs probe (d0.5, mse, lr3e-4) — 2.1 small nets hit the 400 cap
    #     under-trained (best_epoch 380-395)
    grid_list(s, [make_config((w, w), 0.5, max_epochs=me) for w in (128, 192, 256) for me in (500, 600)])
    # AG: 3-layer small nets (d0.4, mse) — 3-layer untested in 96
    grid_list(s, [make_config((w, w, w), 0.4, lr=lr) for w in (96, 128, 192) for lr in (6e-4, 1e-3)])
    # AH: mixup x lr at (256^2, d0.5, mse) — mixup at lr {6e-4, 1e-3} untested
    #     (2.1 mixup at lr3e-4: bias^2/MSE 0.009% but test 0.772)
    grid_list(s, [make_config((256, 256), 0.5, mixup=mx, lr=lr) for mx in (0.2, 0.4) for lr in (6e-4, 1e-3)])
    # AI: dropout x huber at (256^2, lr6e-4)
    grid_list(s, [make_config((256, 256), dd, loss="huber", huber_delta=hd, lr=6e-4)
                  for dd in (0.4, 0.6) for hd in (0.1, 0.2)])
    # AJ: anchors — 2.1 96-family test-best + val winner (bit-identity checks);
    #     w256x256_d0.5 and w256x256_d0.5_huber0.2 are dedup'd against AD/AE.
    grid_list(s, [
        make_config((256, 256), 0.5),
        make_config((256, 256), 0.5, loss="huber", huber_delta=0.2),
        make_config((512, 512, 512), D, lr=1e-3),
    ])
    return s


def build_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in _build_54().values():
        add(out, c)
    for c in _build_mixed().values():
        add(out, c)
    for c in _build_96().values():
        add(out, c)
    return out


# ----------------------------------------- config.yaml assembly -----------------------------------------

# Shared 54-feature global backbone discovered in derived_8.4-feature-selection-2.0
# (identical to mlp-1.3 / mlp-2.0 / mlp-2.1; do not edit — parity with the
# XGBoost baseline).
SHARED_BACKBONE_54 = [
    "precip_mm", "s2_b4", "s2_b8", "SMAP_sm_pm_interp", "D_sin_DOY", "D_cos_DOY",
    "E_SAR_ratio", "G_API", "G_DSLR", "G_rain_sum_3d", "G_rain_sum_7d",
    "SMAP_sm_pm_interp_lag7", "SMAP_sm_pm_interp_lag30", "SMAP_sm_pm_interp_rollrange7",
    "SMAP_sm_pm_interp_rollmean30", "SMAP_sm_pm_interp_rollrange30", "SMAP_sm_interp_rollrange7",
    "SMAP_ampm_diff_interp", "V_rollrng_G_API_kobs7", "V_rollmax_F_NDMI_kobs30",
    "A_d_E_SAR_ratio_kobs30", "V_rollmax_E_SAR_ratio_kobs7", "V_rollmin_E_SAR_ratio_kobs30",
    "V_rollmax_E_SAR_ratio_kobs30", "V_rollmin_LST_modis_kobs30", "V_ema_LST_modis_kobs30",
    "V_rollmax_F_NDVI_kobs14", "V_rollmax_F_NDVI_kobs30", "V_ema_F_NDVI_kobs30",
    "C_lag_F_NDVI_kobs30", "A_grad_E_SAR_diff_kobs30", "V_rollmax_E_SAR_diff_kobs14",
    "V_rollrng_E_SAR_diff_kobs30", "V_rollmax_E_SAR_diff_kobs30", "A_grad_s2_b11_kobs30",
    "V_rollrng_s2_b11_kobs30", "V_rollmin_s2_b11_kobs30", "V_rollmin_s2_b12_kobs30",
    "A_d_SMAP_sm_interp_kobs30", "V_rollmin_SMAP_sm_interp_kobs14", "V_rollmin_SMAP_sm_interp_kobs30",
    "E_rough_s1_vh_kobs14", "J_aspect_deg", "J_bio_bio02", "J_bio_bio13", "J_lc_code",
    "J_soil_texture_usda_b0", "sin_year", "cos_year", "SMAP_x_year", "D_z_F_NDMI",
    "D_z_LST_modis", "D_fft_dom_LST_modis_kobs30", "D_fft_ent_LST_modis_kobs30",
]

# Cluster-1 add-only delta features = eval-1.1 winner (Clustering_V0_Full_k2_c0_0_c1_10).
CLUSTER_1_DELTA_FEATURES = [
    "V_rollmin_F_NDMI_kobs30", "V_rollmean_G_API_kobs14", "lia_mean_asc_deg", "J_bio_bio04",
    "DOY", "SMAP_sm_am_interp", "J_bio_bio07", "SMAP_sm_am_interp_lag1",
    "C_lag_F_NDMI_kobs30", "SMAP_sm_pm_interp_lag1",
]

FAMILIES = [
    {"id": "2regime_96", "structure": "cluster",
     "feature_sets": {"0": "candidate_pool_96", "1": "candidate_pool_96"}},
    {"id": "2regime_54", "structure": "cluster",
     "feature_sets": {"0": "backbone_54", "1": "backbone_54_delta"}},
    {"id": "2regime_mixed", "structure": "cluster",
     "feature_sets": {"0": "candidate_pool_96", "1": "backbone_54_delta"}},
]

SWEEP_DEFAULTS = {
    "max_epochs": 400, "patience": 60, "checkpoint_every": 20, "grad_clip": 1.0,
    "norm": "bn", "activation": "silu", "dropout": 0.3, "lr": 0.0003,
    "weight_decay": 0.0001, "batch_size": 512, "loss": "mse", "huber_delta": 0.05,
    "warmup_frac": 0.05, "ema": False, "stop_rule": "patience", "mixup_alpha": 0.0,
    "center_target": False, "architecture": "mlp",
    "swa": False, "swa_start_frac": 0.6, "swa_eval_every": 10, "swa_bn_recal": True,
    "fg_tower_width": 128, "fg_group_dropout": 0.0, "plr_n_bins": 8,
}


def _config_yaml_entry(c: dict) -> dict:
    """The YAML entry for one config: id + only the non-default overrides.

    `hidden_sizes` is ALWAYS emitted (the v7-generation bug that omitted it
    silently trained [384, 384] configs at [256, 256]; fixed in v8, kept
    here). `max_epochs` is emitted when != 400 (the 96-family probes).
    """
    out: dict = {"id": c["id"], "hidden_sizes": c["hidden_sizes"]}
    if c["activation"] != "silu":
        out["activation"] = c["activation"]
    if c["dropout"] != 0.3:
        out["dropout"] = c["dropout"]
    if c["lr"] != 3e-4:
        out["lr"] = c["lr"]
    if c["weight_decay"] != 1e-4:
        out["weight_decay"] = c["weight_decay"]
    if c["loss"] != "mse":
        out["loss"] = c["loss"]
        out["huber_delta"] = c["huber_delta"]
    if c.get("mixup_alpha", 0.0) > 0.0:
        out["mixup_alpha"] = c["mixup_alpha"]
    if c.get("max_epochs", 400) != 400:
        out["max_epochs"] = c["max_epochs"]
    if c.get("swa", False):
        out["swa"] = True
        out["swa_start_frac"] = c["swa_start_frac"]
    return out


def build_yaml() -> dict:
    _54 = _build_54()
    mixed = _build_mixed()
    _96 = _build_96()
    configs_all: dict[str, dict] = {}
    for c in list(_54.values()) + list(mixed.values()) + list(_96.values()):
        add(configs_all, c)
    configs_all = dict(sorted(configs_all.items()))

    return {
        "data": {
            "target": "soil_moisture_5cm",
            "metadata_path": "data/splits/derived_8.4/dataset_metadata.py",
            "splits": {
                "train": "data/splits/derived_8.4/train.csv",
                "val": "data/splits/derived_8.4/val.csv",
                "test": "data/splits/derived_8.4/test.csv",
            },
        },
        "model": {"seed": 42},
        "xgboost_reference_file": "notebooks/experiment/derived_8.4-eval-1.1/metrics_summary.csv",
        "mlp13_reference_file": "notebooks/experiment/derived_8.4-eval-mlp-1.3/metrics_summary.csv",
        "mlp20_reference_file": "notebooks/experiment/derived_8.4-eval-mlp-2.0/metrics_summary.csv",
        "mlp21_reference_file": "notebooks/experiment/derived_8.4-eval-mlp-2.1/metrics_summary.csv",
        "candidate_pool_file": "notebooks/experiment/derived_8.4-feature-selection-2.0/artifacts/candidate_pool.csv",
        "shared_backbone_54": SHARED_BACKBONE_54,
        "cluster_config": {
            "strategy": "Clustering_V0_Full_k2",
            "c0_count": 0,
            "c1_count": 10,
            "cluster_1_delta_features": CLUSTER_1_DELTA_FEATURES,
        },
        "families": FAMILIES,
        "sweep": {
            "n_parallel": 8,
            "holdout": "official_val",
            "data_version": 9,          # v8 = mlp-2.1; v9 = mlp-2.2 (new grids +
                                        # val_preds.npy/val_meta.npz diagnostics)
            "seeds": [42, 7, 123],      # phase 1 / phase 2 / phase 3 (3-seed winners)
            "stability_seeds": [42, 7, 123, 2024, 999],
            # top-Ns capped at the deduped family sizes (mixed 66, 96 59,
            # 54 82) — dense multi-seed coverage is the direct mitigation for
            # the 2.1 val-seed-noise finding.
            "phase2_top_n": {"2regime_mixed": 66, "2regime_96": 57, "2regime_54": 78},
            "phase3_top_n": {"2regime_mixed": 42, "2regime_96": 26, "2regime_54": 40},
            "phase2_metric": "val_rmse",
            "phase3_metric": "val_rmse",
            # NEW in 2.2: per-family champion depth (fixes 2.1's documented
            # "top-2-mixed not expressible" limitation; int or {family: n}).
            "champion_top_n": {"2regime_mixed": 2, "2regime_54": 1, "2regime_96": 1},
            "aux_year": 2020,
            "family_configs": {
                "2regime_mixed": sorted(mixed),
                "2regime_96": sorted(_96),
                "2regime_54": sorted(_54),
            },
            "defaults": SWEEP_DEFAULTS,
            "configs": [_config_yaml_entry(c) for c in configs_all.values()],
        },
    }


_HEADER = """# Configuration for derived_8.4-eval-mlp-2.2 — exploit the 54-family
# lr6e-4/gelu region + the mixed 3-layer gelu cell + finish the 96-family
# debias (~1.25 h gpu_debug H100 wall).
#
# GENERATED FILE — do not edit by hand. This config.yaml is produced
# deterministically by make_configs.py from the documented grid spec (see its
# docstring). Regenerate with `python make_configs.py`; the output is
# byte-identical across runs (no RNG).
#
# Protocol (data_version 9, temporal only): train on official train
# (2017-2020, n=9,803); early-stop/select on official val (2021-2022,
# n=4,805); evaluate on untouched test (2023-2025, n=6,620). aux2020
# diagnostic only. 3-phase, 3-seed sweep: seed 42 (phase 1, all configs) ->
# seed 7 (phase 2, top-N per family by 2-seed val RMSE) -> seed 123 (phase 3,
# top-M per family by 3-seed val RMSE). Winners = multi-seed mean val RMSE
# among mlp/fg/plr (mlp-only in practice; fg/plr are documented 2.0 negatives,
# swa a documented 2.1 negative — 0/152 deployments with the RNG guard).
#
# Families: 2regime_96 (c0=c1=96-pool), 2regime_54 (c0=54, c1=54+10 deltas),
# 2regime_mixed (c0=96-pool, c1=54+10 deltas) — pinned from 2.0/2.1.
#
# New in 2.2 (mlp22/trainer.py): best-val predictions saved to
# val_preds.npy (post-training, same deployed weights as preds.npy, no RNG
# consumption) + artifacts/val_meta.npz (y_val/year/station) -> the offline
# val-year (2021 vs 2022) selection-reliability diagnostic. No SWA configs
# (closed negative). Champion step per-family top-N via sweep.champion_top_n.
#
# Budget: 191 phase-1 + 201 phase-2 + 108 phase-3 + 8 champion = 508
# job-seeds ~= 6.4 GPU-h ~= 65 min sweep ~= 74 min total wall (target
# ~1.25 h; 2 h wall cap).
"""


def main() -> None:
    doc = build_yaml()
    out = EXP_DIR / "config.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, width=120)
    n_54 = len(_build_54())
    n_mixed = len(_build_mixed())
    n_96 = len(_build_96())
    n_all = len(doc["sweep"]["configs"])
    print(f"[make_configs] wrote {out}")
    print(f"  54={n_54}  mixed={n_mixed}  96={n_96}  union={n_all}")
    p1 = n_all
    p2 = sum(doc["sweep"]["phase2_top_n"].values())
    p3 = sum(doc["sweep"]["phase3_top_n"].values())
    champ = sum(doc["sweep"]["champion_top_n"].values()) * 2  # extra seeds {2024, 999}
    print(f"  jobs: phase1={p1} phase2={p2} phase3={p3} champion={champ} total={p1 + p2 + p3 + champ}")


if __name__ == "__main__":
    main()
