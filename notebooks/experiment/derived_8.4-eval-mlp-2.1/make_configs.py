#!/usr/bin/env python3
"""Deterministic generator for the derived_8.4-eval-mlp-2.1 sweep configs.

Per the reproducibility rule (AGENTS.md), the 178-config sweep table is NOT
hand-typed: it is produced by this committed script from the documented grid
spec below and written to `config.yaml` (committed alongside the generator).
Run `python make_configs.py` after editing the grids; the output is stable
(no RNG), so `config.yaml` is reproducible byte-for-byte.

Grids (all `mlp` architecture — `fg`/`plr` are documented negatives from 2.0
and get no GPU; the winner pool filter in run_mlp_sweep.py stays mlp/fg/plr
so the protocol text is unchanged):

  mixed (100): A shape x lr | B loss x dropout | C wd x mixup |
               D act x depth | E lr x huber | F SWA start-frac x configs |
               G v7 anchors
  96   (46):  H width x dropout x lr | I huber at small nets |
               J width x mixup | K SWA start-frac x configs | L v7 anchors
  54   (32):  M width x lr | N loss x act | O width x huber |
               P SWA start-frac x configs | Q mixup + anchors

Config ids follow the 2.0 naming convention:
  w<W>x<W>..._d<dropout>_<loss|act>[_lr<lr>][_wd<wd>][_mixup<a>][_swa<frac>]
where the loss/act suffix is `huber<delta>` (huber) or `_<act>` (non-silu act
with mse); `_swa` = swa_start_frac 0.6 (2.0 value, kept for the 96 anchor),
`_swa075` etc. = explicit start frac x 100.

Sizing rationale (from 2.0 timing): MLP-only per-seed train time mean 63 s;
at 8 parallel workers / 76 % utilization, 178 phase-1 + 110 phase-2 + 44
phase-3 + 6 champion job-seeds = 338 jobs ~= 6.2 GPU-h ~= 55-60 min wall —
the sweep is sized to spend ~1 h of the 2 h H100 wall allocation.
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


def _fmt_swa(start: float) -> str:
    # 0.6 -> "swa" (2.0 naming); 0.75 -> "swa075" (start frac x 100, no dot);
    # the caller joins id parts with "_", so no leading underscore here.
    if abs(start - 0.6) < 1e-9:
        return "swa"
    return f"swa{int(round(start * 100)):03d}"


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
) -> dict:
    """Build one config dict with a deterministic id (2.0 naming convention)."""
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
        id_parts.append(_fmt_swa(swa_start))
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


def _build_mixed() -> dict:
    s: dict = {}
    shapes = [(256, 256), (320, 320), (384, 384), (448, 448), (512, 512),
              (576, 576), (640, 640), (768, 768),
              (384, 384, 384), (448, 448, 448), (512, 512, 512)]
    lrs = [1e-4, 3e-4, 6e-4, 1e-3]
    # A: shape x lr at (huber0.1, d0.3, wd1e-4)
    grid_list(s, [make_config(w, D, loss="huber", huber_delta=0.1, lr=lr) for w in shapes for lr in lrs])
    # B: loss x dropout at (512^3, lr3e-4, wd1e-4)
    losses = [("mse", 0.05), ("huber", 0.05), ("huber", 0.1), ("huber", 0.2)]
    grid_list(s, [make_config((512, 512, 512), dd, loss=lo, huber_delta=hd) for lo, hd in losses for dd in (0.2, 0.3, 0.4)])
    # C: wd x mixup at (512^3, huber0.1, lr3e-4, d0.3)
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=0.1, wd=wd, mixup=mx)
                  for wd in (1e-4, 1e-3) for mx in (0.0, 0.2, 0.4)])
    # D: act x depth at (448, d0.3, lr3e-4, huber0.1)
    grid_list(s, [make_config(dep, D, loss="huber", huber_delta=0.1, act=ac) for ac in ("gelu", "silu", "relu") for dep in ((448, 448), (448, 448, 448))])
    # E: lr x huber at (512^3, d0.3, wd1e-4)
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=hd, lr=lr) for lr in lrs for hd in (0.05, 0.1, 0.2)])
    # F: SWA start-frac x configs (ids rebuilt by make_config with swa_start)
    swa_bases = [
        make_config((512, 512, 512), D, loss="huber", huber_delta=0.1),
        make_config((512, 512, 512), D, loss="huber", huber_delta=0.2),
        make_config((448, 448), D, loss="huber", huber_delta=0.1),
        make_config((384, 384), D, act="gelu"),
        make_config((448, 448), D, act="gelu"),
        make_config((640, 640), D, loss="huber", huber_delta=0.1),
        make_config((256, 256), 0.4),
    ]
    swa_cfgs = []
    for base in swa_bases:
        for start in (0.7, 0.75, 0.8, 0.85):
            swa_cfgs.append(make_config(
                tuple(base["hidden_sizes"]), base["dropout"],
                loss=base["loss"], huber_delta=base.get("huber_delta", 0.05),
                act=base["activation"], lr=base["lr"], wd=base["weight_decay"],
                mixup=base.get("mixup_alpha", 0.0), swa_start=start,
            ))
    grid_list(s, swa_cfgs)
    # G: v7 anchors (stack check; the two gelu anchors are not in grid A which is huber-only)
    grid_list(s, [
        make_config((384, 384), D, act="gelu"),
        make_config((448, 448), D, act="gelu"),
    ])
    return s


def _build_96() -> dict:
    s: dict = {}
    widths = [(96, 96), (128, 128), (192, 192), (256, 256)]
    # H: width x dropout x lr at (mse, wd1e-4)
    grid_list(s, [make_config(w, dd, lr=lr) for w in widths for dd in (0.4, 0.5, 0.6) for lr in (3e-4, 1e-3)])
    # I: huber at small nets (d0.5, lr3e-4)
    grid_list(s, [make_config(w, 0.5, loss="huber", huber_delta=hd) for w in widths[1:] for hd in (0.1, 0.2)])
    # J: width x mixup (d0.5, lr3e-4, mse)
    grid_list(s, [make_config(w, 0.5, mixup=mx) for w in widths[1:] for mx in (0.2, 0.4)])
    # K: SWA start-frac x configs
    swa_bases = [
        make_config((256, 256), 0.5),
        make_config((192, 192), 0.5),
        make_config((256, 256), 0.5, lr=1e-3),
        make_config((128, 128), 0.5),
    ]
    for base in swa_bases:
        for start in (0.75, 0.85):
            c2 = dict(base)
            c2["swa"] = True
            c2["swa_start_frac"] = float(start)
            c2 = make_config(
                tuple(c2["hidden_sizes"]), c2["dropout"],
                loss=c2["loss"], huber_delta=c2.get("huber_delta", 0.05),
                act=c2["activation"], lr=c2["lr"], wd=c2["weight_decay"],
                mixup=c2.get("mixup_alpha", 0.0), swa_start=c2["swa_start_frac"],
            )
            add(s, c2)
    # L: v7 anchors (w256x256_d0.5 is in grid H; add the two big-net anchors)
    grid_list(s, [
        make_config((512, 512, 512), D, lr=1e-3),
        make_config((512, 512, 512), D, loss="huber", huber_delta=0.1, swa_start=0.6),
    ])
    return s


def _build_54() -> dict:
    s: dict = {}
    widths = [(320, 320), (384, 384), (448, 448), (512, 512), (640, 640)]
    # M: width x lr at (gelu, d0.3, mse)
    grid_list(s, [make_config(w, D, act="gelu", lr=lr) for w in widths for lr in (1e-4, 3e-4, 6e-4)])
    # N: loss x act at (448^2, lr3e-4, d0.3)
    grid_list(s, [make_config((448, 448), D, loss=lo, huber_delta=hd, act=ac)
                  for lo, hd in (("mse", 0.05), ("huber", 0.1), ("huber", 0.2)) for ac in ("gelu", "silu")])
    # O: width x huber at (gelu, d0.3, lr3e-4)
    grid_list(s, [make_config(w, D, loss="huber", huber_delta=hd, act="gelu")
                  for w in ((384, 384), (448, 448), (512, 512)) for hd in (0.1, 0.2)])
    # P: SWA start-frac x configs
    swa_bases = [
        make_config((512, 512, 512), D, loss="huber", huber_delta=0.1),
        make_config((448, 448), D, act="gelu"),
        make_config((384, 384), D, act="gelu"),
    ]
    for base in swa_bases:
        for start in (0.75, 0.85):
            c2 = dict(base)
            c2["swa"] = True
            c2["swa_start_frac"] = float(start)
            c2 = make_config(
                tuple(c2["hidden_sizes"]), c2["dropout"],
                loss=c2["loss"], huber_delta=c2.get("huber_delta", 0.05),
                act=c2["activation"], lr=c2["lr"], wd=c2["weight_decay"],
                mixup=c2.get("mixup_alpha", 0.0), swa_start=c2["swa_start_frac"],
            )
            add(s, c2)
    # Q: mixup + v7 anchors (the two gelu anchors and the huber anchor)
    grid_list(s, [
        make_config((448, 448), D, act="gelu", mixup=0.2),
        make_config((512, 512, 512), D, loss="huber", huber_delta=0.1),
    ])
    return s


def build_all() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in _build_mixed().values():
        add(out, c)
    for c in _build_96().values():
        add(out, c)
    for c in _build_54().values():
        add(out, c)
    return out


# ----------------------------------------- config.yaml assembly -----------------------------------------

# Shared 54-feature global backbone discovered in derived_8.4-feature-selection-2.0
# (identical to mlp-1.3 / mlp-2.0; do not edit — parity with the XGBoost baseline).
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
    """The YAML entry for one config: id + only the non-default overrides (2.0 style).

    `hidden_sizes` is ALWAYS emitted: the sweep defaults have NO hidden_sizes
    key and build_model falls back to [256, 256] when it is absent — omitting
    it (as a v7-generation bug did) silently trained every [384, 384] config
    at [256, 256]. Fixed in data_version 8.
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
    if c.get("swa", False):
        out["swa"] = True
        out["swa_start_frac"] = c["swa_start_frac"]
    return out


def build_yaml() -> dict:
    mixed = _build_mixed()
    _96 = _build_96()
    _54 = _build_54()
    configs_all: dict[str, dict] = {}
    for c in list(mixed.values()) + list(_96.values()) + list(_54.values()):
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
            "data_version": 8,          # v7 = mlp-2.1 first GPU run: BUMPED because the
                                        # v7 generator omitted hidden_sizes for [384,384]
                                        # configs (build_model fell back to [256,256]); the
                                        # fix always emits hidden_sizes. v8 invalidates the
                                        # v7 artifacts so every config re-trains correctly.
            "seeds": [42, 7, 123],      # phase 1 / phase 2 / phase 3 (3-seed winners)
            "stability_seeds": [42, 7, 123, 2024, 999],
            "phase2_top_n": {"2regime_mixed": 60, "2regime_96": 30, "2regime_54": 20},
            "phase3_top_n": {"2regime_mixed": 24, "2regime_96": 12, "2regime_54": 8},
            "phase2_metric": "val_rmse",
            "phase3_metric": "val_rmse",
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


_HEADER = """# Configuration for derived_8.4-eval-mlp-2.1 — optimize the mixed family +
# fair SWA re-test + finish the 96-family debias (~1 h H100 wall).
#
# GENERATED FILE — do not edit by hand. This config.yaml is produced
# deterministically by make_configs.py from the documented grid spec (see its
# docstring). Regenerate with `python make_configs.py`; the output is
# byte-identical across runs (no RNG).
#
# Protocol (data_version 8, temporal only): train on official train
# (2017-2020, n=9,803); early-stop/select on official val (2021-2022,
# n=4,805); evaluate on untouched test (2023-2025, n=6,620). aux2020
# diagnostic only. data_version 8 = the fix for a v7 generator bug that
# omitted hidden_sizes on [384,384] configs (silently trained at [256,256]).
# 3-phase, 3-seed sweep: seed 42 (phase 1, all configs) ->
# seed 7 (phase 2, top-N per family by 2-seed val RMSE) -> seed 123 (phase 3,
# top-M per family by 3-seed val RMSE). Winners = multi-seed mean val RMSE
# among mlp/fg/plr (mlp-only in practice; fg/plr are documented 2.0 negatives).
#
# Families: 2regime_96 (c0=c1=96-pool), 2regime_54 (c0=54, c1=54+10 deltas),
# 2regime_mixed (c0=96-pool, c1=54+10 deltas) — the per-cluster-optimal
# allocation that broke the ceiling in 2.0 (val top-5 ensemble 0.8003).
#
# New in 2.1 (mlp21/trainer.py): SWA re-test with the two 2.0-prescribed
# fixes — swa_start_frac swept {0.7, 0.75, 0.8, 0.85} and an RNG guard around
# the SWA snapshot evaluation (live trajectory bit-identical to the anchor).
#
# Budget: 178 phase-1 + 110 phase-2 + 44 phase-3 + 6 champion = 338 job-seeds
# ~= 6.2 GPU-h ~= 55-60 min wall at 8 parallel H100 workers (2 h wall cap).
"""


def main() -> None:
    doc = build_yaml()
    out = EXP_DIR / "config.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False, width=120)
    n_mixed = len(_build_mixed())
    n_96 = len(_build_96())
    n_54 = len(_build_54())
    n_all = len(doc["sweep"]["configs"])
    print(f"[make_configs] wrote {out}")
    print(f"  mixed={n_mixed}  96={n_96}  54={n_54}  union={n_all}")
    p1 = n_mixed + n_96 + n_54
    p2 = sum(doc["sweep"]["phase2_top_n"].values())
    p3 = sum(doc["sweep"]["phase3_top_n"].values())
    champ = 3 * 2  # top-1 per family (run_slurm.sh --top-n 1) x extra seeds {2024, 999}
    print(f"  jobs: phase1={p1} phase2={p2} phase3={p3} champion={champ} total={p1 + p2 + p3 + champ}")


if __name__ == "__main__":
    main()
