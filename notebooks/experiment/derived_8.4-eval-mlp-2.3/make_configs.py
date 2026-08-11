#!/usr/bin/env python3
"""Deterministic generator for the derived_8.4-eval-mlp-2.3 sweep configs.

Per the reproducibility rule (AGENTS.md), the 2.3 sweep table is NOT
hand-typed: it is produced by this committed script from the documented grid
spec below and written to `config.yaml` (committed alongside the generator).
Run `python make_configs.py` after editing the grids; the output is stable
(no RNG), so `config.yaml` is reproducible byte-for-byte.

Grids (all `mlp` architecture — `fg`/`plr` are documented negatives from 2.0,
`swa` is a documented negative from 2.1 (0/152 deployments with the RNG guard
in place), 54-family 3-layer is a documented 2.2 negative (the val-overfit
trap), and 96 mid-lr/huber/mixup/max_epochs>400 are documented 2.2 negatives;
no config uses them, except the three bit-identity anchors (one per family —
the 2.2 val winners) and the two 3-layer d0.4 probes re-checking the 54
negative at the huber0.2 cell. The winner pool filter in run_mlp_sweep.py
stays mlp/fg/plr so the protocol text is unchanged):

  54   (148):  R2a/b/c/e 320^2 huber delta x lr fine x d (the frontier cell
               w320x320_d0.4_huber0.2_gelu_lr6e-4, test 0.7973) |
               R2d 320^2 mse lr x d fine | S2 width x delta x d @ lr6e-4 |
               S2b width x delta0.2 x lr {4e-4, 8e-4} |
               S2c width x mse x d0.4 x lr {4e-4, 8e-4} |
               U2 small-net mse width x lr x d (lr3e-4/5e-4 untested below
               320^2; the near-unbiased 192^2/224^2 region) |
               U2b/c small-net huber x lr {4e-4, 8e-4} |
               V2 silu probes (is the frontier gelu-specific?) |
               T2b 3-layer negative re-check at d0.4 |
               W2 anchor (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)
  mixed (46):  X3 gelu 3-layer delta x lr at {384^3, 448^3, 512^3}
               (the 2.2 test-best cell w448x448x448_d0.3_huber0.1_gelu
               0.7940; gelu x delta {0.05, 0.2} x lr {2e-4, 4e-4} untested) |
               X3b lr5e-4 probes | X3c d {0.2, 0.4} @ 512^3 |
               X3d gelu 2-layer low-lr probes (2.2 negative check) |
               Z3 silu 512^3 lr {2e-4, 3e-4} | Z3b/c silu 448^3/384^3 lr3e-4 |
               W3 anchor (w512x512x512_d0.3_huber0.03_lr1e-3)
  96   (30):   A3 width x d @ lr3e-4 (the debiased region — 2.2's mid-lr
               pool was MORE biased; w256x256_d0.5 1.1% bias is the anchor) |
               A3b 3-layer @ lr3e-4 | A3c lr2e-4 probes | A3d me600 probes |
               A3e big 2-layer @ lr3e-4 | AJ3 anchors (incl. the 2.2
               test-best w256x256_d0.5 — only 1 seed in 2.2!)

Design rationale (from the 2.2 sweep results; see the plan doc
docs/plans/20260811-mlp-2.3.md):
  - 54: the 2-layer 320^2-hubergelu/lr6e-4 cell is the series' strongest
    single MLP (test 0.7973) but sits at val rank 49/82 — invisible to the
    val selector. 2.3 refines its neighborhood (delta {0.1..0.3}, lr
    {4e-4..8e-4} + 1e-3, d {0.3, 0.4, 0.5}) and the near-unbiased small-net
    region (192^2/224^2 gelu lr4e-4/6e-4 ~0.79, bias^2/MSE share <0.1%).
    54-family 3-layer configs are NOT swept (2.2 documented the
    val-overfit trap: val top-10 dominated, test 0.7596-0.7790) — only the
    2.2 54 val winner is kept as the bit-identity anchor plus two d0.4
    probes re-confirming the negative at the huber0.2 cell.
  - mixed: the gelu 3-layer cell at lr3e-4/4e-4 is the frontier (0.7940 @
    448^3, 0.7928 @ 512^3); delta {0.05, 0.2} x lr {2e-4, 4e-4} and d
    {0.2, 0.4} are untested. 2-layer mixed at lr {6e-4, 1e-3} is a 2.2
    negative (0.71-0.76) — probed at lr3e-4/4e-4 only as a completeness
    check. silu-512^3 at lr3e-4 (the default) was never tested for
    delta {0.03, 0.08}.
  - 96: the lr3e-4 small-net pool IS the answer (w256x256_d0.5 0.7834,
    1.1% bias; w192x192_d0.5 0.7770); every mid-lr/huber/mixup/me-probe
    variant was worse and more biased (2.2 median bias^2/MSE 21.7%). The
    2.3 96 grid is lr3e-4-only by construction.

Config ids follow the 2.2 naming convention, plus `_me<epochs>` for the
max_epochs probes:
  w<W>x<W>..._d<dropout>_<loss|act>[_lr<lr>][_me<epochs>]
where the loss/act suffix is `huber<delta>` (huber) or `_<act>` (non-silu act
with mse).

Sizing rationale (from 2.2's real timings): per-seed train mean 43 s (54) /
36 s (mixed) / 52 s (96); 508 jobs in 3,774 s sweep wall (~0.135 jobs/s at
8 parallel workers / ~5.9 effective). 2.3: ~224 phase-1 configs x 3 seeds
(full 3-seed pool — the user's ask; 2.2 left most configs at 2 seeds) +
12 champion job-seeds ~= 684 jobs ~= 8.5 GPU-h ~= ~85 min sweep ~= ~1.6 h
total wall — sized to spend ~1.75 h of the 2 h gpu_debug wall allocation,
with `--resume` protecting the run.
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
    """Build one config dict with a deterministic id (2.2 naming convention)."""
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


def _fmt_mixup(a: float) -> str:
    return f"{a:.1f}"


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
    # R2a: 320^2 huber delta x lr fine x d (gelu) — the frontier cell
    #    (w320x320_d0.4_huber0.2_gelu_lr6e-4 test 0.7973; delta {0.15, 0.25,
    #    0.3} and lr {4e-4, 5e-4, 7e-4, 8e-4} were never tested at 320^2).
    grid_list(s, [make_config((320, 320), dd, loss="huber", huber_delta=hd, act="gelu", lr=lr)
                  for hd in (0.1, 0.15, 0.2, 0.25, 0.3)
                  for lr in (4e-4, 5e-4, 7e-4, 8e-4)
                  for dd in (0.3, 0.4)])
    # R2b: 320^2 delta fine at lr6e-4 (the tested lr — new deltas only)
    grid_list(s, [make_config((320, 320), dd, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for hd in (0.15, 0.25, 0.3) for dd in (0.3, 0.4)])
    # R2c: 320^2 huber x lr1e-3 (2.2 tested mse at lr1e-3 only)
    grid_list(s, [make_config((320, 320), D, loss="huber", huber_delta=hd, act="gelu", lr=1e-3)
                  for hd in (0.1, 0.2)])
    # R2d: 320^2 mse lr x d fine (d {0.35, 0.45} and lr {5e-4, 7e-4} untested)
    grid_list(s, [make_config((320, 320), dd, act="gelu", lr=lr)
                  for lr in (5e-4, 7e-4) for dd in (0.35, 0.4, 0.45)])
    # R2e: 320^2 d0.5 probes (dropout was capped at 0.4 in 2.2)
    grid_list(s, [make_config((320, 320), 0.5, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for hd in (0.1, 0.2)])
    grid_list(s, [make_config((320, 320), 0.5, act="gelu", lr=6e-4)])
    # S2: width x delta x d at lr6e-4 (gelu) — 2.2 tested delta {0.05, 0.1,
    #    0.2} x d0.3 at {224, 256, 320, 384, 448}; fresh cells only:
    #    {288, 352} (untested widths) x all d, {256, 384} x d0.4, delta0.3.
    grid_list(s, [make_config((w, w), dd, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for w in (288, 352) for hd in (0.1, 0.2, 0.3) for dd in (0.3, 0.4)])
    grid_list(s, [make_config((w, w), 0.4, loss="huber", huber_delta=hd, act="gelu", lr=6e-4)
                  for w in (256, 384) for hd in (0.1, 0.2, 0.3)])
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.3, act="gelu", lr=6e-4)
                  for w in (256, 384)])
    # S2b: width x delta0.2 x lr {4e-4, 8e-4} (huber at those lrs untested)
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.2, act="gelu", lr=lr)
                  for w in (256, 288, 352, 384) for lr in (4e-4, 8e-4)])
    # S2c: width x mse x d0.4 x lr {4e-4, 8e-4} (d0.4 at those lrs untested)
    grid_list(s, [make_config((w, w), 0.4, act="gelu", lr=lr)
                  for w in (256, 288, 352) for lr in (4e-4, 8e-4)])
    # U2: small-net gelu mse width x lr x d — the near-unbiased region
    #    (w192x192_d0.3_gelu_lr4e-4 test 0.7934, bias^2/MSE share 0.02%);
    #    lr3e-4/5e-4 were never tested below 320^2; d0.4 x lr4e-4 untested.
    grid_list(s, [make_config((w, w), 0.3, act="gelu", lr=lr)
                  for w in (128, 160) for lr in (3e-4, 4e-4, 5e-4, 6e-4)])
    grid_list(s, [make_config((w, w), 0.3, act="gelu", lr=lr)
                  for w in (192, 224, 256) for lr in (3e-4, 5e-4)])
    grid_list(s, [make_config((w, w), 0.4, act="gelu", lr=lr)
                  for w in (128, 160) for lr in (3e-4, 4e-4, 5e-4, 6e-4)])
    grid_list(s, [make_config((w, w), 0.4, act="gelu", lr=lr)
                  for w in (192, 224, 256) for lr in (3e-4, 4e-4, 5e-4)])
    # U2b: small-net huber delta0.2 x lr {4e-4, 8e-4} x d (2.2 tested lr6e-4)
    grid_list(s, [make_config((w, w), dd, loss="huber", huber_delta=0.2, act="gelu", lr=lr)
                  for w in (192, 224, 256) for lr in (4e-4, 8e-4) for dd in (0.3, 0.4)])
    # U2c: small-net huber delta0.1 x lr {4e-4, 8e-4} (d0.3)
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.1, act="gelu", lr=lr)
                  for w in (192, 224, 256) for lr in (4e-4, 8e-4)])
    # V2: silu probes — is the gelu frontier architecture-specific?
    grid_list(s, [make_config((w, w), D, act="silu", lr=6e-4) for w in (256, 320)])
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.2, act="silu", lr=6e-4)
                  for w in (256, 320)])
    grid_list(s, [make_config((320, 320), D, act="silu", lr=4e-4)])
    # T2b: 3-layer negative re-check at d0.4 (2.2 tested d0.3 only) — the
    #      54 3-layer cell is a documented 2.2 negative (val-overfit trap).
    grid_list(s, [make_config((w, w, w), 0.4, loss="huber", huber_delta=0.2, act="gelu", lr=6e-4)
                  for w in (320, 384)])
    # W2: anchor — the 2.2 54-family val winner (bit-identity check vs 2.2)
    grid_list(s, [make_config((448, 448, 448), D, loss="huber", huber_delta=0.1, act="gelu", lr=1e-3)])
    return s


def _build_mixed() -> dict:
    s: dict = {}
    # X3: gelu 3-layer delta x lr at {384^3, 448^3, 512^3} (d0.3) — the 2.2
    #     test-best cell w448x448x448_d0.3_huber0.1_gelu (0.7940) + 512^3
    #     (0.7928); delta {0.05, 0.2} x lr {2e-4, 4e-4} untested (2.2 tested
    #     delta0.1 x lr {3e-4, 4e-4} and delta0.05 x lr {6e-4, 1e-3}).
    grid_list(s, [make_config(dep, D, loss="huber", huber_delta=hd, act="gelu", lr=lr)
                  for dep in ((384, 384, 384), (448, 448, 448), (512, 512, 512))
                  for hd in (0.05, 0.1, 0.2) for lr in (2e-4, 3e-4, 4e-4)
                  if not (hd == 0.1 and lr in (3e-4, 4e-4))])
    # X3b: gelu 3-layer lr5e-4 probes at {448^3, 512^3}
    grid_list(s, [make_config(dep, D, loss="huber", huber_delta=hd, act="gelu", lr=5e-4)
                  for dep in ((448, 448, 448), (512, 512, 512)) for hd in (0.05, 0.1)])
    # X3c: gelu 3-layer dropout probes at 512^3 lr3e-4 (2.2's AA grid was silu)
    grid_list(s, [make_config((512, 512, 512), dd, loss="huber", huber_delta=hd, act="gelu", lr=3e-4)
                  for dd in (0.2, 0.4) for hd in (0.05, 0.1)])
    # X3d: gelu 2-layer low-lr probes — the 2.2 negative check (2-layer mixed
    #      at lr {6e-4, 1e-3}: 0.71-0.76); is it lr-dependent?
    grid_list(s, [make_config((w, w), D, loss="huber", huber_delta=0.1, act="gelu", lr=lr)
                  for w in (512, 640) for lr in (3e-4, 4e-4)])
    # Z3: silu 512^3 delta x lr {2e-4, 3e-4} (2.2's Z grid started at lr4e-4)
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=hd, lr=lr)
                  for hd in (0.03, 0.08) for lr in (2e-4, 3e-4)])
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=hd, lr=2e-4)
                  for hd in (0.05, 0.1)])
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=0.05, lr=5e-4)])
    # Z3b: silu 448^3 delta probes at lr3e-4 (2.2 tested delta0.1 only)
    grid_list(s, [make_config((448, 448, 448), D, loss="huber", huber_delta=hd, lr=3e-4)
                  for hd in (0.03, 0.05, 0.08)])
    # Z3c: silu 384^3 delta probes at lr3e-4
    grid_list(s, [make_config((384, 384, 384), D, loss="huber", huber_delta=hd, lr=3e-4)
                  for hd in (0.05, 0.2)])
    # W3: anchor — the 2.2 mixed-family val winner (bit-identity check vs 2.2)
    grid_list(s, [make_config((512, 512, 512), D, loss="huber", huber_delta=0.03, lr=1e-3)])
    return s


def _build_96() -> dict:
    s: dict = {}
    # A3: width x d at lr3e-4 (mse) — the ONLY debiased region per 2.2 (2.2
    #     tested d0.5 at {96, 128, 192, 256, 320}; d0.4/d0.6 and widths
    #     {160, 224, 288} untested; mid-lr {4e-4..8e-4} is a 2.2 negative).
    grid_list(s, [make_config((w, w), dd, lr=3e-4)
                  for w in (96, 128, 160, 192, 224, 256, 288, 320) for dd in (0.4, 0.6)])
    grid_list(s, [make_config((w, w), 0.5, lr=3e-4) for w in (160, 224, 288)])
    # A3b: 3-layer at lr3e-4 (2.2's AG grid was lr {6e-4, 1e-3} only)
    grid_list(s, [make_config((w, w, w), 0.4, lr=3e-4)
                  for w in (96, 128, 192, 256)])
    # A3c: lr2e-4 probes (slower-lr convergence; best_epoch 380-391 suggested
    #      the 400-epoch cap binds at lr3e-4)
    grid_list(s, [make_config((w, w), 0.5, lr=2e-4) for w in (192, 256)])
    # A3d: me600 probes at the cheapest sizes (2.2 tested {128, 192, 256}^2;
    #      the probes HURT at 256^2 — re-check the smallest net only)
    grid_list(s, [make_config((96, 96), 0.5, max_epochs=me) for me in (500, 600)])
    # A3e: big 2-layer at lr3e-4 (2.2's AD grid capped at 320^2)
    grid_list(s, [make_config((w, w), 0.5, lr=3e-4) for w in (352, 384)])
    # AJ3: anchors — 2.2 96-family test-best (w256x256_d0.5, only 1 seed in
    #      2.2!), test-2nd (w192x192_d0.5), and the val winner
    #      (w512x512x512_d0.3_lr1e-3, bit-identity check vs 2.2).
    grid_list(s, [make_config((256, 256), 0.5)])  # dedup'd against A3
    grid_list(s, [make_config((192, 192), 0.5)])  # dedup'd against A3
    grid_list(s, [make_config((512, 512, 512), D, lr=1e-3)])
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
# (identical to mlp-1.3 / mlp-2.0 / mlp-2.1 / mlp-2.2; do not edit — parity with the
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
        "mlp22_reference_file": "notebooks/experiment/derived_8.4-eval-mlp-2.2/metrics_summary.csv",
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
            "data_version": 10,         # v9 = mlp-2.2; v10 = mlp-2.3 (new
                                        # grids + FULL 3-seed pool; the trainer
                                        # is byte-identical to v9, so anchors'
                                        # val curves stay bit-identical)
            "seeds": [42, 7, 123],      # phase 1 / phase 2 / phase 3
            "stability_seeds": [42, 7, 123, 2024, 999],
            # FULL 3-seed pool (NEW in 2.3): phase-2/3 top-Ns are capped at
            # the deduped family sizes, so EVERY config trains seeds
            # {42, 7, 123} — the 3-seed mean val RMSE becomes the honest
            # signal for the entire pool (2.2 gave only the top-M 3 seeds,
            # which limited the val-year/selection diagnostics to a subset
            # and left the 54 winner flip under-documented). Sizes are taken
            # from the built grids so the caps always match the pool.
            "phase2_top_n": {fid: len(fam) for fid, fam in
                             (("2regime_mixed", mixed), ("2regime_96", _96), ("2regime_54", _54))},
            "phase3_top_n": {fid: len(fam) for fid, fam in
                             (("2regime_mixed", mixed), ("2regime_96", _96), ("2regime_54", _54))},
            "phase2_metric": "val_rmse",
            "phase3_metric": "val_rmse",
            # NEW in 2.3: the 54 family gets the deepest champion hedge
            # (top-3) because its val noise is the documented problem.
            "champion_top_n": {"2regime_mixed": 2, "2regime_54": 3, "2regime_96": 1},
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


_HEADER = """# Configuration for derived_8.4-eval-mlp-2.3 — final frontier check for the
# mlp-2.0 architecture: 320^2-hubergelu/lr6e-4 cell refinement + mixed gelu
# 3-layer low-lr + the 96 lr3e-4 debiased pool (~1.75 h gpu_debug H100 wall).
#
# GENERATED FILE — do not edit by hand. This config.yaml is produced
# deterministically by make_configs.py from the documented grid spec (see its
# docstring). Regenerate with `python make_configs.py`; the output is
# byte-identical across runs (no RNG).
#
# Protocol (data_version 10, temporal only): train on official train
# (2017-2020, n=9,803); early-stop/select on official val (2021-2022,
# n=4,805); evaluate on untouched test (2023-2025, n=6,620). aux2020
# diagnostic only. 3-phase, 3-seed sweep: seed 42 (phase 1, all configs) ->
# seed 7 (phase 2, ALL configs — full 3-seed pool) -> seed 123 (phase 3,
# ALL configs). Winners = 3-seed mean val RMSE among mlp/fg/plr (mlp-only in
# practice; fg/plr are documented 2.0 negatives, swa a documented 2.1
# negative, 54 3-layer a documented 2.2 negative — only the 3 anchors and 2
# re-check probes use it).
#
# Families: 2regime_96 (c0=c1=96-pool), 2regime_54 (c0=54, c1=54+10 deltas),
# 2regime_mixed (c0=96-pool, c1=54+10 deltas) — pinned from 2.0/2.1/2.2.
#
# New in 2.3: (a) FULL 3-seed coverage (phase-2/3 top-N = family sizes) —
# the direct mitigation for 2.2's finding that 3-seed means existed only for
# the phase-3 subset; (b) grids around the 2.2 frontiers (54 320^2-hubergelu/
# lr6e-4 cell + small-net region, mixed gelu 3-layer at low lr, 96 lr3e-4
# pool); (c) champion_top_n {mixed: 2, 54: 3, 96: 1} (the 54 hedge). The
# mlp23 trainer is byte-identical to mlp22 (no training-path change) —
# anchors reproduce 2.2 bit-identically (stack check after the sweep).
#
# Budget: ~224 phase-1 x 3 seeds + 12 champion = ~684 job-seeds ~= 8.5 GPU-h
# ~= ~85 min sweep ~= ~1.6 h total wall (target ~1.75 h; 2 h wall cap).
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
