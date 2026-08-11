# Experiment: `derived_8.4-eval-mlp-2.2` — exploit the 54-family lr6e-4/gelu region + the mixed 3-layer gelu cell + finish the 96-family debias (~1.25 h gpu_debug H100 wall)

## Objective

Follow-up to `derived_8.4-eval-mlp-2.1` (an honest negative: the mixed-family
3-seed val winner `w512x512x512_d0.3_huber0.05_lr6e-4` → test R² 0.7844 did
not beat 2.0's 2-seed honest single 0.7903 / val top-5 ensemble 0.8003; SWA
closed with proof — 0/152 deployments under the RNG guard; val selection
shown to be the bottleneck, Spearman(val, test) −0.555 (54) / −0.309 (mixed)
even at 3 seeds). 2.2 is an **optimization + further parameter sweep** of the
2.1 winners' neighborhoods, temporal protocol only (no LOSO, same honest
protocol as 2.0/2.1), **sized to spend ~1.25 h of the 2 h `gpu_debug` H100
wall allocation** (508 job-seeds at 8 workers ≈ 6.4 GPU-h; the allocation is
otherwise wasted).

All numbers below are the stdout of the executed report notebook
(`derived_8.4-eval-mlp-2.2.ipynb`). Weights/checkpoints/test predictions under
`models/`; preprocessed tensors and per-job logs under `artifacts/`; figures at the
experiment root.

## Verdict (TL;DR)

- **The val-selected winners are again below 2.0's 0.8003 — an honest
  negative — but the sweep found the strongest single MLP of the whole
  1.0–2.2 series on test.** Honest 3-seed val winners: mixed
  `w512x512x512_d0.3_huber0.03_lr1e-3` → test R² 0.7809 (below 2.1's
  0.7844), 54 `w448x448x448_d0.3_huber0.1_gelu_lr1e-3` → 0.7596 (below
  2.1's 0.7713), 96 unchanged 0.7595; 2.2's mixed val top-5 ensemble 0.7850
  and cross-family ensemble 0.7885 both sit below 2.0's 0.8003 / 0.7932.
  The **test-best single is the 54-family `w320x320_d0.4_huber0.2_gelu_lr6e-4`
  → 0.7973** (3 seeds, val rank 49/82!), with `w320x320_d0.4_gelu_lr6e-4`
  0.7960 and mixed `w512x512x512_d0.3_huber0.1_gelu` 0.7928 (the untested
  gelu-512³ cell) close behind — the 320²-hubergelu/lr6e-4 cell is the new
  frontier, and it is invisible to the val selector.
- **Val-year diagnostic (NEW, headline finding): the official val split's
  2022 half is the noise source for the 54/96 families.** Spearman(val-2021,
  test) = +0.747 (96, p=1e-11) / +0.454 (54) while val-2022 = +0.106 /
  +0.133 (both ns) — the full-val mean dilutes the reliable 2021 signal;
  selecting on val-2021 only would pick the 54 `w320x320_d0.2_huber0.1_gelu_lr6e-4`
  (test 0.7810) over the full-val winner (0.7596). The mixed family is
  different: BOTH years are negatively correlated with test (−0.249 /
  −0.352) — its val-noise is structural (the c0 = 96-pool half), not a year
  artifact. Diagnostic only; the deployed rule is unchanged.
- **54-family selection signal flipped positive: +0.582 at 3 seeds (2.1:
  −0.555)** — the denser seed coverage and the new pool fixed the direction,
  yet the 2→3-seed flip still moved the 54 winner to a worse-test config
  (0.7772 → 0.7596); the 54 val top-10 is dominated by 3-layer configs that
  fail on test (the 3-layer-val-overfit pattern).
- **Debias: 54 met (median 3.1 %), mixed improved (12.7 → 9.6 %) but still
  >5 %; 96 worsened (13.9 → 21.7 %).** The mid-lr {4e-4, 6e-4, 8e-4}
  small-net configs are more biased than the lr3e-4 anchor (w256x256_d0.5:
  1.1 %), and the max_epochs {500, 600} probes did not help (me500 → 0.7682
  vs the 400-cap 0.7834). The 96 criterion remains unmet.
- **Budget:** sweep 3,774.5 s (62.9 min) + champion/eval/analyses ≈
  1:06:53 total wall on the `gpu_debug` H100 (6.1 GPU-h of training) —
  inside the ~1.25 h target; 3/3 anchors bit-identical vs 2.1 on a different
  node (max|diff| = 0).

## What's new in 2.2

1. **The 54-family lr6e-4 × gelu region (headline lever)** — 2.1's test-best
   was `w320x320_d0.3_gelu_lr6e-4` (0.7935) and the gelu/lr6e-4 combination
   won at every width tested; 2.2 sweeps the untested cells: widths below 320,
   lr {4e-4, 8e-4}, huber δ × lr6e-4, 3-layer × lr6e-4, dropout × width.
2. **The mixed 3-layer gelu cell** — 2.1's mixed test-best was
   `w448x448x448_d0.3_huber0.1_gelu` (0.7940, val rank 34!); gelu was only
   ever tested at lr3e-4, so 2.2 grids act × depth × lr and refines the
   silu-512³ huber-δ × lr surface (δ {0.03, 0.08, 0.15}, lr {4e-4, 8e-4}).
3. **96-family debias + small-net convergence** — small nets hit the
   400-epoch cap under-trained (best_epoch 380–395); 2.2 adds lr {4e-4, 6e-4,
   8e-4} (lr6e-4 never tested for 96), huber × lr, mixup × lr, 3-layer small
   nets, and max_epochs {500, 600} probes. Criterion: median bias²/MSE < 5 %
   (2.1: 13.9 %).
4. **Val-year diagnostic (NEW)** — every job now saves best-val predictions
   (`val_preds.npy`) plus `artifacts/val_meta.npz`; `analyze_val_years.py`
   computes per-config val-2021 vs val-2022 RMSE, per-year Spearman vs test,
   and winner stability under val-year-drop. Diagnostic only — the selection
   rule stays 3-seed mean val RMSE (protocol unchanged).
5. **Densest seed coverage yet** — phase-2/3 top-Ns are capped at the family
   sizes (66/57/82 phase-2; 42/26/40 phase-3), the direct mitigation for
   2.1's val-seed-noise finding; the champion step gets per-family top-N
   (`sweep.champion_top_n`: mixed top-2 + 54 top-1 + 96 top-1), fixing 2.1's
   documented "top-2-mixed not expressible" limitation.
6. **No SWA re-spend** — SWA is a closed negative (0/152 deployments, RNG
   guard proof); no SWA configs run in 2.2. `fg`/`plr` stay closed negatives.

Documented negatives honored (no GPU re-spent): no calibration, no trainval
retrain, patience-60 kept, aux2020 diagnostic-only, batch 512, no new
routers / station embeddings / feature selection, lr1e-4 dropped (2.1
negative), mixup/wd1e-3 at mixed-512³ dropped (2.1 negative).

## Protocol (data_version 9, temporal only — same honest protocol as 2.1)

Train on the official train split (2017–2020, n=9,803); early-stop / select on the
official val split (2021–2022, n=4,805); evaluate on the untouched test split
(2023–2025, n=6,620). aux2020 (2020 slice of train, n=2,519) diagnostic only.
Winners selected by **3-seed mean val RMSE** among mlp/fg/plr (phase 1 = seed 42 for
all configs; phase 2 = seed 7 for the top-N per family; phase 3 = seed 123 for the
top-M per family). Patience-60; AdamW + warmup 5% + cosine; grad clip 1.0;
median-impute → StandardScaler → clip [−5, 5] fit on train only; target in original
units; `cudnn.deterministic=True`.

**data_version 9 (v8 → v9):** new sweep grids (section below), the trainer now
saves best-val predictions (`val_preds.npy`, post-training eval-mode forward —
the training path is byte-identical to v8, so anchors' val curves stay
bit-identical), and `build_all_tensors` saves `artifacts/val_meta.npz` for the
val-year diagnostic.

**Cross-node bit-identity caveat:** v8 (2.1) reproduced v6 (2.0)'s anchor curve
bit-identically on a different node (offline comparison, max|diff| = 0); 2.2
re-checks the same way against 2.1 (`compare_anchor_vs_2.1.py` →
`artifacts/anchor_vs_21_comparison.json`). General cross-node bit-identity is
still not guaranteed (PTX-JIT/driver/cuDNN), but the observed reproductions
have been exact.

## Sweep design

191 phase-1 configs (all `mlp`), generated by `make_configs.py` from the
documented grids below; `config.yaml` is the committed output. See
`make_configs.py` for the full spec and the per-family id lists in
`config.yaml`.

| family | n phase-1 | grids (axes) | phase-2 top-N | phase-3 top-N |
|---|---:|---|---:|---:|
| `2regime_54` (lr6e-4/gelu region) | 82 | width × lr fine; width × huber @ lr6e-4; 3-layer × lr; 3-layer × huber @ lr6e-4; dropout × width; dropout × huber; anchor | 78 | 40 |
| `2regime_mixed` (3-layer gelu cell) | 66 | act × depth × lr; act × depth × lr fine; gelu × huber × lr @ 448³; silu-512³ δ × lr fine; δ 0.15 probe; dropout × δ; gelu 2-layer × lr; gelu/silu × 3-layer × δ 0.05; gelu 2-layer × δ 0.05 | 66 | 42 |
| `2regime_96` (debias) | 59 | width × lr fine; huber × lr; max_epochs {500, 600} probe; 3-layer small nets; mixup × lr; dropout × huber; anchors | 57 | 26 |

Job count: 191 phase-1 + 201 phase-2 + 108 phase-3 + 8 champion = **508 job-seeds**.
Budget math (from 2.1 timing): per-seed mean 45 s; at 8 workers / ~5.9
effective ≈ 6.4 GPU-h ≈ **~65 min sweep ≈ ~74 min total wall**. Resumable;
`--phase2-top-n` / `--phase3-top-n` / `--families` / `--only` trims keep the
session inside the 2 h wall cap.

## Selection Protocol v9 Diagnostic

Selection = multi-seed mean val RMSE among the honest architectures (mlp / fg / plr — 2.2 runs only mlp configs, so the pool is mlp-only in practice). 2.2 keeps the 3-phase, 3-seed sweep {42, 7, 123} from 2.1 with the **densest coverage yet** (phase-2/3 top-Ns capped at the family sizes) because 2.1 documented that the mixed/54 families' val ranking is noisy even at 3 seeds (Spearman(val, test) = -0.309 / -0.555). aux2020 stays diagnostic-only (measures train fit). This section reports the val ranking, the Spearman correlations vs test at 1-/2-/3-seed aggregation, and the phase-stability table from `analyze_selection.py`.

### Selection Protocol v9 Diagnostic (selection = multi-seed mean val RMSE; mlp/fg/plr pool)

#### 2-Regime-96 — top-10 by val RMSE
| config_id                     | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w512x512x512_d0.3_lr1e-3      | mlp            |         3 |  0.0484652 |  0.0260292 |  0.759483 |   0.0499584 |   0.0188537 |
| w320x320_d0.5_huber0.1_lr1e-3 | mlp            |         3 |  0.0493431 |  0.0300213 |  0.729662 |   0.0529651 |   0.0280154 |
| w320x320_d0.5_huber0.2_lr1e-3 | mlp            |         3 |  0.0497279 |  0.031144  |  0.716225 |   0.0542654 |   0.0302069 |
| w320x320_d0.5_huber0.1_lr6e-4 | mlp            |         3 |  0.0497972 |  0.032205  |  0.724091 |   0.053508  |   0.0286928 |
| w320x320_d0.5_huber0.2_lr6e-4 | mlp            |         3 |  0.0504033 |  0.0321439 |  0.715018 |   0.0543807 |   0.0302253 |
| w128x128x128_d0.4_lr1e-3      | mlp            |         3 |  0.0504319 |  0.034475  |  0.728667 |   0.0530624 |   0.0241718 |
| w320x320_d0.5_lr8e-4          | mlp            |         3 |  0.0504604 |  0.0338634 |  0.722144 |   0.0536965 |   0.0291776 |
| w256x256_d0.4_huber0.1_lr6e-4 | mlp            |         3 |  0.0505146 |  0.0300834 |  0.711224 |   0.0547414 |   0.030543  |
| w320x320_d0.5_lr6e-4          | mlp            |         3 |  0.050579  |  0.0338381 |  0.7223   |   0.0536814 |   0.0291325 |
| w128x128x128_d0.4_lr6e-4      | mlp            |         3 |  0.0505914 |  0.0336181 |  0.720764 |   0.0538297 |   0.0256773 |
  Spearman(val_rmse, test_r2) = +0.566 (p=0.000, n=59)
  val winner (honest) : w512x512x512_d0.3_lr1e-3 (test_r2=0.7595)
  test best (ref)     : w256x256_d0.5 (test_r2=0.7834)

#### 2-Regime-54 — top-10 by val RMSE
| config_id                               | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:----------------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w448x448x448_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |  0.0544122 |  0.0213436 |  0.759564 |   0.04995   |  0.0106746  |
| w448x448x448_d0.3_huber0.05_gelu_lr6e-4 | mlp            |         3 |  0.0546208 |  0.0200238 |  0.772386 |   0.0485999 |  0.00851276 |
| w384x384x384_d0.3_huber0.05_gelu_lr6e-4 | mlp            |         3 |  0.0546954 |  0.018177  |  0.777157 |   0.0480878 |  0.00900293 |
| w384x384x384_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |  0.0547516 |  0.0197445 |  0.77048  |   0.048803  |  0.0118001  |
| w512x512x512_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |  0.0551325 |  0.0194012 |  0.779028 |   0.0478855 |  0.00732942 |
| w384x384x384_d0.3_huber0.1_gelu_lr6e-4  | mlp            |         3 |  0.0551467 |  0.0195225 |  0.774073 |   0.0484194 |  0.00930099 |
| w448x448x448_d0.3_huber0.1_gelu_lr6e-4  | mlp            |         3 |  0.055182  |  0.0212696 |  0.769301 |   0.0489281 |  0.00802879 |
| w320x320x320_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |  0.0553146 |  0.0174075 |  0.769007 |   0.0489593 |  0.00911682 |
| w320x320_d0.3_huber0.05_gelu_lr6e-4     | mlp            |         3 |  0.0553399 |  0.0202901 |  0.784954 |   0.0472391 |  0.0126206  |
| w384x384x384_d0.3_huber0.2_gelu_lr6e-4  | mlp            |         3 |  0.0555267 |  0.0215351 |  0.769672 |   0.0488887 |  0.0106387  |
  Spearman(val_rmse, test_r2) = +0.413 (p=0.000, n=82)
  val winner (honest) : w448x448x448_d0.3_huber0.1_gelu_lr1e-3 (test_r2=0.7596)
  test best (ref)     : w320x320_d0.4_huber0.2_gelu_lr6e-4 (test_r2=0.7973)

#### 2-Regime-Mixed — top-10 by val RMSE
| config_id                          | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:-----------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w512x512x512_d0.3_huber0.03_lr1e-3 | mlp            |         3 |  0.0477502 |  0.0202982 |  0.78085  |   0.0476877 |   0.0140565 |
| w512x512x512_d0.3_huber0.03_lr6e-4 | mlp            |         3 |  0.0479339 |  0.0195794 |  0.781923 |   0.0475708 |   0.0123782 |
| w512x512x512_d0.3_huber0.05_lr6e-4 | mlp            |         3 |  0.0480034 |  0.0210395 |  0.7844   |   0.0472999 |   0.0123237 |
| w512x512x512_d0.3_huber0.03_lr8e-4 | mlp            |         3 |  0.0480289 |  0.0192288 |  0.781254 |   0.0476438 |   0.0100759 |
| w512x512x512_d0.3_huber0.05_lr8e-4 | mlp            |         3 |  0.0480483 |  0.0220867 |  0.786733 |   0.0470433 |   0.0124685 |
| w512x512x512_d0.3_huber0.08_lr1e-3 | mlp            |         3 |  0.0480727 |  0.0214041 |  0.788461 |   0.0468523 |   0.0100503 |
| w512x512x512_d0.2_huber0.05_lr6e-4 | mlp            |         3 |  0.0481297 |  0.0177169 |  0.775861 |   0.0482274 |   0.0130511 |
| w512x512x512_d0.3_huber0.05_lr1e-3 | mlp            |         3 |  0.0481478 |  0.0207832 |  0.778976 |   0.0478912 |   0.0147049 |
| w512x512x512_d0.3_huber0.05_lr4e-4 | mlp            |         3 |  0.0482153 |  0.0202201 |  0.778599 |   0.047932  |   0.0152692 |
| w512x512x512_d0.3_huber0.1_lr1e-3  | mlp            |         3 |  0.0482586 |  0.0207169 |  0.783607 |   0.0473868 |   0.0122037 |
  Spearman(val_rmse, test_r2) = -0.413 (p=0.001, n=66)
  val winner (honest) : w512x512x512_d0.3_huber0.03_lr1e-3 (test_r2=0.7809)
  test best (ref)     : w448x448x448_d0.3_huber0.1_gelu (test_r2=0.7940)

### Selection-reliability summary (analyze_selection.py)
#### Spearman(val, test) by aggregation depth
| family        | aggregation       |   n_configs |   spearman_val_test |     p_value |   median_abs_delta_val |   mean_abs_delta_val |   config_id |   val_rmse |   test_r2 |
|:--------------|:------------------|------------:|--------------------:|------------:|-----------------------:|---------------------:|------------:|-----------:|----------:|
| 2regime_96    | 1-seed (42)       |          59 |            0.61467  | 2.23196e-07 |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_96    | 2-seed (42,7)     |          57 |            0.514973 | 4.15555e-05 |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_96    | 3-seed (42,7,123) |          26 |            0.356581 | 0.07376     |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_54    | 1-seed (42)       |          82 |            0.278567 | 0.0112711   |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_54    | 2-seed (42,7)     |          78 |            0.418986 | 0.000134601 |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_54    | 3-seed (42,7,123) |          40 |            0.582176 | 8.12258e-05 |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_mixed | 1-seed (42)       |          66 |           -0.100344 | 0.422754    |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_mixed | 2-seed (42,7)     |          66 |           -0.372049 | 0.00209779  |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_mixed | 3-seed (42,7,123) |          42 |           -0.409934 | 0.00701495  |                    nan |                  nan |         nan |        nan |       nan |

#### Phase stability — winner at each seed depth
| family        | aggregation              |   n_configs |   spearman_val_test |   p_value |   median_abs_delta_val |   mean_abs_delta_val | config_id                               |   val_rmse |   test_r2 |
|:--------------|:-------------------------|------------:|--------------------:|----------:|-----------------------:|---------------------:|:----------------------------------------|-----------:|----------:|
| 2regime_96    | winner|1-seed (42)       |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_lr1e-3                |  0.0478642 |  0.759483 |
| 2regime_96    | winner|2-seed (42,7)     |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_lr1e-3                |  0.0482834 |  0.759483 |
| 2regime_96    | winner|3-seed (42,7,123) |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_lr1e-3                |  0.0484652 |  0.759483 |
| 2regime_54    | winner|1-seed (42)       |         nan |                 nan |       nan |                    nan |                  nan | w384x384x384_d0.3_huber0.05_gelu_lr6e-4 |  0.0538335 |  0.777157 |
| 2regime_54    | winner|2-seed (42,7)     |         nan |                 nan |       nan |                    nan |                  nan | w384x384x384_d0.3_huber0.05_gelu_lr6e-4 |  0.054405  |  0.777157 |
| 2regime_54    | winner|3-seed (42,7,123) |         nan |                 nan |       nan |                    nan |                  nan | w448x448x448_d0.3_huber0.1_gelu_lr1e-3  |  0.0544122 |  0.759564 |
| 2regime_mixed | winner|1-seed (42)       |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_huber0.03_lr6e-4      |  0.0476532 |  0.781923 |
| 2regime_mixed | winner|2-seed (42,7)     |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_huber0.03_lr6e-4      |  0.0477616 |  0.781923 |
| 2regime_mixed | winner|3-seed (42,7,123) |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_huber0.03_lr1e-3      |  0.0477502 |  0.78085  |

## Val-Year Selection Reliability (NEW in 2.2)

2.1's headline negative: the mixed/54 families' val ranking is noisy even at 3-seed aggregation (Spearman(val, test) = -0.309 / -0.555), and more seeds on the same val split did not fix it. This diagnostic splits the official val set (2021–2022) by YEAR and asks which val year is the better proxy for test, and whether the val-selected winner is stable under leave-one-val-year-out selection. It is made possible by the new `val_preds.npy` (best-val predictions per job, saved by the mlp22 trainer) + `artifacts/val_meta.npz`. **Diagnostic only** — the deployed selection rule stays 3-seed mean val RMSE on the FULL official val (protocol unchanged). Backed by `analyze_val_years.py` (`val_year_summary.csv`).

### Val-year diagnostic (val-2021 vs val-2022; 3-seed mean val RMSE per config)

#### 2-Regime-96 — top-10 by full-val RMSE (with per-year val RMSE)
| config_id                     |   n_seeds |   val_rmse |   val_2021_rmse |   val_2022_rmse |   test_r2 |
|:------------------------------|----------:|-----------:|----------------:|----------------:|----------:|
| w512x512x512_d0.3_lr1e-3      |         3 |  0.0484652 |       0.0403419 |       0.0557868 |  0.759483 |
| w320x320_d0.5_huber0.1_lr1e-3 |         3 |  0.0493431 |       0.0411788 |       0.0567137 |  0.729662 |
| w320x320_d0.5_huber0.2_lr1e-3 |         3 |  0.0497279 |       0.0412356 |       0.057361  |  0.716225 |
| w320x320_d0.5_huber0.1_lr6e-4 |         3 |  0.0497972 |       0.0413    |       0.0574368 |  0.724091 |
| w320x320_d0.5_huber0.2_lr6e-4 |         3 |  0.0504033 |       0.0411828 |       0.0585985 |  0.715018 |
| w128x128x128_d0.4_lr1e-3      |         3 |  0.0504319 |       0.040164  |       0.059393  |  0.728667 |
| w320x320_d0.5_lr8e-4          |         3 |  0.0504604 |       0.0417095 |       0.0583057 |  0.722144 |
| w256x256_d0.4_huber0.1_lr6e-4 |         3 |  0.0505146 |       0.04124   |       0.0587559 |  0.711224 |
| w320x320_d0.5_lr6e-4          |         3 |  0.050579  |       0.0413052 |       0.058818  |  0.7223   |
| w128x128x128_d0.4_lr6e-4      |         3 |  0.0505914 |       0.0399604 |       0.0598201 |  0.720764 |

#### 2-Regime-54 — top-10 by full-val RMSE (with per-year val RMSE)
| config_id                               |   n_seeds |   val_rmse |   val_2021_rmse |   val_2022_rmse |   test_r2 |
|:----------------------------------------|----------:|-----------:|----------------:|----------------:|----------:|
| w448x448x448_d0.3_huber0.1_gelu_lr1e-3  |         3 |  0.0544122 |       0.0451344 |       0.0627473 |  0.759564 |
| w448x448x448_d0.3_huber0.05_gelu_lr6e-4 |         3 |  0.0546208 |       0.0454681 |       0.0628669 |  0.772386 |
| w384x384x384_d0.3_huber0.05_gelu_lr6e-4 |         3 |  0.0546954 |       0.0458932 |       0.0626644 |  0.777157 |
| w384x384x384_d0.3_huber0.1_gelu_lr1e-3  |         3 |  0.0547516 |       0.0456318 |       0.0629815 |  0.77048  |
| w512x512x512_d0.3_huber0.1_gelu_lr1e-3  |         3 |  0.0551325 |       0.0467645 |       0.062787  |  0.779028 |
| w384x384x384_d0.3_huber0.1_gelu_lr6e-4  |         3 |  0.0551467 |       0.0464641 |       0.0630362 |  0.774073 |
| w448x448x448_d0.3_huber0.1_gelu_lr6e-4  |         3 |  0.055182  |       0.046252  |       0.0632725 |  0.769301 |
| w320x320x320_d0.3_huber0.1_gelu_lr1e-3  |         3 |  0.0553146 |       0.045296  |       0.0642386 |  0.769007 |
| w320x320_d0.3_huber0.05_gelu_lr6e-4     |         3 |  0.0553399 |       0.0455343 |       0.0640933 |  0.784954 |
| w384x384x384_d0.3_huber0.2_gelu_lr6e-4  |         3 |  0.0555267 |       0.0463243 |       0.06381   |  0.769672 |

#### 2-Regime-Mixed — top-10 by full-val RMSE (with per-year val RMSE)
| config_id                          |   n_seeds |   val_rmse |   val_2021_rmse |   val_2022_rmse |   test_r2 |
|:-----------------------------------|----------:|-----------:|----------------:|----------------:|----------:|
| w512x512x512_d0.3_huber0.03_lr1e-3 |         3 |  0.0477502 |       0.040137  |       0.054661  |  0.78085  |
| w512x512x512_d0.3_huber0.03_lr6e-4 |         3 |  0.0479339 |       0.040206  |       0.0549321 |  0.781923 |
| w512x512x512_d0.3_huber0.05_lr6e-4 |         3 |  0.0480034 |       0.04065   |       0.0547193 |  0.7844   |
| w512x512x512_d0.3_huber0.03_lr8e-4 |         3 |  0.0480289 |       0.0400353 |       0.0552403 |  0.781254 |
| w512x512x512_d0.3_huber0.05_lr8e-4 |         3 |  0.0480483 |       0.0403884 |       0.054999  |  0.786733 |
| w512x512x512_d0.3_huber0.08_lr1e-3 |         3 |  0.0480727 |       0.0405998 |       0.0548693 |  0.788461 |
| w512x512x512_d0.2_huber0.05_lr6e-4 |         3 |  0.0481297 |       0.0399386 |       0.0554971 |  0.775861 |
| w512x512x512_d0.3_huber0.05_lr1e-3 |         3 |  0.0481478 |       0.0400186 |       0.0554633 |  0.778976 |
| w512x512x512_d0.3_huber0.05_lr4e-4 |         3 |  0.0482153 |       0.0402891 |       0.0553773 |  0.778599 |
| w512x512x512_d0.3_huber0.1_lr1e-3  |         3 |  0.0482586 |       0.0406402 |       0.0551782 |  0.783607 |

### Spearman(val signal, test R2) per family (phase-1 pool, 3-seed aggregation)
| family        | signal        |   n_configs |   spearman |   p_value |
|:--------------|:--------------|------------:|-----------:|----------:|
| 2regime_96    | val_rmse      |          59 |      0.566 |    0      |
| 2regime_96    | val_2021_rmse |          59 |      0.747 |    0      |
| 2regime_96    | val_2022_rmse |          59 |      0.106 |    0.4252 |
| 2regime_54    | val_rmse      |          82 |      0.413 |    0.0001 |
| 2regime_54    | val_2021_rmse |          82 |      0.454 |    0      |
| 2regime_54    | val_2022_rmse |          82 |      0.133 |    0.235  |
| 2regime_mixed | val_rmse      |          66 |     -0.413 |    0.0006 |
| 2regime_mixed | val_2021_rmse |          66 |     -0.249 |    0.0434 |
| 2regime_mixed | val_2022_rmse |          66 |     -0.352 |    0.0037 |

### Winner stability under leave-one-val-year-out selection (3-seed means)
| family        | selected_by   | winner                                  |   winner_test_r2 |
|:--------------|:--------------|:----------------------------------------|-----------------:|
| 2regime_96    | val_rmse      | w512x512x512_d0.3_lr1e-3                |           0.7595 |
| 2regime_96    | val_2021_rmse | w128x128x128_d0.4_lr6e-4                |           0.7208 |
| 2regime_96    | val_2022_rmse | w512x512x512_d0.3_lr1e-3                |           0.7595 |
| 2regime_54    | val_rmse      | w448x448x448_d0.3_huber0.1_gelu_lr1e-3  |           0.7596 |
| 2regime_54    | val_2021_rmse | w320x320_d0.2_huber0.1_gelu_lr6e-4      |           0.781  |
| 2regime_54    | val_2022_rmse | w384x384x384_d0.3_huber0.05_gelu_lr6e-4 |           0.7772 |
| 2regime_mixed | val_rmse      | w512x512x512_d0.3_huber0.03_lr1e-3      |           0.7809 |
| 2regime_mixed | val_2021_rmse | w512x512x512_d0.3_huber0.15_lr1e-3      |           0.7779 |
| 2regime_mixed | val_2022_rmse | w512x512x512_d0.3_huber0.03_lr1e-3      |           0.7809 |

## Overall Model Leaderboard

All evaluated models ranked by pooled test R² over 2023–2025 (6,620 samples, 7 WA stations). MLP rows carry the sweep `config_id` and `n_seeds`; `(val top-k avg)` rows are offline seed-averaged ensembles of the top-k val-selected honest configs (no extra training); `(5-seed champ, ...)` rows are 5-seed champion ensembles of the val-selected winners (extra stability seeds, no trainval retrain — documented negative); `cross-family` rows average the val-selected winners across families. XGBoost rows are the eval-1.1 references; `MLP-1.3` / `MLP-2.0` / `MLP-2.1` rows are the previous experiments' val-selected winners + test-best references (2.0's mixed val top-5 ensemble 0.8003 is the number 2.2 must beat; 2.1's mixed val winner 0.7844 and test-best 0.7940 are the nearer bars); `test-best` rows are reporting only (selection on test would be leakage).

### Overall Leaderboard (2023-2025 Test Set)
| model_name                                                                                                                                                                 | strategy_name          |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                                                                                                                                 | XGBoost_Reference      |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| MLP-2.0 2-Regime-Mixed (val top-5 avg)                                                                                                                                     | MLP_2.0_Reference      |    0.800323 |     0.0455197 |       0.0434056 |    0.0137112  |    0.034711  |         0.90468  |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_huber0.2_gelu_lr6e-4)                                                                                                            | MLP_testbest_reference |    0.797318 |     0.045861  |       0.0456567 |    0.00432342 |    0.0355598 |         0.893942 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_gelu_lr6e-4)                                                                                                                     | MLP_testbest_reference |    0.796035 |     0.0460059 |       0.0459125 |    0.00293122 |    0.0356722 |         0.892721 |
| MLP 2-Regime-54 (test-best, w320x320_d0.3_huber0.2_gelu_lr6e-4)                                                                                                            | MLP_testbest_reference |    0.794266 |     0.046205  |       0.0453869 |    0.00865647 |    0.0356409 |         0.895408 |
| MLP 2-Regime-Mixed (test-best, w448x448x448_d0.3_huber0.1_gelu)                                                                                                            | MLP_testbest_reference |    0.793991 |     0.0462358 |       0.0450644 |    0.0103416  |    0.0350157 |         0.897313 |
| MLP-2.1 2-Regime-Mixed (test_best: w448x448x448_d0.3_huber0.1_gelu)                                                                                                        | MLP_2.1_Reference      |    0.793991 |     0.0462358 |       0.0450644 |    0.0103416  |    0.0350157 |         0.897313 |
| MLP-2.1 2-Regime-54 (test_best: w320x320_d0.3_gelu_lr6e-4)                                                                                                                 | MLP_2.1_Reference      |    0.793502 |     0.0462908 |       0.0456942 |    0.00740775 |    0.0359189 |         0.893767 |
| MLP-2.0 cross-family (val winners)                                                                                                                                         | MLP_2.0_Reference      |    0.793243 |     0.0463198 |       0.0455726 |    0.00828602 |    0.035305  |         0.894921 |
| MLP-2.1 cross-family (val winners)                                                                                                                                         | MLP_2.1_Reference      |    0.793069 |     0.0463393 |       0.0445616 |    0.0127116  |    0.0352024 |         0.900164 |
| MLP 2-Regime-Mixed (test-best, w512x512x512_d0.3_huber0.1_gelu)                                                                                                            | MLP_testbest_reference |    0.792799 |     0.0463694 |       0.0453242 |    0.00978997 |    0.0350111 |         0.896882 |
| MLP 2-Regime-Mixed (test-best, w448x448x448_d0.3_huber0.1_gelu_lr4e-4)                                                                                                     | MLP_testbest_reference |    0.791894 |     0.0464706 |       0.0452626 |    0.0105268  |    0.0352433 |         0.896426 |
| MLP-2.0 2-Regime-Mixed (test_best: w512x512x512_d0.3_huber0.1_swa)                                                                                                         | MLP_2.0_Reference      |    0.790253 |     0.0466534 |       0.0449702 |    0.0124186  |    0.0354233 |         0.898277 |
| MLP-1.3 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        | MLP_1.3_Reference      |    0.788821 |     0.0468125 |       0.0467953 |    0.00126699 |    0.0362252 |         0.888558 |
| MLP-2.1 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_2.1_Reference      |    0.788526 |     0.0468451 |       0.0462892 |    0.0071954  |    0.0362901 |         0.891197 |
| MLP cross-family (val winners: 2regime_96/w512x512x512_d0.3_lr1e-3 + 2regime_54/w448x448x448_d0.3_huber0.1_gelu_lr1e-3 + 2regime_mixed/w512x512x512_d0.3_huber0.03_lr1e-3) | MLP_cross_family       |    0.788487 |     0.0468495 |       0.0445399 |    0.0145283  |    0.0355868 |         0.900107 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr8e-4)                                                                                                                    | MLP_2regime_mixed      |    0.786733 |     0.0470433 |       0.0453609 |    0.0124685  |    0.0356718 |         0.895979 |
| MLP-2.0 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        | MLP_2.0_Reference      |    0.786493 |     0.0470697 |       0.046945  |    0.0034241  |    0.0364015 |         0.887498 |
| MLP-2.0 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_2.0_Reference      |    0.785392 |     0.047191  |       0.0461896 |    0.00967028 |    0.0364359 |         0.892043 |
| MLP 2-Regime-Mixed (val top-5 avg)                                                                                                                                         | MLP_2regime_mixed      |    0.784973 |     0.047237  |       0.0456182 |    0.0122606  |    0.0359353 |         0.894566 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                                    | MLP_2regime_mixed      |    0.7844   |     0.0472999 |       0.0456662 |    0.0123237  |    0.0359433 |         0.89492  |
| MLP-2.1 2-Regime-Mixed (val_sel: w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                       | MLP_2.1_Reference      |    0.7844   |     0.0472999 |       0.0456662 |    0.0123237  |    0.0359433 |         0.89492  |
| MLP 2-Regime-Mixed (val top-10 avg)                                                                                                                                        | MLP_2regime_mixed      |    0.78428  |     0.047313  |       0.0455883 |    0.0126582  |    0.036046  |         0.894662 |
| MLP 2-Regime-Mixed (val top-3 avg)                                                                                                                                         | MLP_2regime_mixed      |    0.783969 |     0.0473472 |       0.0455505 |    0.0129195  |    0.0360732 |         0.894928 |
| MLP-2.1 2-Regime-Mixed (val top-3 avg)                                                                                                                                     | MLP_2.1_Reference      |    0.783877 |     0.0473572 |       0.0455158 |    0.0130774  |    0.0360972 |         0.894986 |
| MLP-1.3 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_1.3_Reference      |    0.783404 |     0.047409  |       0.0471517 |    0.00493258 |    0.0360624 |         0.887208 |
| MLP-2.1 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_2.1_Reference      |    0.783404 |     0.047409  |       0.0471517 |    0.00493258 |    0.0360624 |         0.887208 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                                                                                                                                 | MLP_testbest_reference |    0.783404 |     0.047409  |       0.0471517 |    0.00493258 |    0.0360624 |         0.887208 |
| MLP-1.3 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_1.3_Reference      |    0.782533 |     0.0475043 |       0.0470162 |    0.0067925  |    0.0369583 |         0.888139 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr6e-4)                                                                                                                    | MLP_2regime_mixed      |    0.781923 |     0.0475708 |       0.0459322 |    0.0123782  |    0.0364047 |         0.893092 |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr6e-4)                                                                                                      | MLP_2regime_mixed      |    0.781829 |     0.047581  |       0.0461576 |    0.0115514  |    0.036438  |         0.891699 |
| MLP-2.1 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                  | MLP_2.1_Reference      |    0.781771 |     0.0475874 |       0.0459221 |    0.0124787  |    0.0363492 |         0.89312  |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                      | MLP_2regime_mixed      |    0.78165  |     0.0476006 |       0.0460759 |    0.011951   |    0.0362704 |         0.891943 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr8e-4)                                                                                                                    | MLP_2regime_mixed      |    0.781254 |     0.0476438 |       0.0465661 |    0.0100759  |    0.036249  |         0.889768 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                                    | MLP_2regime_mixed      |    0.78085  |     0.0476877 |       0.045569  |    0.0140565  |    0.0364254 |         0.894551 |
| MLP-2.0 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_2.0_Reference      |    0.780505 |     0.0477252 |       0.0464651 |    0.0108946  |    0.0372804 |         0.889914 |
| Global Single Model (54 Backbone)                                                                                                                                          | XGBoost_Reference      |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   | MLP_2regime_54         |    0.779028 |     0.0478855 |       0.0473213 |    0.00732942 |    0.0366639 |         0.887871 |
| MLP 2-Regime-54 (w384x384x384_d0.3_huber0.05_gelu_lr6e-4)                                                                                                                  | MLP_2regime_54         |    0.777157 |     0.0480878 |       0.0472375 |    0.00900293 |    0.0369984 |         0.887627 |
| MLP 2-Regime-54 (val top-10 avg)                                                                                                                                           | MLP_2regime_54         |    0.777    |     0.0481048 |       0.0471161 |    0.00970258 |    0.0368603 |         0.889244 |
| MLP 2-Regime-96 (test-best, w192x192_d0.5)                                                                                                                                 | MLP_testbest_reference |    0.776985 |     0.0481064 |       0.0463387 |    0.0129212  |    0.0375806 |         0.89203  |
| MLP 2-Regime-54 (val top-5 avg)                                                                                                                                            | MLP_2regime_54         |    0.774382 |     0.0483863 |       0.0474517 |    0.00946398 |    0.0370492 |         0.88815  |
| MLP 2-Regime-54 (w448x448x448_d0.3_huber0.05_gelu_lr6e-4)                                                                                                                  | MLP_2regime_54         |    0.772386 |     0.0485999 |       0.0478486 |    0.00851276 |    0.0372759 |         0.886778 |
| MLP-1.3 2-Regime-96 (val top-10 avg)                                                                                                                                       | MLP_1.3_Reference      |    0.772329 |     0.048606  |       0.0446198 |    0.0192772  |    0.0375105 |         0.899578 |
| MLP 2-Regime-54 (val top-3 avg)                                                                                                                                            | MLP_2regime_54         |    0.771955 |     0.0486459 |       0.0477297 |    0.00939678 |    0.0372582 |         0.887219 |
| MLP-2.1 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                             | MLP_2.1_Reference      |    0.771661 |     0.0486772 |       0.0480595 |    0.00772988 |    0.0380031 |         0.886477 |
| MLP-2.1 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  | MLP_2.1_Reference      |    0.771284 |     0.0487174 |       0.048218  |    0.00695755 |    0.0380165 |         0.886103 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5_huber0.2)                                                                                                                        | MLP_testbest_reference |    0.770703 |     0.0487793 |       0.045896  |    0.0165218  |    0.0380062 |         0.893051 |
| MLP 2-Regime-54 (w384x384x384_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   | MLP_2regime_54         |    0.77048  |     0.048803  |       0.0473549 |    0.0118001  |    0.0373547 |         0.888677 |
| MLP-2.0 2-Regime-96 (val top-10 avg)                                                                                                                                       | MLP_2.0_Reference      |    0.769168 |     0.0489422 |       0.0450254 |    0.0191846  |    0.0379532 |         0.897463 |
| MLP 2-Regime-54 (5-seed champ, w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                     | MLP_2regime_54         |    0.766788 |     0.0491939 |       0.0479627 |    0.0109373  |    0.0375111 |         0.886572 |
| MLP-2.0 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  | MLP_2.0_Reference      |    0.76511  |     0.0493706 |       0.0488979 |    0.00681535 |    0.0385003 |         0.882441 |
| MLP-1.3 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  | MLP_1.3_Reference      |    0.76511  |     0.0493706 |       0.0488979 |    0.00681535 |    0.0385003 |         0.882441 |
| MLP-2.0 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_2.0_Reference      |    0.761018 |     0.0497987 |       0.0465842 |    0.0176019  |    0.0384117 |         0.890751 |
| MLP-1.3 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_1.3_Reference      |    0.761018 |     0.0497987 |       0.0465842 |    0.0176019  |    0.0384117 |         0.890751 |
| MLP-2.0 2-Regime-Mixed (val_sel: fg_w512x512_d0.3_huber0.1_swa)                                                                                                            | MLP_2.0_Reference      |    0.759881 |     0.049917  |       0.0499151 |    0.00044081 |    0.038269  |         0.872243 |
| MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   | MLP_2regime_54         |    0.759564 |     0.04995   |       0.0487961 |    0.0106746  |    0.0382266 |         0.884994 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                                                                                                                                 | MLP_2regime_96         |    0.759483 |     0.0499584 |       0.0462642 |    0.0188537  |    0.0387226 |         0.891821 |
| MLP-2.1 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_2.1_Reference      |    0.759483 |     0.0499584 |       0.0462642 |    0.0188537  |    0.0387226 |         0.891821 |
| MLP-2.1 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                               | MLP_2.1_Reference      |    0.75656  |     0.050261  |       0.0459346 |    0.0204007  |    0.0391776 |         0.892873 |
| MLP 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                                   | MLP_2regime_96         |    0.75656  |     0.050261  |       0.0459346 |    0.0204007  |    0.0391776 |         0.892873 |
| MLP-2.1 2-Regime-96 (val top-3 avg)                                                                                                                                        | MLP_2.1_Reference      |    0.753121 |     0.0506148 |       0.0448707 |    0.0234197  |    0.0400017 |         0.897911 |
| MLP 2-Regime-96 (val top-3 avg)                                                                                                                                            | MLP_2regime_96         |    0.744187 |     0.0515225 |       0.0446598 |    0.025692   |    0.0409332 |         0.898879 |
| MLP 2-Regime-96 (val top-5 avg)                                                                                                                                            | MLP_2regime_96         |    0.736517 |     0.0522892 |       0.0446585 |    0.0271988  |    0.0418427 |         0.89879  |
| MLP 2-Regime-96 (val top-10 avg)                                                                                                                                           | MLP_2regime_96         |    0.735509 |     0.0523892 |       0.0446099 |    0.0274696  |    0.0418711 |         0.899048 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.1_lr1e-3)                                                                                                                            | MLP_2regime_96         |    0.729662 |     0.0529651 |       0.0449493 |    0.0280154  |    0.0426531 |         0.897386 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.1_lr6e-4)                                                                                                                            | MLP_2regime_96         |    0.724091 |     0.053508  |       0.0451644 |    0.0286928  |    0.0429847 |         0.896344 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.2_lr1e-3)                                                                                                                            | MLP_2regime_96         |    0.716225 |     0.0542654 |       0.0450807 |    0.0302069  |    0.0437254 |         0.897076 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.2_lr6e-4)                                                                                                                            | MLP_2regime_96         |    0.715018 |     0.0543807 |       0.0452072 |    0.0302253  |    0.044076  |         0.896187 |

## Hyperparameter Sweep Summary

191 curated phase-1 configs (all `mlp` — `fg`/`plr` are documented negatives from 2.0, `swa` a documented negative from 2.1, and none get GPU) generated deterministically by `make_configs.py` from 2-factor grids around the 2.1 winners: 54 gelu/lr6e-4 width × lr fine / width × huber / 3-layer × lr / 3-layer × huber / dropout × width / dropout × huber; mixed act × depth × lr / gelu × huber × lr / silu-512³ δ × lr fine / dropout × δ / gelu 2-layer × lr / δ-0.05 probes; 96 small-net width × lr fine / huber × lr / max_epochs {500, 600} / 3-layer small nets / mixup × lr / dropout × huber. 8 parallel H100 workers; configs ranked by **multi-seed mean val RMSE** (honest signal); test R² for reference. Phase-2 configs carry `n_seeds=2`, phase-3 configs `n_seeds=3` (top-Ns capped at the family sizes — the densest seed coverage yet).

### Sweep Top-10 — 2-Regime-96 (by val RMSE, the honest selection signal)
| config_id                     | architecture   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |   best_epoch |   train_time_s | deployed   |
|:------------------------------|:---------------|----------:|----------:|-------:|:-------|-----------:|-----------:|----------:|------------:|------------:|-------------:|---------------:|:-----------|
| w512x512x512_d0.3_lr1e-3      | mlp            |         3 |       0.3 | 0.001  | mse    |  0.0484652 |  0.0260292 |  0.759483 |   0.0499584 |   0.0188537 |          263 |        124.873 | live       |
| w320x320_d0.5_huber0.1_lr1e-3 | mlp            |         3 |       0.5 | 0.001  | huber  |  0.0493431 |  0.0300213 |  0.729662 |   0.0529651 |   0.0280154 |          199 |        103.297 | live       |
| w320x320_d0.5_huber0.2_lr1e-3 | mlp            |         3 |       0.5 | 0.001  | huber  |  0.0497279 |  0.031144  |  0.716225 |   0.0542654 |   0.0302069 |          199 |        109.567 | live       |
| w320x320_d0.5_huber0.1_lr6e-4 | mlp            |         3 |       0.5 | 0.0006 | huber  |  0.0497972 |  0.032205  |  0.724091 |   0.053508  |   0.0286928 |          219 |        117.283 | live       |
| w320x320_d0.5_huber0.2_lr6e-4 | mlp            |         3 |       0.5 | 0.0006 | huber  |  0.0504033 |  0.0321439 |  0.715018 |   0.0543807 |   0.0302253 |          207 |        125.337 | live       |
| w128x128x128_d0.4_lr1e-3      | mlp            |         3 |       0.4 | 0.001  | mse    |  0.0504319 |  0.034475  |  0.728667 |   0.0530624 |   0.0241718 |          241 |        126.291 | live       |
| w320x320_d0.5_lr8e-4          | mlp            |         3 |       0.5 | 0.0008 | mse    |  0.0504604 |  0.0338634 |  0.722144 |   0.0536965 |   0.0291776 |          206 |        120.975 | live       |
| w256x256_d0.4_huber0.1_lr6e-4 | mlp            |         3 |       0.4 | 0.0006 | huber  |  0.0505146 |  0.0300834 |  0.711224 |   0.0547414 |   0.030543  |          224 |        117.768 | live       |
| w320x320_d0.5_lr6e-4          | mlp            |         3 |       0.5 | 0.0006 | mse    |  0.050579  |  0.0338381 |  0.7223   |   0.0536814 |   0.0291325 |          291 |        132.411 | live       |
| w128x128x128_d0.4_lr6e-4      | mlp            |         3 |       0.4 | 0.0006 | mse    |  0.0505914 |  0.0336181 |  0.720764 |   0.0538297 |   0.0256773 |          286 |        153.404 | live       |

### Sweep Top-10 — 2-Regime-54 (by val RMSE, the honest selection signal)
| config_id                               | architecture   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |   best_epoch |   train_time_s | deployed   |
|:----------------------------------------|:---------------|----------:|----------:|-------:|:-------|-----------:|-----------:|----------:|------------:|------------:|-------------:|---------------:|:-----------|
| w448x448x448_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0544122 |  0.0213436 |  0.759564 |   0.04995   |  0.0106746  |          187 |        92.1484 | live       |
| w448x448x448_d0.3_huber0.05_gelu_lr6e-4 | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0546208 |  0.0200238 |  0.772386 |   0.0485999 |  0.00851276 |          229 |       122.592  | live       |
| w384x384x384_d0.3_huber0.05_gelu_lr6e-4 | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0546954 |  0.018177  |  0.777157 |   0.0480878 |  0.00900293 |          294 |       132.988  | live       |
| w384x384x384_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0547516 |  0.0197445 |  0.77048  |   0.048803  |  0.0118001  |          226 |       100.588  | live       |
| w512x512x512_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0551325 |  0.0194012 |  0.779028 |   0.0478855 |  0.00732942 |          322 |       113.742  | live       |
| w384x384x384_d0.3_huber0.1_gelu_lr6e-4  | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0551467 |  0.0195225 |  0.774073 |   0.0484194 |  0.00930099 |          294 |       131.517  | live       |
| w448x448x448_d0.3_huber0.1_gelu_lr6e-4  | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.055182  |  0.0212696 |  0.769301 |   0.0489281 |  0.00802879 |          229 |       133.058  | live       |
| w320x320x320_d0.3_huber0.1_gelu_lr1e-3  | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0553146 |  0.0174075 |  0.769007 |   0.0489593 |  0.00911682 |          293 |       135.797  | live       |
| w320x320_d0.3_huber0.05_gelu_lr6e-4     | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0553399 |  0.0202901 |  0.784954 |   0.0472391 |  0.0126206  |          213 |       114.317  | live       |
| w384x384x384_d0.3_huber0.2_gelu_lr6e-4  | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0555267 |  0.0215351 |  0.769672 |   0.0488887 |  0.0106387  |          261 |       127.629  | live       |

### Sweep Top-10 — 2-Regime-Mixed (by val RMSE, the honest selection signal)
| config_id                          | architecture   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |   best_epoch |   train_time_s | deployed   |
|:-----------------------------------|:---------------|----------:|----------:|-------:|:-------|-----------:|-----------:|----------:|------------:|------------:|-------------:|---------------:|:-----------|
| w512x512x512_d0.3_huber0.03_lr1e-3 | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0477502 |  0.0202982 |  0.78085  |   0.0476877 |   0.0140565 |          126 |        83.7996 | live       |
| w512x512x512_d0.3_huber0.03_lr6e-4 | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0479339 |  0.0195794 |  0.781923 |   0.0475708 |   0.0123782 |          161 |       101.944  | live       |
| w512x512x512_d0.3_huber0.05_lr6e-4 | mlp            |         3 |       0.3 | 0.0006 | huber  |  0.0480034 |  0.0210395 |  0.7844   |   0.0472999 |   0.0123237 |          180 |       102.037  | live       |
| w512x512x512_d0.3_huber0.03_lr8e-4 | mlp            |         3 |       0.3 | 0.0008 | huber  |  0.0480289 |  0.0192288 |  0.781254 |   0.0476438 |   0.0100759 |          161 |        97.1513 | live       |
| w512x512x512_d0.3_huber0.05_lr8e-4 | mlp            |         3 |       0.3 | 0.0008 | huber  |  0.0480483 |  0.0220867 |  0.786733 |   0.0470433 |   0.0124685 |          160 |        95.1038 | live       |
| w512x512x512_d0.3_huber0.08_lr1e-3 | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0480727 |  0.0214041 |  0.788461 |   0.0468523 |   0.0100503 |          164 |        94.0493 | live       |
| w512x512x512_d0.2_huber0.05_lr6e-4 | mlp            |         3 |       0.2 | 0.0006 | huber  |  0.0481297 |  0.0177169 |  0.775861 |   0.0482274 |   0.0130511 |          164 |        94.6627 | live       |
| w512x512x512_d0.3_huber0.05_lr1e-3 | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0481478 |  0.0207832 |  0.778976 |   0.0478912 |   0.0147049 |          164 |        87.7447 | live       |
| w512x512x512_d0.3_huber0.05_lr4e-4 | mlp            |         3 |       0.3 | 0.0004 | huber  |  0.0482153 |  0.0202201 |  0.778599 |   0.047932  |   0.0152692 |          225 |       131.012  | live       |
| w512x512x512_d0.3_huber0.1_lr1e-3  | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0482586 |  0.0207169 |  0.783607 |   0.0473868 |   0.0122037 |          164 |        95.378  | live       |

## Per-Regime Performance Breakdown

Cluster 0 holds 73 % of the test rows, so it dominates the pooled R². Per-cluster test metrics for the val top-3 honest configs per family (the mixed family's c1 = 54+10 specialist is expected to hold the ~0.83 R² of the 54-family's c1 while c0 gains the 96-pool fit), the XGBoost references, and the 1.3 / 2.0 / 2.1 reference winners.

### Per-Regime Performance Breakdown
| strategy_name     | model_name                                                |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |         bias |       mae |
|:------------------|:----------------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|-------------:|----------:|
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         0 |      7156 |     4817 | 0.751308 | 0.0498896 | 0.0471617 |  0.0162712   | 0.0390899 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         1 |      2647 |     1803 | 0.77822  | 0.0501416 | 0.0430228 |  0.0257531   | 0.0377413 |
| MLP_2regime_96    | MLP 2-Regime-96 (w320x320_d0.5_huber0.1_lr1e-3)           |         0 |      7156 |     4817 | 0.714825 | 0.0534239 | 0.0457827 |  0.0275327   | 0.043497  |
| MLP_2regime_96    | MLP 2-Regime-96 (w320x320_d0.5_huber0.1_lr1e-3)           |         1 |      2647 |     1803 | 0.764043 | 0.0517194 | 0.0426159 |  0.029305    | 0.0403985 |
| MLP_2regime_96    | MLP 2-Regime-96 (w320x320_d0.5_huber0.2_lr1e-3)           |         0 |      7156 |     4817 | 0.693136 | 0.0554182 | 0.0459571 |  0.0309697   | 0.0452389 |
| MLP_2regime_96    | MLP 2-Regime-96 (w320x320_d0.5_huber0.2_lr1e-3)           |         1 |      2647 |     1803 | 0.77004  | 0.0510579 | 0.0425844 |  0.0281688   | 0.0396818 |
| MLP_2regime_54    | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)  |         0 |      7156 |     4817 | 0.739603 | 0.0510502 | 0.0501782 |  0.00939558  | 0.0401103 |
| MLP_2regime_54    | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)  |         1 |      2647 |     1803 | 0.8061   | 0.0468841 | 0.0447162 |  0.0140919   | 0.0331941 |
| MLP_2regime_54    | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.05_gelu_lr6e-4) |         0 |      7156 |     4817 | 0.748671 | 0.0501534 | 0.0495604 |  0.00769018  | 0.0391885 |
| MLP_2regime_54    | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.05_gelu_lr6e-4) |         1 |      2647 |     1803 | 0.827803 | 0.0441824 | 0.0428646 |  0.0107104   | 0.0321661 |
| MLP_2regime_54    | MLP 2-Regime-54 (w384x384x384_d0.3_huber0.05_gelu_lr6e-4) |         0 |      7156 |     4817 | 0.762332 | 0.0487713 | 0.0483099 |  0.00669324  | 0.0383801 |
| MLP_2regime_54    | MLP 2-Regime-54 (w384x384x384_d0.3_huber0.05_gelu_lr6e-4) |         1 |      2647 |     1803 | 0.811618 | 0.0462122 | 0.04365   |  0.0151736   | 0.0333069 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)   |         0 |      7156 |     4817 | 0.759582 | 0.0490527 | 0.0470685 |  0.0138103   | 0.0386101 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)   |         1 |      2647 |     1803 | 0.830518 | 0.0438328 | 0.0412892 |  0.0147143   | 0.0305884 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr6e-4)   |         0 |      7156 |     4817 | 0.766096 | 0.0483837 | 0.0472695 |  0.0103233   | 0.037682  |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr6e-4)   |         1 |      2647 |     1803 | 0.81876  | 0.0453277 | 0.0416573 |  0.0178682   | 0.0329921 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)   |         0 |      7156 |     4817 | 0.767059 | 0.0482839 | 0.0470642 |  0.0107839   | 0.0375314 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)   |         1 |      2647 |     1803 | 0.824812 | 0.0445645 | 0.0414222 |  0.0164375   | 0.0317004 |
| XGBoost_Reference | Global Single Model (54 Backbone)                         |         0 |     14608 |     6620 | 0.77923  | 0.0478636 | 0.0466868 |  0.0105484   | 0.0370592 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)                |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 |  0.00861491  | 0.0359221 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)                |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 |  0.000797068 | 0.0278349 |
| MLP_1.3_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         0 |      7156 |     4817 | 0.754287 | 0.0495899 | 0.0472413 |  0.0150802   | 0.0389389 |
| MLP_1.3_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         1 |      2647 |     1803 | 0.776352 | 0.0503523 | 0.0440792 |  0.0243389   | 0.0370033 |
| MLP_1.3_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |         0 |      7156 |     4817 | 0.736751 | 0.051329  | 0.0510691 |  0.0051591   | 0.0408763 |
| MLP_1.3_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |         1 |      2647 |     1803 | 0.831465 | 0.0437101 | 0.0422401 |  0.0112403   | 0.0321524 |
| MLP_2.0_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         0 |      7156 |     4817 | 0.754287 | 0.0495899 | 0.0472413 |  0.0150802   | 0.0389389 |
| MLP_2.0_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         1 |      2647 |     1803 | 0.776352 | 0.0503523 | 0.0440792 |  0.0243389   | 0.0370033 |
| MLP_2.0_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |         0 |      7156 |     4817 | 0.736751 | 0.051329  | 0.0510691 |  0.0051591   | 0.0408763 |
| MLP_2.0_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |         1 |      2647 |     1803 | 0.831465 | 0.0437101 | 0.0422401 |  0.0112403   | 0.0321524 |
| MLP_2.0_Reference | MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa)        |         0 |      7156 |     4817 | 0.73214  | 0.0517766 | 0.0515325 | -0.00502148  | 0.0403861 |
| MLP_2.0_Reference | MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa)        |         1 |      2647 |     1803 | 0.824767 | 0.0445702 | 0.041958  |  0.0150342   | 0.0326127 |
| MLP_2.1_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         0 |      7156 |     4817 | 0.751308 | 0.0498896 | 0.0471617 |  0.0162712   | 0.0390899 |
| MLP_2.1_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                |         1 |      2647 |     1803 | 0.77822  | 0.0501416 | 0.0430228 |  0.0257531   | 0.0377413 |
| MLP_2.1_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |         0 |      7156 |     4817 | 0.744135 | 0.050604  | 0.0503453 |  0.00511076  | 0.0404377 |
| MLP_2.1_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)              |         1 |      2647 |     1803 | 0.8348   | 0.0432754 | 0.0416095 |  0.0118915   | 0.0315477 |
| MLP_2.1_Reference | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)   |         0 |      7156 |     4817 | 0.767059 | 0.0482839 | 0.0470642 |  0.0107839   | 0.0375314 |
| MLP_2.1_Reference | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)   |         1 |      2647 |     1803 | 0.824812 | 0.0445645 | 0.0414222 |  0.0164375   | 0.0317004 |

## Yearly Performance Breakdown

Year-by-year R² on the 2023–2025 test period. 2.0 fixed the historically weak 2025 year for the mixed family's ensembles (2025 R² 0.8336 — best of any model); 2.1's mixed val winner held 2025 at 0.8185. This table tracks whether the 2.2 winners hold that year.

### Year-by-Year R² Breakdown
| model_name                                                                                                                                                                 |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                                                                                                                                 |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| MLP-2.0 2-Regime-Mixed (val top-5 avg)                                                                                                                                     |    0.800323 |       0.745352 |       0.825612 |       0.832424 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_huber0.2_gelu_lr6e-4)                                                                                                            |    0.797318 |       0.771995 |       0.813612 |       0.803729 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_gelu_lr6e-4)                                                                                                                     |    0.796035 |       0.776739 |       0.811801 |       0.795928 |
| MLP 2-Regime-54 (test-best, w320x320_d0.3_huber0.2_gelu_lr6e-4)                                                                                                            |    0.794266 |       0.746759 |       0.829883 |       0.807392 |
| MLP 2-Regime-Mixed (test-best, w448x448x448_d0.3_huber0.1_gelu)                                                                                                            |    0.793991 |       0.74719  |       0.811376 |       0.824133 |
| MLP-2.1 2-Regime-Mixed (test_best: w448x448x448_d0.3_huber0.1_gelu)                                                                                                        |    0.793991 |       0.74719  |       0.811376 |       0.824133 |
| MLP-2.1 2-Regime-54 (test_best: w320x320_d0.3_gelu_lr6e-4)                                                                                                                 |    0.793502 |       0.758868 |       0.820591 |       0.800015 |
| MLP-2.0 cross-family (val winners)                                                                                                                                         |    0.793243 |       0.74832  |       0.812752 |       0.819099 |
| MLP-2.1 cross-family (val winners)                                                                                                                                         |    0.793069 |       0.740408 |       0.818901 |       0.821706 |
| MLP 2-Regime-Mixed (test-best, w512x512x512_d0.3_huber0.1_gelu)                                                                                                            |    0.792799 |       0.747266 |       0.808386 |       0.823188 |
| MLP 2-Regime-Mixed (test-best, w448x448x448_d0.3_huber0.1_gelu_lr4e-4)                                                                                                     |    0.791894 |       0.743514 |       0.807827 |       0.825214 |
| MLP-2.0 2-Regime-Mixed (test_best: w512x512x512_d0.3_huber0.1_swa)                                                                                                         |    0.790253 |       0.735215 |       0.807409 |       0.830042 |
| MLP-1.3 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        |    0.788821 |       0.773579 |       0.818284 |       0.770357 |
| MLP-2.1 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.788526 |       0.75198  |       0.818135 |       0.79462  |
| MLP cross-family (val winners: 2regime_96/w512x512x512_d0.3_lr1e-3 + 2regime_54/w448x448x448_d0.3_huber0.1_gelu_lr1e-3 + 2regime_mixed/w512x512x512_d0.3_huber0.03_lr1e-3) |    0.788487 |       0.737554 |       0.812476 |       0.816765 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr8e-4)                                                                                                                    |    0.786733 |       0.728391 |       0.804696 |       0.829441 |
| MLP-2.0 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        |    0.786493 |       0.767355 |       0.826921 |       0.76174  |
| MLP-2.0 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.785392 |       0.763461 |       0.817843 |       0.771644 |
| MLP 2-Regime-Mixed (val top-5 avg)                                                                                                                                         |    0.784973 |       0.734463 |       0.801995 |       0.819457 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                                    |    0.7844   |       0.73328  |       0.802516 |       0.818504 |
| MLP-2.1 2-Regime-Mixed (val_sel: w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                       |    0.7844   |       0.73328  |       0.802516 |       0.818504 |
| MLP 2-Regime-Mixed (val top-10 avg)                                                                                                                                        |    0.78428  |       0.732848 |       0.801268 |       0.819845 |
| MLP 2-Regime-Mixed (val top-3 avg)                                                                                                                                         |    0.783969 |       0.734502 |       0.801373 |       0.816834 |
| MLP-2.1 2-Regime-Mixed (val top-3 avg)                                                                                                                                     |    0.783877 |       0.73011  |       0.801706 |       0.821313 |
| MLP-1.3 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP-2.1 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                                                                                                                                 |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP-1.3 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.782533 |       0.749643 |       0.803851 |       0.792293 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr6e-4)                                                                                                                    |    0.781923 |       0.736367 |       0.794995 |       0.814422 |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr6e-4)                                                                                                      |    0.781829 |       0.737827 |       0.793869 |       0.813533 |
| MLP-2.1 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                  |    0.781771 |       0.734857 |       0.795061 |       0.815627 |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                      |    0.78165  |       0.739578 |       0.798031 |       0.806861 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr8e-4)                                                                                                                    |    0.781254 |       0.735784 |       0.795173 |       0.812801 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                                    |    0.78085  |       0.728664 |       0.801999 |       0.813099 |
| MLP-2.0 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.780505 |       0.736315 |       0.827429 |       0.778244 |
| Global Single Model (54 Backbone)                                                                                                                                          |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   |    0.779028 |       0.752109 |       0.783873 |       0.797862 |
| MLP 2-Regime-54 (w384x384x384_d0.3_huber0.05_gelu_lr6e-4)                                                                                                                  |    0.777157 |       0.750466 |       0.780913 |       0.796729 |
| MLP 2-Regime-54 (val top-10 avg)                                                                                                                                           |    0.777    |       0.739206 |       0.785624 |       0.80468  |
| MLP 2-Regime-96 (test-best, w192x192_d0.5)                                                                                                                                 |    0.776985 |       0.751922 |       0.813254 |       0.762843 |
| MLP 2-Regime-54 (val top-5 avg)                                                                                                                                            |    0.774382 |       0.739662 |       0.778871 |       0.802452 |
| MLP 2-Regime-54 (w448x448x448_d0.3_huber0.05_gelu_lr6e-4)                                                                                                                  |    0.772386 |       0.735171 |       0.769353 |       0.81064  |
| MLP-1.3 2-Regime-96 (val top-10 avg)                                                                                                                                       |    0.772329 |       0.72545  |       0.797894 |       0.793806 |
| MLP 2-Regime-54 (val top-3 avg)                                                                                                                                            |    0.771955 |       0.738461 |       0.771908 |       0.802957 |
| MLP-2.1 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                             |    0.771661 |       0.723384 |       0.781407 |       0.810216 |
| MLP-2.1 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  |    0.771284 |       0.730569 |       0.781896 |       0.800208 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5_huber0.2)                                                                                                                        |    0.770703 |       0.737319 |       0.809457 |       0.763563 |
| MLP 2-Regime-54 (w384x384x384_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   |    0.77048  |       0.724608 |       0.786676 |       0.799891 |
| MLP-2.0 2-Regime-96 (val top-10 avg)                                                                                                                                       |    0.769168 |       0.720824 |       0.814159 |       0.773225 |
| MLP 2-Regime-54 (5-seed champ, w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                     |    0.766788 |       0.721024 |       0.7697   |       0.808946 |
| MLP-2.0 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  |    0.76511  |       0.723072 |       0.775216 |       0.79585  |
| MLP-1.3 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  |    0.76511  |       0.723072 |       0.775216 |       0.79585  |
| MLP-2.0 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.761018 |       0.706877 |       0.786985 |       0.790134 |
| MLP-1.3 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.761018 |       0.706877 |       0.786985 |       0.790134 |
| MLP-2.0 2-Regime-Mixed (val_sel: fg_w512x512_d0.3_huber0.1_swa)                                                                                                            |    0.759881 |       0.744054 |       0.757107 |       0.772643 |
| MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   |    0.759564 |       0.722393 |       0.758606 |       0.795294 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                                                                                                                                 |    0.759483 |       0.711309 |       0.782285 |       0.784721 |
| MLP-2.1 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.759483 |       0.711309 |       0.782285 |       0.784721 |
| MLP-2.1 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                               |    0.75656  |       0.708198 |       0.776529 |       0.784688 |
| MLP 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                                   |    0.75656  |       0.708198 |       0.776529 |       0.784688 |
| MLP-2.1 2-Regime-96 (val top-3 avg)                                                                                                                                        |    0.753121 |       0.709325 |       0.778246 |       0.770787 |
| MLP 2-Regime-96 (val top-3 avg)                                                                                                                                            |    0.744187 |       0.695093 |       0.77868  |       0.758521 |
| MLP 2-Regime-96 (val top-5 avg)                                                                                                                                            |    0.736517 |       0.687764 |       0.775031 |       0.746256 |
| MLP 2-Regime-96 (val top-10 avg)                                                                                                                                           |    0.735509 |       0.6887   |       0.771595 |       0.745332 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.1_lr1e-3)                                                                                                                            |    0.729662 |       0.685333 |       0.77179  |       0.730494 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.1_lr6e-4)                                                                                                                            |    0.724091 |       0.674203 |       0.770383 |       0.727105 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.2_lr1e-3)                                                                                                                            |    0.716225 |       0.663849 |       0.758266 |       0.726012 |
| MLP 2-Regime-96 (w320x320_d0.5_huber0.2_lr6e-4)                                                                                                                            |    0.715018 |       0.668961 |       0.758509 |       0.716014 |

## Systematic-Bias Diagnostic (headline)

mlp-1.2/1.3 documented the 96-family's systematic positive test bias (bias² ≈ 10–17 % of MSE); 2.0 got the 54-family under 5 %; 2.1 got the 54-family to 0.8 % but the 96 (13.9 %) and mixed (12.7 %) families still miss the <5 % criterion. 2.2's debias lever for 96: the small-net pool (width 96–320, dropout 0.4–0.6, lr {3e-4, 4e-4, 6e-4, 8e-4} — lr6e-4 never tested before), huber × lr, mixup × lr, 3-layer small nets, and max_epochs {500, 600} probes for the under-trained small nets. Success criterion: per-family median bias²/MSE < 5 % for ALL three families. Backed by `analyze_bias.py`.

### Per-family median bias^2/MSE share (honest architectures)
| family     | n_configs |   med_bias2_mse_share |   med_test_bias |   med_test_r2 |
|:-----------|----------:|----------------------:|----------------:|--------------:|
| 2regime_96 | 59 | 0.2169 | 0.0243 | 0.7355 |
| 2regime_54 | 82 | 0.0307 | 0.0084 | 0.7815 |
| 2regime_mixed | 66 | 0.0964 | 0.0149 | 0.7786 |

### Worst 8 configs by bias^2/MSE share (all architectures)
| family        | config_id                          | architecture   |   test_r2 |   test_rmse |   test_bias |   bias2_mse_share |
|:--------------|:-----------------------------------|:---------------|----------:|------------:|------------:|------------------:|
| 2regime_96    | w96x96x96_d0.4_lr1e-3              | mlp            |  0.684285 |   0.0572378 |   0.0323514 |          0.319462 |
| 2regime_96    | w256x256_d0.4_huber0.1_lr6e-4      | mlp            |  0.711224 |   0.0547414 |   0.030543  |          0.311309 |
| 2regime_96    | w320x320_d0.5_huber0.2_lr1e-3      | mlp            |  0.716225 |   0.0542654 |   0.0302069 |          0.309861 |
| 2regime_96    | w320x320_d0.5_huber0.2_lr6e-4      | mlp            |  0.715018 |   0.0543807 |   0.0302253 |          0.308925 |
| 2regime_96    | w192x192_d0.5_huber0.1_lr6e-4      | mlp            |  0.720906 |   0.0538159 |   0.0297061 |          0.304698 |
| 2regime_mixed | w512x512_d0.3_huber0.1_gelu_lr1e-3 | mlp            |  0.707055 |   0.0551352 |   0.0302146 |          0.300313 |
| 2regime_96    | w256x256_d0.4_huber0.2_lr6e-4      | mlp            |  0.716301 |   0.0542581 |   0.0296308 |          0.298235 |
| 2regime_96    | w192x192x192_d0.4_lr6e-4           | mlp            |  0.70121  |   0.0556825 |   0.0302792 |          0.2957   |

### Best 8 configs by bias^2/MSE share (all architectures)
| family     | config_id                       | architecture   |   test_r2 |   test_rmse |   test_bias |   bias2_mse_share |
|:-----------|:--------------------------------|:---------------|----------:|------------:|------------:|------------------:|
| 2regime_54 | w192x192_d0.3_gelu_lr4e-4       | mlp            |  0.793364 |   0.0463062 | 0.000575134 |       0.000154262 |
| 2regime_54 | w256x256_d0.4_gelu_lr6e-4       | mlp            |  0.781208 |   0.0476488 | 0.000623366 |       0.000171152 |
| 2regime_54 | w256x256_d0.3_gelu_lr4e-4       | mlp            |  0.780325 |   0.0477449 | 0.000794159 |       0.000276669 |
| 2regime_54 | w512x512_d0.3_gelu_lr4e-4       | mlp            |  0.766509 |   0.0492234 | 0.001005    |       0.000416859 |
| 2regime_54 | w224x224_d0.3_gelu_lr4e-4       | mlp            |  0.790938 |   0.0465772 | 0.00115967  |       0.000619902 |
| 2regime_54 | w320x320x320_d0.3_huber0.1_gelu | mlp            |  0.770323 |   0.0488196 | 0.00191464  |       0.0015381   |
| 2regime_54 | w288x288_d0.3_gelu_lr4e-4       | mlp            |  0.781688 |   0.0475964 | 0.00226842  |       0.00227142  |
| 2regime_54 | w448x448x448_d0.3_huber0.1_gelu | mlp            |  0.769091 |   0.0489504 | 0.00248504  |       0.00257723  |

### Per-cluster median bias^2/MSE share (honest architectures)
| family     | cluster |   med_bias2_mse_share |   med_test_bias |   med_test_r2 |
|:-----------|--------:|----------------------:|----------------:|--------------:|
| 2regime_96 | 0 | 0.1922 | 0.0236 | 0.7169 |
| 2regime_96 | 1 | 0.3041 | 0.0279 | 0.7732 |
| 2regime_54 | 0 | 0.0200 | 0.0070 | 0.7595 |
| 2regime_54 | 1 | 0.0683 | 0.0111 | 0.8337 |
| 2regime_mixed | 0 | 0.0912 | 0.0149 | 0.7600 |
| 2regime_mixed | 1 | 0.0933 | 0.0137 | 0.8264 |

## FeatureGroupedMLP / PLR — documented negatives, not re-run in 2.2

2.0 established that the grouped-tower (`fg`, best 0.782) and PLR-encoding (`plr`, best 0.720) architectures underperform the plain MLP (0.790) at this scale — the winning lever was the *feature allocation* (the `2regime_mixed` family), not the tower structure. Per the no-re-spend rule, **2.2 runs no fg/plr configs**; the classes and the validated semantic grouping remain available in `mlp22/feature_groups.py`. The grouping table for the union of the three families' features is printed for reference.

Union of the 3 families' features: 116
| group_id | group | n_features | features |
|---|---|---|---|
| 0 | smap | 26 | A_d_SMAP_sm_interp_kobs14, A_d_SMAP_sm_interp_kobs30, A_grad_SMAP_sm_interp_kobs14, A_grad_SMAP_sm_interp_kobs30, A_grad_SMAP_sm_interp_kobs7, C_lag_SMAP_sm_interp_kobs12, C_lag_SMAP_sm_interp_kobs30, SMAP_ampm_diff_interp, SMAP_sm_am_interp, SMAP_sm_am_interp_lag1, SMAP_sm_am_interp_lag30, SMAP_sm_am_interp_rollrange30, SMAP_sm_am_interp_rollrange7, SMAP_sm_interp_lag7, SMAP_sm_interp_rollrange30, SMAP_sm_interp_rollrange7, SMAP_sm_pm_interp, SMAP_sm_pm_interp_lag1, SMAP_sm_pm_interp_lag30, SMAP_sm_pm_interp_lag7, SMAP_sm_pm_interp_rollmean30, SMAP_sm_pm_interp_rollrange30, SMAP_sm_pm_interp_rollrange7, V_ema_SMAP_sm_interp_kobs30, V_rollmin_SMAP_sm_interp_kobs14, V_rollmin_SMAP_sm_interp_kobs30 |
| 1 | optical | 7 | A_grad_s2_b11_kobs30, V_rollmin_s2_b11_kobs14, V_rollmin_s2_b11_kobs30, V_rollmin_s2_b12_kobs30, V_rollrng_s2_b11_kobs30, s2_b4, s2_b8 |
| 2 | vegetation | 17 | C_lag_F_NDMI_kobs30, C_lag_F_NDVI_kobs30, D_sa_F_NDMI, D_z_F_NDMI, F_MSI, V_ema_F_NDVI_kobs30, V_rollmax_F_NDMI_kobs14, V_rollmax_F_NDMI_kobs30, V_rollmax_F_NDVI_kobs14, V_rollmax_F_NDVI_kobs30, V_rollmin_F_NDMI_kobs30, V_rollmin_F_NDVI_kobs14, V_rollmin_F_NDVI_kobs30, V_rollmin_F_NDVI_kobs7, V_rollrng_F_NDVI_kobs14, V_rollrng_F_NDVI_kobs30, V_rollstd_F_NDVI_kobs30 |
| 3 | sar | 16 | A_d_E_SAR_ratio_kobs30, A_grad_E_SAR_diff_kobs30, C_lag_E_SAR_ratio_kobs30, D_z_E_SAR_ratio, E_SAR_ratio, E_rough_s1_vh_kobs14, V_ema_E_SAR_ratio_kobs30, V_rollmax_E_SAR_diff_kobs14, V_rollmax_E_SAR_diff_kobs30, V_rollmax_E_SAR_ratio_kobs30, V_rollmax_E_SAR_ratio_kobs7, V_rollmin_E_SAR_diff_kobs14, V_rollmin_E_SAR_diff_kobs30, V_rollmin_E_SAR_ratio_kobs30, V_rollrng_E_SAR_diff_kobs30, V_rollrng_E_SAR_ratio_kobs30 |
| 4 | thermal | 11 | A_d_LST_modis_kobs30, C_lag_LST_modis_kobs30, D_fft_dom_LST_modis_kobs30, D_fft_ent_LST_modis_kobs30, D_sa_LST_modis, D_z_LST_modis, V_ema_LST_modis_kobs30, V_rollmax_LST_modis_kobs14, V_rollmax_LST_modis_kobs30, V_rollmean_LST_modis_kobs30, V_rollmin_LST_modis_kobs30 |
| 5 | meteo | 17 | G_API, G_DSLR, G_rain_sum_30d, G_rain_sum_3d, G_rain_sum_7d, V_ema_G_API_kobs14, V_ema_G_API_kobs30, V_ema_G_API_kobs7, V_rollmax_G_API_kobs14, V_rollmax_G_API_kobs30, V_rollmax_G_API_kobs7, V_rollmean_G_API_kobs14, V_rollmean_G_API_kobs30, V_rollmin_G_API_kobs14, V_rollmin_G_API_kobs30, V_rollrng_G_API_kobs7, precip_mm |
| 6 | static | 15 | J_aspect_deg, J_bio_bio02, J_bio_bio03, J_bio_bio04, J_bio_bio06, J_bio_bio07, J_bio_bio13, J_bio_bio14, J_lc_code, J_soil_texture_usda_b0, aspect, elev, latitude, lia_mean_asc_deg, longitude |
| 7 | temporal | 7 | DOY, D_cos_DOY, D_sin_DOY, SMAP_x_year, cos_year, sin_year, year_frac |

## Early-Stopping Replay (patience-60 re-check, tag 21)

Offline replay of honest epoch-selection rules on the saved per-epoch curves (`analyze_stopping.py --tag 21`). 1.2/1.3 established that patience-60 is the best honest rule; the `swa_val` rule rows are replayed for completeness (no SWA configs run in 2.2 — the curves are the live ones). 2.2 re-checks patience-60 on the new grids, including the max_epochs {500, 600} probes (does extending the cap help the 96 small nets?).

### Stopping-rule aggregates (mean pooled test RMSE; lower is better; oracle = unreachable bound)
| family        | rule             |   mean_test_rmse |   median_test_rmse |   n |
|:--------------|:-----------------|-----------------:|-------------------:|----:|
| 2regime_96    | patience60       |        0.0531403 |          0.0532437 | 144 |
| 2regime_96    | patience20       |        0.0531403 |          0.0532437 | 144 |
| 2regime_96    | patience40       |        0.0531403 |          0.0532437 | 144 |
| 2regime_96    | val_aux          |        0.0556548 |          0.0556567 | 144 |
| 2regime_96    | swa_val          |        0.0531403 |          0.0532437 | 144 |
| 2regime_96    | plateau_w20e1e-4 |        0.0713855 |          0.0696428 | 144 |
| 2regime_96    | plateau_w40e1e-4 |        0.0639534 |          0.0600721 | 144 |
| 2regime_96    | plateau_w40e3e-4 |        0.0639534 |          0.0600721 | 144 |
| 2regime_96    | plateau_w60e1e-4 |        0.0559729 |          0.0527998 | 144 |
| 2regime_96    | oracle           |        0.0475741 |          0.0467573 | 144 |
| 2regime_54    | patience60       |        0.0486722 |          0.0486635 | 202 |
| 2regime_54    | patience20       |        0.0486722 |          0.0486635 | 202 |
| 2regime_54    | patience40       |        0.0486722 |          0.0486635 | 202 |
| 2regime_54    | val_aux          |        0.0490989 |          0.0490892 | 202 |
| 2regime_54    | swa_val          |        0.0486722 |          0.0486635 | 202 |
| 2regime_54    | plateau_w20e1e-4 |        0.0620983 |          0.061006  | 202 |
| 2regime_54    | plateau_w40e1e-4 |        0.0559576 |          0.0546964 | 202 |
| 2regime_54    | plateau_w40e3e-4 |        0.0559576 |          0.0546964 | 202 |
| 2regime_54    | plateau_w60e1e-4 |        0.0521962 |          0.0514149 | 202 |
| 2regime_54    | oracle           |        0.0469713 |          0.0469359 | 202 |
| 2regime_mixed | patience60       |        0.0495597 |          0.0488734 | 178 |
| 2regime_mixed | patience20       |        0.0495597 |          0.0488734 | 178 |
| 2regime_mixed | patience40       |        0.0495597 |          0.0488734 | 178 |
| 2regime_mixed | val_aux          |        0.0505185 |          0.0497882 | 178 |
| 2regime_mixed | swa_val          |        0.0495597 |          0.0488734 | 178 |
| 2regime_mixed | plateau_w20e1e-4 |        0.0558747 |          0.0537151 | 178 |
| 2regime_mixed | plateau_w40e1e-4 |        0.0512684 |          0.0505939 | 178 |
| 2regime_mixed | plateau_w40e3e-4 |        0.0512684 |          0.0505939 | 178 |
| 2regime_mixed | plateau_w60e1e-4 |        0.0502772 |          0.0498574 | 178 |
| 2regime_mixed | oracle           |        0.0455368 |          0.0455231 | 178 |

## Extrapolation (OOD) Check

588/6,620 test rows (8.9 %) are OOD on ≥1 top-10 gain feature (same definition as mlp-1.0–2.1). The pure-96 family keeps its OOD strength; the mixed family is in-distribution-strong but OOD-weak (its c1 = 54+10 half carries the 54-family's weak OOD). The 2.2 winners' OOD behavior is reported for the record (family allocation is pinned, so this is a tracking table, not a target).

### Extrapolation check (OOD test slices)
| model                                                   | slice           |    n |       r2 |      rmse |        bias |       mae |
|:--------------------------------------------------------|:----------------|-----:|---------:|----------:|------------:|----------:|
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3)               | all             | 6620 | 0.759483 | 0.0499584 |  0.0188537  | 0.0387226 |
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3)               | in_distribution | 6032 | 0.753545 | 0.0514519 |  0.0213449  | 0.0401715 |
| MLP 2regime-96 (w512x512x512_d0.3_lr1e-3)               | ood             |  588 | 0.757848 | 0.0306936 | -0.0067026  | 0.0238591 |
| MLP 2regime-96 (5-seed champ)                           | all             | 6620 | 0.75656  | 0.050261  |  0.0204007  | 0.0391776 |
| MLP 2regime-96 (5-seed champ)                           | in_distribution | 6032 | 0.749983 | 0.0518224 |  0.0227929  | 0.0407259 |
| MLP 2regime-96 (5-seed champ)                           | ood             |  588 | 0.770967 | 0.0298506 | -0.00414004 | 0.0232951 |
| MLP 2regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3) | all             | 6620 | 0.759564 | 0.04995   |  0.0106746  | 0.0382266 |
| MLP 2regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3) | in_distribution | 6032 | 0.756491 | 0.0511435 |  0.0131292  | 0.0391035 |
| MLP 2regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3) | ood             |  588 | 0.67684  | 0.0354578 | -0.0145052  | 0.0292309 |
| MLP 2regime-54 (5-seed champ)                           | all             | 6620 | 0.766788 | 0.0491939 |  0.0109373  | 0.0375111 |
| MLP 2regime-54 (5-seed champ)                           | in_distribution | 6032 | 0.764773 | 0.0502663 |  0.0137066  | 0.0382613 |
| MLP 2regime-54 (5-seed champ)                           | ood             |  588 | 0.659193 | 0.0364131 | -0.0174722  | 0.0298149 |
| MLP 2regime-mixed (w512x512x512_d0.3_huber0.03_lr1e-3)  | all             | 6620 | 0.78085  | 0.0476877 |  0.0140565  | 0.0364254 |
| MLP 2regime-mixed (w512x512x512_d0.3_huber0.03_lr1e-3)  | in_distribution | 6032 | 0.775347 | 0.0491234 |  0.0157311  | 0.0377422 |
| MLP 2regime-mixed (w512x512x512_d0.3_huber0.03_lr1e-3)  | ood             |  588 | 0.781981 | 0.029124  | -0.00312246 | 0.0229168 |
| MLP 2regime-mixed (5-seed champ)                        | all             | 6620 | 0.78165  | 0.0476006 |  0.011951   | 0.0362704 |
| MLP 2regime-mixed (5-seed champ)                        | in_distribution | 6032 | 0.777184 | 0.0489223 |  0.0141968  | 0.0374722 |
| MLP 2regime-mixed (5-seed champ)                        | ood             |  588 | 0.75399  | 0.0309371 | -0.0110874  | 0.0239423 |
| XGBoost Global (54)                                     | all             | 6620 | 0.77923  | 0.0478636 |  0.0105484  | 0.0370592 |
| XGBoost Global (54)                                     | in_distribution | 6032 | 0.780849 | 0.0485182 |  0.0145436  | 0.0373064 |
| XGBoost Global (54)                                     | ood             |  588 | 0.577509 | 0.0405427 | -0.0304369  | 0.0345237 |
| XGBoost 2-Regime (Winner)                               | all             | 6620 | 0.81496  | 0.0438196 |  0.00648567 | 0.0337195 |
| XGBoost 2-Regime (Winner)                               | in_distribution | 6032 | 0.81728  | 0.0443022 |  0.00971871 | 0.0338159 |
| XGBoost 2-Regime (Winner)                               | ood             |  588 | 0.618589 | 0.0385212 | -0.0266805  | 0.0327308 |

## Overfitting-Symptom Analysis

From the saved artifacts (no retraining), via `analyze_overfitting.py`: train-fit vs held-out gap (aux2020 = train-fit), capacity vs test transfer, and the per-epoch curve shape for each family's val winner. 2.1's winners show the familiar pattern (test min early, val flat, train-fit improving); the 2.2 mitigations in play are the denser multi-seed selection and (for the 96-family) the lr6e-4 / max_epochs small-net levers.

### Overfitting symptoms (analyze_overfitting.py)

#### 1. Train-fit vs held-out gap (median RMSE)
| family     |   aux2020 (train-fit) |   val |   test |   val/train ratio |
|:-----------|----------------------:|------:|-------:|------------------:|
| 2regime_96 | 0.0357 | 0.0517 | 0.0524 | 1.4x |
| 2regime_54 | 0.0228 | 0.0576 | 0.0476 | 2.5x |
| 2regime_mixed | 0.0217 | 0.0490 | 0.0479 | 2.3x |

#### 2. Capacity vs test transfer (median by n_params bucket)
| family     | capacity   |   n_configs |   med_val_rmse |   med_test_r2 |   med_test_bias |
|:-----------|:-----------|------------:|---------------:|--------------:|----------------:|
| 2regime_96 | <200k | 48 | 0.0519 | 0.7356 | 0.0240 |
| 2regime_96 | 200-500k | 10 | 0.0505 | 0.7269 | 0.0284 |
| 2regime_96 | 1M+ | 1 | 0.0485 | 0.7595 | 0.0189 |
| 2regime_54 | <200k | 22 | 0.0585 | 0.7870 | 0.0066 |
| 2regime_54 | 200-500k | 42 | 0.0576 | 0.7835 | 0.0088 |
| 2regime_54 | 500k-1M | 14 | 0.0556 | 0.7718 | 0.0092 |
| 2regime_54 | 1M+ | 4 | 0.0561 | 0.7685 | 0.0071 |
| 2regime_mixed | 200-500k | 6 | 0.0500 | 0.7445 | 0.0240 |
| 2regime_mixed | 500k-1M | 29 | 0.0499 | 0.7781 | 0.0151 |
| 2regime_mixed | 1M+ | 31 | 0.0485 | 0.7808 | 0.0141 |

#### 3. Per-epoch curve shape for the val winner (cluster-0 specialist)
| family     | config_id |   aux_ep100 |   aux_ep260 |   val_plateau |   test_min |   test_min_epoch |   test_at_best_val |   test_final |   test_rise_after_min |
|:-----------|:----------|------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|-------------:|----------------------:|
| 2regime_96 | w512x512x512_d0.3_lr1e-3 | 0.0262 | 0.0179 | 0.0531 | 0.0451 | 90 | 0.0491 | 0.0489 | 0.0037 |
| 2regime_54 | w448x448x448_d0.3_huber0.1_gelu_lr1e-3 | 0.0292 | 0.0166 | 0.0602 | 0.0488 | 98 | 0.0516 | 0.0492 | 0.0004 |
| 2regime_mixed | w512x512x512_d0.3_huber0.03_lr1e-3 | 0.0187 | 0.0206 | 0.0534 | 0.0464 | 84 | 0.0482 | 0.0504 | 0.0040 |

#### 4. Systematic bias on test (MLP vs XGBoost references)
MLP 2regime_96 median test bias: 0.0243
MLP 2regime_54 median test bias: 0.0084
MLP 2regime_mixed median test bias: 0.0149
XGBoost references (eval-1.1): 2-regime 0.0065, global 0.0105

The sweep is sized to spend ~1.25 h of the 2 h `gpu_debug` H100 wall allocation: 191 phase-1 + 201 phase-2 + 108 phase-3 + 8 champion job-seeds ≈ 508 jobs at 8 parallel workers (2.1's per-seed mean was 45 s at ~5.9 effective workers).

### Timing (H100 PCIe 80 GB, 8 parallel workers)
Total sweep wall time: 3774.5 s
Total training time (all jobs, GPU-seconds): 21888 s
Eval wall time: 8.5 s

Slowest jobs (3-seed config train_time_s):
  2regime_mixed/w512x512x512_d0.3_huber0.15_lr4e-4           163.3s  n_seeds=3
  2regime_54/w384x384x384_d0.3_huber0.1_gelu                 160.8s  n_seeds=3
  2regime_96/w128x128_d0.5_me500                             160.2s  n_seeds=2
  2regime_96/w128x128x128_d0.4_lr6e-4                        153.4s  n_seeds=3
  2regime_mixed/w512x512x512_d0.3_huber0.1                   152.9s  n_seeds=3

## Key Takeaways

1. **The val-selected winners are again below 2.0's 0.8003 — but the sweep
   found the strongest single MLP of the whole 1.0–2.2 series on test.**
   Honest 3-seed val winners: mixed `w512x512x512_d0.3_huber0.03_lr1e-3`
   → test 0.7809 (below 2.1's 0.7844), 54 `w448x448x448_d0.3_huber0.1_gelu_lr1e-3`
   → 0.7596 (below 2.1's 0.7713), 96 unchanged 0.7595; 2.2's mixed val top-5
   ensemble 0.7850 and cross-family 0.7885 both sit below 2.0's 0.8003 /
   0.7932. Meanwhile the **test-best single is the 54-family
   `w320x320_d0.4_huber0.2_gelu_lr6e-4` → 0.7973** (3 seeds, val rank 49/82!),
   with `w320x320_d0.4_gelu_lr6e-4` 0.7960 and mixed `w512x512x512_d0.3_huber0.1_gelu`
   0.7928 (the untested gelu-512³ cell) close behind. The 320²-hubergelu/lr6e-4
   cell is the new frontier — and it is invisible to the val selector.
2. **Val-year diagnostic (NEW): the official val split's 2022 half is the
   noise source for the 54/96 families.** Spearman(val-2021, test) = +0.747
   (96, p=1e-11) / +0.454 (54) while val-2022 = +0.106 / +0.133 (both ns) —
   the full-val mean dilutes the reliable 2021 signal. Selecting on val-2021
   only would pick the 54 `w320x320_d0.2_huber0.1_gelu_lr6e-4` (test 0.7810)
   over the full-val winner (0.7596). The mixed family is different: BOTH
   years are negatively correlated with test (-0.249 / -0.352) — its val-noise
   is structural (the c0 = 96-pool half), not a year artifact. Diagnostic
   only; the deployed rule is unchanged.
3. **54-family selection signal flipped positive: +0.582 at 3 seeds (2.1:
   −0.555)** — the denser seed coverage and the new pool fixed the direction,
   yet the 2→3-seed flip still moved the 54 winner to a worse-test config
   (0.7772 → 0.7596): per-config seed noise and the 3-layer-val-overfit
   pattern (the 54 val top-10 is dominated by 3-layer configs that fail on
   test) remain.
4. **Debias: 54 met (median 3.1 %), mixed improved (12.7 → 9.6 %) but still
   >5 %; 96 worsened (13.9 → 21.7 %).** The mid-lr {4e-4, 6e-4, 8e-4} small-net
   configs are more biased than the lr3e-4 anchor (w256x256_d0.5: 1.1 %), and
   the max_epochs {500, 600} probes did not help (me500 → 0.7682 vs the
   400-cap 0.7834). The 96 criterion remains unmet — capacity control without
   lr3e-4 does not debias this family.
5. **The deliverables that hold up:** the val-year diagnostic (the first
   structural explanation of the val-noise), the 54 `w320x320_d0.4_huber0.2_gelu_lr6e-4`
   test-best reference (0.7973), the 3/3 bit-identical anchors vs 2.1 (max|diff|
   = 0 on a different node), and a fully reproducible v9 sweep (191 configs,
   508 job-seeds, 63 min sweep, 1:06:53 total wall — inside the ~1.25 h target).

All numbers above are the stdout of this notebook; weights/checkpoints/test
predictions under `models/`; preprocessed tensors and per-job logs under
`artifacts/`; figures at the experiment root. See README.md for the full
reproducibility checklist and caveats.



## Reproducibility

```bash
cd notebooks/experiment/derived_8.4-eval-mlp-2.2
uv run --no-sync python make_configs.py            # deterministic -> config.yaml (committed)
uv run --no-sync python run_mlp_sweep.py --resume  # phases 1-3 (~65 min wall, 8 workers, H100)
uv run --no-sync python run_mlp_champion.py        # 5-seed champion ensembles (per-family top-N)
uv run --no-sync python run_mlp_eval.py            # leaderboard, per-regime, ensembles, figures
uv run --no-sync python compare_anchor_vs_2.1.py   # cross-version anchor bit-identity evidence
uv run --no-sync python analyze_bias.py            # bias^2/MSE diagnostic
uv run --no-sync python analyze_selection.py       # selection reliability (1/2/3-seed)
uv run --no-sync python analyze_val_years.py       # val-2021 vs val-2022 diagnostic (NEW)
uv run --no-sync python analyze_overfitting.py     # overfitting-symptom analysis
uv run --no-sync python analyze_extrapolation.py   # OOD check
uv run --no-sync python analyze_stopping.py --tag 21  # stopping-rule replay
cd notebooks && nb execute experiment/derived_8.4-eval-mlp-2.2/derived_8.4-eval-mlp-2.2.ipynb --uv
uv run --no-sync python generate_readme.py         # regenerate this README from the notebook
```

- The anchor-vs-2.1 offline comparison (one anchor per family — 54/
  `w512x512x512_d0.3_huber0.1`, mixed/`w512x512x512_d0.3_huber0.05_lr6e-4`,
  96/`w512x512x512_d0.3_lr1e-3`; seed 42, spec 0) compares the v9 val curves
  against the v8 ones over the overlapping epochs (max|diff| = 0 target) —
  reproducible via `compare_anchor_vs_2.1.py` →
  `artifacts/anchor_vs_21_comparison.json`.
- Configurations pinned in `config.yaml` (generated by `make_configs.py`); seeds
  {42, 7, 123} for the sweep, {42, 7, 123, 2024, 999} for the champion step;
  `data_version: 9`. No SWA configs (closed 2.1 negative).
- Artifacts: `models/`, `artifacts/`, `sweep_results.csv`, `metrics_summary.csv`,
  `per_regime_metrics_summary.csv`, `bias_summary.csv`, `ood_summary.csv`,
  `stopping_21_*.csv`, `selection_summary.csv`, `val_year_summary.csv`,
  `timing_log.json`, `artifacts/anchor_vs_21_comparison.json`, figures, and the
  report notebook. All numbers in this README come from the executed notebook.

## Caveats

- The XGBoost 2-regime reference (0.815) was itself test-selected in eval-1.1; all
  honest MLP claims use val-based selection. `test-best` rows are reporting only.
- The mixed family's c1 (54+10) half inherits the 54-family's weak OOD extrapolation;
  the pure-96 family remains the best OOD model.
- 2025 test coverage is partial for several stations; year-2025 numbers should be read
  with the same caution as 1.x/2.0/2.1.
- Val selection remains the bottleneck: the 54-family 3-seed val winner (0.7596) is
  a test loser while the test-best (0.7973) sits at val rank 49/82; the mixed family's
  val ranking is negatively correlated with test in both val years. The val-year
  diagnostic (val-2021 reliable, val-2022 noise for 54/96) is the actionable finding —
  the selection rule itself is unchanged (protocol).
- The 96-family median bias²/MSE worsened to 21.7 % (2.1: 13.9 %): the mid-lr
  {4e-4, 6e-4, 8e-4} small-net pool is more biased than the lr3e-4 anchor; the
  max_epochs {500, 600} probes did not help. The <5 % criterion remains unmet for
  96 and mixed (9.6 %).
- The champion step runs the per-family `sweep.champion_top_n` (mixed top-2 +
  54 top-1 + 96 top-1) × extra seeds {2024, 999}; `--top-n N` CLI overrides
  (uniform, 2.1 parity). See `docs/plans/20260810-mlp-2.2.md`.
