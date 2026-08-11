# Experiment: `derived_8.4-eval-mlp-2.3` — final frontier check for the mlp-2.0 architecture: 320²-hubergelu/lr6e-4 cell refinement + the mixed gelu 3-layer at low lr + the 96 lr3e-4 debiased pool (~1.75 h gpu_debug H100 wall)

## Objective

Follow-up to `derived_8.4-eval-mlp-2.2` (an honest negative on the
val-selected winners — below 2.0's 0.8003 — but the series' strongest single
MLP on test: the 54-family `w320x320_d0.4_huber0.2_gelu_lr6e-4` → 0.7973,
val rank 49/82, invisible to the val selector; the val-year diagnostic showed
val-2021 is the reliable proxy for 54/96 while val-2022 is noise; the 54
val top-10 was dominated by 3-layer configs that overfit val and fail on
test; the 96-family is only debiased in the lr3e-4 small-net region).
2.3 asks **"is the mlp-2.0 (2-regime) architecture still meaningful?"** — an
**optimization + further parameter sweep** of the 2.2 frontiers, temporal
protocol only (no LOSO, same honest protocol as 2.0/2.1/2.2), **sized to
spend ~1.75 h of the 2 h `gpu_debug` H100 wall allocation** (~681 job-seeds
at 8 workers ≈ 8.5 GPU-h; the allocation is otherwise wasted). NEW: the
**full 3-seed pool** — every config trains seeds {42, 7, 123} (2.2 gave only
the phase-3 top-M a 3rd seed), so the 3-seed mean val RMSE and the val-year
diagnostic now cover the entire pool.

All numbers below are the stdout of the executed report notebook
(`derived_8.4-eval-mlp-2.3.ipynb`). Weights/checkpoints/test predictions under
`models/`; preprocessed tensors and per-job logs under `artifacts/`; figures at the
experiment root.

## Verdict (TL;DR)

- **The mlp-2.0 architecture IS still meaningful — on test: the series' first
  single MLP above 2.0's 0.8003.** The mixed test-best
  `w384x384x384_d0.3_huber0.1_gelu_lr2e-4` → test R² **0.8014** (bias 0.0018)
  is the first MLP single config above 2.0's mixed val top-5 ensemble
  (0.8003); the 54 test-best `w320x320_d0.4_huber0.15_gelu_lr5e-4` → **0.7980**
  (new 54 record, bias 0.0020) and the 96 test-best `w256x256_d0.5_lr2e-4` →
  **0.7897** (new 96 record, NEGATIVE bias −0.0028) beat their families' 2.2
  records. `test-best` rows are reporting only (selection on test would be
  leakage) — but the frontier exists.
- **Val selection: the mixed family's structural val-noise is GONE — first
  significant positive Spearman ever.** Full-val +0.318 (p=0.031; 2.1: −0.309,
  2.2: −0.413), BOTH val years positive (+0.263 ns / +0.408 p=0.005), and the
  winner is stable under leave-one-val-year-out. The 2.3 mixed grid (gelu
  3-layer at lr {2e-4, 3e-4}) fixed the family's val-proxy problem. The mixed
  val winner is unchanged (`w512x512x512_d0.3_huber0.03_lr1e-3`, 0.7809) but
  mixed val top-5 avg rose to 0.7895 (2.2: 0.7850).
- **54-family val selection did NOT recover: −0.055 (2.2: +0.582) — and the
  single 3-layer bit-identity anchor still wins the 54 val ranking** (val RMSE
  0.0544 vs the best 2-layer 0.0556): the 3-layer val-overfit is structural,
  not pool-composition luck (test 0.7596). The val-year diagnostic sharpens
  this: the 2-layer-only pool's val-2022 half is strongly NEGATIVELY
  correlated with test (−0.472, p=2e-9). The 54 val top-10 avg still improved
  to 0.7834 (2.2: 0.7770) — the 2-layer ensembles are the real 54 gains.
- **96 val selection weakened (+0.157 ns vs +0.566)** — val-2021 stays strong
  (+0.566, p=7e-4) while val-2022 turned negative (−0.316, p=0.078); the
  big-net `w512x512x512_d0.3_lr1e-3` remains the val winner (stable under all
  selectors, test 0.7595) while the 96 test-best is the lr2e-4 small net
  (0.7897). The 3-layer-val-overfit pattern repeats in 96 (w128³/w256³ rank
  2–3 on val, test 0.738–0.741).
- **Debias: 54 met with the best margin yet (median bias²/MSE 1.25 %;
  w128x128_d0.3_gelu_lr4e-4 bias −1.6e-5); mixed improved to 6.2 % (2.2:
  9.6 %) but still >5 %; 96 improved to 13.5 % (2.2: 21.7 %) but unmet** —
  within 96, the d0.4 variants are the biased cells (0.20–0.25 share) while
  lr2e-4 goes negative-bias.
- **Budget:** sweep 5,287.5 s (88.1 min) wall / 8.8 GPU-h for 681 job-seeds;
  total job 1:31:47 inside the 2 h `gpu_debug` cap (target ~1.75 h);
  3/3 anchors bit-identical vs 2.2 (max|diff| = 0).

## What's new in 2.3

1. **Full 3-seed pool (the user's ask; the structural mitigation for 2.2's
   val-noise findings)** — phase-2/3 top-Ns are set to the deduped family
   sizes, so EVERY config trains seeds {42, 7, 123}. The 3-seed mean val
   RMSE becomes the honest signal for the entire pool, and the
   Spearman-by-depth + val-year diagnostics cover all configs at all three
   seed depths (2.2: only top-42/26/40 per family had 3 seeds). Selection
   rule unchanged (3-seed mean val RMSE on the full official val).
2. **The 54-family 320²-hubergelu/lr6e-4 frontier refinement** — the 2.2
   test-best cell (0.7973) is refined: huber δ {0.1, 0.15, 0.2, 0.25, 0.3} ×
   lr {4e-4, 5e-4, 7e-4, 8e-4, 1e-3} × d {0.3, 0.4, 0.5}; widths
   {256, 288, 352, 384} × huber × d; and the near-unbiased small-net region
   (128²–256² gelu, lr {3e-4, 4e-4, 5e-4, 6e-4} — w192x192_d0.3_gelu_lr4e-4
   hit 0.7934 with a bias²/MSE share of 0.02 % in 2.2).
3. **The mixed gelu 3-layer cell at low lr** — 2.2's mixed test-best was
   `w448x448x448_d0.3_huber0.1_gelu` (0.7940, only 2 seeds in 2.2!); 2.3
   grids δ {0.05, 0.1, 0.2} × lr {2e-4, 3e-4, 4e-4, 5e-4} at {384³, 448³,
   512³}, d {0.2, 0.4} probes, and the untested silu-512³/448³ lr3e-4 cells.
4. **The 96-family lr3e-4-only pool** — 2.2 showed the mid-lr {4e-4, 6e-4,
   8e-4}/huber/mixup/max_epochs variants are all worse AND more biased (96
   median bias²/MSE 21.7 %); the debiased region is lr3e-4 small nets
   (w256x256_d0.5: 1.1 %). 2.3's 96 grid stays at lr3e-4: widths
   {96..320} × d {0.4, 0.5, 0.6}, 3-layer probes, lr2e-4, me600 at 96² —
   and finally gives the 2.2 test-best w256x256_d0.5 (1 seed in 2.2!) its
   full 3-seed coverage.
5. **54-family 3-layer is a documented 2.2 negative** (the val-overfit trap)
   — not swept in 2.3; only the 2.2 54 val winner stays as the bit-identity
   anchor plus two d0.4 re-check probes at the huber0.2 cell.
6. **No training-path changes** — the mlp23 trainer is byte-identical to
   mlp22 (the val_preds.npy save already exists); anchors reproduce 2.2
   bit-identically (stack check via `compare_anchor_vs_2.2.py`).

Documented negatives honored (no GPU re-spent): no calibration, no trainval
retrain, patience-60 kept, aux2020 diagnostic-only, batch 512, no new
routers / station embeddings / feature selection, SWA (2.1 negative), fg/plr
(2.0 negatives), lr1e-4 (2.1 negative), 96 mid-lr/huber/mixup/me{500,600}
(2.2 negative), 2-layer mixed at lr {6e-4, 1e-3} (2.2 negative — probed at
lr3e-4/4e-4 only as a completeness check).

## Protocol (data_version 10, temporal only — same honest protocol as 2.2)

Train on the official train split (2017–2020, n=9,803); early-stop / select on the
official val split (2021–2022, n=4,805); evaluate on the untouched test split
(2023–2025, n=6,620). aux2020 (2020 slice of train, n=2,519) diagnostic only.
Winners selected by **3-seed mean val RMSE** among mlp/fg/plr (phase 1 = seed 42 for
all configs; phase 2 = seed 7 for ALL configs — full 3-seed pool; phase 3 = seed 123
for ALL configs). Patience-60; AdamW + warmup 5% + cosine; grad clip 1.0;
median-impute → StandardScaler → clip [−5, 5] fit on train only; target in original
units; `cudnn.deterministic=True`.

**data_version 10 (v9 → v10):** new sweep grids (section below) and the full
3-seed pool; the trainer is byte-identical to v9 (mlp23 = mlp22), so the
anchors' val curves stay bit-identical (stack check via
`compare_anchor_vs_2.2.py`).

**Cross-node bit-identity caveat:** v9 (2.2) reproduced v8 (2.1)'s anchor
curve bit-identically on a different node (offline comparison, max|diff| =
0); 2.3 re-checks the same way against 2.2. General cross-node bit-identity
is still not guaranteed (PTX-JIT/driver/cuDNN), but the observed
reproductions have been exact.

## Sweep design

223 phase-1 configs (all `mlp`), generated by `make_configs.py` from the
documented grids below; `config.yaml` is the committed output. See
`make_configs.py` for the full spec and the per-family id lists in
`config.yaml`.

| family | n phase-1 | grids (axes) | phase-2 top-N | phase-3 top-N |
|---|---:|---|---:|---:|
| `2regime_54` (320²-hubergelu/lr6e-4 frontier + small nets) | 145 | 320² δ × lr fine × d; 320² δ @ lr6e-4; 320² huber × lr1e-3; 320² mse lr × d fine; 320² d0.5; width × δ × d @ lr6e-4; width × δ0.2 × lr {4e-4, 8e-4}; width × mse × d0.4; small-net mse w × lr × d; small-net huber; silu probes; 3-layer re-checks; anchor | 145 | 145 |
| `2regime_mixed` (gelu 3-layer low-lr cell) | 46 | gelu 3-layer δ × lr @ {384³, 448³, 512³}; lr5e-4 probes; d probes @ 512³; gelu 2-layer low-lr negative check; silu 512³ δ × lr {2e-4, 3e-4}; silu 448³/384³ lr3e-4; anchor | 46 | 46 |
| `2regime_96` (lr3e-4 debiased pool) | 32 | width × d @ lr3e-4; 3-layer @ lr3e-4; lr2e-4 probes; me600 probes; big 2-layer @ lr3e-4; anchors | 32 | 32 |

Job count: 223 phase-1 + 223 phase-2 + 223 phase-3 + 12 champion = **681 job-seeds**.
Budget math (from 2.2 timing): per-seed mean 43 s; at 8 workers / ~5.9
effective ≈ 8.5 GPU-h ≈ **~85 min sweep ≈ ~1.6 h total wall** (target
~1.75 h; 2:00:00 partition hard cap). Resumable;
`--phase2-top-n` / `--phase3-top-n` / `--families` / `--only` trims keep the
session inside the 2 h wall cap (note: trims would break the full-3-seed
promise — prefer accepting a partial phase-3 tail + `--resume` follow-up).

## Selection Protocol v10 Diagnostic

Selection = multi-seed mean val RMSE among the honest architectures (mlp / fg / plr — 2.3 runs only mlp configs, so the pool is mlp-only in practice). 2.3 introduces the **full 3-seed pool**: phase-2/3 top-Ns are capped at the deduped family sizes, so EVERY config trains seeds {42, 7, 123}. This is the direct mitigation for 2.2's findings — the 54-family val ranking was noisy even at 3 seeds (Spearman(val, test) −0.555 in 2.1 → +0.582 in 2.2, but the 2→3-seed flip still moved the 54 winner to a worse-test config), the 54 val top-10 was dominated by 3-layer val-overfitters (removed from the 2.3 pool), and the 3-seed diagnostics previously covered only the phase-3 subset. aux2020 stays diagnostic-only (measures train fit). This section reports the val ranking, the Spearman correlations vs test at 1-/2-/3-seed aggregation over the FULL pool, and the phase-stability table from `analyze_selection.py`.

### Selection Protocol v10 Diagnostic (selection = multi-seed mean val RMSE; mlp/fg/plr pool)

#### 2-Regime-96 — top-10 by val RMSE
| config_id                | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:-------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w512x512x512_d0.3_lr1e-3 | mlp            |         3 |  0.0484652 |  0.0260292 |  0.759483 |   0.0499584 |   0.0188537 |
| w128x128x128_d0.4        | mlp            |         3 |  0.0513085 |  0.0372981 |  0.738032 |   0.0521387 |   0.0198161 |
| w256x256x256_d0.4        | mlp            |         3 |  0.0513375 |  0.0332079 |  0.741311 |   0.0518113 |   0.0228407 |
| w320x320_d0.4            | mlp            |         3 |  0.0518897 |  0.0375998 |  0.734852 |   0.0524542 |   0.02608   |
| w352x352_d0.5            | mlp            |         3 |  0.0520095 |  0.038937  |  0.755707 |   0.0503491 |   0.0215775 |
| w384x384_d0.5            | mlp            |         3 |  0.0521039 |  0.040366  |  0.764861 |   0.0493967 |   0.0186354 |
| w288x288_d0.4            | mlp            |         3 |  0.0523438 |  0.0379067 |  0.744757 |   0.0514651 |   0.023371  |
| w224x224_d0.4            | mlp            |         3 |  0.0526765 |  0.0405141 |  0.758012 |   0.050111  |   0.0187004 |
| w320x320_d0.6            | mlp            |         3 |  0.0527247 |  0.0426692 |  0.765928 |   0.0492845 |   0.0166944 |
| w96x96_d0.5_me600        | mlp            |         3 |  0.0528893 |  0.0380364 |  0.727164 |   0.0532092 |   0.0251972 |
  Spearman(val_rmse, test_r2) = +0.157 (p=0.392, n=32)
  val winner (honest) : w512x512x512_d0.3_lr1e-3 (test_r2=0.7595)
  test best (ref)     : w256x256_d0.5_lr2e-4 (test_r2=0.7897)

#### 2-Regime-54 — top-10 by val RMSE
| config_id                              | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:---------------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w448x448x448_d0.3_huber0.1_gelu_lr1e-3 | mlp            |         3 |  0.0544122 |  0.0213436 |  0.759564 |   0.04995   |   0.0106746 |
| w320x320_d0.3_huber0.1_gelu_lr8e-4     | mlp            |         3 |  0.05561   |  0.0207162 |  0.780476 |   0.0477284 |   0.0145637 |
| w320x320_d0.3_huber0.1_gelu_lr7e-4     | mlp            |         3 |  0.0557855 |  0.019243  |  0.782779 |   0.0474774 |   0.0136559 |
| w320x320_d0.3_huber0.1_gelu_lr1e-3     | mlp            |         3 |  0.055825  |  0.0200434 |  0.777579 |   0.0480423 |   0.0163324 |
| w320x320_d0.4_huber0.1_gelu_lr8e-4     | mlp            |         3 |  0.0559707 |  0.0203148 |  0.780282 |   0.0477495 |   0.0141917 |
| w320x320_d0.3_huber0.2_gelu_lr1e-3     | mlp            |         3 |  0.0559858 |  0.020419  |  0.782092 |   0.0475525 |   0.0138873 |
| w224x224_d0.3_huber0.1_gelu_lr8e-4     | mlp            |         3 |  0.0560242 |  0.0204446 |  0.782251 |   0.0475351 |   0.0101633 |
| w320x320_d0.4_huber0.1_gelu_lr7e-4     | mlp            |         3 |  0.0560419 |  0.021556  |  0.785305 |   0.0472005 |   0.0124189 |
| w384x384x384_d0.4_huber0.2_gelu_lr6e-4 | mlp            |         3 |  0.0560497 |  0.023581  |  0.767588 |   0.0491094 |   0.0110827 |
| w320x320_d0.3_huber0.15_gelu_lr8e-4    | mlp            |         3 |  0.056129  |  0.0200333 |  0.781633 |   0.0476025 |   0.0134109 |
  Spearman(val_rmse, test_r2) = -0.055 (p=0.510, n=145)
  val winner (honest) : w448x448x448_d0.3_huber0.1_gelu_lr1e-3 (test_r2=0.7596)
  test best (ref)     : w320x320_d0.4_huber0.15_gelu_lr5e-4 (test_r2=0.7980)

#### 2-Regime-Mixed — top-10 by val RMSE
| config_id                               | architecture   |   n_seeds |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |
|:----------------------------------------|:---------------|----------:|-----------:|-----------:|----------:|------------:|------------:|
| w512x512x512_d0.3_huber0.03_lr1e-3      | mlp            |         3 |  0.0477502 |  0.0202982 |  0.78085  |   0.0476877 |  0.0140565  |
| w512x512x512_d0.2_huber0.05_gelu        | mlp            |         3 |  0.0483729 |  0.020265  |  0.786059 |   0.0471176 |  0.00891077 |
| w512x512x512_d0.3_huber0.05_gelu_lr5e-4 | mlp            |         3 |  0.0484101 |  0.0202909 |  0.78659  |   0.047059  |  0.00848842 |
| w512x512x512_d0.3_huber0.03             | mlp            |         3 |  0.048411  |  0.0218556 |  0.789424 |   0.0467455 |  0.0118217  |
| w512x512x512_d0.3_huber0.05_lr5e-4      | mlp            |         3 |  0.0484269 |  0.0215119 |  0.784303 |   0.0473105 |  0.0149688  |
| w512x512x512_d0.3_huber0.05_gelu_lr4e-4 | mlp            |         3 |  0.0488974 |  0.0224191 |  0.784596 |   0.0472784 |  0.0116668  |
| w512x512x512_d0.3_huber0.08             | mlp            |         3 |  0.0490405 |  0.0235137 |  0.779052 |   0.047883  |  0.0162419  |
| w448x448x448_d0.3_huber0.03             | mlp            |         3 |  0.0490983 |  0.0221754 |  0.781621 |   0.0476037 |  0.0123493  |
| w512x512x512_d0.2_huber0.1_gelu         | mlp            |         3 |  0.0491787 |  0.0208873 |  0.778842 |   0.0479057 |  0.0122854  |
| w448x448x448_d0.3_huber0.05_gelu_lr4e-4 | mlp            |         3 |  0.0493193 |  0.0224976 |  0.787546 |   0.0469535 |  0.0101402  |
  Spearman(val_rmse, test_r2) = +0.318 (p=0.031, n=46)
  val winner (honest) : w512x512x512_d0.3_huber0.03_lr1e-3 (test_r2=0.7809)
  test best (ref)     : w384x384x384_d0.3_huber0.1_gelu_lr2e-4 (test_r2=0.8014)

### Selection-reliability summary (analyze_selection.py)
#### Spearman(val, test) by aggregation depth (full 3-seed pool)
| family        | aggregation       |   n_configs |   spearman_val_test |    p_value |   median_abs_delta_val |   mean_abs_delta_val |   config_id |   val_rmse |   test_r2 |
|:--------------|:------------------|------------:|--------------------:|-----------:|-----------------------:|---------------------:|------------:|-----------:|----------:|
| 2regime_96    | 1-seed (42)       |          32 |           0.27456   | 0.128323   |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_96    | 2-seed (42,7)     |          32 |           0.242669  | 0.180821   |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_96    | 3-seed (42,7,123) |          32 |           0.156525  | 0.392273   |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_54    | 1-seed (42)       |         145 |          -0.110735  | 0.184853   |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_54    | 2-seed (42,7)     |         145 |          -0.144973  | 0.0818936  |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_54    | 3-seed (42,7,123) |         145 |          -0.0552039 | 0.509579   |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_mixed | 1-seed (42)       |          46 |           0.435461  | 0.00248889 |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_mixed | 2-seed (42,7)     |          46 |           0.299167  | 0.0434114  |                    nan |                  nan |         nan |        nan |       nan |
| 2regime_mixed | 3-seed (42,7,123) |          46 |           0.318039  | 0.0312438  |                    nan |                  nan |         nan |        nan |       nan |

#### Phase stability — winner at each seed depth
| family        | aggregation              |   n_configs |   spearman_val_test |   p_value |   median_abs_delta_val |   mean_abs_delta_val | config_id                              |   val_rmse |   test_r2 |
|:--------------|:-------------------------|------------:|--------------------:|----------:|-----------------------:|---------------------:|:---------------------------------------|-----------:|----------:|
| 2regime_96    | winner|1-seed (42)       |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_lr1e-3               |  0.0478642 |  0.759483 |
| 2regime_96    | winner|2-seed (42,7)     |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_lr1e-3               |  0.0482834 |  0.759483 |
| 2regime_96    | winner|3-seed (42,7,123) |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_lr1e-3               |  0.0484652 |  0.759483 |
| 2regime_54    | winner|1-seed (42)       |         nan |                 nan |       nan |                    nan |                  nan | w320x320_d0.3_huber0.1_gelu_lr8e-4     |  0.0546592 |  0.780476 |
| 2regime_54    | winner|2-seed (42,7)     |         nan |                 nan |       nan |                    nan |                  nan | w448x448x448_d0.3_huber0.1_gelu_lr1e-3 |  0.05457   |  0.759564 |
| 2regime_54    | winner|3-seed (42,7,123) |         nan |                 nan |       nan |                    nan |                  nan | w448x448x448_d0.3_huber0.1_gelu_lr1e-3 |  0.0544122 |  0.759564 |
| 2regime_mixed | winner|1-seed (42)       |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_huber0.03_lr1e-3     |  0.0477661 |  0.78085  |
| 2regime_mixed | winner|2-seed (42,7)     |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_huber0.03_lr1e-3     |  0.0480794 |  0.78085  |
| 2regime_mixed | winner|3-seed (42,7,123) |         nan |                 nan |       nan |                    nan |                  nan | w512x512x512_d0.3_huber0.03_lr1e-3     |  0.0477502 |  0.78085  |

## Val-Year Selection Reliability (full 3-seed pool)

2.2's headline finding: the official val split's 2022 half is the noise source for the 54/96 families (Spearman(val-2021, test) = +0.747 (96) / +0.454 (54) vs val-2022 = +0.106 / +0.133, both ns), while the mixed family is negatively correlated with test in BOTH years (structural, the c0 = 96-pool half). This diagnostic splits the official val set (2021–2022) by YEAR and asks which val year is the better proxy for test, and whether the val-selected winner is stable under leave-one-val-year-out selection. It is made possible by `val_preds.npy` (best-val predictions per job, saved by the mlp23 trainer — byte-identical to mlp22) + `artifacts/val_meta.npz`. **NEW in 2.3: with the full 3-seed pool the diagnostic covers the ENTIRE config pool at 3-seed aggregation** (2.2's was restricted to the phase-3 subset, n=26–42/family). **Diagnostic only** — the deployed selection rule stays 3-seed mean val RMSE on the FULL official val (protocol unchanged). Backed by `analyze_val_years.py` (`val_year_summary.csv`).

### Val-year diagnostic (val-2021 vs val-2022; 3-seed mean val RMSE per config)

#### 2-Regime-96 — top-10 by full-val RMSE (with per-year val RMSE)
| config_id                |   n_seeds |   val_rmse |   val_2021_rmse |   val_2022_rmse |   test_r2 |
|:-------------------------|----------:|-----------:|----------------:|----------------:|----------:|
| w512x512x512_d0.3_lr1e-3 |         3 |  0.0484652 |       0.0403419 |       0.0557868 |  0.759483 |
| w128x128x128_d0.4        |         3 |  0.0513085 |       0.0415644 |       0.0599037 |  0.738032 |
| w256x256x256_d0.4        |         3 |  0.0513375 |       0.0414355 |       0.060066  |  0.741311 |
| w320x320_d0.4            |         3 |  0.0518897 |       0.0449803 |       0.0583087 |  0.734852 |
| w352x352_d0.5            |         3 |  0.0520095 |       0.0448905 |       0.0586049 |  0.755707 |
| w384x384_d0.5            |         3 |  0.0521039 |       0.0457668 |       0.0580459 |  0.764861 |
| w288x288_d0.4            |         3 |  0.0523438 |       0.0450731 |       0.059062  |  0.744757 |
| w224x224_d0.4            |         3 |  0.0526765 |       0.0460327 |       0.0588844 |  0.758012 |
| w320x320_d0.6            |         3 |  0.0527247 |       0.045282  |       0.0595936 |  0.765928 |
| w96x96_d0.5_me600        |         3 |  0.0528893 |       0.0436484 |       0.0611648 |  0.727164 |

#### 2-Regime-54 — top-10 by full-val RMSE (with per-year val RMSE)
| config_id                              |   n_seeds |   val_rmse |   val_2021_rmse |   val_2022_rmse |   test_r2 |
|:---------------------------------------|----------:|-----------:|----------------:|----------------:|----------:|
| w448x448x448_d0.3_huber0.1_gelu_lr1e-3 |         3 |  0.0544122 |       0.0451344 |       0.0627473 |  0.759564 |
| w320x320_d0.3_huber0.1_gelu_lr8e-4     |         3 |  0.05561   |       0.0454357 |       0.0646224 |  0.780476 |
| w320x320_d0.3_huber0.1_gelu_lr7e-4     |         3 |  0.0557855 |       0.0452592 |       0.0650782 |  0.782779 |
| w320x320_d0.3_huber0.1_gelu_lr1e-3     |         3 |  0.055825  |       0.0444269 |       0.0657599 |  0.777579 |
| w320x320_d0.4_huber0.1_gelu_lr8e-4     |         3 |  0.0559707 |       0.0455538 |       0.0651889 |  0.780282 |
| w320x320_d0.3_huber0.2_gelu_lr1e-3     |         3 |  0.0559858 |       0.0459212 |       0.0649277 |  0.782092 |
| w224x224_d0.3_huber0.1_gelu_lr8e-4     |         3 |  0.0560242 |       0.0476662 |       0.0636603 |  0.782251 |
| w320x320_d0.4_huber0.1_gelu_lr7e-4     |         3 |  0.0560419 |       0.0464128 |       0.0646844 |  0.785305 |
| w384x384x384_d0.4_huber0.2_gelu_lr6e-4 |         3 |  0.0560497 |       0.0477276 |       0.0636663 |  0.767588 |
| w320x320_d0.3_huber0.15_gelu_lr8e-4    |         3 |  0.056129  |       0.0461972 |       0.0649592 |  0.781633 |

#### 2-Regime-Mixed — top-10 by full-val RMSE (with per-year val RMSE)
| config_id                               |   n_seeds |   val_rmse |   val_2021_rmse |   val_2022_rmse |   test_r2 |
|:----------------------------------------|----------:|-----------:|----------------:|----------------:|----------:|
| w512x512x512_d0.3_huber0.03_lr1e-3      |         3 |  0.0477502 |       0.040137  |       0.054661  |  0.78085  |
| w512x512x512_d0.2_huber0.05_gelu        |         3 |  0.0483729 |       0.0410694 |       0.0550412 |  0.786059 |
| w512x512x512_d0.3_huber0.05_gelu_lr5e-4 |         3 |  0.0484101 |       0.0404276 |       0.0556016 |  0.78659  |
| w512x512x512_d0.3_huber0.03             |         3 |  0.048411  |       0.0409206 |       0.0552418 |  0.789424 |
| w512x512x512_d0.3_huber0.05_lr5e-4      |         3 |  0.0484269 |       0.0407704 |       0.0553716 |  0.784303 |
| w512x512x512_d0.3_huber0.05_gelu_lr4e-4 |         3 |  0.0488974 |       0.0407716 |       0.0562047 |  0.784596 |
| w512x512x512_d0.3_huber0.08             |         3 |  0.0490405 |       0.0409355 |       0.0563612 |  0.779052 |
| w448x448x448_d0.3_huber0.03             |         3 |  0.0490983 |       0.0420688 |       0.0555692 |  0.781621 |
| w512x512x512_d0.2_huber0.1_gelu         |         3 |  0.0491787 |       0.0413236 |       0.0562979 |  0.778842 |
| w448x448x448_d0.3_huber0.05_gelu_lr4e-4 |         3 |  0.0493193 |       0.041857  |       0.0561334 |  0.787546 |

### Spearman(val signal, test R2) per family (FULL pool, 3-seed aggregation)
| family        | signal        |   n_configs |   spearman |   p_value |
|:--------------|:--------------|------------:|-----------:|----------:|
| 2regime_96    | val_rmse      |          32 |      0.157 |    0.3923 |
| 2regime_96    | val_2021_rmse |          32 |      0.566 |    0.0007 |
| 2regime_96    | val_2022_rmse |          32 |     -0.316 |    0.0777 |
| 2regime_54    | val_rmse      |         145 |     -0.055 |    0.5096 |
| 2regime_54    | val_2021_rmse |         145 |      0.068 |    0.4182 |
| 2regime_54    | val_2022_rmse |         145 |     -0.472 |    0      |
| 2regime_mixed | val_rmse      |          46 |      0.318 |    0.0312 |
| 2regime_mixed | val_2021_rmse |          46 |      0.263 |    0.0771 |
| 2regime_mixed | val_2022_rmse |          46 |      0.408 |    0.0048 |

### Winner stability under leave-one-val-year-out selection (3-seed means)
| family        | selected_by   | winner                                 |   winner_test_r2 |
|:--------------|:--------------|:---------------------------------------|-----------------:|
| 2regime_96    | val_rmse      | w512x512x512_d0.3_lr1e-3               |           0.7595 |
| 2regime_96    | val_2021_rmse | w512x512x512_d0.3_lr1e-3               |           0.7595 |
| 2regime_96    | val_2022_rmse | w512x512x512_d0.3_lr1e-3               |           0.7595 |
| 2regime_54    | val_rmse      | w448x448x448_d0.3_huber0.1_gelu_lr1e-3 |           0.7596 |
| 2regime_54    | val_2021_rmse | w320x320_d0.3_huber0.1_gelu_lr1e-3     |           0.7776 |
| 2regime_54    | val_2022_rmse | w448x448x448_d0.3_huber0.1_gelu_lr1e-3 |           0.7596 |
| 2regime_mixed | val_rmse      | w512x512x512_d0.3_huber0.03_lr1e-3     |           0.7809 |
| 2regime_mixed | val_2021_rmse | w512x512x512_d0.3_huber0.03_lr1e-3     |           0.7809 |
| 2regime_mixed | val_2022_rmse | w512x512x512_d0.3_huber0.03_lr1e-3     |           0.7809 |

## Overall Model Leaderboard

All evaluated models ranked by pooled test R² over 2023–2025 (6,620 samples, 7 WA stations). MLP rows carry the sweep `config_id` and `n_seeds`; `(val top-k avg)` rows are offline seed-averaged ensembles of the top-k val-selected honest configs (no extra training); `(5-seed champ, ...)` rows are 5-seed champion ensembles of the val-selected winners (extra stability seeds, no trainval retrain — documented negative); `cross-family` rows average the val-selected winners across families. XGBoost rows are the eval-1.1 references; `MLP-1.3` / `MLP-2.0` / `MLP-2.1` / `MLP-2.2` rows are the previous experiments' val-selected winners + test-best references (2.0's mixed val top-5 ensemble 0.8003 is the number 2.3 must beat; 2.2's mixed val winner 0.7809, cross-family 0.7885, and the series-best single test row 0.7973 are the nearer bars); `test-best` rows are reporting only (selection on test would be leakage).

### Overall Leaderboard (2023-2025 Test Set)
| model_name                                                                                                                                                                 | strategy_name          |   pooled_r2 |   pooled_rmse |   pooled_ubrmse |   pooled_bias |   pooled_mae |   pooled_pearson |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------|------------:|--------------:|----------------:|--------------:|-------------:|-----------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                                                                                                                                 | XGBoost_Reference      |    0.81496  |     0.0438196 |       0.043337  |    0.00648567 |    0.0337195 |         0.905594 |
| MLP 2-Regime-Mixed (test-best, w384x384x384_d0.3_huber0.1_gelu_lr2e-4)                                                                                                     | MLP_testbest_reference |    0.80135  |     0.0454025 |       0.0453656 |    0.00182917 |    0.0344527 |         0.895533 |
| MLP-2.0 2-Regime-Mixed (val top-5 avg)                                                                                                                                     | MLP_2.0_Reference      |    0.800323 |     0.0455197 |       0.0434056 |    0.0137112  |    0.034711  |         0.90468  |
| MLP 2-Regime-Mixed (test-best, w384x384x384_d0.3_huber0.05_gelu_lr2e-4)                                                                                                    | MLP_testbest_reference |    0.799074 |     0.0456619 |       0.045163  |    0.00673168 |    0.0343176 |         0.896684 |
| MLP 2-Regime-Mixed (test-best, w512x512x512_d0.3_huber0.05_gelu_lr2e-4)                                                                                                    | MLP_testbest_reference |    0.798248 |     0.0457557 |       0.0451452 |    0.00744944 |    0.0346151 |         0.897498 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_huber0.15_gelu_lr5e-4)                                                                                                           | MLP_testbest_reference |    0.79802  |     0.0457815 |       0.0457386 |    0.00198307 |    0.0354808 |         0.893533 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_huber0.1_gelu_lr5e-4)                                                                                                            | MLP_testbest_reference |    0.797993 |     0.0457845 |       0.0455617 |    0.00451169 |    0.0354582 |         0.894551 |
| MLP 2-Regime-54 (test-best, w320x320_d0.3_huber0.2_gelu_lr5e-4)                                                                                                            | MLP_testbest_reference |    0.797526 |     0.0458374 |       0.0455935 |    0.0047227  |    0.0353179 |         0.894281 |
| MLP-2.2 2-Regime-54 (test_best: w320x320_d0.4_huber0.2_gelu_lr6e-4)                                                                                                        | MLP_2.2_Reference      |    0.797318 |     0.045861  |       0.0456567 |    0.00432342 |    0.0355598 |         0.893942 |
| MLP-2.2 2-Regime-Mixed (test_best: w448x448x448_d0.3_huber0.1_gelu)                                                                                                        | MLP_2.2_Reference      |    0.793991 |     0.0462358 |       0.0450644 |    0.0103416  |    0.0350157 |         0.897313 |
| MLP-2.1 2-Regime-Mixed (test_best: w448x448x448_d0.3_huber0.1_gelu)                                                                                                        | MLP_2.1_Reference      |    0.793991 |     0.0462358 |       0.0450644 |    0.0103416  |    0.0350157 |         0.897313 |
| MLP-2.1 2-Regime-54 (test_best: w320x320_d0.3_gelu_lr6e-4)                                                                                                                 | MLP_2.1_Reference      |    0.793502 |     0.0462908 |       0.0456942 |    0.00740775 |    0.0359189 |         0.893767 |
| MLP-2.0 cross-family (val winners)                                                                                                                                         | MLP_2.0_Reference      |    0.793243 |     0.0463198 |       0.0455726 |    0.00828602 |    0.035305  |         0.894921 |
| MLP-2.1 cross-family (val winners)                                                                                                                                         | MLP_2.1_Reference      |    0.793069 |     0.0463393 |       0.0445616 |    0.0127116  |    0.0352024 |         0.900164 |
| MLP-2.0 2-Regime-Mixed (test_best: w512x512x512_d0.3_huber0.1_swa)                                                                                                         | MLP_2.0_Reference      |    0.790253 |     0.0466534 |       0.0449702 |    0.0124186  |    0.0354233 |         0.898277 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5_lr2e-4)                                                                                                                          | MLP_testbest_reference |    0.78972  |     0.0467127 |       0.0466311 |   -0.00276097 |    0.0353476 |         0.890537 |
| MLP 2-Regime-Mixed (val top-5 avg)                                                                                                                                         | MLP_2regime_mixed      |    0.789465 |     0.046741  |       0.045266  |    0.0116492  |    0.0355006 |         0.896314 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03)                                                                                                                           | MLP_2regime_mixed      |    0.789424 |     0.0467455 |       0.045226  |    0.0118217  |    0.0355475 |         0.89698  |
| MLP-1.3 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        | MLP_1.3_Reference      |    0.788821 |     0.0468125 |       0.0467953 |    0.00126699 |    0.0362252 |         0.888558 |
| MLP-2.1 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_2.1_Reference      |    0.788526 |     0.0468451 |       0.0462892 |    0.0071954  |    0.0362901 |         0.891197 |
| MLP-2.2 cross-family (val winners)                                                                                                                                         | MLP_2.2_Reference      |    0.788487 |     0.0468495 |       0.0445399 |    0.0145283  |    0.0355868 |         0.900107 |
| MLP cross-family (val winners: 2regime_96/w512x512x512_d0.3_lr1e-3 + 2regime_54/w448x448x448_d0.3_huber0.1_gelu_lr1e-3 + 2regime_mixed/w512x512x512_d0.3_huber0.03_lr1e-3) | MLP_cross_family       |    0.788487 |     0.0468495 |       0.0445399 |    0.0145283  |    0.0355868 |         0.900107 |
| MLP 2-Regime-Mixed (val top-10 avg)                                                                                                                                        | MLP_2regime_mixed      |    0.788413 |     0.0468576 |       0.0452703 |    0.012093   |    0.0356354 |         0.896518 |
| MLP 2-Regime-Mixed (val top-3 avg)                                                                                                                                         | MLP_2regime_mixed      |    0.788361 |     0.0468634 |       0.0456754 |    0.0104852  |    0.0355345 |         0.89424  |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                                                                                                                                 | MLP_testbest_reference |    0.786949 |     0.0470194 |       0.0461698 |    0.00889795 |    0.0362064 |         0.892146 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_gelu_lr5e-4)                                                                                                               | MLP_2regime_mixed      |    0.78659  |     0.047059  |       0.0462872 |    0.00848842 |    0.0356326 |         0.891175 |
| MLP-2.0 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        | MLP_2.0_Reference      |    0.786493 |     0.0470697 |       0.046945  |    0.0034241  |    0.0364015 |         0.887498 |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.2_huber0.05_gelu)                                                                                                        | MLP_2regime_mixed      |    0.786306 |     0.0470904 |       0.0467351 |    0.00577356 |    0.0354361 |         0.889671 |
| MLP 2-Regime-Mixed (w512x512x512_d0.2_huber0.05_gelu)                                                                                                                      | MLP_2regime_mixed      |    0.786059 |     0.0471176 |       0.0462673 |    0.00891077 |    0.0357033 |         0.892097 |
| MLP-2.0 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_2.0_Reference      |    0.785392 |     0.047191  |       0.0461896 |    0.00967028 |    0.0364359 |         0.892043 |
| MLP-2.2 2-Regime-Mixed (val top-5 avg)                                                                                                                                     | MLP_2.2_Reference      |    0.784973 |     0.047237  |       0.0456182 |    0.0122606  |    0.0359353 |         0.894566 |
| MLP-2.1 2-Regime-Mixed (val_sel: w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                       | MLP_2.1_Reference      |    0.7844   |     0.0472999 |       0.0456662 |    0.0123237  |    0.0359433 |         0.89492  |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr5e-4)                                                                                                                    | MLP_2regime_mixed      |    0.784303 |     0.0473105 |       0.0448801 |    0.0149688  |    0.0362003 |         0.898246 |
| MLP-2.1 2-Regime-Mixed (val top-3 avg)                                                                                                                                     | MLP_2.1_Reference      |    0.783877 |     0.0473572 |       0.0455158 |    0.0130774  |    0.0360972 |         0.894986 |
| MLP 2-Regime-54 (val top-10 avg)                                                                                                                                           | MLP_2regime_54         |    0.78341  |     0.0474084 |       0.0455803 |    0.0130381  |    0.0367489 |         0.894692 |
| MLP-2.1 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_2.1_Reference      |    0.783404 |     0.047409  |       0.0471517 |    0.00493258 |    0.0360624 |         0.887208 |
| MLP-2.2 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_2.2_Reference      |    0.783404 |     0.047409  |       0.0471517 |    0.00493258 |    0.0360624 |         0.887208 |
| MLP-1.3 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             | MLP_1.3_Reference      |    0.783404 |     0.047409  |       0.0471517 |    0.00493258 |    0.0360624 |         0.887208 |
| MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr7e-4)                                                                                                                       | MLP_2regime_54         |    0.782779 |     0.0474774 |       0.0454711 |    0.0136559  |    0.036974  |         0.894869 |
| MLP 2-Regime-54 (5-seed champ, w320x320_d0.3_huber0.1_gelu_lr7e-4)                                                                                                         | MLP_2regime_54         |    0.782733 |     0.0474824 |       0.0451309 |    0.0147574  |    0.0369749 |         0.896505 |
| MLP-1.3 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_1.3_Reference      |    0.782533 |     0.0475043 |       0.0470162 |    0.0067925  |    0.0369583 |         0.888139 |
| MLP 2-Regime-96 (test-best, w192x192_d0.5_lr2e-4)                                                                                                                          | MLP_testbest_reference |    0.782428 |     0.0475157 |       0.046627  |    0.00914698 |    0.036741  |         0.891355 |
| MLP-2.2 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr6e-4)                                                                                                  | MLP_2.2_Reference      |    0.781829 |     0.047581  |       0.0461576 |    0.0115514  |    0.036438  |         0.891699 |
| MLP-2.1 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                  | MLP_2.1_Reference      |    0.781771 |     0.0475874 |       0.0459221 |    0.0124787  |    0.0363492 |         0.89312  |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                      | MLP_2regime_mixed      |    0.78165  |     0.0476006 |       0.0460759 |    0.011951   |    0.0362704 |         0.891943 |
| MLP 2-Regime-54 (val top-5 avg)                                                                                                                                            | MLP_2regime_54         |    0.781169 |     0.047653  |       0.0455857 |    0.0138837  |    0.0369125 |         0.89473  |
| MLP-2.2 2-Regime-Mixed (val_sel: w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                       | MLP_2.2_Reference      |    0.78085  |     0.0476877 |       0.045569  |    0.0140565  |    0.0364254 |         0.894551 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                                    | MLP_2regime_mixed      |    0.78085  |     0.0476877 |       0.045569  |    0.0140565  |    0.0364254 |         0.894551 |
| MLP 2-Regime-54 (val top-3 avg)                                                                                                                                            | MLP_2regime_54         |    0.780538 |     0.0477216 |       0.0459268 |    0.0129648  |    0.0367954 |         0.893544 |
| MLP-2.0 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_2.0_Reference      |    0.780505 |     0.0477252 |       0.0464651 |    0.0108946  |    0.0372804 |         0.889914 |
| MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr8e-4)                                                                                                                       | MLP_2regime_54         |    0.780476 |     0.0477284 |       0.0454522 |    0.0145637  |    0.0370385 |         0.895092 |
| MLP 2-Regime-54 (w320x320_d0.4_huber0.1_gelu_lr8e-4)                                                                                                                       | MLP_2regime_54         |    0.780282 |     0.0477495 |       0.0455918 |    0.0141917  |    0.0372159 |         0.89432  |
| MLP 2-Regime-54 (5-seed champ, w320x320_d0.3_huber0.1_gelu_lr8e-4)                                                                                                         | MLP_2regime_54         |    0.780117 |     0.0477674 |       0.045064  |    0.0158416  |    0.0370607 |         0.896959 |
| Global Single Model (54 Backbone)                                                                                                                                          | XGBoost_Reference      |    0.77923  |     0.0478636 |       0.0466868 |    0.0105484  |    0.0370592 |         0.889432 |
| MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                       | MLP_2regime_54         |    0.777579 |     0.0480423 |       0.0451809 |    0.0163324  |    0.0375872 |         0.896343 |
| MLP-2.2 2-Regime-54 (val top-10 avg)                                                                                                                                       | MLP_2.2_Reference      |    0.777    |     0.0481048 |       0.0471161 |    0.00970258 |    0.0368603 |         0.889244 |
| MLP-1.3 2-Regime-96 (val top-10 avg)                                                                                                                                       | MLP_1.3_Reference      |    0.772329 |     0.048606  |       0.0446198 |    0.0192772  |    0.0375105 |         0.899578 |
| MLP-2.1 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                             | MLP_2.1_Reference      |    0.771661 |     0.0486772 |       0.0480595 |    0.00772988 |    0.0380031 |         0.886477 |
| MLP-2.2 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                             | MLP_2.2_Reference      |    0.771661 |     0.0486772 |       0.0480595 |    0.00772988 |    0.0380031 |         0.886477 |
| MLP-2.1 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  | MLP_2.1_Reference      |    0.771284 |     0.0487174 |       0.048218  |    0.00695755 |    0.0380165 |         0.886103 |
| MLP-2.0 2-Regime-96 (val top-10 avg)                                                                                                                                       | MLP_2.0_Reference      |    0.769168 |     0.0489422 |       0.0450254 |    0.0191846  |    0.0379532 |         0.897463 |
| MLP 2-Regime-54 (5-seed champ, w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                     | MLP_2regime_54         |    0.766788 |     0.0491939 |       0.0479627 |    0.0109373  |    0.0375111 |         0.886572 |
| MLP-1.3 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  | MLP_1.3_Reference      |    0.76511  |     0.0493706 |       0.0488979 |    0.00681535 |    0.0385003 |         0.882441 |
| MLP-2.0 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  | MLP_2.0_Reference      |    0.76511  |     0.0493706 |       0.0488979 |    0.00681535 |    0.0385003 |         0.882441 |
| MLP 2-Regime-96 (val top-3 avg)                                                                                                                                            | MLP_2regime_96         |    0.761378 |     0.0497613 |       0.0453408 |    0.0205035  |    0.0385944 |         0.895528 |
| MLP-1.3 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_1.3_Reference      |    0.761018 |     0.0497987 |       0.0465842 |    0.0176019  |    0.0384117 |         0.890751 |
| MLP-2.0 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_2.0_Reference      |    0.761018 |     0.0497987 |       0.0465842 |    0.0176019  |    0.0384117 |         0.890751 |
| MLP 2-Regime-96 (val top-5 avg)                                                                                                                                            | MLP_2regime_96         |    0.760726 |     0.0498292 |       0.0447911 |    0.0218336  |    0.0390424 |         0.898255 |
| MLP 2-Regime-96 (val top-10 avg)                                                                                                                                           | MLP_2regime_96         |    0.760333 |     0.0498701 |       0.0451506 |    0.0211766  |    0.0392596 |         0.896745 |
| MLP-2.0 2-Regime-Mixed (val_sel: fg_w512x512_d0.3_huber0.1_swa)                                                                                                            | MLP_2.0_Reference      |    0.759881 |     0.049917  |       0.0499151 |    0.00044081 |    0.038269  |         0.872243 |
| MLP-2.2 2-Regime-54 (val_sel: w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                      | MLP_2.2_Reference      |    0.759564 |     0.04995   |       0.0487961 |    0.0106746  |    0.0382266 |         0.884994 |
| MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   | MLP_2regime_54         |    0.759564 |     0.04995   |       0.0487961 |    0.0106746  |    0.0382266 |         0.884994 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                                                                                                                                 | MLP_2regime_96         |    0.759483 |     0.0499584 |       0.0462642 |    0.0188537  |    0.0387226 |         0.891821 |
| MLP-2.1 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_2.1_Reference      |    0.759483 |     0.0499584 |       0.0462642 |    0.0188537  |    0.0387226 |         0.891821 |
| MLP-2.2 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    | MLP_2.2_Reference      |    0.759483 |     0.0499584 |       0.0462642 |    0.0188537  |    0.0387226 |         0.891821 |
| MLP 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                                   | MLP_2regime_96         |    0.75656  |     0.050261  |       0.0459346 |    0.0204007  |    0.0391776 |         0.892873 |
| MLP-2.2 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                               | MLP_2.2_Reference      |    0.75656  |     0.050261  |       0.0459346 |    0.0204007  |    0.0391776 |         0.892873 |
| MLP-2.1 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                               | MLP_2.1_Reference      |    0.75656  |     0.050261  |       0.0459346 |    0.0204007  |    0.0391776 |         0.892873 |
| MLP 2-Regime-96 (w352x352_d0.5)                                                                                                                                            | MLP_2regime_96         |    0.755707 |     0.0503491 |       0.0454911 |    0.0215775  |    0.0399151 |         0.89495  |
| MLP-2.1 2-Regime-96 (val top-3 avg)                                                                                                                                        | MLP_2.1_Reference      |    0.753121 |     0.0506148 |       0.0448707 |    0.0234197  |    0.0400017 |         0.897911 |
| MLP-2.2 2-Regime-96 (val top-3 avg)                                                                                                                                        | MLP_2.2_Reference      |    0.744187 |     0.0515225 |       0.0446598 |    0.025692   |    0.0409332 |         0.898879 |
| MLP 2-Regime-96 (w256x256x256_d0.4)                                                                                                                                        | MLP_2regime_96         |    0.741311 |     0.0518113 |       0.046505  |    0.0228407  |    0.0406736 |         0.889761 |
| MLP 2-Regime-96 (w128x128x128_d0.4)                                                                                                                                        | MLP_2regime_96         |    0.738032 |     0.0521387 |       0.0482262 |    0.0198161  |    0.0403889 |         0.880891 |
| MLP 2-Regime-96 (w320x320_d0.4)                                                                                                                                            | MLP_2regime_96         |    0.734852 |     0.0524542 |       0.0455112 |    0.02608    |    0.0418635 |         0.894667 |

## Hyperparameter Sweep Summary

223 curated phase-1 configs (all `mlp` — `fg`/`plr` are documented negatives from 2.0, `swa` a documented negative from 2.1, 54-family 3-layer a documented 2.2 negative, and none get GPU except the 3 bit-identity anchors + 2 re-check probes) generated deterministically by `make_configs.py` from 2-factor grids around the 2.2 frontiers: 54 320²-hubergelu/lr6e-4 δ × lr × d fine / width × δ × d / small-net mse + huber / silu probes; mixed gelu 3-layer δ × lr @ {384³, 448³, 512³} / d probes / silu lr3e-4 cells; 96 lr3e-4-only width × d / 3-layer / lr2e-4 / me600 probes. 8 parallel H100 workers; **every config trains seeds {42, 7, 123} (full 3-seed pool)**; configs ranked by **multi-seed mean val RMSE** (honest signal); test R² for reference.

### Sweep Top-10 — 2-Regime-96 (by val RMSE, the honest selection signal)
| config_id                | architecture   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |   best_epoch |   train_time_s | deployed   |
|:-------------------------|:---------------|----------:|----------:|-------:|:-------|-----------:|-----------:|----------:|------------:|------------:|-------------:|---------------:|:-----------|
| w512x512x512_d0.3_lr1e-3 | mlp            |         3 |       0.3 | 0.001  | mse    |  0.0484652 |  0.0260292 |  0.759483 |   0.0499584 |   0.0188537 |          263 |        130.289 | live       |
| w128x128x128_d0.4        | mlp            |         3 |       0.4 | 0.0003 | mse    |  0.0513085 |  0.0372981 |  0.738032 |   0.0521387 |   0.0198161 |          390 |        202.054 | live       |
| w256x256x256_d0.4        | mlp            |         3 |       0.4 | 0.0003 | mse    |  0.0513375 |  0.0332079 |  0.741311 |   0.0518113 |   0.0228407 |          300 |        183.179 | live       |
| w320x320_d0.4            | mlp            |         3 |       0.4 | 0.0003 | mse    |  0.0518897 |  0.0375998 |  0.734852 |   0.0524542 |   0.02608   |          351 |        176.439 | live       |
| w352x352_d0.5            | mlp            |         3 |       0.5 | 0.0003 | mse    |  0.0520095 |  0.038937  |  0.755707 |   0.0503491 |   0.0215775 |          375 |        194.14  | live       |
| w384x384_d0.5            | mlp            |         3 |       0.5 | 0.0003 | mse    |  0.0521039 |  0.040366  |  0.764861 |   0.0493967 |   0.0186354 |          300 |        168.591 | live       |
| w288x288_d0.4            | mlp            |         3 |       0.4 | 0.0003 | mse    |  0.0523438 |  0.0379067 |  0.744757 |   0.0514651 |   0.023371  |          372 |        189.006 | live       |
| w224x224_d0.4            | mlp            |         3 |       0.4 | 0.0003 | mse    |  0.0526765 |  0.0405141 |  0.758012 |   0.050111  |   0.0187004 |          373 |        183.943 | live       |
| w320x320_d0.6            | mlp            |         3 |       0.6 | 0.0003 | mse    |  0.0527247 |  0.0426692 |  0.765928 |   0.0492845 |   0.0166944 |          363 |        180.865 | live       |
| w96x96_d0.5_me600        | mlp            |         3 |       0.5 | 0.0003 | mse    |  0.0528893 |  0.0380364 |  0.727164 |   0.0532092 |   0.0251972 |          582 |        260.4   | live       |

### Sweep Top-10 — 2-Regime-54 (by val RMSE, the honest selection signal)
| config_id                              | architecture   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |   best_epoch |   train_time_s | deployed   |
|:---------------------------------------|:---------------|----------:|----------:|-------:|:-------|-----------:|-----------:|----------:|------------:|------------:|-------------:|---------------:|:-----------|
| w448x448x448_d0.3_huber0.1_gelu_lr1e-3 | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0544122 |  0.0213436 |  0.759564 |   0.04995   |   0.0106746 |          187 |        92.7435 | live       |
| w320x320_d0.3_huber0.1_gelu_lr8e-4     | mlp            |         3 |       0.3 | 0.0008 | huber  |  0.05561   |  0.0207162 |  0.780476 |   0.0477284 |   0.0145637 |          187 |        95.4127 | live       |
| w320x320_d0.3_huber0.1_gelu_lr7e-4     | mlp            |         3 |       0.3 | 0.0007 | huber  |  0.0557855 |  0.019243  |  0.782779 |   0.0474774 |   0.0136559 |          218 |       115.243  | live       |
| w320x320_d0.3_huber0.1_gelu_lr1e-3     | mlp            |         3 |       0.3 | 0.001  | huber  |  0.055825  |  0.0200434 |  0.777579 |   0.0480423 |   0.0163324 |          180 |        88.0438 | live       |
| w320x320_d0.4_huber0.1_gelu_lr8e-4     | mlp            |         3 |       0.4 | 0.0008 | huber  |  0.0559707 |  0.0203148 |  0.780282 |   0.0477495 |   0.0141917 |          214 |       118.713  | live       |
| w320x320_d0.3_huber0.2_gelu_lr1e-3     | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0559858 |  0.020419  |  0.782092 |   0.0475525 |   0.0138873 |          218 |       102.168  | live       |
| w224x224_d0.3_huber0.1_gelu_lr8e-4     | mlp            |         3 |       0.3 | 0.0008 | huber  |  0.0560242 |  0.0204446 |  0.782251 |   0.0475351 |   0.0101633 |          246 |       103.035  | live       |
| w320x320_d0.4_huber0.1_gelu_lr7e-4     | mlp            |         3 |       0.4 | 0.0007 | huber  |  0.0560419 |  0.021556  |  0.785305 |   0.0472005 |   0.0124189 |          218 |       115.74   | live       |
| w384x384x384_d0.4_huber0.2_gelu_lr6e-4 | mlp            |         3 |       0.4 | 0.0006 | huber  |  0.0560497 |  0.023581  |  0.767588 |   0.0491094 |   0.0110827 |          272 |       137.919  | live       |
| w320x320_d0.3_huber0.15_gelu_lr8e-4    | mlp            |         3 |       0.3 | 0.0008 | huber  |  0.056129  |  0.0200333 |  0.781633 |   0.0476025 |   0.0134109 |          218 |       109.208  | live       |

### Sweep Top-10 — 2-Regime-Mixed (by val RMSE, the honest selection signal)
| config_id                               | architecture   |   n_seeds |   dropout |     lr | loss   |   val_rmse |   aux_rmse |   test_r2 |   test_rmse |   test_bias |   best_epoch |   train_time_s | deployed   |
|:----------------------------------------|:---------------|----------:|----------:|-------:|:-------|-----------:|-----------:|----------:|------------:|------------:|-------------:|---------------:|:-----------|
| w512x512x512_d0.3_huber0.03_lr1e-3      | mlp            |         3 |       0.3 | 0.001  | huber  |  0.0477502 |  0.0202982 |  0.78085  |   0.0476877 |  0.0140565  |          126 |        83.0465 | live       |
| w512x512x512_d0.2_huber0.05_gelu        | mlp            |         3 |       0.2 | 0.0003 | huber  |  0.0483729 |  0.020265  |  0.786059 |   0.0471176 |  0.00891077 |          228 |       110.965  | live       |
| w512x512x512_d0.3_huber0.05_gelu_lr5e-4 | mlp            |         3 |       0.3 | 0.0005 | huber  |  0.0484101 |  0.0202909 |  0.78659  |   0.047059  |  0.00848842 |          208 |       114.224  | live       |
| w512x512x512_d0.3_huber0.03             | mlp            |         3 |       0.3 | 0.0003 | huber  |  0.048411  |  0.0218556 |  0.789424 |   0.0467455 |  0.0118217  |          260 |       141.768  | live       |
| w512x512x512_d0.3_huber0.05_lr5e-4      | mlp            |         3 |       0.3 | 0.0005 | huber  |  0.0484269 |  0.0215119 |  0.784303 |   0.0473105 |  0.0149688  |          178 |       105.154  | live       |
| w512x512x512_d0.3_huber0.05_gelu_lr4e-4 | mlp            |         3 |       0.3 | 0.0004 | huber  |  0.0488974 |  0.0224191 |  0.784596 |   0.0472784 |  0.0116668  |          184 |       118.744  | live       |
| w512x512x512_d0.3_huber0.08             | mlp            |         3 |       0.3 | 0.0003 | huber  |  0.0490405 |  0.0235137 |  0.779052 |   0.047883  |  0.0162419  |          260 |       153.743  | live       |
| w448x448x448_d0.3_huber0.03             | mlp            |         3 |       0.3 | 0.0003 | huber  |  0.0490983 |  0.0221754 |  0.781621 |   0.0476037 |  0.0123493  |          345 |       167.483  | live       |
| w512x512x512_d0.2_huber0.1_gelu         | mlp            |         3 |       0.2 | 0.0003 | huber  |  0.0491787 |  0.0208873 |  0.778842 |   0.0479057 |  0.0122854  |          209 |       113.141  | live       |
| w448x448x448_d0.3_huber0.05_gelu_lr4e-4 | mlp            |         3 |       0.3 | 0.0004 | huber  |  0.0493193 |  0.0224976 |  0.787546 |   0.0469535 |  0.0101402  |          260 |       132.34   | live       |

## Per-Regime Performance Breakdown

Cluster 0 holds 73 % of the test rows, so it dominates the pooled R². Per-cluster test metrics for the val top-3 honest configs per family (the mixed family's c1 = 54+10 specialist is expected to hold the ~0.83 R² of the 54-family's c1 while c0 gains the 96-pool fit), the XGBoost references, and the 1.3 / 2.0 / 2.1 / 2.2 reference winners.

### Per-Regime Performance Breakdown
| strategy_name     | model_name                                                   |   cluster |   n_train |   n_test |       r2 |      rmse |    ubrmse |         bias |       mae |
|:------------------|:-------------------------------------------------------------|----------:|----------:|---------:|---------:|----------:|----------:|-------------:|----------:|
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         0 |      7156 |     4817 | 0.751308 | 0.0498896 | 0.0471617 |  0.0162712   | 0.0390899 |
| MLP_2regime_96    | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         1 |      2647 |     1803 | 0.77822  | 0.0501416 | 0.0430228 |  0.0257531   | 0.0377413 |
| MLP_2regime_96    | MLP 2-Regime-96 (w128x128x128_d0.4)                          |         0 |      7156 |     4817 | 0.706123 | 0.0542328 | 0.0500799 |  0.0208134   | 0.0426266 |
| MLP_2regime_96    | MLP 2-Regime-96 (w128x128x128_d0.4)                          |         1 |      2647 |     1803 | 0.812698 | 0.0460795 | 0.0427685 |  0.0171515   | 0.0344104 |
| MLP_2regime_96    | MLP 2-Regime-96 (w256x256x256_d0.4)                          |         0 |      7156 |     4817 | 0.722746 | 0.0526766 | 0.0476607 |  0.0224342   | 0.0413775 |
| MLP_2regime_96    | MLP 2-Regime-96 (w256x256x256_d0.4)                          |         1 |      2647 |     1803 | 0.784512 | 0.0494252 | 0.0432476 |  0.0239268   | 0.0387931 |
| MLP_2regime_54    | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)     |         0 |      7156 |     4817 | 0.739603 | 0.0510502 | 0.0501782 |  0.00939558  | 0.0401103 |
| MLP_2regime_54    | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)     |         1 |      2647 |     1803 | 0.8061   | 0.0468841 | 0.0447162 |  0.0140919   | 0.0331941 |
| MLP_2regime_54    | MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr8e-4)         |         0 |      7156 |     4817 | 0.758635 | 0.0491492 | 0.0469066 |  0.014677    | 0.0392832 |
| MLP_2regime_54    | MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr8e-4)         |         1 |      2647 |     1803 | 0.831491 | 0.0437068 | 0.0413147 |  0.0142611   | 0.0310414 |
| MLP_2regime_54    | MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr7e-4)         |         0 |      7156 |     4817 | 0.761049 | 0.0489028 | 0.0469738 |  0.0135997   | 0.0391794 |
| MLP_2regime_54    | MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr7e-4)         |         1 |      2647 |     1803 | 0.833539 | 0.0434403 | 0.0411879 |  0.0138063   | 0.031082  |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)      |         0 |      7156 |     4817 | 0.759582 | 0.0490527 | 0.0470685 |  0.0138103   | 0.0386101 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)      |         1 |      2647 |     1803 | 0.830518 | 0.0438328 | 0.0412892 |  0.0147143   | 0.0305884 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.2_huber0.05_gelu)        |         0 |      7156 |     4817 | 0.770437 | 0.0479325 | 0.0472237 |  0.00821291  | 0.0375094 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.2_huber0.05_gelu)        |         1 |      2647 |     1803 | 0.82242  | 0.0448677 | 0.0435546 |  0.0107752   | 0.0308782 |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_gelu_lr5e-4) |         0 |      7156 |     4817 | 0.772384 | 0.0477289 | 0.0471255 |  0.00756496  | 0.037041  |
| MLP_2regime_mixed | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_gelu_lr5e-4) |         1 |      2647 |     1803 | 0.819614 | 0.0452209 | 0.0438737 |  0.0109556   | 0.0318698 |
| XGBoost_Reference | Global Single Model (54 Backbone)                            |         0 |     14608 |     6620 | 0.77923  | 0.0478636 | 0.0466868 |  0.0105484   | 0.0370592 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)                   |         0 |     10624 |     4817 | 0.80246  | 0.0444639 | 0.0436213 |  0.00861491  | 0.0359221 |
| XGBoost_Reference | Clustering_V0_Full_k2 (Winner c0=0, c1=10)                   |         1 |      3984 |     1803 | 0.844023 | 0.0420501 | 0.0420426 |  0.000797068 | 0.0278349 |
| MLP_1.3_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         0 |      7156 |     4817 | 0.754287 | 0.0495899 | 0.0472413 |  0.0150802   | 0.0389389 |
| MLP_1.3_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         1 |      2647 |     1803 | 0.776352 | 0.0503523 | 0.0440792 |  0.0243389   | 0.0370033 |
| MLP_1.3_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                 |         0 |      7156 |     4817 | 0.736751 | 0.051329  | 0.0510691 |  0.0051591   | 0.0408763 |
| MLP_1.3_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                 |         1 |      2647 |     1803 | 0.831465 | 0.0437101 | 0.0422401 |  0.0112403   | 0.0321524 |
| MLP_2.0_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         0 |      7156 |     4817 | 0.754287 | 0.0495899 | 0.0472413 |  0.0150802   | 0.0389389 |
| MLP_2.0_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         1 |      2647 |     1803 | 0.776352 | 0.0503523 | 0.0440792 |  0.0243389   | 0.0370033 |
| MLP_2.0_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                 |         0 |      7156 |     4817 | 0.736751 | 0.051329  | 0.0510691 |  0.0051591   | 0.0408763 |
| MLP_2.0_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                 |         1 |      2647 |     1803 | 0.831465 | 0.0437101 | 0.0422401 |  0.0112403   | 0.0321524 |
| MLP_2.0_Reference | MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa)           |         0 |      7156 |     4817 | 0.73214  | 0.0517766 | 0.0515325 | -0.00502148  | 0.0403861 |
| MLP_2.0_Reference | MLP 2-Regime-Mixed (fg_w512x512_d0.3_huber0.1_swa)           |         1 |      2647 |     1803 | 0.824767 | 0.0445702 | 0.041958  |  0.0150342   | 0.0326127 |
| MLP_2.1_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         0 |      7156 |     4817 | 0.751308 | 0.0498896 | 0.0471617 |  0.0162712   | 0.0390899 |
| MLP_2.1_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         1 |      2647 |     1803 | 0.77822  | 0.0501416 | 0.0430228 |  0.0257531   | 0.0377413 |
| MLP_2.1_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                 |         0 |      7156 |     4817 | 0.744135 | 0.050604  | 0.0503453 |  0.00511076  | 0.0404377 |
| MLP_2.1_Reference | MLP 2-Regime-54 (w512x512x512_d0.3_huber0.1)                 |         1 |      2647 |     1803 | 0.8348   | 0.0432754 | 0.0416095 |  0.0118915   | 0.0315477 |
| MLP_2.1_Reference | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)      |         0 |      7156 |     4817 | 0.767059 | 0.0482839 | 0.0470642 |  0.0107839   | 0.0375314 |
| MLP_2.1_Reference | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr6e-4)      |         1 |      2647 |     1803 | 0.824812 | 0.0445645 | 0.0414222 |  0.0164375   | 0.0317004 |
| MLP_2.2_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         0 |      7156 |     4817 | 0.751308 | 0.0498896 | 0.0471617 |  0.0162712   | 0.0390899 |
| MLP_2.2_Reference | MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                   |         1 |      2647 |     1803 | 0.77822  | 0.0501416 | 0.0430228 |  0.0257531   | 0.0377413 |
| MLP_2.2_Reference | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)     |         0 |      7156 |     4817 | 0.739603 | 0.0510502 | 0.0501782 |  0.00939558  | 0.0401103 |
| MLP_2.2_Reference | MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)     |         1 |      2647 |     1803 | 0.8061   | 0.0468841 | 0.0447162 |  0.0140919   | 0.0331941 |
| MLP_2.2_Reference | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)      |         0 |      7156 |     4817 | 0.759582 | 0.0490527 | 0.0470685 |  0.0138103   | 0.0386101 |
| MLP_2.2_Reference | MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)      |         1 |      2647 |     1803 | 0.830518 | 0.0438328 | 0.0412892 |  0.0147143   | 0.0305884 |

## Yearly Performance Breakdown

Year-by-year R² on the 2023–2025 test period. 2.0 fixed the historically weak 2025 year for the mixed family's ensembles (2025 R² 0.8336 — best of any model); 2.1's mixed val winner held 2025 at 0.8185; 2.2's winners held 2025 at ~0.81. This table tracks whether the 2.3 winners hold that year.

### Year-by-Year R² Breakdown
| model_name                                                                                                                                                                 |   pooled_r2 |   year_2023_r2 |   year_2024_r2 |   year_2025_r2 |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------:|---------------:|---------------:|---------------:|
| Clustering_V0_Full_k2 (Winner c0=0, c1=10)                                                                                                                                 |    0.81496  |       0.822971 |       0.783256 |       0.83029  |
| MLP 2-Regime-Mixed (test-best, w384x384x384_d0.3_huber0.1_gelu_lr2e-4)                                                                                                     |    0.80135  |       0.788075 |       0.811327 |       0.80011  |
| MLP-2.0 2-Regime-Mixed (val top-5 avg)                                                                                                                                     |    0.800323 |       0.745352 |       0.825612 |       0.832424 |
| MLP 2-Regime-Mixed (test-best, w384x384x384_d0.3_huber0.05_gelu_lr2e-4)                                                                                                    |    0.799074 |       0.761872 |       0.809406 |       0.825158 |
| MLP 2-Regime-Mixed (test-best, w512x512x512_d0.3_huber0.05_gelu_lr2e-4)                                                                                                    |    0.798248 |       0.755776 |       0.807679 |       0.831298 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_huber0.15_gelu_lr5e-4)                                                                                                           |    0.79802  |       0.783526 |       0.805243 |       0.800771 |
| MLP 2-Regime-54 (test-best, w320x320_d0.4_huber0.1_gelu_lr5e-4)                                                                                                            |    0.797993 |       0.768907 |       0.819623 |       0.803572 |
| MLP 2-Regime-54 (test-best, w320x320_d0.3_huber0.2_gelu_lr5e-4)                                                                                                            |    0.797526 |       0.766465 |       0.823651 |       0.800981 |
| MLP-2.2 2-Regime-54 (test_best: w320x320_d0.4_huber0.2_gelu_lr6e-4)                                                                                                        |    0.797318 |       0.771995 |       0.813612 |       0.803729 |
| MLP-2.2 2-Regime-Mixed (test_best: w448x448x448_d0.3_huber0.1_gelu)                                                                                                        |    0.793991 |       0.74719  |       0.811376 |       0.824133 |
| MLP-2.1 2-Regime-Mixed (test_best: w448x448x448_d0.3_huber0.1_gelu)                                                                                                        |    0.793991 |       0.74719  |       0.811376 |       0.824133 |
| MLP-2.1 2-Regime-54 (test_best: w320x320_d0.3_gelu_lr6e-4)                                                                                                                 |    0.793502 |       0.758868 |       0.820591 |       0.800015 |
| MLP-2.0 cross-family (val winners)                                                                                                                                         |    0.793243 |       0.74832  |       0.812752 |       0.819099 |
| MLP-2.1 cross-family (val winners)                                                                                                                                         |    0.793069 |       0.740408 |       0.818901 |       0.821706 |
| MLP-2.0 2-Regime-Mixed (test_best: w512x512x512_d0.3_huber0.1_swa)                                                                                                         |    0.790253 |       0.735215 |       0.807409 |       0.830042 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5_lr2e-4)                                                                                                                          |    0.78972  |       0.781446 |       0.817718 |       0.76464  |
| MLP 2-Regime-Mixed (val top-5 avg)                                                                                                                                         |    0.789465 |       0.743772 |       0.803715 |       0.821233 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03)                                                                                                                           |    0.789424 |       0.736345 |       0.803833 |       0.829601 |
| MLP-1.3 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        |    0.788821 |       0.773579 |       0.818284 |       0.770357 |
| MLP-2.1 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.788526 |       0.75198  |       0.818135 |       0.79462  |
| MLP-2.2 cross-family (val winners)                                                                                                                                         |    0.788487 |       0.737554 |       0.812476 |       0.816765 |
| MLP cross-family (val winners: 2regime_96/w512x512x512_d0.3_lr1e-3 + 2regime_54/w448x448x448_d0.3_huber0.1_gelu_lr1e-3 + 2regime_mixed/w512x512x512_d0.3_huber0.03_lr1e-3) |    0.788487 |       0.737554 |       0.812476 |       0.816765 |
| MLP 2-Regime-Mixed (val top-10 avg)                                                                                                                                        |    0.788413 |       0.7398   |       0.802583 |       0.823608 |
| MLP 2-Regime-Mixed (val top-3 avg)                                                                                                                                         |    0.788361 |       0.747378 |       0.80194  |       0.815281 |
| MLP 2-Regime-96 (test-best, w256x256_d0.5)                                                                                                                                 |    0.786949 |       0.767397 |       0.819748 |       0.770157 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_gelu_lr5e-4)                                                                                                               |    0.78659  |       0.74618  |       0.795177 |       0.817671 |
| MLP-2.0 2-Regime-54 (test_best: w384x384_d0.3_gelu)                                                                                                                        |    0.786493 |       0.767355 |       0.826921 |       0.76174  |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.2_huber0.05_gelu)                                                                                                        |    0.786306 |       0.766272 |       0.795747 |       0.792907 |
| MLP 2-Regime-Mixed (w512x512x512_d0.2_huber0.05_gelu)                                                                                                                      |    0.786059 |       0.756936 |       0.796001 |       0.802704 |
| MLP-2.0 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.785392 |       0.763461 |       0.817843 |       0.771644 |
| MLP-2.2 2-Regime-Mixed (val top-5 avg)                                                                                                                                     |    0.784973 |       0.734463 |       0.801995 |       0.819457 |
| MLP-2.1 2-Regime-Mixed (val_sel: w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                       |    0.7844   |       0.73328  |       0.802516 |       0.818504 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.05_lr5e-4)                                                                                                                    |    0.784303 |       0.733405 |       0.800673 |       0.819853 |
| MLP-2.1 2-Regime-Mixed (val top-3 avg)                                                                                                                                     |    0.783877 |       0.73011  |       0.801706 |       0.821313 |
| MLP 2-Regime-54 (val top-10 avg)                                                                                                                                           |    0.78341  |       0.724703 |       0.818659 |       0.809512 |
| MLP-2.1 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP-2.2 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP-1.3 2-Regime-96 (test_best: w256x256_d0.5)                                                                                                                             |    0.783404 |       0.763386 |       0.82079  |       0.762541 |
| MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr7e-4)                                                                                                                       |    0.782779 |       0.717594 |       0.824606 |       0.809936 |
| MLP 2-Regime-54 (5-seed champ, w320x320_d0.3_huber0.1_gelu_lr7e-4)                                                                                                         |    0.782733 |       0.718559 |       0.832595 |       0.800853 |
| MLP-1.3 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.782533 |       0.749643 |       0.803851 |       0.792293 |
| MLP 2-Regime-96 (test-best, w192x192_d0.5_lr2e-4)                                                                                                                          |    0.782428 |       0.760066 |       0.820244 |       0.763828 |
| MLP-2.2 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr6e-4)                                                                                                  |    0.781829 |       0.737827 |       0.793869 |       0.813533 |
| MLP-2.1 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.05_lr6e-4)                                                                                                  |    0.781771 |       0.734857 |       0.795061 |       0.815627 |
| MLP 2-Regime-Mixed (5-seed champ, w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                      |    0.78165  |       0.739578 |       0.798031 |       0.806861 |
| MLP 2-Regime-54 (val top-5 avg)                                                                                                                                            |    0.781169 |       0.717676 |       0.818574 |       0.810634 |
| MLP-2.2 2-Regime-Mixed (val_sel: w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                       |    0.78085  |       0.728664 |       0.801999 |       0.813099 |
| MLP 2-Regime-Mixed (w512x512x512_d0.3_huber0.03_lr1e-3)                                                                                                                    |    0.78085  |       0.728664 |       0.801999 |       0.813099 |
| MLP 2-Regime-54 (val top-3 avg)                                                                                                                                            |    0.780538 |       0.72283  |       0.811937 |       0.809149 |
| MLP-2.0 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.780505 |       0.736315 |       0.827429 |       0.778244 |
| MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr8e-4)                                                                                                                       |    0.780476 |       0.712013 |       0.826929 |       0.806825 |
| MLP 2-Regime-54 (w320x320_d0.4_huber0.1_gelu_lr8e-4)                                                                                                                       |    0.780282 |       0.709719 |       0.823068 |       0.812651 |
| MLP 2-Regime-54 (5-seed champ, w320x320_d0.3_huber0.1_gelu_lr8e-4)                                                                                                         |    0.780117 |       0.709504 |       0.833607 |       0.802064 |
| Global Single Model (54 Backbone)                                                                                                                                          |    0.77923  |       0.750748 |       0.770077 |       0.813582 |
| MLP 2-Regime-54 (w320x320_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                       |    0.777579 |       0.7049   |       0.827209 |       0.80561  |
| MLP-2.2 2-Regime-54 (val top-10 avg)                                                                                                                                       |    0.777    |       0.739206 |       0.785624 |       0.80468  |
| MLP-1.3 2-Regime-96 (val top-10 avg)                                                                                                                                       |    0.772329 |       0.72545  |       0.797894 |       0.793806 |
| MLP-2.1 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                             |    0.771661 |       0.723384 |       0.781407 |       0.810216 |
| MLP-2.2 2-Regime-54 (5-seed champ, w512x512x512_d0.3_huber0.1)                                                                                                             |    0.771661 |       0.723384 |       0.781407 |       0.810216 |
| MLP-2.1 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  |    0.771284 |       0.730569 |       0.781896 |       0.800208 |
| MLP-2.0 2-Regime-96 (val top-10 avg)                                                                                                                                       |    0.769168 |       0.720824 |       0.814159 |       0.773225 |
| MLP 2-Regime-54 (5-seed champ, w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                     |    0.766788 |       0.721024 |       0.7697   |       0.808946 |
| MLP-1.3 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  |    0.76511  |       0.723072 |       0.775216 |       0.79585  |
| MLP-2.0 2-Regime-54 (val_sel: w512x512x512_d0.3_huber0.1)                                                                                                                  |    0.76511  |       0.723072 |       0.775216 |       0.79585  |
| MLP 2-Regime-96 (val top-3 avg)                                                                                                                                            |    0.761378 |       0.710766 |       0.79078  |       0.783049 |
| MLP-1.3 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.761018 |       0.706877 |       0.786985 |       0.790134 |
| MLP-2.0 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.761018 |       0.706877 |       0.786985 |       0.790134 |
| MLP 2-Regime-96 (val top-5 avg)                                                                                                                                            |    0.760726 |       0.719159 |       0.794446 |       0.767657 |
| MLP 2-Regime-96 (val top-10 avg)                                                                                                                                           |    0.760333 |       0.725155 |       0.796496 |       0.757451 |
| MLP-2.0 2-Regime-Mixed (val_sel: fg_w512x512_d0.3_huber0.1_swa)                                                                                                            |    0.759881 |       0.744054 |       0.757107 |       0.772643 |
| MLP-2.2 2-Regime-54 (val_sel: w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                      |    0.759564 |       0.722393 |       0.758606 |       0.795294 |
| MLP 2-Regime-54 (w448x448x448_d0.3_huber0.1_gelu_lr1e-3)                                                                                                                   |    0.759564 |       0.722393 |       0.758606 |       0.795294 |
| MLP 2-Regime-96 (w512x512x512_d0.3_lr1e-3)                                                                                                                                 |    0.759483 |       0.711309 |       0.782285 |       0.784721 |
| MLP-2.1 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.759483 |       0.711309 |       0.782285 |       0.784721 |
| MLP-2.2 2-Regime-96 (val_sel: w512x512x512_d0.3_lr1e-3)                                                                                                                    |    0.759483 |       0.711309 |       0.782285 |       0.784721 |
| MLP 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                                   |    0.75656  |       0.708198 |       0.776529 |       0.784688 |
| MLP-2.2 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                               |    0.75656  |       0.708198 |       0.776529 |       0.784688 |
| MLP-2.1 2-Regime-96 (5-seed champ, w512x512x512_d0.3_lr1e-3)                                                                                                               |    0.75656  |       0.708198 |       0.776529 |       0.784688 |
| MLP 2-Regime-96 (w352x352_d0.5)                                                                                                                                            |    0.755707 |       0.728858 |       0.794892 |       0.740046 |
| MLP-2.1 2-Regime-96 (val top-3 avg)                                                                                                                                        |    0.753121 |       0.709325 |       0.778246 |       0.770787 |
| MLP-2.2 2-Regime-96 (val top-3 avg)                                                                                                                                        |    0.744187 |       0.695093 |       0.77868  |       0.758521 |
| MLP 2-Regime-96 (w256x256x256_d0.4)                                                                                                                                        |    0.741311 |       0.696068 |       0.774859 |       0.752005 |
| MLP 2-Regime-96 (w128x128x128_d0.4)                                                                                                                                        |    0.738032 |       0.684655 |       0.76653  |       0.762987 |
| MLP 2-Regime-96 (w320x320_d0.4)                                                                                                                                            |    0.734852 |       0.705517 |       0.775525 |       0.719898 |

## Systematic-Bias Diagnostic (headline)

mlp-1.2/1.3 documented the 96-family's systematic positive test bias (bias² ≈ 10–17 % of MSE); 2.0 got the 54-family under 5 %; 2.1 got the 54-family to 0.8 %; 2.2 met the 54 criterion (3.1 %) and improved mixed (12.7 → 9.6 %) but the 96-family WORSENED to 21.7 % — the mid-lr {4e-4, 6e-4, 8e-4} small-net pool is more biased than the lr3e-4 anchor (w256x256_d0.5: 1.1 %). 2.3's 96 pool is therefore **lr3e-4-only by construction** (widths {96..320} × d {0.4, 0.5, 0.6}, 3-layer probes, lr2e-4, me600 at 96²). Success criterion: per-family median bias²/MSE < 5 % for ALL three families. Backed by `analyze_bias.py`.

### Per-family median bias^2/MSE share (honest architectures)
| family     | n_configs |   med_bias2_mse_share |   med_test_bias |   med_test_r2 |
|:-----------|----------:|----------------------:|----------------:|--------------:|
| 2regime_96 | 32 | 0.1353 | 0.0183 | 0.7511 |
| 2regime_54 | 145 | 0.0125 | 0.0047 | 0.7870 |
| 2regime_mixed | 46 | 0.0616 | 0.0117 | 0.7871 |

### Worst 8 configs by bias^2/MSE share (all architectures)
| family        | config_id                          | architecture   |   test_r2 |   test_rmse |   test_bias |   bias2_mse_share |
|:--------------|:-----------------------------------|:---------------|----------:|------------:|------------:|------------------:|
| 2regime_96    | w320x320_d0.4                      | mlp            |  0.734852 |   0.0524542 |   0.02608   |          0.247204 |
| 2regime_96    | w128x128_d0.4                      | mlp            |  0.717528 |   0.0541407 |   0.0265355 |          0.240218 |
| 2regime_96    | w96x96_d0.5_me600                  | mlp            |  0.727164 |   0.0532092 |   0.0251972 |          0.224249 |
| 2regime_96    | w192x192_d0.4                      | mlp            |  0.744075 |   0.0515338 |   0.02355   |          0.208832 |
| 2regime_96    | w288x288_d0.4                      | mlp            |  0.744757 |   0.0514651 |   0.023371  |          0.206219 |
| 2regime_mixed | w640x640_d0.3_huber0.1_gelu        | mlp            |  0.736682 |   0.0522728 |   0.0236545 |          0.204773 |
| 2regime_mixed | w640x640_d0.3_huber0.1_gelu_lr4e-4 | mlp            |  0.744087 |   0.0515326 |   0.0232953 |          0.204349 |
| 2regime_96    | w128x128_d0.6                      | mlp            |  0.699865 |   0.0558077 |   0.0248434 |          0.198169 |

### Best 8 configs by bias^2/MSE share (all architectures)
| family        | config_id                              | architecture   |   test_r2 |   test_rmse |    test_bias |   bias2_mse_share |
|:--------------|:---------------------------------------|:---------------|----------:|------------:|-------------:|------------------:|
| 2regime_54    | w128x128_d0.3_gelu_lr4e-4              | mlp            |  0.78485  |   0.0472505 | -1.60805e-05 |       1.15821e-07 |
| 2regime_54    | w320x320_d0.5_gelu_lr6e-4              | mlp            |  0.785553 |   0.0471733 | -0.000212898 |       2.03682e-05 |
| 2regime_54    | w352x352_d0.4_gelu_lr4e-4              | mlp            |  0.787149 |   0.0469974 |  0.000378894 |       6.49964e-05 |
| 2regime_54    | w320x320_d0.4_gelu_lr5e-4              | mlp            |  0.792034 |   0.0464549 | -0.00038276  |       6.78876e-05 |
| 2regime_54    | w320x320_d0.4_huber0.15_gelu_lr4e-4    | mlp            |  0.79446  |   0.0461832 | -0.000432632 |       8.77545e-05 |
| 2regime_mixed | w512x512x512_d0.3_huber0.2_gelu_lr2e-4 | mlp            |  0.788713 |   0.0468244 |  0.000451644 |       9.30352e-05 |
| 2regime_54    | w256x256_d0.4_huber0.1_gelu_lr6e-4     | mlp            |  0.788916 |   0.0468019 |  0.000619086 |       0.000174974 |
| 2regime_54    | w128x128_d0.4_gelu_lr5e-4              | mlp            |  0.78036  |   0.047741  |  0.000693442 |       0.000210978 |

### Per-cluster median bias^2/MSE share (honest architectures)
| family     | cluster |   med_bias2_mse_share |   med_test_bias |   med_test_r2 |
|:-----------|--------:|----------------------:|----------------:|--------------:|
| 2regime_96 | 0 | 0.1135 | 0.0174 | 0.7386 |
| 2regime_96 | 1 | 0.1879 | 0.0207 | 0.7977 |
| 2regime_54 | 0 | 0.0086 | 0.0029 | 0.7627 |
| 2regime_54 | 1 | 0.0449 | 0.0089 | 0.8482 |
| 2regime_mixed | 0 | 0.0656 | 0.0123 | 0.7709 |
| 2regime_mixed | 1 | 0.0404 | 0.0089 | 0.8294 |

## FeatureGroupedMLP / PLR — documented negatives, not re-run in 2.3

2.0 established that the grouped-tower (`fg`, best 0.782) and PLR-encoding (`plr`, best 0.720) architectures underperform the plain MLP (0.790) at this scale — the winning lever was the *feature allocation* (the `2regime_mixed` family), not the tower structure. Per the no-re-spend rule, **2.3 runs no fg/plr configs**; the classes and the validated semantic grouping remain available in `mlp23/feature_groups.py`. The grouping table for the union of the three families' features is printed for reference.

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

## Early-Stopping Replay (patience-60 re-check, tag 22)

Offline replay of honest epoch-selection rules on the saved per-epoch curves (`analyze_stopping.py --tag 22`). 1.2/1.3 established that patience-60 is the best honest rule; the `swa_val` rule rows are replayed for completeness (no SWA configs run in 2.3 — the curves are the live ones). 2.3 re-checks patience-60 on the new grids, including the 96-family me600 probes at 96² (does extending the cap help the smallest nets?).

### Stopping-rule aggregates (mean pooled test RMSE; lower is better; oracle = unreachable bound)
| family        | rule             |   mean_test_rmse |   median_test_rmse |   n |
|:--------------|:-----------------|-----------------:|-------------------:|----:|
| 2regime_96    | patience60       |        0.0518455 |          0.051905  |  98 |
| 2regime_96    | patience20       |        0.0518455 |          0.051905  |  98 |
| 2regime_96    | patience40       |        0.0518455 |          0.051905  |  98 |
| 2regime_96    | val_aux          |        0.0526654 |          0.052617  |  98 |
| 2regime_96    | swa_val          |        0.0518455 |          0.051905  |  98 |
| 2regime_96    | plateau_w20e1e-4 |        0.0834599 |          0.080948  |  98 |
| 2regime_96    | plateau_w40e1e-4 |        0.0766095 |          0.0730849 |  98 |
| 2regime_96    | plateau_w40e3e-4 |        0.0766095 |          0.0730849 |  98 |
| 2regime_96    | plateau_w60e1e-4 |        0.0669366 |          0.0637704 |  98 |
| 2regime_96    | oracle           |        0.0489282 |          0.0484569 |  98 |
| 2regime_54    | patience60       |        0.0482185 |          0.0479484 | 441 |
| 2regime_54    | patience20       |        0.0482185 |          0.0479484 | 441 |
| 2regime_54    | patience40       |        0.0482185 |          0.0479484 | 441 |
| 2regime_54    | val_aux          |        0.048458  |          0.0482376 | 441 |
| 2regime_54    | swa_val          |        0.0482185 |          0.0479484 | 441 |
| 2regime_54    | plateau_w20e1e-4 |        0.0676412 |          0.0668449 | 441 |
| 2regime_54    | plateau_w40e1e-4 |        0.0603548 |          0.0591855 | 441 |
| 2regime_54    | plateau_w40e3e-4 |        0.0603548 |          0.0591855 | 441 |
| 2regime_54    | plateau_w60e1e-4 |        0.0550168 |          0.0539754 | 441 |
| 2regime_54    | oracle           |        0.0470233 |          0.0467485 | 441 |
| 2regime_mixed | patience60       |        0.0486452 |          0.0483324 | 142 |
| 2regime_mixed | patience20       |        0.0486452 |          0.0483324 | 142 |
| 2regime_mixed | patience40       |        0.0486452 |          0.0483324 | 142 |
| 2regime_mixed | val_aux          |        0.0489917 |          0.0484705 | 142 |
| 2regime_mixed | swa_val          |        0.0486452 |          0.0483324 | 142 |
| 2regime_mixed | plateau_w20e1e-4 |        0.068978  |          0.0687667 | 142 |
| 2regime_mixed | plateau_w40e1e-4 |        0.0589758 |          0.0581882 | 142 |
| 2regime_mixed | plateau_w40e3e-4 |        0.0589758 |          0.0581882 | 142 |
| 2regime_mixed | plateau_w60e1e-4 |        0.0527995 |          0.0520641 | 142 |
| 2regime_mixed | oracle           |        0.0460527 |          0.0458846 | 142 |

## Extrapolation (OOD) Check

588/6,620 test rows (8.9 %) are OOD on ≥1 top-10 gain feature (same definition as mlp-1.0–2.2). The pure-96 family keeps its OOD strength; the mixed family is in-distribution-strong but OOD-weak (its c1 = 54+10 half carries the 54-family's weak OOD). The 2.3 winners' OOD behavior is reported for the record (family allocation is pinned, so this is a tracking table, not a target).

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

From the saved artifacts (no retraining), via `analyze_overfitting.py`: train-fit vs held-out gap (aux2020 = train-fit), capacity vs test transfer, and the per-epoch curve shape for each family's val winner. 2.2's winners show the familiar pattern (test min early, val flat, train-fit improving); the 2.3 mitigations in play are the full 3-seed pool, the removal of the 54 3-layer val-overfitters, and the 96 lr3e-4-only pool.

### Overfitting symptoms (analyze_overfitting.py)

#### 1. Train-fit vs held-out gap (median RMSE)
| family     |   aux2020 (train-fit) |   val |   test |   val/train ratio |
|:-----------|----------------------:|------:|-------:|------------------:|
| 2regime_96 | 0.0423 | 0.0535 | 0.0508 | 1.3x |
| 2regime_54 | 0.0263 | 0.0584 | 0.0470 | 2.2x |
| 2regime_mixed | 0.0254 | 0.0503 | 0.0470 | 2.0x |

#### 2. Capacity vs test transfer (median by n_params bucket)
| family     | capacity   |   n_configs |   med_val_rmse |   med_test_r2 |   med_test_bias |
|:-----------|:-----------|------------:|---------------:|--------------:|----------------:|
| 2regime_96 | <200k | 23 | 0.0538 | 0.7496 | 0.0178 |
| 2regime_96 | 200-500k | 8 | 0.0522 | 0.7603 | 0.0201 |
| 2regime_96 | 1M+ | 1 | 0.0485 | 0.7595 | 0.0189 |
| 2regime_54 | <200k | 56 | 0.0597 | 0.7847 | 0.0018 |
| 2regime_54 | 200-500k | 87 | 0.0579 | 0.7898 | 0.0063 |
| 2regime_54 | 500k-1M | 2 | 0.0552 | 0.7636 | 0.0109 |
| 2regime_mixed | 500k-1M | 25 | 0.0511 | 0.7875 | 0.0123 |
| 2regime_mixed | 1M+ | 21 | 0.0496 | 0.7866 | 0.0117 |

#### 3. Per-epoch curve shape for the val winner (cluster-0 specialist)
| family     | config_id |   aux_ep100 |   aux_ep260 |   val_plateau |   test_min |   test_min_epoch |   test_at_best_val |   test_final |   test_rise_after_min |
|:-----------|:----------|------------:|------------:|--------------:|-----------:|-----------------:|-------------------:|-------------:|----------------------:|
| 2regime_96 | w512x512x512_d0.3_lr1e-3 | 0.0262 | 0.0179 | 0.0531 | 0.0451 | 90 | 0.0491 | 0.0489 | 0.0037 |
| 2regime_54 | w448x448x448_d0.3_huber0.1_gelu_lr1e-3 | 0.0292 | 0.0166 | 0.0602 | 0.0488 | 98 | 0.0516 | 0.0492 | 0.0004 |
| 2regime_mixed | w512x512x512_d0.3_huber0.03_lr1e-3 | 0.0187 | 0.0206 | 0.0534 | 0.0464 | 84 | 0.0482 | 0.0504 | 0.0040 |

#### 4. Systematic bias on test (MLP vs XGBoost references)
MLP 2regime_96 median test bias: 0.0183
MLP 2regime_54 median test bias: 0.0047
MLP 2regime_mixed median test bias: 0.0117
XGBoost references (eval-1.1): 2-regime 0.0065, global 0.0105

## Timing

The sweep is sized to spend ~1.75 h of the 2 h `gpu_debug` H100 wall allocation: 223 phase-1 × 3 seeds (full 3-seed pool) + 12 champion job-seeds ≈ 681 jobs at 8 parallel workers (2.2's per-seed mean was 43 s at ~5.9 effective workers).

### Timing (H100 PCIe 80 GB, 8 parallel workers)
Total sweep wall time: 5287.5 s
Total training time (all jobs, GPU-seconds): 31826 s
Eval wall time: 8.7 s

Slowest jobs (3-seed config train_time_s):
  2regime_96/w96x96_d0.5_me600                               260.4s  n_seeds=3
  2regime_96/w96x96_d0.5_me500                               224.7s  n_seeds=3
  2regime_96/w96x96x96_d0.4                                  219.0s  n_seeds=3
  2regime_96/w192x192_d0.4                                   212.1s  n_seeds=3
  2regime_96/w192x192x192_d0.4                               212.1s  n_seeds=3

## Key Takeaways

1. **The mlp-2.0 architecture IS still meaningful — on test.** The mixed test-best `w384x384x384_d0.3_huber0.1_gelu_lr2e-4` → **0.8014** (bias 0.0018) is the first MLP single config above 2.0's mixed val top-5 ensemble (0.8003); the 54 test-best `w320x320_d0.4_huber0.15_gelu_lr5e-4` → **0.7980** (new 54 record, bias 0.0020) and the 96 test-best `w256x256_d0.5_lr2e-4` → **0.7897** (new 96 record, NEGATIVE bias −0.0028) also beat their families' 2.2 records. `test-best` rows are reporting only (selection on test would be leakage) — but the frontier exists: the 2-regime MLP can clear 0.80 as a single model.
2. **Val selection: the mixed family's structural val-noise is GONE — first significant positive Spearman ever.** Full-val +0.318 (p=0.031; 2.1: −0.309, 2.2: −0.413), BOTH val years positive (+0.263 ns / +0.408 p=0.005), and the winner is stable under leave-one-val-year-out. The 2.3 mixed grid (gelu 3-layer at lr {2e-4, 3e-4}) fixed the family's val-proxy problem. The mixed val winner is unchanged (`w512x512x512_d0.3_huber0.03_lr1e-3`, 0.7809) but mixed val top-5 avg rose to 0.7895 (2.2: 0.7850).
3. **54-family val selection did NOT recover: −0.055 (2.2: +0.582) — and the single 3-layer bit-identity anchor still wins the 54 val ranking** (val RMSE 0.0544 vs the best 2-layer 0.0556). The 3-layer val-overfit is structural, not pool-composition luck: even one 3-layer config dominates 54 val selection while failing on test (0.7596). The val-year diagnostic sharpens this: the 2-layer-only pool's val-2022 half is strongly NEGATIVELY correlated with test (−0.472, p=2e-9). The 54 val top-10 avg still improved to 0.7834 (2.2: 0.7770) — the 2-layer ensembles are the real 54 gains.
4. **96 val selection weakened (+0.157 ns vs +0.566)** — val-2021 stays strong (+0.566, p=7e-4) while val-2022 turned negative (−0.316, p=0.078); the big-net `w512x512x512_d0.3_lr1e-3` remains the val winner (stable under all selectors, test 0.7595) while the 96 test-best is the lr2e-4 small net (0.7897). The 3-layer-val-overfit pattern repeats in 96 (w128³/w256³ rank 2–3 on val, test 0.738–0.741).
5. **Debias: 54 met with the best margin yet (median bias²/MSE 1.25 %; w128x128_d0.3_gelu_lr4e-4 bias −1.6e-5); mixed improved to 6.2 % (2.2: 9.6 %) but still >5 %; 96 improved to 13.5 % (2.2: 21.7 %) but unmet** — within 96, the d0.4 variants are the biased cells (0.20–0.25 share) while lr2e-4 goes negative-bias.
6. **Reproducibility:** 3/3 anchors bit-identical vs 2.2 (max|diff| = 0); sweep 5,287.5 s (88.1 min) wall / 8.8 GPU-h for 681 job-seeds — total job 1:31:47, inside the 2 h `gpu_debug` cap (target ~1.75 h).
7. **The honest answer to "is the mlp-2.0 architecture still meaningful?":** YES on capacity — the 2-regime MLP exceeds 0.80 as a single config for the first time in the 1.0–2.3 series, and the mixed family's val-selection signal is now positive. NO on 54/96 val-selection — the deployed rule still lands at 0.7596/0.7595, with the 3-layer anchor (54) and the big net (96) as the persistent val favorites. The val-year diagnostic localizes the remaining noise to the 54/96 val-2022 half.



## Reproducibility

```bash
cd notebooks/experiment/derived_8.4-eval-mlp-2.3
uv run --no-sync python make_configs.py            # deterministic -> config.yaml (committed)
uv run --no-sync python run_mlp_sweep.py --resume  # phases 1-3, FULL 3-seed pool (~85 min wall, 8 workers, H100)
uv run --no-sync python run_mlp_champion.py        # 5-seed champion ensembles (per-family top-N)
uv run --no-sync python run_mlp_eval.py            # leaderboard, per-regime, ensembles, figures
uv run --no-sync python compare_anchor_vs_2.2.py   # cross-version anchor bit-identity evidence
uv run --no-sync python analyze_bias.py            # bias^2/MSE diagnostic
uv run --no-sync python analyze_selection.py       # selection reliability (1/2/3-seed, full pool)
uv run --no-sync python analyze_val_years.py       # val-2021 vs val-2022 diagnostic (full pool)
uv run --no-sync python analyze_overfitting.py     # overfitting-symptom analysis
uv run --no-sync python analyze_extrapolation.py   # OOD check
uv run --no-sync python analyze_stopping.py --tag 22  # stopping-rule replay
cd notebooks && nb execute experiment/derived_8.4-eval-mlp-2.3/derived_8.4-eval-mlp-2.3.ipynb --uv
uv run --no-sync python generate_readme.py         # regenerate this README from the notebook
```

- The anchor-vs-2.2 offline comparison (one anchor per family — 54/
  `w448x448x448_d0.3_huber0.1_gelu_lr1e-3`, mixed/`w512x512x512_d0.3_huber0.03_lr1e-3`,
  96/`w512x512x512_d0.3_lr1e-3`; seed 42, spec 0) compares the v10 val curves
  against the v9 ones over the overlapping epochs (max|diff| = 0 target) —
  reproducible via `compare_anchor_vs_2.2.py` →
  `artifacts/anchor_vs_22_comparison.json`.
- Configurations pinned in `config.yaml` (generated by `make_configs.py`); seeds
  {42, 7, 123} for the sweep (ALL configs — full 3-seed pool),
  {42, 7, 123, 2024, 999} for the champion step;
  `data_version: 10`. No SWA / fg / plr / 54-3-layer configs (closed negatives).
- Artifacts: `models/`, `artifacts/`, `sweep_results.csv`, `metrics_summary.csv`,
  `per_regime_metrics_summary.csv`, `bias_summary.csv`, `ood_summary.csv`,
  `stopping_22_*.csv`, `selection_summary.csv`, `val_year_summary.csv`,
  `timing_log.json`, `artifacts/anchor_vs_22_comparison.json`, figures, and the
  report notebook. All numbers in this README come from the executed notebook.

## Caveats

- The XGBoost 2-regime reference (0.815) was itself test-selected in eval-1.1; all
  honest MLP claims use val-based selection. `test-best` rows are reporting only.
- The mixed family's c1 (54+10) half inherits the 54-family's weak OOD extrapolation;
  the pure-96 family remains the best OOD model.
- 2025 test coverage is partial for several stations; year-2025 numbers should be read
  with the same caution as 1.x/2.0/2.1/2.2.
- Val selection remains the bottleneck (2.2: the 54-family 3-seed val winner 0.7596
  was a test loser while the test-best 0.7973 sat at val rank 49/82; the mixed
  family's val ranking is negatively correlated with test in both val years).
  2.3's mitigations: the full 3-seed pool, the 3-layer val-overfitters removed
  from the 54 pool, and the deeper 54 champion hedge (top-3). The val-year
  diagnostic (val-2021 reliable, val-2022 noise for 54/96) is diagnostic only —
  the selection rule itself is unchanged (protocol).
- The 96-family median bias²/MSE was 21.7 % in 2.2 (criterion < 5 %); the 2.3
  96 pool is lr3e-4-small-net-dominated by construction (the only region 2.2
  found debiased) — reported honestly either way.
- The champion step runs the per-family `sweep.champion_top_n` (mixed top-2 +
  54 top-3 + 96 top-1) × extra seeds {2024, 999}; `--top-n N` CLI overrides
  (uniform, 2.1 parity). See `docs/plans/20260811-mlp-2.3.md`.
