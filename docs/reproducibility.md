
## Short answer

**Not mainly “seed not fixed.”** Seeds *are* fixed (`SEED = 42`, XGBoost `random_state=42`). The mismatch is a mix of:

1. **Known env/hardware non-determinism** (~0.002–0.003 R² on the true v25 config)
2. **Protocol drift between notebooks** (bigger; can move R² by ~0.007+)
3. **Comparing different “SOTA configs”** without always saying so

---

## What the original SOTA actually was

In `MDR-v25.ipynb` (Mac M2 Pro, Python 3.10, NumPy 1.26.4, XGBoost 3.2.0, **CPU**):

| Config | Objective | R² | RMSE | ubRMSE | Bias | MAE | MedAE |
|--------|-----------|-----|------|--------|------|-----|-------|
| No weight, old hparams | `reg:absoluteerror`, 5500 trees, lr=0.04 | **0.8142** | 0.04059 | 0.04049 | −0.00284 | 0.03039 | 0.02369 |
| **Drift β=0.2, old hparams (reported SOTA)** | `reg:pseudohubererror`, 5500, lr=0.04 | **0.8224** | 0.03968 | 0.03956 | −0.00320 | 0.02832 | 0.02032 |

Seed setup in v25: `random.seed`, `np.random.seed`, `PYTHONHASHSEED`, XGB `random_state=SEED`. No torch CUDA seed (irrelevant for XGB). No `deterministic_histogram` / single-thread lock.

---

## Replication gap (same config, new machine)

`derived_8.0-optimization-1.0` already diagnosed this carefully:

| Run | Device | R² (Model 4 = drift + old feats + old hparams) |
|-----|--------|-----------------------------------------------|
| Original MDR-v25 | Mac M2 CPU | **0.8224** |
| Re-run MDR-v25 in current env | (their env) | **0.8207** |
| opt-1.0 | GPU (`device="cuda"`) | **0.8190** |
| opt-1.0 | CPU | **0.8194** |

They verified **identical** train/val/test rows, feature matrices, and weight vector. So this ~0.002–0.003 hit is **not data**, **not missing seed**, and **not a different feature set**.

Likely causes (all real for XGBoost hist):

| Factor | Evidence |
|--------|----------|
| **CPU arch / FP / threading** | M2 NEON vs x86 AVX; `n_jobs=-1`; different split tie-breaking |
| **Python/NumPy stack** | 3.10+np1.26 → **3.12+np2.4** in current `notebooks` uv env |
| **GPU vs CPU** | GPU hist builds are known to be slightly non-identical to CPU |
| Seed | Fixed everywhere; residual non-determinism is from **parallel/hist/device**, not unset `random_state` |

**Verdict:** environment/hardware can explain the **0.8224 → ~0.819–0.821** band. That is a mild crisis for “exact number in the paper,” not for “model is broken.”

---

## Bigger issue: protocol mismatches (not seed)

### 1. Weight **normalization** differs (explains drift gap between experiments)

| Notebook | Formula |
|----------|---------|
| MDR-v25 & opt-1.0 | `w = exp(β·(year−tmax)) / mean(w)` → **mean weight = 1** |
| feature-selection-2.0 `eval.ipynb` | `w = exp(−β·(tmax−year))` → **not mean-normalized** |

Same β, same shape of decay, but **different absolute weight scale**. With `min_child_weight=10`, that changes leaf constraints and tree structure.

That lines up with the numbers:

| Protocol | opt-1.0 (mean-normalized, often CUDA) | fs-2.0 hand (no mean-norm, CPU hist) | Δ R² |
|----------|----------------------------------------|--------------------------------------|------|
| **No drift** (weights off) | 0.8222 | 0.8240 | **+0.0017** (close) |
| **With drift** | **0.8253** | **0.8178** | **−0.0076** |

Unweighted matches well; weighted diverges. That is protocol, not RNG.

### 2. Other silent differences in fs-2.0 eval

- **Median imputation** on trainval medians (v25 / opt-1.0 do not)
- **Bias sign flipped**: fs uses `mean(pred − true)`; v25/opt use `mean(true − pred)` → Bias looks opposite; R²/RMSE unchanged
- **ubRMSE formula** not identical across notebooks (`std(err)` vs `√(RMSE²−bias²)` vs centered residual form in hyperparams-1.3-lite)
- No `device=` in fs-2.0 → CPU `tree_method="hist"`; opt-1.0 forces CUDA when available

### 3. “SOTA 0.825” vs “hand 0.818” is often **different hparams**

- Classic v25 SOTA: **old** hparams + drift → ~0.822
- Peak in opt-1.0: **new** 1.3-lite hparams + drift + old features → **0.8253**
- fs-2.0 hand baseline: **1.3-lite only**, with the unnormalized weights → **0.8178** under “with drift”

So gates like “within 0.01 of hand ~0.825” mixed numbers from different training protocols.

---

## Other metrics: do they “fluctuate” less than R²?

### Hardware/replication noise (orig 0.8224 vs opt GPU 0.8190, same Model-4 config)

| Metric | Original | opt-1.0 | Δ | \|rel\| |
|--------|----------|---------|---|--------|
| **R²** | 0.82239 | 0.81902 | −0.0034 | **0.41%** |
| **RMSE** | 0.03968 | 0.04006 | +0.00038 | **0.95%** |
| **ubRMSE** | 0.03956 | 0.03992 | +0.00036 | **0.91%** |
| **MAE** | 0.02832 | 0.02845 | +0.00013 | **0.46%** |
| **Med\|Err\|** | 0.02032 | 0.02057 | +0.00025 | **1.2%** |
| **Bias** | −0.00320 | −0.00332 | −0.00012 | **3.6%** |

So:

- **MAE** is about as stable as R² in relative terms (and more interpretable in m³/m³).
- **RMSE / ubRMSE** move a bit more relatively (~1%) but stay in a tight absolute band (~0.0004).
- **Bias** is the noisiest *relatively* because it is near zero; small FP/tree differences show up as large % on Bias.
- **Pearson** (where logged) barely moves: opt M4 ≈ 0.906, M5 ≈ 0.909; fs hand drift ≈ 0.906 — correlation is the most stable ranking metric.

### Year-to-year (Model 5, opt-1.0) — this is where R² “jumps”

| Year | R² | RMSE | MAE | Pearson |
|------|-----|------|-----|---------|
| 2023 | 0.815 | 0.0408 | 0.0288 | 0.906 |
| 2024 | 0.833 | 0.0365 | 0.0272 | 0.913 |
| 2025 | 0.825 | 0.0411 | 0.0285 | 0.916 |

Year R² range ≈ **0.018**; RMSE range ≈ **0.0046**. That is **data/regime**, not seed. Overall R² is a pooled summary and will look jumpy if you stare only at yearly R².

### Across *real* config changes (opt-1.0 old features only)

R² range 0.813–0.825; Pearson 0.903–0.909. Rank order of models is consistent on R², RMSE, MAE, Pearson; Bias alone is a poor leaderboard metric.

---

## Crisis level (honest triage)

| Claim | Severity | Cause |
|-------|----------|--------|
| Can’t hit **exact** 0.8224 from M2 notebook | Low–medium | Hardware + stack; seeds already fixed |
| opt-1.0 hand-with-drift **0.825** vs fs-2.0 hand-with-drift **0.818** | **High (protocol)** | Weight mean-normalization + impute + device |
| Auto FS fails gate vs “0.825 hand” | Partly **self-inflicted** | Gate baseline and train protocol not locked to one harness |
| R² “fluctuates a lot” | Mixed | Year splits & protocols ≫ RNG |

This is a **reproducibility-process** crisis more than a “forgot the seed” crisis.

---

## What to lock down (recommended harness)

One shared eval function used by every experiment:

1. **Weights:** always mean-normalize (match v25/opt-1.0), or never — pick one and document it  
2. **Device:** default **CPU** for paper numbers; log GPU as exploratory only  
3. **No median impute** unless it was in the original training path  
4. **Bias definition:** `mean(y − ŷ)` everywhere  
5. **ubRMSE:** one formula (`std(y−ŷ)` or `√(RMSE²−bias²)` — they are almost the same if bias is mean residual)  
6. **Report a metric bundle**, not R² alone: **R² + RMSE + MAE + Pearson (+ Bias)**  
7. Treat **ΔR² < ~0.003** as env noise on this stack; require larger gaps for “beats SOTA”

---

## Bottom line

- **Seeds are fixed**; residual gap to 0.8224 is mostly **arch/stack/GPU**, ~0.002–0.003 R².  
- **Larger mismatches** between opt-1.0 and feature-selection-2.0 come from **weight normalization, imputation, and device**, not RNG.  
- **Other metrics:** MAE tracks R² for stability; RMSE/ubRMSE slightly more sensitive but absolute-stable; Pearson very stable; Bias relatively noisy; yearly R² is legitimately variable and should not be used alone as the reproducibility thermometer.

I can next either (a) patch `eval.ipynb` to match v25/opt-1.0 weight + metrics definitions and re-score the fs-2.0 leaderboard, or (b) draft a short `docs/` note on the locked eval protocol for the team.