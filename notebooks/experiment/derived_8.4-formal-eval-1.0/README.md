# Experiment: `derived_8.4-formal-eval-1.0` — Formal Statistical Evaluation of the Two-Regime Clustering Model

## Objective

Publication-oriented statistical evaluation of the claim established in `derived_8.4-eval-1.1` / `-1.3`:
**a two-regime (KMeans k=2) clustering model beats the single-regime global model and the trained-gating
model**, on the frozen temporal split (2023–2025 test) and under leave-one-station-out (LOSO) spatial
generalization. All numbers in this README are copied verbatim from the stdout of the executed report
notebook (`derived_8.4-formal-eval-1.0.ipynb`, executed with `nb execute --uv` from `notebooks/`).

## Configurations (20)

14 requested configurations (test-selected deltas pinned from `derived_8.4-eval-1.1`'s delta grid,
identical parsing to eval-1.3; `none` = c0=c1=0) + 6 val-selected winners (`select_deltas_val.py`):

| config_id | strategy | c0 | c1 | delta_source |
|---|---|---|---|---|
| *(populated from `pinned_configs.csv` / notebook stdout after the run)* | | | | |

## Protocol

- **Temporal (primary):** experts trained on trainval (train 2017–2020 + val 2021–2022, 14,608 rows),
  evaluated on the frozen test set (2023–2025, 6,620 rows, 7 WA stations), **30 random seeds**
  (seed 42 included as exact replication anchor vs eval-1.1 / eval-1.3).
- **LOSO (secondary):** same 20 configurations × **5 seeds** × 7 held-out stations; router refitted per
  fold on the 6-station trainval (no held-out-station leakage into routing).
- **Delta-robustness:** per-regime delta features from three selection sources — *test-selected*,
  *val-selected* (re-ranked on validation-period residuals, train-only fits), *none*.
- **Seed scope:** only the XGBoost expert regressors' `random_state` varies; routers (KMeans / gating
  classifier) stay at seed 42 because the delta additions are tied to the seed-42 cluster labels.
- **Statistics:** seed-level (mean ± std, median, 95% t-CI, paired t-test, Wilcoxon signed-rank,
  % seeds A better), sample-level (paired cluster bootstrap over (station, month) blocks, percentile
  95% CI + bootstrap p), Benjamini–Hochberg FDR over the pre-specified comparison family, LOSO
  per-station win counts + two-sided sign test (n = 7; 7/7 → p ≈ 0.016, 6/7 → p = 0.125).

## Temporal results

### Seed-level summary (mean ± std over seeds, [95% t-CI])

*(tables copied from notebook stdout — `temporal_config_summary.csv`)*

### Focused pairwise comparisons (paired t / Wilcoxon / FDR)

*(tables copied from notebook stdout — `temporal_pairwise_focused.csv`)*

### Sample-level bootstrap CIs (station, month blocks)

*(tables copied from notebook stdout — `temporal_bootstrap.csv`)*

## LOSO results

### Per-configuration summary (mean / median over stations)

*(tables copied from notebook stdout — `loso_config_summary.csv`)*

### Per-station pairwise tests (wins "k of 7 stations", sign test, paired tests)

*(tables copied from notebook stdout — `loso_pairwise_focused.csv`)*

## Delta-robustness

*(table copied from notebook stdout — `delta_robustness_summary.csv`)*

## Replication checks (seed 42)

*(checks copied from notebook stdout)*

## Methods & caveats

See the notebook's final markdown cell (as it will appear in the paper):
statistical-test semantics, the three known leakage sources (delta selection on test — addressed by the
ablation; (c0, c1) winner choice on test — addressed by the val protocol; backbone/V0 feature selection
targeting the test period — accepted as a caveat, shared by all models), n = 7 station power limits,
partial 2025 coverage, and the deliberate seed-scope decision for routing.

## Reproducibility

```bash
cd notebooks/experiment/derived_8.4-formal-eval-1.0
mkdir -p artifacts/slurm && sbatch run_slurm.sh   # GPU: val selection -> temporal (30 seeds) -> LOSO (5 seeds)
# smoke (CPU, n_estimators=100, data_version=-1, never reused by the real run):
uv run python select_deltas_val.py --smoke
uv run python run_temporal.py --smoke --config-id <id> --seeds 42 7 --n-parallel 4
uv run python run_loso.py --smoke --config-id <id> --seeds 42 7 --max-stations 2 --n-parallel 4
uv run python -m eval_formal.stats          # statistical self-tests
# report:
cd ../.. && nb execute experiment/derived_8.4-formal-eval-1.0/derived_8.4-formal-eval-1.0.ipynb --uv
```

- Configurations are pinned in `pinned_configurations.json` before any run (audit trail).
- Per-seed artifacts use cache-safe naming `<config_id>__s<seed>__<station>` for weights, predictions
  and job meta; jobs resume via `artifacts/jobs/*/meta.json` (data_version + file-presence match).
- `--smoke` uses data_version=-1 so smoke artifacts are never reused by the real run.
- Seed-42 temporal rows must reproduce eval-1.1 pooled test R² (e.g. V0_Full (0,10) = 0.814960,
  Global_54 = 0.779230, Baseline_V0_50 = 0.760447) to |diff| < 1e-6; LOSO seed-42 rows must reproduce
  eval-1.2/-1.3 loso_mean_r2 to |diff| < 1e-3 (printed by the drivers and the notebook).
