#!/usr/bin/env python3
"""Regenerate README.md for derived_8.4-eval-mlp-2.1.

AGENTS.md rule: README tables come strictly from the executed notebook's
stdout; the report notebook is the single source of result tables. This script
assembles the README from:

  1. PREAMBLE — hand-maintained prose (title, objective, verdict, what's new,
     protocol, sweep design). No result tables live here (the sweep-design
     table is a spec, auditable against config.yaml / make_configs.py).
  2. Notebook sections — every markdown cell (except the title cell) is copied
     verbatim, followed by the stdout of the code cell(s) below it, verbatim:
     exactly what the executed notebook printed.
  3. CLOSING — hand-maintained prose (reproducibility checklist, caveats).

Deterministic: regenerating after an `nb execute` reproduces the same file.

Usage:
    uv run --no-sync python generate_readme.py
"""

from __future__ import annotations

import json
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
NB_PATH = EXP_DIR / "derived_8.4-eval-mlp-2.1.ipynb"
OUT_PATH = EXP_DIR / "README.md"

PREAMBLE = r"""# Experiment: `derived_8.4-eval-mlp-2.1` — optimize the mixed family + fair SWA re-test + finish the 96-family debias (~1 h H100 wall)

## Objective

Follow-up to `derived_8.4-eval-mlp-2.0` (the `2regime_mixed` family broke the
plain-MLP ceiling: honest val top-5 / top-3 ensembles → test R² **0.8003**, the
2-seed honest single `w512x512x512_d0.3_huber0.1_swa` (mixed) → **0.7903**; XGBoost
2-regime 0.815, itself test-selected; `fg`/`plr` documented negatives; the 2.0 SWA
recipe a negative with two prescribed fixes). 2.1 is an **optimization + further
parameter sweep** of the 2.0 winners, temporal protocol only (no LOSO), **sized to
spend ~1 h of the 2 h H100 wall allocation** (338 job-seeds at 8 workers ≈ 6.2 GPU-h;
the allocation is otherwise wasted — 2.0's whole sweep took 13 min of wall).

All numbers below are the stdout of the executed report notebook
(`derived_8.4-eval-mlp-2.1.ipynb`). Weights/checkpoints/test predictions under
`models/`; preprocessed tensors and per-job logs under `artifacts/`; figures at the
experiment root.

## Verdict (TL;DR)

- **The mixed-family optimization did not beat 2.0 — an honest negative.**
  The 3-seed val winner `w512x512x512_d0.3_huber0.05_lr6e-4` (mixed) reaches
  test R² **0.7844**, below 2.0's mixed 2-seed honest single (0.7903) and its
  val top-5 ensemble (0.8003); 2.1's mixed val top-3/5/10 ensembles
  (0.7839/0.7831/0.7839) also sit below 0.8003, and the 2.1 cross-family
  val-winner ensemble (0.7931) is a draw with 2.0's (0.7932). The best single
  MLP on test across 2.1 is the 54-family `w320x320_d0.3_gelu_lr6e-4`
  (0.7935) — a *test-best*, not val-selected; the 54-family val winner is the
  same `w512x512x512_d0.3_huber0.1` as 1.3/2.0 (0.7713).
- **SWA stays a documented negative — now provably so.** With the RNG guard in
  place, **136/136** (swa-live vs anchor) val-curve pairs are bit-identical
  (max|diff| = 0) and **0/152** (seed, specialist) jobs deployed the SWA
  snapshot across the swept starts {0.7, 0.75, 0.8, 0.85} + the 0.6 re-check.
  Late-start SWA never beats the live best on val; both 2.0-prescribed fixes
  are discharged and the "gains un-attributable" caveat is gone — there are no
  gains to attribute.
- **Bias²/MSE criterion met only for `2regime_54`** (median 0.8% vs 2.0's
  3.7%); the 96 (13.9%) and mixed (12.7%) families still miss the <5%
  success criterion. The small-net grid produced near-unbiased configs
  (`w256x256_d0.5_mixup0.2` 0.009%, `w512x512_d0.3_huber0.1_gelu` 0.004%) but
  the val-selected winners are not the near-unbiased ones.
- **Selection reliability is the headline negative finding.** 3-seed
  aggregation flips the 54-family winner (1-seed `w320x320_d0.3_gelu_lr6e-4`
  → 2/3-seed `w512x512x512_d0.3_huber0.1`) and the mixed winner (1-seed
  `w512x512x512_d0.2_huber0.05` → 2/3-seed `w512x512x512_d0.3_huber0.05_lr6e-4`);
  only the 96 winner is stable across seed depths. Spearman(val, test) is
  +0.504 (96), **−0.555 (54)** and **−0.309 (mixed)** on the final aggregated
  data — val RMSE is a weak/unreliable selector for the 54 and mixed families.
- **Budget:** sweep 2,596 s (43.3 min) wall + 8 s eval = 15,275 GPU-s of
  training at 8 workers — ~4.2 GPU-h of the 2 h H100 allocation, inside the
  ~1 h plan; the RNG-guard verify step passed on the H100 node (max|diff| = 0).

## What's new in 2.1

1. **Fair SWA re-test (headline)** — the two 2.0-prescribed fixes in
   `mlp21/trainer.py`: `swa_start_frac` swept {0.7, 0.75, 0.8, 0.85} (2.0 hard-coded
   0.6) and an **RNG guard** around the SWA snapshot evaluation, so a `swa=true`
   job's **live trajectory is bit-identical to its `swa=false` anchor** (2.0's "gains
   are live-trajectory artifacts" caveat is gone). Deployment stays honest: SWA is
   deployed iff its best val RMSE beats the live best, per seed and per specialist.
   Verified by `analyze_swa.py`'s bit-identity stack check.
2. **3-phase, 3-seed sweep {42, 7, 123}** — phase 3 (new) adds a 3rd seed to the
   top-M configs per family → 3-seed mean val RMSE winner selection. This is the honest
   answer to 2.0's val-noise finding (mixed Spearman(val, test) = -0.455). Per-family
   phase depth (`sweep.phase2_top_n` / `phase3_top_n`, int-or-dict): the mixed winner
   gets the densest coverage.
3. **No fg/plr re-spend** — documented negatives get no GPU; all 178 phase-1 configs
   are plain MLP (the winner-pool filter stays mlp/fg/plr so the protocol text is
   unchanged).
4. **96-family debias grid** — small-net neighborhood (width 96–256, dropout 0.4–0.6,
   lr {3e-4, 1e-3}, huber, mixup, late SWA): 2.0 showed <200k-param nets are
   near-unbiased (`w256x256_d0.5`: bias²/MSE 1.0 %, test R² 0.7854).
5. **Configs generated by code** — `make_configs.py` produces the 178-config grid
   deterministically into `config.yaml` (reproducibility rule: constants generated by
   committed code, not hand-typed).

Documented negatives honored (no GPU re-spent): no calibration, no trainval retrain,
patience-60 kept, aux2020 diagnostic-only, batch 512 (batch-256 is a 1.3 negative),
no new routers / station embeddings / feature selection.

## Protocol (data_version 8, temporal only — same honest protocol as 2.0, denser seeds)

Train on the official train split (2017–2020, n=9,803); early-stop / select on the
official val split (2021–2022, n=4,805); evaluate on the untouched test split
(2023–2025, n=6,620). aux2020 (2020 slice of train, n=2,519) diagnostic only.
Winners selected by **3-seed mean val RMSE** among mlp/fg/plr (phase 1 = seed 42 for
all configs; phase 2 = seed 7 for the top-N per family; phase 3 = seed 123 for the
top-M per family). Patience-60; AdamW + warmup 5% + cosine; grad clip 1.0;
median-impute → StandardScaler → clip [−5, 5] fit on train only; target in original
units; `cudnn.deterministic=True`.

**data_version 8 (v7 → v8):** the v7 config generator omitted `hidden_sizes` for
`[384, 384]` configs (the sweep defaults have no `hidden_sizes` key and `build_model`
falls back to `[256, 256]`), so those 16 configs (9 mixed + 7 fifty-four) were
silently trained at [256, 256]. Fixed in `make_configs.py` (hidden_sizes is always
emitted) and `data_version` bumped to 8 so the v7 artifacts are invalidated and every
config re-trains with its intended architecture. The v7 run's non-`[384,384]` results
(including all reported winners) are unaffected.

**Cross-node bit-identity caveat:** the v8 run reproduced 2.0's anchor curve
bit-identically on a different node (offline comparison via
`compare_anchor_vs_2.0.py` → `artifacts/anchor_vs_20_comparison.json`, max|diff| = 0;
the earlier ~4-5% relative mismatch was the v7 `hidden_sizes` bug, not cross-node
nondeterminism). General cross-node bit-identity is still not guaranteed
(PTX-JIT/driver/cuDNN), but the observed reproduction was exact for the compared
config, and the RNG-guard live-vs-anchor bit-identity (within-run) is exact by proof.

## Sweep design

178 phase-1 configs (all `mlp`), generated by `make_configs.py` from the documented
grids below; `config.yaml` is the committed output. See `make_configs.py` for the
full spec and the per-family id lists in `config.yaml`.

| family | n phase-1 | grids (axes) | phase-2 top-N | phase-3 top-N |
|---|---:|---|---:|---:|
| `2regime_mixed` (winner) | 100 | shape × lr; loss × dropout; wd × mixup; act × depth; lr × huber; SWA start-frac × configs; anchors | 60 | 24 |
| `2regime_96` (debias) | 46 | width × dropout × lr; huber at small nets; width × mixup; SWA start-frac × configs; anchors | 30 | 12 |
| `2regime_54` (refine) | 32 | width × lr; loss × act; width × huber; SWA start-frac × configs; mixup + anchors | 20 | 8 |

Job count: 178 phase-1 + 110 phase-2 + 44 phase-3 + 6 champion ≈ **338 job-seeds**.
Budget math (from 2.0 timing): MLP-only per-seed mean 63 s; at 8 workers / 76 %
utilization ≈ 6.2 GPU-h ≈ **~55–60 min wall**. Resumable; `--phase2-top-n` /
`--phase3-top-n` / `--families` / `--only` trims keep the session inside the 2 h wall cap.
"""

CLOSING = r"""## Reproducibility

```bash
cd notebooks/experiment/derived_8.4-eval-mlp-2.1
uv run --no-sync python make_configs.py            # deterministic -> config.yaml (committed)
uv run --no-sync python verify_rng_guard.py        # pre-GPU RNG-guard proof (CPU, ~1 min)
uv run --no-sync python run_mlp_sweep.py --resume  # phases 1-3 (~1 h wall, 8 workers, H100)
uv run --no-sync python run_mlp_champion.py --top-n 1   # 5-seed champion ensembles
uv run --no-sync python run_mlp_eval.py            # leaderboard, per-regime, ensembles, figures
uv run --no-sync python compare_anchor_vs_2.0.py   # cross-version anchor bit-identity evidence
uv run --no-sync python analyze_bias.py            # bias^2/MSE diagnostic
uv run --no-sync python analyze_swa.py             # SWA deployment + bit-identity stack check
uv run --no-sync python analyze_selection.py       # selection reliability (1/2/3-seed)
uv run --no-sync python analyze_overfitting.py     # overfitting-symptom analysis
uv run --no-sync python analyze_extrapolation.py   # OOD check
uv run --no-sync python analyze_stopping.py --tag 20  # stopping-rule + SWA-val replay
cd notebooks && nb execute experiment/derived_8.4-eval-mlp-2.1/derived_8.4-eval-mlp-2.1.ipynb --uv
uv run --no-sync python generate_readme.py         # regenerate this README from the notebook
```

- `verify_rng_guard.py` proves the RNG guard: a `swa=true` job's LIVE val curve
  is bit-identical to its `swa=false` anchor's (max|diff| = 0 on `w384x384_d0.3_gelu`
  vs `w384x384_d0.3_gelu_swa085`, 2regime_54 cluster 0, seed 42 — both on CPU in the
  pre-GPU check and on the H100 node inside the GPU job). The real-sweep proof is the
  `analyze_swa.py` bit-identity stack check (136/136 pairs, see the SWA section).
  The anchor-vs-2.0 offline comparison (2regime_54 `w384x384_d0.3_gelu`, seed 42,
  spec 0) is bit-identical over the overlapping epochs (max|diff| = 0.0; best val
  0.06624 @ epoch 240 in both; reproducible via `compare_anchor_vs_2.0.py` →
  `artifacts/anchor_vs_20_comparison.json`) — retroactively confirming the v7
  mismatch was the `hidden_sizes` bug, not cross-node nondeterminism. The 6-epoch
  verify cap itself remains a sanity figure only (the cap changes the LR warmup vs
  the real runs).
- Configurations pinned in `config.yaml` (generated by `make_configs.py`); seeds
  {42, 7, 123} for the sweep, {42, 7, 123, 2024, 999} for the champion step;
  `data_version: 8`. The non-SWA anchors reproduce 2.0 bit-identically (stack
  check on the H100); a `swa_late` config's live val curve equals its anchor's
  (RNG-guard proof, verified on CPU).
- Artifacts: `models/`, `artifacts/`, `verify_rng_guard_out/` (gitignored, except
  `summary.json`), `sweep_results.csv`, `metrics_summary.csv`,
  `per_regime_metrics_summary.csv`, `bias_summary.csv`, `ood_summary.csv`,
  `stopping_20_*.csv`, `selection_summary.csv`, `swa_seed_meta.csv`,
  `swa_bit_identity.csv`, `timing_log.json`, `artifacts/anchor_vs_20_comparison.json`,
  figures, and the report notebook. All numbers in this README come from the
  executed notebook.

## Caveats

- The XGBoost 2-regime reference (0.815) was itself test-selected in eval-1.1; all
  honest MLP claims use val-based selection. `test-best` rows are reporting only.
- The mixed family's c1 (54+10) half inherits the 54-family's weak OOD extrapolation;
  the pure-96 family remains the best OOD model.
- 2025 test coverage is partial for several stations; year-2025 numbers should be read
  with the same caution as 1.x/2.0.
- The `_swa` (start 0.6) configs are the 2.0-recipe re-check under the RNG guard; the
  swept `_swa07x/075/080/085` configs are the 2.1 story. Neither deploys (0/152).
- The champion step ran `--top-n 1` (top-1 per family × extra seeds {2024, 999} = 6
  jobs); the plan's "top-2 mixed + top-1 96 + top-1 54" was not expressible because
  `--top-n` applies uniformly per family (documented deviation, see
  `docs/plans/20260810-mlp-2.1.md`).
"""


def notebook_sections(nb_path: Path) -> list[tuple[str, str]]:
    """Return [(markdown source, joined stdout)] for each notebook section.

    The first markdown cell (title) is skipped (the PREAMBLE carries the title
    and intro). Every following markdown cell starts a section; the stdout of
    the code cell(s) below it (until the next markdown cell) is appended.
    """
    cells = json.loads(nb_path.read_text(encoding="utf-8"))["cells"]
    sections: list[tuple[str, str]] = []
    for cell in cells:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown":
            if src.startswith("# "):  # title cell -> skip
                continue
            sections.append((src.rstrip(), ""))
        elif sections:
            stdout = []
            for out in cell.get("outputs", []):
                if out.get("output_type") == "stream" and out.get("name") == "stdout":
                    stdout.append("".join(out.get("text", [])))
            header, body = sections[-1]
            sections[-1] = (header, body + "".join(stdout))
    return sections


def main() -> None:
    parts = [PREAMBLE.rstrip(), ""]
    for header, stdout in notebook_sections(NB_PATH):
        parts.append(header)
        parts.append("")
        parts.append(stdout.rstrip())
        parts.append("")
    parts.append(CLOSING.rstrip())
    OUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    n_sections = len(notebook_sections(NB_PATH))
    print(f"[generate-readme] wrote {OUT_PATH.relative_to(EXP_DIR)} "
          f"({n_sections} notebook sections, {len(PREAMBLE) + len(CLOSING)} chars of prose)")


if __name__ == "__main__":
    main()
