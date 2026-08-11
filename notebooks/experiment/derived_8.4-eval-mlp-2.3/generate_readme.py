#!/usr/bin/env python3
"""Regenerate README.md for derived_8.4-eval-mlp-2.3.

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
NB_PATH = EXP_DIR / "derived_8.4-eval-mlp-2.3.ipynb"
OUT_PATH = EXP_DIR / "README.md"

PREAMBLE = r"""# Experiment: `derived_8.4-eval-mlp-2.3` — final frontier check for the mlp-2.0 architecture: 320²-hubergelu/lr6e-4 cell refinement + the mixed gelu 3-layer at low lr + the 96 lr3e-4 debiased pool (~1.75 h gpu_debug H100 wall)

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

[FILLED AFTER THE SWEEP FROM THE EXECUTED NOTEBOOK — see the notebook stdout]

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
"""

CLOSING = r"""## Reproducibility

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
