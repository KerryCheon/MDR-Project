#!/usr/bin/env python3
"""Regenerate README.md for derived_8.4-eval-mlp-2.2.

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
NB_PATH = EXP_DIR / "derived_8.4-eval-mlp-2.2.ipynb"
OUT_PATH = EXP_DIR / "README.md"

PREAMBLE = r"""# Experiment: `derived_8.4-eval-mlp-2.2` — exploit the 54-family lr6e-4/gelu region + the mixed 3-layer gelu cell + finish the 96-family debias (~1.25 h gpu_debug H100 wall)

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

<!-- VERDICT — filled from the executed notebook after the GPU run -->

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
"""

CLOSING = r"""## Reproducibility

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
- Val selection was shown noisy for the 54/mixed families in 2.1 (Spearman −0.555 /
  −0.309 even at 3 seeds); 2.2's mitigation is denser seed coverage (phase-2/3
  top-Ns capped at family sizes) plus the val-year diagnostic — the selection rule
  itself is unchanged (protocol).
- The champion step runs the per-family `sweep.champion_top_n` (mixed top-2 +
  54 top-1 + 96 top-1) × extra seeds {2024, 999}; `--top-n N` CLI overrides
  (uniform, 2.1 parity). See `docs/plans/20260810-mlp-2.2.md`.
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
