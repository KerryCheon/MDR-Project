# paper2-final-evidence-1.2 — Paired Global_Single re-run

Replaces the sensitivity-anchored `Global_Single` rows (copied from
`derived_8.4-formal-eval-1.0`) with a fresh refit inside this harness, under
the **same protocol** as `paper2-final-evidence-1.0` Guarded/Backbone runs:

- Same `derived_8.4` splits (trainval 14608 / test 6620; LOSO folds per split).
- Same shared 54-feature backbone (`config.yaml:shared_backbone_54`).
- Same `exact_params` (2500 trees, lr 0.005, d9, `hist/cuda`) and `router_seed 42`.
- Same seed lists: 30 temporal `[42, 7, 13, ...]` (42 first), 5 LOSO `[42, 7, 13, 21, 55]`.
- Same drivers: `run_temporal.py` / `run_loso.py` / `run_worker.py` verbatim copies of 1.0.

Only `Global_Single_54` is run here (`--config-id Global_Single_54`):
30 temporal jobs + 35 LOSO jobs (5 seeds x 7 stations). Guarded / Backbone /
StationMean are **not** refit; their paired rows stay sourced from 1.0.

## Outputs (replace formal-eval sensitivity Global)

- `temporal_seed_summary.csv` (30 rows) + `temporal_seed_station/year/cluster.csv`
- `loso_seed_station.csv` (35 rows) + `loso_seed_year/cluster.csv`
- `global_paired_temporal_seed.csv` / `global_paired_temporal_summary.csv`
- `global_paired_loso_seed_station.csv` / `global_paired_loso_summary.csv`
  (each with `evidence_role=paired_in_harness`, `source_harness=paper2-final-evidence-1.2`)
- `pinned_configurations.json` / `pinned_configs.csv` (1 pinned config)

Built by `build_paired_global.py` (saved here, not inline):

```bash
cd notebooks/experiment/paper2-final-evidence-1.2
sbatch run_temporal_gpu_debug.sbatch
sbatch --dependency=afterok:<temporal-jobid> run_loso_gpu_debug.sbatch
uv run --project "$PWD/../../.." --no-sync python build_paired_global.py
```

Downstream `writeup2/report-assemble-1.4` sources Guarded/Backbone from 1.0
and Global from here; seed-by-seed pairing holds because seed lists and
protocol are identical. Formal-eval Global rows are then fully superseded.

## Completed run (2026-10-03, Slurm jobs 2185508/2185509, ac055 H100)

- Temporal: 30/30 ok (~10 min). `mean_r2 0.779794, sd 0.001279, rmse 0.047802` —
  bit-identical to formal-eval sensitivity rows on all 30 seeds.
- LOSO: 35/35 ok (~11 min). Station-mean `r2 0.579737, rmse 0.060990` —
  identical to formal `loso_seed_station.csv` Global rows (diffs ≤ 6e-17).
  Note: formal `loso_config_summary.csv` says 0.579500, but the mean of its own
  seed rows is 0.579737; the summary value looks stale.
- Paired check vs 1.0 Guarded (same 30 seeds): mean Guarded−Global R²
  `+0.032049`, Guarded higher on 30/30 seeds — now a true seed-paired result.
- Partition was moved `gpu_debug` → `gpu` (H100 node ac098 down); same H100 arch.
