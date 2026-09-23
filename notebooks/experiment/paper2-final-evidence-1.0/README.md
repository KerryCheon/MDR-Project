# Paper 2 final evidence run

Versioned reproduction and supplementary evidence for `writeup2/outline-v6.md`.
This directory contains run code and machine-readable evidence; it does not
assemble the paper.

## Runs

- `run_temporal_gpu_debug.sbatch` and `run_loso_gpu_debug.sbatch` reproduce the
  pinned routing experiment on one H100 each. Submit LOSO with an `afterok`
  dependency on temporal; both stages fit within the two-hour `gpu_debug`
  limit. The submitted job IDs are recorded in `run_manifest.json`.
  `analyze_agreement.py` exports router agreement and K-sweep CSVs.
- `ece_guarded/run_cpu.sbatch` extends the ECE salvage harness with the Guarded
  shared-54 router. Washington train/validation data are the only fit and
  calibration source; canonical ECE v3 is evaluation-only.
- `notebooks/paper2_main_figures.ipynb` produces the temporal and LOSO evidence
  figure. `notebooks/paper2_ece_supplement.ipynb` produces the ECE diagnostic
  panel. Both notebooks read this directory's CSV outputs.

## Outline-v6 work covered

- W4′: add the Guarded shared-54 family to the five-seed ECE evaluation,
  preserve salvage C0 for V0/Backbone and Guarded c1 for Guarded, and retain
  the Washington-only synthetic-SMAP-mask calibration evidence.
- W8: reproduce router agreement, temporal T1, and LOSO L1; execute the main
  F1/T8 and ECE T7/T9/F5 evidence notebooks from the recorded CSVs.
- Optional F4/F6 and the contingent Guarded-versus-global paired LOSO follow-up
  are outside this run. No paper draft or final paper files are assembled.

## Guarded expert-label mapping

Policy names `c0_only` and `c1_only` are semantic dry-oracle and wet-diagnostic
aliases, but their family-local indices follow the documented family convention:
V0 and Backbone preserve salvage C0=dry (local index 0), while Guarded uses its
fit-frame canonicalization (local index 1=drier). Both WA-only canonical
feature means and WA target means are lower for local index 1 in all three
families, so the inherited V0/Backbone salvage labels conflict with those
mean-based orderings. This conflict is recorded rather than resolved from ECE
outcomes. The exact per-family mapping and WA feature/target means are saved to
`ece_guarded/routing_audit.json`.

## Machine-readable outputs

Routing CSVs are written at this directory's root: agreement summary and
pairwise tables, the LOSO fold agreement audit, K-sweep quality, and the
temporal/LOSO seed, station, year, and cluster outputs. ECE base outputs are
written under `ece_guarded/`: `summary.csv`, `seed_metrics.csv`,
`station_metrics.csv`, `predictions_v3.csv`, `wa_calibration.csv`, and
`routing_audit.json`. The ECE notebook adds `policy_crosswalk.csv`, site and
policy summary tables, semantic routing-share tables,
`salvage_c0_dry_comparison_by_seed.csv` and its summary, and
`f5_guarded_daily_overlay.csv`. Figures are saved under `figures/`.
SLURM stdout/stderr are kept under `artifacts/slurm/` and `ece_guarded/slurm/`.

## Main evidence observed

The following values are copied from the executed main notebook stdout. T1
intervals summarize expert-seed variation; comparisons against V0 remain
sensitivity-anchored, while Guarded and Backbone are paired in this harness.

| T1 strategy | Evidence role | Seeds | Mean R² | Seed-level 95% CI | Mean RMSE |
|---|---|---:|---:|---:|---:|
| Guarded backbone | Paired in harness | 30 | 0.811843 | [0.811332, 0.812354] | 0.044187 |
| Backbone | Paired in harness | 30 | 0.811724 | [0.811213, 0.812235] | 0.044201 |
| V0 | Sensitivity-anchored | 30 | 0.811843 | [0.811332, 0.812354] | 0.044187 |

T8 pairs Guarded and Backbone on the same five seeds and seven held-out
stations. Guarded gains are on SourdoughGulch (+0.059106), Spokane (+0.051448),
and Paradise (+0.015444), with five wins on each; the other four stations tie
on all five seeds. These fold counts are descriptive at seven stations.

## ECE label audit and interpretation

The corrected five-seed run preserves salvage C0=dry for V0/Backbone and
Guarded local c1=dry. The WA audit places local 1 lower on both the canonical
SMAP feature (0.1801 vs 0.4360) and target mean (0.2094 vs 0.2216) for all three
routers; therefore the V0/Backbone mapping is intentionally historical and
does not match those WA mean orderings. The notebook exports this crosswalk,
the WA calibration table, and a seed-paired comparison against salvage C0.

Observed pooled ECE v3 RMSE means (five expert seeds) are:

| Family | Static route | `auto_hard` / `auto_soft` | `c0_only` oracle | `c1_only` diagnostic |
|---|---:|---:|---:|---:|
| V0 | 0.1659 | 0.0578 | 0.0578 | 0.1925 |
| Backbone54 | 0.1674 | 0.0578 | 0.0578 | 0.1925 |
| Guarded_Backbone54 | 0.1674 | 0.1925 | 0.1925 | 0.0578 |
| Global_Single_54 | direct: 0.0586 | — | — | — |

Every oracle and diagnostic row is non-deployable. Guarded local c1 (the
approved semantic dry mapping) has RMSE 0.1925 versus 0.0578 for salvage
Backbone C0, a mean paired difference of +0.1348 across five seeds. The
WA synthetic-mask validation comparison is mixed: the auxiliary router is
better for Backbone54 (0.0740 vs static 0.0869), while static is better for
Guarded (0.0869 vs auxiliary 0.1172) and V0 (0.0681 vs auxiliary 0.0728).
These results expose the preserved label convention and Guarded routing
failure; they do not justify changing labels from ECE outcomes. ECE targets
are used for evaluation only.

To reproduce the GPU stages from this directory:

```bash
temporal_job=$(sbatch --parsable run_temporal_gpu_debug.sbatch)
temporal_job=${temporal_job%%;*}
sbatch --parsable --dependency=afterok:${temporal_job} run_loso_gpu_debug.sbatch
```

The ECE runner is CPU-pinned in `ece_guarded/config.yaml` and is submitted with
`cd ece_guarded && sbatch run_cpu.sbatch`.

`comparators/` contains a frozen copy of the formal-eval 1.0 temporal seed
table and its config/readme provenance. Those rows are sensitivity references,
not paired Guarded comparisons. The paired primary ablation comes from the
two routing reproduction outputs in this directory.
