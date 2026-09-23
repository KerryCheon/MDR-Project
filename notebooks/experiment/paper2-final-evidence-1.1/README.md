# Paper 2 ECE final evidence 1.1

Reproduction of the missingness-aware ECE v3 routing evidence with audited,
Washington-calibrated G_API class-to-local-expert mappings. This bundle is
separate from v1.0; it does not assemble or draft the paper.

## Protocol

- Fit static routers and experts on Washington `derived_8.4` trainval. Fit
  auxiliary calibration experts on WA train and choose calibration parameters
  and Guarded's class alignment on WA validation. Use the ECE
  `derived_8.4_ece_v3` test split only for final evaluation.
- Evaluate V0, Backbone54, Guarded_Backbone54, and the direct Global_Single_54
  reference with seeds `[42, 7, 13, 101, 123]` and policies
  `as_routed`, `auto_hard`, `auto_soft`, `c0_only`, and `c1_only`.
- Preserve salvage 2.0's G_API class-to-local-index behavior for V0 and
  Backbone. For Guarded, compare both class-to-local-index permutations using
  synthetic-SMAP-masked WA validation and freeze the lower-RMSE mapping before
  ECE evaluation; an exact tie selects class 0 → local c0.
- Keep the historical/canonical dry-assigned oracle index convention separate
  from the G_API mapping. `c0_only` and `c1_only` are non-deployable reference
  rows; their physical interpretation is not inferred from ECE results.
- The gate uses router-input availability only. ECE targets do not affect
  fitting, calibration, label alignment, thresholds, or routing.

## Reproduction

Submit the CPU-pinned ECE run from its runner directory:

```bash
cd notebooks/experiment/paper2-final-evidence-1.1/ece_guarded
sbatch run_cpu.sbatch
```

Execute the CSV-backed T7/T9/F5 supplement from `notebooks/`:

```bash
nb execute experiment/paper2-final-evidence-1.1/notebooks/paper2_ece_supplement.ipynb --uv --timeout 3600
```

## Outputs

- `ece_guarded/summary.csv`, `seed_metrics.csv`, `station_metrics.csv`, and
  `predictions_v3.csv` contain pooled, seed, station, and row-level evidence.
- `ece_guarded/wa_calibration.csv` records WA-only calibration and both
  Guarded class mappings; `routing_audit.json` and `routing_audit.csv` record
  the selected mapping, gate coverage, family-local indices, WA feature/target
  means, and non-use of ECE for fitting.
- The executed notebook exports policy crosswalks and derived tables under
  `ece_guarded/`, and T7/T9/F5 figures under `figures/`. The oracle chart labels
  use the approved family-specific dry-assigned convention; `policy_crosswalk.csv`
  records the local index behind each row (Guarded dry-assigned is local c1).
- The SLURM script and logs are retained under `ece_guarded/` and
  `ece_guarded/slurm/`; `run_manifest.json` records job and artifact status.

## Executed notebook summary

These values are copied from the executed T7 notebook output (pooled means over
five seeds; RMSE, bias, and ubRMSE are in soil-moisture units):

| Family | Policy | RMSE (seed SD) | Bias | ubRMSE | Deployable |
|---|---|---:|---:|---:|---|
| V0 | `as_routed` | 0.165908 (0.003051) | 0.117727 | 0.116899 | yes |
| V0 | `auto_hard` / `auto_soft` | 0.057768 (0.000617) | 0.028654 / 0.028655 | 0.050158 | yes |
| V0 | dry-assigned / complementary | 0.057768 / 0.192536 | 0.028654 / 0.185650 | 0.050158 / 0.051020 | no |
| Backbone | `as_routed` | 0.167431 (0.003401) | 0.141932 | 0.088816 | yes |
| Backbone | `auto_hard` / `auto_soft` | 0.057768 (0.000617) | 0.028654 / 0.028655 | 0.050158 | yes |
| Backbone | dry-assigned / complementary | 0.057768 / 0.192536 | 0.028654 / 0.185650 | 0.050158 / 0.051020 | no |
| Guarded | `as_routed` | 0.167431 (0.003401) | 0.141932 | 0.088816 | yes |
| Guarded | `auto_hard` / `auto_soft` | 0.057768 (0.000617) | 0.028654 / 0.028655 | 0.050158 | yes |
| Guarded | dry-assigned / complementary | 0.192536 / 0.057768 | 0.185650 / 0.028654 | 0.051020 / 0.050158 | no |
| Global | direct | 0.058634 (0.000735) | 0.015122 | 0.056635 | yes |

Guarded's G_API class-0-to-local-c0 mapping was selected using synthetic-SMAP-
masked WA validation (RMSE 0.073965; the class-0-to-local-c1 candidate scored
0.117180). All 150 ECE rows activate the availability gate and have a 100%
SMAP-missing router block. The approved dry-assigned oracle index is local c0
for V0/Backbone and local c1 for Guarded. WA means show local c1 lower on the
canonical SMAP feature (0.180078 vs 0.435981) and target (0.209373 vs
0.221648); keep these expert indices as family conventions, not physical class
labels. The notebook verifies all 750 Guarded seed-row G_API predictions are
class 0 and that `auto_hard` matches the complementary local c0 diagnostic on
this ECE set; the diagnostic is still non-deployable.

## Interpretation

The ECE split contains five disjoint in-situ stations and a 30-day evaluation
window. Report results as a small, family-specific diagnostic under missing
SMAP router inputs. Do not treat the local expert indices, station descriptors,
or one-season errors as evidence of universal physical regimes or causal
geography effects.
