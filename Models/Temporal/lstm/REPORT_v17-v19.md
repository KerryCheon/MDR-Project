# LSTM Experiments — Commit Summary (v17–v19 + split audit)

**Commit:** `fd985ea` "lstm experiments" — Kerry Cheon, 2026-06-26
**Dataset:** derived_8.0 (train 2017–2020 / val 2021–2022 / test 2023–2025)
**Motivation:** Act on REPORT.md §7 step #1 (audit the split) and test whether the
val→test gap is fixable, after v7–v16 plateaued at v9's test R² = 0.747.

---

## 1. Split audit (`audit_split.py` → `outputs_audit/`)

The audit **overturns the original report's headline diagnosis.** The report assumed
test stations were held-out/spatially different. They are not.

- **Split is TEMPORAL, not spatial.** The same 5 stations (Darrington, Quinault,
  SourdoughGulch, Spokane, Touchet) appear in train, val, and test — only the years
  differ. Station overlap train∩test = 5/5.
- **Target drift is modest.** Mean |per-station train→test shift| = **0.0247 m³/m³**;
  2/5 stations exceed 0.02. Largest: **Quinault +0.050**, **Touchet +0.045**.
- **SMAP features are the biggest OOD source:** ~**17–20% of test rows** fall outside
  the train [p1,p99] range for `SMAP_sm_am/pm_interp`. Static features (lat, elev) 0% OOD.
- **Touchet caveat:** val_rows = 0 and test_rows = 170 with collapsed variance — its
  per-station R² is unreliable and drags the aggregate.
- No monotonic annual trend in the target (2017–2025 annual means oscillate 0.18–0.23).

**Conclusion:** the gap is a **temporal / feature-range shift** (esp. SMAP), not a
station-population shift.

---

## 2. Variants

| Variant | Idea | Test R² | vs v9 (0.747) |
|---|---|---|---|
| **v17** | v9 + **per-station target normalization** (subtract per-station train mean, predict residual, re-add) | **0.713** | −0.034 |
| **v18** | **Rolling window** — move 2021 into train, val 2022, test 2023–2025 | **0.683** | −0.064 |
| **v19** | **3-fold temporal CV** (expanding window) | A 0.719 / B 0.683 / C 0.718 (mean ≈ 0.707) | −0.03 to −0.06 |

Details:
- **v17** — station means saved to `station_means.json`; reconstructed test R² = 0.713,
  bias −0.0028 (near-unbiased). Normalizing the per-station mean did **not** recover test
  performance — consistent with the earlier finding that error is variance, not bias.
- **v18** — adding a year of training data and shifting the val year made test **worse**.
- **v19** — three expanding-window folds (train→2020/21/22) all land in **0.68–0.72**
  regardless of where the temporal cut is, confirming a stable ceiling.

---

## 3. Bottom line

- The split audit **confirms the shift is temporal (SMAP feature range) not spatial** —
  correcting the original REPORT.md hypothesis.
- **None of v17–v19 beat v9 (0.747).** Per-station mean normalization, a rolling window,
  and temporal cross-validation all reproduce the same ~0.68–0.72 test ceiling.
- The systematic (per-station mean) component is small; the residual **variance** is what
  caps test R². De-biasing / re-centering can't fix it.
- **v9 remains the production candidate.** Cracking 0.80 will require attacking the SMAP
  feature-range OOD directly (domain adaptation / feature robustification / more recent
  training data), not another normalization or temporal reslice.
