# Fix derived_8.3 Post-Review (Simplified)

## Fix 1 — dataset_metadata.py line 1

**File:** `data/splits/derived_8.3/dataset_metadata.py`

**Change:** `derived_8.2 splits` → `derived_8.3 splits`

**Tool:** One `edit` call.

---

## Fix 2 — Restore config.yaml station blocks as comments

**File:** `data/splits/derived_8.3/config.yaml`

The 3 station blocks were **deleted** during generation. They need to be **re-added as commented-out** blocks, matching the style of `device_4` (lines 312–338 in the source `derived_8.2/config.yaml`).

**Action:** Manually insert the 3 blocks back with all content lines prefixed by `# ` after the 2-space indent. The 3 blocks to copy from `derived_8.2/config.yaml`:

| Block | Source Lines | Insert Before |
|-------|-------------|--------------|
| Touchet | 273–311 | current `# STATION 6` header (which was originally the next line after touchet) |
| BurntMountain | 474–512 | So it's commented between Paradise and BeaverPass |
| HartsPass | 552–590 | So it's commented between BeaverPass and MartenRidge |

Each block should look like this after commenting:
```yaml
  # -----------------------------------------------------------
  # STATION 5: Touchet SNOTEL #824
  # -----------------------------------------------------------
  # touchnet:
  #   request:
  #     base_url: null # SNOTEL data - no download needed
  #     station: "Touchet_WA_824"
  #     ...
  #     index: false
```

---

## Fix 3 — Strip config.yaml section from make script

**File:** `data/splits/derived_8.3/make_derived_8.3.py`

Remove lines 41–86 (the `# 2. Filter config.yaml` section). Since config handling is a one-time manual step, the script shouldn't have fragile auto-generated config modification code. Also remove the `import yaml` if it was added (the current script doesn't have it).

---

## Verification

1. Open `config.yaml` — the 3 stations are present but fully commented out
2. `dataset_metadata.py:1` reads `derived_8.3`
