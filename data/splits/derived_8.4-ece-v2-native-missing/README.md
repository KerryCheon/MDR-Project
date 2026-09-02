# `derived_8.4-ece-v2-native-missing`

Corrected 499-column ECE evaluation split derived from `derived_8.4-ece`.

The parent split converted unavailable SMAP AM/PM retrievals to physical
`0.0` values before generating masks, lags, rolling statistics, and other
descendants. This version preserves unavailable SMAP values as `NaN` and sets
the three SMAP observation masks to `0`. The original split is unchanged.

## Rebuild

From the repository root:

```powershell
py -m uv run --project notebooks python data/splits/derived_8.4-ece-v2-native-missing/build_from_existing_split.py
```

The recovery builder verifies that all 150 evaluation rows are covered by
committed ECE satellite-cache intervals whose SMAP AM and PM retrieval values
are null. It aborts if a finite cached SMAP value is found or a date is not
covered.

## Result

- 150 rows and 499 columns, in the same order as the parent split.
- 85 SMAP-related columns corrected.
- 82 SMAP value/derived columns contain no finite values.
- 3 SMAP observation-mask columns are zero for every row.
- Non-SMAP features and targets are inherited unchanged from the parent split.

The raw and processed ECE station files are absent from this checkout, so the
canonical builder cannot presently be rerun from raw inputs. Its zero-fill bug
has also been corrected for future raw-data rebuilds.
