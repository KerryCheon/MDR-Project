# Feature Availability Warning

## SMAP (Soil Moisture Active Passive)

- SMAP mission launched in **2015**
- All data prior to 2015 contains **true N/A values** for SMAP-derived features
- Pre-2015 missingness is due to **sensor unavailability**, not imputation failure
- This introduces a structural temporal shift in the feature space
- Missingness is correlated with year and may implicitly encode time regime

This should be explicitly acknowledged when interpreting model performance across years

---

**Inlcude this in the paper or mention when describing the dataset and temporal splits.**

_Jakob Balkovec_
