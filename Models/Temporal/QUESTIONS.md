Dr. Zhou,

I’ve been pushing this pretty hard over the past few weeks...running a range of modeling experiments (gradient boosting, stacking, post-hoc calibration, SMAP fusion, strict temporal splits, etc.). Performance has improved incrementally, but I’m starting to feel like we’ve extracted most of what we can from purely algorithmic tuning.

I keep going back and forth, and at this point, it seems less like an ML optimization problem and more like a physics alignment problem. I’d really value your perspective on whether the next gains should come from better retrieval assumptions, signal decomposition, or structural modeling changes rather than additional hyperparameter work.

Here’s what I’d love your thoughts on:

---

## A. Retrieval + Physics Constraints

1. Given Sentinel-1 C-band’s limited penetration over vegetation, what signal strategy would you consider best practice?
   - VV only?
   - VV/VH ratio?
   - RVI?
   - Or something closer to canopy correction?

2. Would you recommend applying something like a Water Cloud Model correction (even approximate) before ML, or is it better to let the model learn vegetation attenuation implicitly?

3. On geometry:
   - Should I explicitly include local incidence angle / orbit direction / look angle?
   - Or are DEM-derived slope/aspect features enough?

4. Surface roughness:
   I’m currently using rolling abs change of VV/VH and spike timing as roughness proxies. Do those make sense physically, or is there a better roughness feature family I should be looking at?

5. Freeze/thaw + snow:
   If I’m staying “no ground sensors”, what screening or flags would you apply?
   - Snow cover?
   - Temperature?
   - SWE?
   - Something else?

6. Temporal lag:
   In your experience, what lag between rainfall and C-band response is actually meaningful at 5 cm? I’ve tried short lags, rolling sums, API-style accumulation, etc., but I’m curious what’s physically realistic.

---

## B. SMAP Fusion (recent addition)

We recently added SMAP and it helps slightly, but not dramatically (I expected a different outcome, but oh well...). I’m trying to understand how to use it properly.

7. When fusing SMAP with S1, do you prefer:
   - SMAP as a prior (baseline + residual learning)?
   - Or just another feature in the model?

8. AM vs PM:
   Is one generally more stable or less biased?
   Would you recommend:
   - AM only?
   - PM only?
   - Or combining them?

9. Gap filling:
   I’m currently imputing SMAP and adding mask features.
   Is that the right approach, or would you avoid imputation and only train on valid pixels?

10. Spatial mismatch:
    Given SMAP’s footprint vs station scale, would you recommend:

- Lightweight downscaling?
- Or treating SMAP strictly as a coarse context feature?

---

## C. Modeling Strategy + Evaluation

For context: temporal split, limited stations, tabular features, mostly boosting + RF so far.

11. If this were your setup, what model family would you bet on?

- Gradient boosting?
- Linear + engineered interactions?
- State-space / dynamic model?
- Something else?

12. Station generalization:
    Should I do leave-one-station-out as a stress test, even if the score drops, just to test transferability?

13. Target transform:
    Would you recommend predicting anomalies (remove seasonal climatology per station) and adding back later, versus predicting raw soil moisture?

14. Leakage audit:
    I’m using train-only seasonal anomaly and z-score computation. Are there subtle leakage patterns you’ve seen in similar workflows that I should double-check?

15. Feature sanity check:
    If the model is behaving physically, what should rise to the top?

- VV?
- API?
- SMAP?
- NDMI?
- Something else?

---

## D. Concrete Next Experiments

16. If you had to suggest 2–3 diagnostic ablations, what would they be?

- Remove SMAP entirely?
- VV only?
- Remove optical?
- Something more targeted?

17. If performance improves with SMAP, how would you prove it’s not just learning a trivial “SMAP ≈ truth” mapping?

18. Beyond R², what metrics would you consider essential for soil moisture?

- Bias?
- Seasonal phase error?
- Event response fidelity?
- Something more physically grounded?

---

Big picture: we’re hovering just below what feels like a structural ceiling. I’m trying to make sure the next improvements come from better physics alignment rather than just more ML tuning.

Appreciate any direction you think is most promising.

_Jakob & Kerry_
