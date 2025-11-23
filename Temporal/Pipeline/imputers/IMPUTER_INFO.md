# Drawing Board

**Jakob Balkovec**

## Voting Strategy for Filling Missing MDR Data

We have missing values in many features. There is no single “best” imputation method across all stations, seasons, or sensors. Some methods work well on short gaps, some only work on long gaps, and some collapse completely when data gets messy.

So instead of picking one, we let multiple methods attempt to fill the same gap and then vote...

---

### 1. Goal

Produce a single “best guess” for each missing point by combining several imputation techniques:

- Simple methods
  - Linear interpolation
  - Forward or backward fill
  - Rolling mean
- Seasonal or climatology methods
  - Day-of-year averages
  - Seasonal smoothing
- Heavier statistical methods
  - KNN imputation
  - Low-rank matrix factorization
- Model-based methods (optional)
  - Small regression model trained on nearby stations
  - A tiny temporal model

Each one is treated as an independent **imputer**.

---

### 2. Common interface for all imputers

Each imputer returns two things:

1. A filled value
2. A confidence score (the imputer's belief that its answer is reasonable)

Confidence might come from:

- Gap length (shorter gaps mean higher confidence)
- Historical error of the method on similar conditions
- Smoothness of the filled segment
- Seasonal alignment
- Local variance (stable periods are easier to fill)

The voting system only requires that each method defines some confidence value.

---

### 3. Base weighting per imputer

Before filling anything, we evaluate all imputers on a known region with no missing data by **artificially masking** sections and testing how well each method recovers the truth.

From that, each imputer gets a **base weight**:

- High weight if consistently accurate
- Medium weight if it performs well only under some conditions
- Low weight if it’s weak but still useful as a safety net

These are global weights that never change during actual imputation.

---

### 4. Sample-level voting

For a missing value at time _t_ and station _s_:

1. Every imputer attempts to fill it.
2. Every imputer produces a confidence score for that specific gap.
3. Effective weight = base weight \* confidence.
4. Combine values with weighted average: $$\text{filled value} = \frac{sum(w_i * v_i)}{sum(w_i)}$$

If a method is confident and historically good, it influences a lot. If it’s shaky, it still contributes but barely.

---

### 5. Outlier suppression

Sometimes one imputer outputs nonsense (for example linear interpolation during a huge weather jump).
To protect against this:

- Compute the median of all imputed values.
- If an imputer’s value is far outside a small band around the median, clamp its weight heavily.
- Keep it in the vote but at a tiny influence so it can’t distort the result.

---

### 6. Special cases

Certain gap structures require special treatment:

**Short gaps (1–3 days)**

- Prefer linear interpolation, rolling means, short-term smoothing.
- Confidence for simple methods goes up.

**Long gaps (20+ days)**

- Prefer climatology, seasonal decomposition, maybe KNN across stations.
- Interpolation confidence collapses.

**High variance periods**

- Confidence for rolling mean drops.
- Confidence for seasonal or neighbor regression increases.

The voting system automatically adapts as long as each method adjusts its confidence correctly.

---

### 7. Temporal smoothing after voting

The filled series should behave like real environmental data.
After combining:

- Apply optional light smoothing.
- Do not oversmooth since that hides real events.
- Just enough to remove single-day spikes caused by imperfect imputers.

---

### 8. Why this strategy matters

- No single imputation method is reliable on its own.
- Voting lets us combine strengths and cancel weaknesses.
- We get station-level continuity even for nasty gaps.
- This keeps downstream ML models from getting destabilized by garbage imputations.

Overall, the voting system gives us a flexible and robust way to fill missing MDR data before we even start the soil moisture modeling.
