## Residual Learning with Two XGB Models (Notes)

**High-level thought:**
Instead of forcing one model to learn _everything_, split the job into two parts.

---

### 1. Base / Physics Model (Model A)

Goal:
Capture the **obvious, dominant soil moisture drivers**.

Train Model A using:

- API
- rain accumulations (3/7/30d)
- lagged soil moisture features
- DOY / seasonality features

Mental model:

> This model learns the _bulk moisture state_
> aka “how wet should the soil be given recent weather + season”

---

### 2. Residual Model (Model B)

After Model A:

- Compute residuals:

$$r_t = y_t - \hat{y}_t^{(A)}$$

Now train Model B **only on what Model A missed**.

Features for Model B:

- Radar:
  - `SAR_ratio`
  - `SAR_diff`
- Optical:
  - `NDMI`
  - `NDVI`
- Dynamics:
  - gradients
  - rolling std / volatility features

---

### Final Prediction

Combine both models:

$$\hat{y}_t = \hat{y}_t^{(A)} + \hat{r}_t^{(B)}$$

---

### Why this is powerful

- Model A eats the easy physics (rain + memory + seasonality)
- Model B focuses on subtle corrections
- Radar and optical signals stop being drowned out by rainfall dominance
- Much cleaner signal separation than a single monolithic model

---

### Important notes

- Target stays **soil moisture**
- No hyperparameter gymnastics
- No fancy tricks
- This is a **structural change**, not tuning

This is a legit, high-impact modeling move in environmental ML.
