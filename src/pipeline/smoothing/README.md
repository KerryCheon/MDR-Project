# Whittaker smoothing

This file documents the `whittaker.py` implementation in this folder. The code provides a simple, pure NumPy implementation of Whittaker smoothing (Eilers, 2003) which solves a penalised least-squares problem to produce a smooth estimate of an input series while handling missing values.

## Mathematical formulation

Given observed values $y_i$ for $i=1\dots n$ with (non-negative) weights $w_i$ (where $w_i=0$ marks a missing observation), the Whittaker smoother finds a vector $z\in\mathbb{R}^n$ minimizing the objective:

$$\operatorname{argmin}_z \; \sum_{i=1}^n w_i (y_i - z_i)^2 + \lambda \sum_{j=1}^{n-2} (\Delta^2 z_j)^2$$

where $\lambda\ge 0$ is the smoothing parameter and $\Delta^2 z_j$ is the second difference operator:

$$\Delta^2 z_j = z_j - 2 z_{j+1} + z_{j+2}, \qquad j = 1,\dots,n-2.$$

Writing the weights as the diagonal matrix $W = \operatorname{diag}(w_1,\dots,w_n)$ and forming the discrete second-difference matrix $D\in\mathbb{R}^{(n-2)\times n}$ (each row of $D$ contains the pattern $[\dots, 1, -2, 1, \dots]$ over consecutive positions), the objective becomes

$$J(z) = (y - z)^T W (y - z) + \lambda z^T D^T D z.$$

Differentiating with respect to $z$ and setting the gradient to zero yields the linear normal equations:

$$ (W + \lambda D^T D) z = W y. $$

Note about the implementation: the code sets `y = mask * x` (so missing observations are zeroed) and `W = diag(mask)`. Since the mask is binary (`0`/`1`), we have `W @ (mask * x) = mask * x`, so the code solves the equivalent linear system `A z = y` where `A = W + \lambda D^T D` and `y = W x`.

## Implementation notes

- Input: A 1D array-like `x` (floats), optional `mask` (1 for observed values, 0 for missing), and `lmbda` smoothing parameter.
- Handling missing values: If `mask` is not supplied, it is inferred from `~np.isnan(x)`. The implementation substitutes NaN observations with 0 in the RHS `y = mask * x` and uses `W = diag(mask)`. This yields the same linear system because `W @ y = W @ (mask * x) = mask^2 * x = mask * x` for binary masks.
- Constructing the penalty: The matrix `D` is constructed with shape `(n-2, n)` and each row `i` sets `D[i, i:i+3] = [1, -2, 1]`. The penalty term is then `lambda * D^T @ D`.
- Solving: The equations are solved via `np.linalg.solve(A, y)` where `A = W + lambda * (D.T @ D)`. If a `LinAlgError` occurs (e.g. singular matrix), a least-squares fallback with `np.linalg.lstsq` is used.

## Edge cases and behavior

- For short series: If `n < 3` the second-difference matrix has shape `(n-2, n)`; e.g., for `n=2` this is `(0, 2)` and the penalty vanishes. For `n=1` the D construction would attempt to create a shape of `( -1, 1 )` which raises an error from NumPy. The implementation therefore expects `n >= 2`, with meaningful smoothing only when `n >= 3`.
- Missing values: Observations with zero weight (`mask == 0`) are treated as missing. The solution still finds `z` that minimizes the objective using only observed terms.
- Computational complexity: The matrix `A` is dense and the linear solve is therefore O(n^3) in the worst case. In practice for long series, it's recommended to use a banded or sparse solver because `D^T D` is a banded matrix with bandwidth 5 (due to the second-difference structure); using sparse linear algebra gives O(n) or O(n log n) solves for large `n`.

## Parameter notes

- `\lambda`: Controls the trade-off between fidelity to the data and smoothness. Larger $\lambda$ yields a smoother `z` (more emphasis on penalising second differences). Setting `\lambda = 0` returns `z = y` (no smoothing for fully observed points).

## Python API / usage

The function signature:

```
whittaker_smooth(x, lmbda=10, mask=None)
```

- `x`: 1D array-like input signal containing floats, can include `np.nan` for missing values.
- `lmbda`: smoothing parameter (float, default 10).
- `mask`: optional 1D array-like of same length as `x` with 1 for observed values and 0 for missing; if omitted, `~np.isnan(x)` is used.

Example usage:

```
import numpy as np
from smoothing.whittaker import whittaker_smooth

x = np.array([0.0, 1.0, np.nan, 0.5, 1.2, 1.0])
z = whittaker_smooth(x, lmbda=100)
print(z)
```

## Practical tips

- Selecting `\lambda`: There's no single universally optimal value. Cross-validation, generalized cross-validation (GCV), or visual assessment (and domain expertise) are common approaches. Smaller `\lambda` values preserve detail; larger `\lambda` values smooth more aggressively.
- Speed and memory: For very long series, convert `D` into a sparse matrix and use banded or sparse solvers (`scipy.sparse` and `scipy.sparse.linalg`) as `D^T D` is banded; this avoids forming dense matrices and reduces memory and runtime.
- Alternatives: The Whittaker smoother is related to smoothing splines and Kalman smoothing; for time series with structure (seasonality or trends), consider combining with decomposition or a model-based imputer.

## References

- P. H. C. Eilers (2003). "A Perfect Smoother". Analytical Chemistry.
