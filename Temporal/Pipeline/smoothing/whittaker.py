# Jakob Balkovec
# Now 27th 2025
# whittaker.py
#
# Pure Whittaker smoothing implementation
# Reference: Eilers (2003), "A Perfect Smoother"

import numpy as np

def whittaker_smooth(x, lmbda=10, mask=None):
    # pre: x is a 1D array-like of floats (with possible NaNs)
    #      lmbda is the smoothing parameter (higher = smoother)
    #      mask is an optional 1D array-like of same length as x,
    #           with 1 for observed values and 0 for missing
    # post: returns a 1D numpy array of smoothed values
    # desc: performs Whittaker smoothing on the input data

    x = np.asarray(x, dtype=float)
    n = len(x)

    # default mask: observed where not NaN
    if mask is None:
        mask = ~np.isnan(x)

    mask = np.asarray(mask, dtype=float)

    # replace NaN with zeros for solving
    y = np.where(mask == 1, x, 0.0)

    # build penalty matrix (second order differences)
    # D is (n-2) x n:
    # each row: [0,...,1,-2,1,...,0] over consecutive triples
    D = np.zeros((n-2, n))
    for i in range(n-2):
        D[i, i:i+3] = [1, -2, 1]

    # left-hand side: diagonal weights + lambda * D^T D
    W = np.diag(mask)
    A = W + lmbda * (D.T @ D)

    # solve linear system
    try:
        z = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        # fallback: least-squares
        z, *_ = np.linalg.lstsq(A, y, rcond=None)

    return z
