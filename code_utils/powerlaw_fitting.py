import numpy as np
import pandas as pd

from typing import Optional, Union


def degree_frequency_distribution(
    sample : Union[pd.Series, np.ndarray, list],
    trunc_threshold : Optional[Union[float, int]] = None
):
    """
    Compute the empirical degree frequency distribution (normalized histogram / PMF)
    for discrete observations.

    Parameters
    ----------
    sample : pandas.Series or array-like
        Observations (e.g., node degrees). Values are treated as discrete bins.
    trunc_threshold : float or int, optional
        If provided, only keep bins with x >= trunc_threshold.

    Returns
    -------
    x : np.ndarray
        Sorted unique values (bin centers).
    y : np.ndarray
        Normalized frequencies (probabilities) for each x.

    Notes
    -----
    - This function only computes the empirical distribution; it does not plot.
    - Values are treated as discrete categories. If you pass floats, you may
      unintentionally create many bins.
    """
    # Normalize input to a pandas Series for convenience
    if isinstance(sample, pd.Series):
        s = sample
    else:
        # Accept lists, numpy arrays, iterables
        try:
            s = pd.Series(sample)
        except Exception as e:
            raise TypeError("sample must be a pandas Series or array-like.") from e

    # Drop missing values early
    s = s.dropna()

    # Empirical PMF: counts per value, then normalize
    freq = s.value_counts().sort_index()
    y = (freq / freq.sum()).to_numpy(dtype=float)
    x = freq.index.to_numpy(dtype=float)

    # Optional truncation: keep only x >= threshold
    if trunc_threshold is not None:
        thr = float(trunc_threshold)
        mask = x >= thr
        x, y = x[mask], y[mask]

    return x, y
# =====================================================================
def fit_powerlaw(x, y):
    """
    Fit a power-law of the form: y = C * x^(-gamma)

    Taking logs:
        log(y) = log(C) - gamma * log(x)

    Parameters
    ----------
    x, y : array-like
        Data to fit. Must be positive after filtering.

    Returns
    -------
    gamma : float
        Power-law exponent.
    const : float
        By default, const = log(C). If return_C=True, const = C.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Build mask for valid points in log-domain
    mask = (x > 0) & (y > 0)

    x_fit = x[mask]
    y_fit = y[mask]

    if x_fit.size < 2:
        raise ValueError("Not enough valid points to fit (need at least 2).")

    # Log-transform
    lx = np.log(x_fit)
    ly = np.log(y_fit)

    # Design matrix for linear regression with intercept:
    # ly ≈ b0 + b1 * lx  where b0 = log(C), b1 = -gamma
    A = np.column_stack([np.ones_like(lx), lx])

    # Least squares fit
    coeffs, residuals, rank, s = np.linalg.lstsq(A, ly, rcond=None)
    logC, slope = coeffs
    gamma = -slope

    return gamma, logC
# =====================================================================
def powerlaw_func(x, gamma, const=1.0):
    """
    Compute a power-law (power) distribution:

        y = const * x^(-gamma)

    Parameters
    ----------
    x : float or np.ndarray
        Input values. Must be strictly positive (x > 0).
    gamma : float
        Power-law exponent.
    const : float, optional
        Multiplicative constant (default is 1.0).

    Returns
    -------
    y : float or np.ndarray
        Power-law evaluated at x.

    Notes
    -----
    - This function assumes x > 0. Values of x <= 0 will produce
      infinities or NaNs.
    - Large gamma or large x may lead to numerical underflow.
    """
    # Convert input to NumPy array for safe vectorized operations
    x = np.asarray(x, dtype=float)

    # Optional safety check (uncomment if you want strict enforcement)
    # if np.any(x <= 0):
    #     raise ValueError("power_distribution is only defined for x > 0.")

    # Compute power-law
    y = const * np.power(x, -gamma)

    return y
# =====================================================================