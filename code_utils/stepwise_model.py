import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from typing import List, Optional, Union, Dict, Any, Tuple



def _mnl_fit(
    Y: pd.Series,
    X: pd.DataFrame,
    **kwargs,
) -> Tuple[Optional[Any], float]:
    """
    Fit a multinomial logistic regression (MNLogit) and return (result, AIC).

    If the fit fails (singular matrix, separation, non-convergence, etc.),
    returns (None, +inf).
    """
    try:
        model = sm.MNLogit(endog=Y, exog=X)

        # You can silence optimizer output with disp=False in fit_kwargs
        res = model.fit(**kwargs)
        return res, float(res.aic)

    except Exception as e:
        warnings.warn(
            f"MNLogit failed for features {X.columns.to_list()}: {type(e).__name__}: {e}",
            RuntimeWarning,
        )
        return None, float("inf")
# -------------------------------------------------
def _find_best_feature(
    Y: pd.Series,
    X: pd.DataFrame,
    selected: List[str],
    remaining: List[str],
    direction: str = "minimize",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], float, Optional[Any]]:
    """
    Evaluate adding each feature in `remaining` to the currently `selected` set,
    fit an MNL model, and choose the best feature according to `direction`.

    Parameters
    ----------
    Y : pd.Series
        Endogenous/categorical target.
    X : pd.DataFrame
        Design matrix containing all candidate columns (and possibly constant).
    selected : List[str]
        Already-selected columns (must be present in X).
    remaining : List[str]
        Candidate columns to try adding next (must be present in X).
    direction : {"minimize","min","maximize","max"}
        Whether to minimize or maximize the criterion (AIC is minimized).
    fit_kwargs : dict, optional
        Passed through to `_mnl_fit` (which passes to statsmodels fit).

    Returns
    -------
    (best_feature, best_criterion, best_result)
        - best_feature: the feature name chosen (None if no valid candidate)
        - best_criterion: best criterion value (inf or -inf if none found, depending on direction)
        - best_result: statsmodels fitted result object for the best candidate (None if none found)

    Notes
    -----
    This function assumes `_mnl_fit(Y, X_subset, fit_kwargs=...)` returns (result, criterion),
    and returns (None, inf) or similar when fitting fails.
    """
    if not remaining:
        raise ValueError("`remaining` must contain at least one feature.")

    fit_kwargs = fit_kwargs or {}

    # Initialize best values depending on optimization direction
    if direction in {"minimize", "min"}:
        best_criterion = float("inf")
        better_func = lambda cur, best: cur < best
    elif direction in {"maximize", "max"}:
        best_criterion = float("-inf")
        better_func = lambda cur, best: cur > best
    else:
        raise ValueError(f"Unknown direction: {direction}. Use 'minimize' or 'maximize'.")

    best_feat: Optional[str] = None
    best_res: Optional[Any] = None

    for feat in remaining:
        cols = selected + [feat]

        # Fit model on current selected + candidate feature
        res, cur_criterion = _mnl_fit(Y, X[cols].copy(), **fit_kwargs)

        # If the fit failed or criterion is not finite, skip this candidate.
        # This prevents failures from being accidentally chosen in "maximize" mode.
        if (not np.isfinite(cur_criterion)):
            continue

        # Update best if improvement
        if better_func(cur_criterion, best_criterion):
            best_criterion = cur_criterion
            best_feat = feat
            best_res = res

    return best_feat, best_criterion, best_res
# ==================================================================
def stepwise_MNL(
    Y: pd.Series,
    X: pd.DataFrame,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Forward stepwise multinomial logistic regression (statsmodels.MNLogit) using AIC.

    Procedure:
    - Start with intercept-only model.
    - Repeatedly add the single feature that produces the lowest AIC.
    - Stop when no candidate addition improves AIC.

    Outputs:
    - selected_features: best feature set (IN ORIGINAL INPUT ORDER, excludes 'const')
    - logs: step-by-step logs (entered feature, AIC changes)
    - result: fitted statsmodels result for the final selected model
    - final_aic: AIC of final model

    Important:
    - Requires no missing values in endog/exog. (You can adapt to drop NA if desired.)
    - MNLogit may fail for some feature sets due to separation or collinearity; those
      candidates are treated as AIC=inf and ignored.
    """
    fit_kwargs = fit_kwargs or {}

    if not isinstance(X, pd.DataFrame):
        raise TypeError("`X` must be a pandas DataFrame (needed for feature names/order).")

    # Defensive copy (avoid mutating caller's exog)
    Y = Y.copy()
    X = X.copy()

    # Preserve the original feature order (per your requirement)
    original_feature_order: List[str] = list(X.columns)

    # Add intercept
    X = sm.add_constant(X, has_constant="add")

    # Initialize selection with intercept-only model
    selected: List[str] = ["const"]
    remaining: List[str] = [f for f in original_feature_order]  # candidates exclude const

    # Fit baseline model
    best_res, best_criterion = _mnl_fit(Y.copy(), X[selected].copy(), **fit_kwargs)

    logs: List[Dict[str, Any]] = []
    logs.append( {
        "step": 0,
        "entered_feature": "const",
        "criterion": best_criterion,
    })

    step = 0
    while remaining:
        step += 1

        if len(remaining) == 0:
            break

        # Find best next feature to add (lowest AIC)
        enter_feat, enter_criterion, _ = _find_best_feature(
            Y = Y.copy(),
            X = X.copy(),
            selected = selected,
            remaining = remaining,
            direction = "minimize",
            fit_kwargs = fit_kwargs,)

        logs.append( {
            "step": step,
            "entered_feature": enter_feat,
            "criterion": enter_criterion,
        })

        # Accept feature
        selected.append(enter_feat)
        remaining.remove(enter_feat)

    return logs
# =========================================================================================================