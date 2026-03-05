import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_correlation_matrix(data, triangle=True, ax=None, **kwargs):
    """
    Plot a correlation matrix heatmap for a pandas DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data containing numeric columns to correlate.
    triangle : bool, default=True
        If True, masks the upper triangle (shows lower triangle only).
        If False, shows the full correlation matrix.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on. If None, a new figure/axes is created.
    **kwargs
        Additional keyword arguments forwarded to seaborn.heatmap.
        Common options: cmap, annot, fmt, linewidths, cbar_kws, etc.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the heatmap.
    """
    # Compute correlation matrix once
    corr = data.corr()

    # Create axes if not provided
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Mask upper triangle to avoid duplicate information (optional)
    mask = None
    if triangle:
        # True values in mask will be hidden by seaborn
        mask = np.triu(np.ones_like(corr, dtype=bool))  # k=1 keeps diagonal visible

    # Set sensible defaults but allow user override via **kwargs
    heatmap_defaults = dict(
        annot=True,           # show correlation values
        fmt=".2f",            # 2 decimal places
        cmap="coolwarm",
        vmin = -1, vmax = 1,
        square = True,
        linewidths = 1, linecolor = "white",
        cbar_kws = {"shrink": 0.75, "aspect": 30, "pad": 0.02},
    )
    # user-provided kwargs should override defaults
    heatmap_defaults.update(kwargs)

    sns.heatmap(
        corr,
        mask = mask,
        ax = ax,
        ** heatmap_defaults
    )

    # Clean up ticks (optional aesthetic choice)
    ax.tick_params(which="both", top=False, right=False, left=False, bottom=False)
    ax.set_title("Correlation Matrix")

    return ax
# ==================================================================================