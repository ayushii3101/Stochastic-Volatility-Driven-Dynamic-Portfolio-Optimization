"""src/utils/return_statistics.py

Utility functions for computing return statistics and plotting return distributions.

Provides:
- compute_log_returns(price_paths): compute log returns from price paths.
- estimate_expected_returns(returns): estimate expected returns (mean across sample axis).
- estimate_covariance_matrix(returns): estimate covariance matrix of return columns.
- plot_return_distribution(returns, ...): quick histogram + KDE plot for returns.

This module uses numpy, pandas and matplotlib.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_log_returns(price_paths: np.ndarray) -> np.ndarray:
    """Compute log returns from price paths.

    Parameters
    ----------
    price_paths : array-like
        Price series or Monte Carlo price paths. Accepts:
        - 1D array of length T+1 (single path),
        - 2D array of shape (n_paths, T+1) where axis 1 is the time axis.

    Returns
    -------
    returns : np.ndarray
        Log returns with shape:
        - (T,) for single-path input,
        - (n_paths, T) for multi-path input.
        Computed as r_t = log(S_{t+1} / S_t).
    """
    arr = np.asarray(price_paths, dtype=float)

    if arr.ndim == 1:
        if arr.size < 2:
            raise ValueError("price_paths must contain at least two time points")
        logp = np.log(arr)
        return logp[1:] - logp[:-1]

    if arr.ndim == 2:
        if arr.shape[1] < 2:
            raise ValueError("price_paths must contain at least two time points along axis=1")
        logp = np.log(arr)
        return logp[:, 1:] - logp[:, :-1]

    raise ValueError("price_paths must be a 1D or 2D array")


def estimate_expected_returns(returns: np.ndarray, axis: int = 0) -> np.ndarray:
    """Estimate expected returns from a sample of returns.

    Parameters
    ----------
    returns : array-like
        Returns matrix. Common shapes:
        - (n_paths, T) where columns are time steps and rows are Monte Carlo samples,
        - (T,) single return series.
    axis : int, default 0
        Axis over which to average to produce expected returns. Default 0 averages
        across rows (e.g., across Monte Carlo paths) and returns a vector over time.

    Returns
    -------
    expected : np.ndarray
        Expected returns (mean) along the specified axis.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size == 0:
        raise ValueError("returns array is empty")
    return np.mean(arr, axis=axis)


def estimate_covariance_matrix(returns: np.ndarray, rowvar: bool = False, ddof: int = 1) -> np.ndarray:
    """Estimate the covariance matrix of returns.

    Parameters
    ----------
    returns : array-like
        Returns data. If 2D, columns are treated as variables when rowvar=False.
        If 1D, the function returns a 1x1 covariance matrix.
    rowvar : bool, default False
        If True, each row represents a variable, with observations in the columns.
        If False (default), each column represents a variable, with observations in the rows.
    ddof : int, default 1
        Delta degrees of freedom for the covariance estimator (1 for sample covariance).

    Returns
    -------
    cov : np.ndarray
        Covariance matrix (K x K) where K is the number of variables (columns when
        rowvar=False).
    """
    arr = np.asarray(returns, dtype=float)

    if arr.ndim == 1:
        # Return 1x1 covariance for a single series
        var = np.var(arr, ddof=ddof)
        return np.array([[var]])

    if arr.ndim == 2:
        # np.cov handles the rowvar flag directly
        return np.cov(arr, rowvar=rowvar, ddof=ddof)

    raise ValueError("returns must be a 1D or 2D array")


def plot_return_distribution(
    returns: np.ndarray,
    time_index: Optional[int] = None,
    bins: int = 50,
    figsize: Tuple[int, int] = (8, 5),
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Axes:
    """Plot the distribution of returns as a histogram with an overlaid KDE.

    Parameters
    ----------
    returns : array-like
        Returns data. If 2D, shape is (n_paths, T). If time_index is provided,
        the column at that index is plotted; otherwise the data are flattened.
    time_index : int or None, default None
        If provided and `returns` is 2D, select the column at this index to plot.
    bins : int, default 50
        Number of histogram bins.
    figsize : tuple, default (8, 5)
        Figure size passed to matplotlib.
    ax : matplotlib.axes.Axes or None
        Optional axes to draw on. If None, a new figure and axes are created.
    title : str or None
        Optional title for the plot.
    show : bool, default True
        Whether to call plt.show() before returning the Axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    arr = np.asarray(returns, dtype=float)

    # Select data to plot
    if arr.ndim == 1:
        data = arr
        t_label = "series"
    elif arr.ndim == 2:
        if time_index is None:
            data = arr.flatten()
            t_label = "flattened"
        else:
            if not (0 <= time_index < arr.shape[1]):
                raise IndexError(f"time_index must be in [0, {arr.shape[1] - 1}]")
            data = arr[:, time_index]
            t_label = f"time_index {time_index}"
    else:
        raise ValueError("returns must be a 1D or 2D array")

    # Convert to pandas Series for convenient plotting utilities (and potential future extensions)
    series = pd.Series(data).dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Histogram (density)
    ax.hist(series, bins=bins, density=True, alpha=0.6, color="C0", label="Histogram")

    # Kernel density estimate via pandas (matplotlib backend)
    try:
        series.plot(kind="kde", ax=ax, color="C1", label="KDE")
    except Exception:
        # Fallback: skip KDE if something goes wrong
        pass

    ax.set_xlabel("Return")
    ax.set_ylabel("Density")
    if title is None:
        ax.set_title(f"Return distribution ({t_label})")
    else:
        ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()

    if show:
        plt.show()

    return ax