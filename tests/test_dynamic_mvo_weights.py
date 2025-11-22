"""
tests/test_dynamic_mvo_weights.py

Test DynamicMVO optimizer returns valid weight series:
- weights are non-negative (long-only)
- each weight vector sums to 1 (simplex)
This test constructs a small synthetic multi-asset price matrix using geometric
Brownian increments to exercise the rolling-horizon optimizer.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# Ensure repository root on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.optimization.dynamic_mvo import DynamicMVO
except Exception as exc:  # pragma: no cover
    raise ImportError("Failed to import DynamicMVO from src.optimization.dynamic_mvo") from exc


def generate_synthetic_price_matrix(n_assets: int, T: int, s0: float = 100.0, seed: int | None = None) -> np.ndarray:
    """Generate a synthetic price matrix with shape (n_assets, T+1)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / 252.0
    mu = 0.0005  # daily drift
    sigma = 0.01  # daily vol
    prices = np.empty((n_assets, T + 1), dtype=float)
    for i in range(n_assets):
        prices[i, 0] = s0 * (1.0 + 0.02 * (i / max(1, n_assets - 1)))  # small differences in starting prices
        logp = np.log(prices[i, 0])
        # simulate log-returns
        eps = rng.normal(loc=(mu - 0.5 * sigma ** 2) * dt, scale=sigma * np.sqrt(dt), size=T)
        logp_series = np.concatenate(([logp], logp + np.cumsum(eps)))
        prices[i] = np.exp(logp_series)
    return prices


def test_dynamic_mvo_weights_simple_case():
    n_assets = 4
    T = 50
    seed = 2025

    price_matrix = generate_synthetic_price_matrix(n_assets=n_assets, T=T, seed=seed)

    optimizer = DynamicMVO(risk_aversion=5.0, regularization=1e-8, bounds=(0.0, 1.0))

    weights_ts, portfolio_values = optimizer.optimize_over_time(price_matrix, window=10)

    # Shapes
    assert weights_ts.shape == (n_assets, T)
    assert portfolio_values.shape == (T + 1,)

    # Non-negativity and simplex constraint (columns sum to 1)
    # allow a tiny numerical tolerance for floating point arithmetic
    assert np.all(weights_ts >= -1e-12), "Found negative weight(s) in weights time series"
    col_sums = weights_ts.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-10), f"Weight columns do not sum to 1 (min/max sums = {col_sums.min()}/{col_sums.max()})"

    # Portfolio values should remain finite and positive
    assert np.all(np.isfinite(portfolio_values))
    assert np.all(portfolio_values > 0.0)