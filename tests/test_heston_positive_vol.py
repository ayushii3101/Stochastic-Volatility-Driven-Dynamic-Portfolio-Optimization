"""
tests/test_heston_positive_vol.py

Smoke test for the HestonModel simulator: ensure variance paths remain non-negative
for a reasonably stable parameterization and fixed RNG seed.

Notes
-----
- The test uses conservative Heston parameters (fast mean reversion, low vol-of-vol)
  and a small time step to reduce the probability of negative variance under the
  Euler discretization used in the implementation.
- The project layout expects the `src` directory to be importable from the test
  runner; we prepend the repository root to sys.path to help Python locate `src`.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Ensure the repository root is on sys.path so `src` can be imported
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from src.models.heston import HestonModel
except Exception as exc:  # pragma: no cover - helpful import error in CI
    raise ImportError("Failed to import HestonModel from src.models.heston") from exc


def test_heston_variance_non_negative():
    """Simulate a modest ensemble and assert that all variance values are non-negative."""
    seed = 12345
    n_paths = 50
    n_steps = 200

    # Conservative Heston parameters to reduce chance of negative variance with Euler steps
    kappa = 10.0
    theta = 0.04
    sigma = 0.1
    rho = -0.3
    v0 = 0.04
    s0 = 100.0
    r = 0.0
    dt = 1.0 / 252.0

    model = HestonModel(kappa=kappa, theta=theta, sigma=sigma, rho=rho, v0=v0, s0=s0, r=r, dt=dt)

    S_paths, v_paths = model.simulate_paths(n_paths=n_paths, n_steps=n_steps, seed=seed)

    # Basic shape checks
    assert S_paths.shape == (n_paths, n_steps + 1)
    assert v_paths.shape == (n_paths, n_steps + 1)

    # Check non-negativity (allow an extremely small negative tolerance for floating error)
    min_v = float(np.min(v_paths))
    assert min_v >= -1e-12, f"Found negative variance value: {min_v}"

    # Also assert typical variance range (sane upper bound)
    assert np.all(np.isfinite(v_paths)), "Non-finite variance values encountered"