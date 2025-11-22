"""src/models/heston.py

Numba-accelerated Heston stochastic volatility model.

This file defines a HestonModel class with an Euler-Maruyama discretization
and correlated Brownian motions. The core time-stepping loop is accelerated
with numba.
"""

from __future__ import annotations

import numpy as np
import numba as nb
from typing import Optional, Tuple


class HestonModel:
    """Heston stochastic volatility model simulator.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of the variance process v_t.
    theta : float
        Long-run variance level.
    sigma : float
        Volatility of volatility (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions (in [-1,1]).
    v0 : float
        Initial variance (>= 0).
    s0 : float
        Initial asset price (> 0).
    r : float
        Risk-free rate (drift of the asset price).
    dt : float
        Time step for Euler discretization (positive).

    Notes
    -----
    - Euler-Maruyama discretization with full truncation is used for the
      variance process to avoid taking sqrt of negative values.
    - The inner integrator is jitted with numba for performance on large
      Monte Carlo ensembles.
    """

    def __init__(self,
        kappa: float,
        theta: float,
        sigma: float,
        rho: float,
        v0: float,
        s0: float,
        r: float,
        dt: float,
    ) -> None:
        if not -1.0 <= rho <= 1.0:
            raise ValueError("rho must be between -1 and 1")
        if v0 < 0.0:
            raise ValueError("v0 must be non-negative")
        if s0 <= 0.0:
            raise ValueError("s0 must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.v0 = float(v0)
        self.s0 = float(s0)
        self.r = float(r)
        self.dt = float(dt)

    def simulate_paths(self, n_paths: int, n_steps: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate asset price and variance paths under the Heston model.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo sample paths to simulate.
        n_steps : int
            Number of time steps (returns arrays of shape (n_paths, n_steps+1)).
        seed : Optional[int]
            RNG seed for reproducibility.

        Returns
        -------
        S_paths : np.ndarray
            Simulated asset price paths, shape (n_paths, n_steps+1).
        v_paths : np.ndarray
            Simulated variance paths, shape (n_paths, n_steps+1).
        """
        n_paths = int(n_paths)
        n_steps = int(n_steps)
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive integers")

        rng = np.random.default_rng(seed)
        z1 = rng.standard_normal(size=(n_paths, n_steps))
        z2 = rng.standard_normal(size=(n_paths, n_steps))

        S_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
        v_paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)

        S_paths[:, 0] = self.s0
        v_paths[:, 0] = self.v0

        _heston_euler(
            S_paths,
            v_paths,
            z1,
            z2,
            self.kappa,
            self.theta,
            self.sigma,
            self.rho,
            self.r,
            self.dt,
        )

        return S_paths, v_paths


@nb.njit(parallel=True)
def _heston_euler(
    S_paths: np.ndarray,
    v_paths: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    kappa: float,
    theta: float,
    sigma: float,
    rho: float,
    r: float,
    dt: float,
) -> None:
    """Numba-jitted Euler-Maruyama integrator for the Heston SDE.

    The correlation between Brownian increments is enforced via:
        dW_v = sqrt(dt) * z1
        dW_s = sqrt(dt) * (rho * z1 + sqrt(1 - rho^2) * z2)

    Full truncation is applied when taking the square root of v.
    """
    n_paths, n_steps = z1.shape
    sqrt_dt = np.sqrt(dt)
    one_minus_rho2 = max(0.0, 1.0 - rho * rho)

    for i in nb.prange(n_paths):
        S = S_paths[i, 0]
        v = v_paths[i, 0]
        for t in range(n_steps):
            z1_t = z1[i, t]
            z2_t = z2[i, t]

            dW_v = sqrt_dt * z1_t
            dW_s = sqrt_dt * (rho * z1_t + np.sqrt(one_minus_rho2) * z2_t)

            sqrt_v = np.sqrt(max(0.0, v))
            dv = kappa * (theta - v) * dt + sigma * sqrt_v * dW_v
            v_new = v + dv

            S_new = S + r * S * dt + sqrt_v * S * dW_s

            v = v_new
            S = S_new

            S_paths[i, t + 1] = S
            v_paths[i, t + 1] = v

    return
