"""src/models/regime_switching.py

RegimeSwitchingModel: simple discrete-time Markov chain helper for regimes.

This module provides a lightweight, well-documented class to represent a finite
Markov chain with a transition matrix and regime-specific volatilities.

Only numpy is required.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class RegimeSwitchingModel:
    """Finite-state Markov regime switching model.

    Parameters
    ----------
    P : array-like, shape (N, N)
        Transition probability matrix. Rows correspond to current state i and
        contain probabilities of transitioning to state j in the next step:
        P[i, j] = P(r_{t+1} = j | r_t = i). Rows should sum to 1 (will be
        normalized if numerical tolerance is off).
    vols : sequence of float, length N
        Regime-specific volatilities (one value per regime). Example:
        [low_vol, medium_vol, high_vol].
    validate : bool, default True
        If True, perform basic validation on P and vols (shape checks, row sums).
    """

    def __init__(self, P: np.ndarray, vols: Sequence[float], validate: bool = True) -> None:
        self.P = np.asarray(P, dtype=float)
        self.vols = np.asarray(vols, dtype=float)

        if validate:
            self._validate_shapes_and_probs()

        self.n_regimes = self.P.shape[0]

    def _validate_shapes_and_probs(self) -> None:
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("Transition matrix P must be a square matrix of shape (N, N).")
        n = self.P.shape[0]
        if self.vols.shape[0] != n:
            raise ValueError("Length of vols must match the number of regimes (rows of P).")

        # Ensure non-negative entries
        if np.any(self.P < 0):
            raise ValueError("Transition matrix P must have non-negative entries.")

        # Normalize rows if they do not sum exactly to 1 (numerical tolerance)
        row_sums = self.P.sum(axis=1)
        close = np.isclose(row_sums, 1.0, atol=1e-10)
        if not np.all(close):
            # Avoid division by zero
            if np.any(row_sums == 0):
                raise ValueError("A row of the transition matrix sums to zero; invalid probabilities.")
            self.P = (self.P.T / row_sums).T  # normalize rows

    def simulate_regimes(self, n_steps: int, start_state: Optional[int] = None, seed: Optional[int] = None) -> np.ndarray:
        """Simulate a discrete-time Markov chain of regimes.

        Parameters
        ----------
        n_steps : int
            Number of time steps to simulate. The returned array has length n_steps
            and contains regime indices in {0, ..., N-1} for times t = 0, ..., n_steps-1.
        start_state : int, optional
            Initial regime index at time t=0. If None, the initial state is sampled
            from the (approximate) stationary distribution of P.
        seed : int, optional
            RNG seed for reproducibility.

        Returns
        -------
        regimes : np.ndarray, shape (n_steps,)
            Integer array of simulated regime indices.

        Notes
        -----
        - The simulation is a simple stepwise sampling using the row of P for the
          current state to determine the next state's distribution.
        - If you need multiple independent paths, call this method multiple times
          with different seeds or implement a vectorized sampler at a higher level.
        """
        n_steps = int(n_steps)
        if n_steps <= 0:
            raise ValueError("n_steps must be a positive integer")

        rng = np.random.default_rng(seed)

        N = self.P.shape[0]

        # Determine initial state
        if start_state is None:
            pi = self._stationary_distribution()
            # fallback: if stationary failed (e.g., complex numerical issues), use uniform
            if pi is None or not np.isfinite(pi).all():
                start_state = int(rng.integers(0, N))
            else:
                start_state = int(rng.choice(N, p=pi))
        else:
            if not (0 <= start_state < N):
                raise ValueError(f"start_state must be in [0, {N-1}]")

        regimes = np.empty(n_steps, dtype=np.int64)
        state = int(start_state)
        for t in range(n_steps):
            regimes[t] = state
            # sample next state using current state's transition row
            state = int(rng.choice(N, p=self.P[state]))

        return regimes

    def get_vol_for_regime(self, regime: int) -> float:
        """Return the volatility associated with a given regime.

        Parameters
        ----------
        regime : int
            Regime index.

        Returns
        -------
        vol : float
            Volatility scalar associated with the regime.
        """
        if not (0 <= regime < self.vols.shape[0]):
            raise IndexError(f"regime must be between 0 and {self.vols.shape[0] - 1}")
        return float(self.vols[regime])

    def _stationary_distribution(self) -> Optional[np.ndarray]:
        """Compute the stationary distribution of the Markov chain P.

        Returns
        -------
        pi : np.ndarray or None
            Stationary distribution (row vector) if computable, otherwise None.
        """
        # Solve left eigenvector: pi = pi P  ->  (P^T - I)x = 0
        try:
            vals, vecs = np.linalg.eig(self.P.T)
            # Find eigenvalue 1
            idx = np.argmin(np.abs(vals - 1.0))
            v = np.real(vecs[:, idx])
            if np.allclose(v, 0):
                return None
            # Ensure non-negative and normalize
            v = np.abs(v)
            s = v.sum()
            if s == 0:
                return None
            return v / s
        except Exception:
            return None