"""src/models/hybrid_vol_model.py

HybridVolModel: combine Heston stochastic volatility with Markov regime scaling.

This module defines HybridVolModel which composes a HestonModel and a
RegimeSwitchingModel. It first simulates Heston paths (S and v), then applies
a regime-dependent multiplicative scaling factor to the variance process to
produce a hybrid variance time series for each Monte Carlo path.

Only numpy is required at the interface level; the HestonModel may use numba.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

# Relative imports from same package
from .heston import HestonModel
from .regime_switching import RegimeSwitchingModel


class HybridVolModel:
    """Hybrid volatility model that merges Heston variance with regime scaling.

    Parameters
    ----------
    heston_model : HestonModel
        Pre-configured HestonModel instance used to simulate base paths.
    regime_model : RegimeSwitchingModel
        Pre-configured RegimeSwitchingModel that provides regime draws and
        regime-specific vol scalars.
    vols_are_multiplicative : bool, default True
        If True, the `vols` in regime_model are treated as multiplicative scaling
        factors applied to the Heston variance v_t:
            v_hybrid(t) = v_heston(t) * vol_factor(regime_t)
        If False, the `vols` are treated as absolute volatilities and used to
        scale relative to the Heston model's initial variance:
            v_hybrid(t) = v_heston(t) * (vol / baseline)
        where baseline defaults to heston_model.v0.
    """

    def __init__(
        self,
        heston_model: HestonModel,
        regime_model: RegimeSwitchingModel,
        vols_are_multiplicative: bool = True,
    ) -> None:
        self.heston = heston_model
        self.regime = regime_model
        self.vols_are_multiplicative = bool(vols_are_multiplicative)

        # Baseline for converting absolute vols to multipliers when needed
        self._baseline = float(self.heston.v0) if self.heston.v0 > 0.0 else 1.0

    def simulate_hybrid_paths(
        self,
        n_paths: int,
        n_steps: int,
        start_state: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate hybrid volatility and asset paths.

        Workflow
        --------
        1. Simulate S_paths and v_paths from the underlying HestonModel:
             S_paths, v_paths = heston.simulate_paths(n_paths, n_steps, seed=seed_heston)
           where v_paths has shape (n_paths, n_steps+1) (includes t=0).
        2. For each Monte Carlo path, simulate a regime sequence of length n_steps
           (regimes for times t = 0, ..., n_steps-1). The regime at the final
           time index n_steps is taken equal to the regime at n_steps-1.
        3. Convert regime vol entries into multiplicative scaling factors and
           apply them to the Heston v_paths for each time-step, producing
           hybrid_v_paths with shape (n_paths, n_steps+1).
        4. Return hybrid_v_paths, S_paths and regimes_paths.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths to simulate.
        n_steps : int
            Number of time steps; the returned arrays have length n_steps+1 in time.
        start_state : int or None
            Optional initial regime index for all simulated paths. If None, each
            path draws its own initial state (via the regime model's stationary or
            random initialization).
        seed : int or None
            RNG seed for reproducibility. When provided, it is used to derive
            independent RNG streams for the Heston simulation and for each
            regime path to ensure reproducibility.

        Returns
        -------
        hybrid_v_paths : np.ndarray, shape (n_paths, n_steps+1)
            Hybrid variance time series after applying regime-dependent scaling.
        S_paths : np.ndarray, shape (n_paths, n_steps+1)
            Asset price paths simulated by the HestonModel (unadjusted S).
        regimes_paths : np.ndarray, shape (n_paths, n_steps)
            Integer regime indices for each path and time t=0..n_steps-1.

        Notes
        -----
        - The HestonModel.simulate_paths method is expected to return (S_paths, v_paths).
        - The regime model produces regimes for times t=0..n_steps-1; the final
          timepoint (t=n_steps) uses the last regime value appended to match v_paths.
        - When regime_model.vols are not multiplicative (vols_are_multiplicative=False),
          they are converted to multipliers relative to the Heston baseline v0.
        """
        n_paths = int(n_paths)
        n_steps = int(n_steps)
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive integers")

        # Create reproducible RNG and split seeds
        rng = np.random.default_rng(seed)

        # Use one seed for the Heston simulator and derive per-path seeds for regimes
        seed_heston = int(rng.integers(0, 2**31 - 1))
        # Derive per-path seeds for independent regime draws
        regime_seeds = rng.integers(0, 2**31 - 1, size=n_paths)

        # 1) Simulate base Heston paths
        S_paths, v_paths = self.heston.simulate_paths(n_paths=n_paths, n_steps=n_steps, seed=seed_heston)
        # v_paths shape: (n_paths, n_steps+1)

        # 2) Simulate regimes: one sequence per Monte Carlo path
        regimes_paths = np.empty((n_paths, n_steps), dtype=np.int64)
        for i in range(n_paths):
            # Provide start_state to simulate_regimes if caller supplied it,
            # otherwise allow the regime model to sample its own start.
            regimes_paths[i] = self.regime.simulate_regimes(n_steps=n_steps, start_state=start_state, seed=int(regime_seeds[i]))

        # 3) Convert regime vols to multiplicative scaling factors
        #    If vols_are_multiplicative, use vols directly; otherwise normalize by baseline.
        regs = self.regime.vols  # array shape (N,)
        if self.vols_are_multiplicative:
            multipliers = regs.astype(float)
        else:
            # avoid division by zero
            baseline = self._baseline if self._baseline > 0.0 else 1.0
            multipliers = (regs.astype(float) / baseline)

        # 4) Build hybrid variance by applying multipliers per time index
        hybrid_v_paths = np.empty_like(v_paths)
        # For each path, build an extended regime index array of length n_steps+1
        for i in range(n_paths):
            regimes = regimes_paths[i]  # length n_steps
            regimes_ext = np.empty(n_steps + 1, dtype=np.int64)
            regimes_ext[:-1] = regimes
            regimes_ext[-1] = regimes[-1]  # carry last regime forward for final timepoint

            # map regimes to multipliers for each time index
            time_multipliers = multipliers[regimes_ext]  # shape (n_steps+1,)

            # Apply multiplier to Heston variance (element-wise)
            hybrid_v_paths[i] = v_paths[i] * time_multipliers

        return hybrid_v_paths, S_paths, regimes_paths