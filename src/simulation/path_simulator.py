"""src/simulation/path_simulator.py

PathSimulator: convenience wrapper that drives simulations for Heston,
regime-switching, and the hybrid model. Adds logging and progress bars.

This module expects to be used from the repository package (src as package root).
It attempts relative imports first and falls back to absolute imports for flexibility.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

# Try relative imports (package usage); fall back to absolute imports for ad-hoc script usage.
try:
    from ..models.heston import HestonModel
    from ..models.regime_switching import RegimeSwitchingModel
    from ..models.hybrid_vol_model import HybridVolModel
except Exception:  # pragma: no cover - fallback for different import contexts
    from src.models.heston import HestonModel  # type: ignore
    from src.models.regime_switching import RegimeSwitchingModel  # type: ignore
    from src.models.hybrid_vol_model import HybridVolModel  # type: ignore


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class PathSimulator:
    """High-level simulator that coordinates model components and provides progress feedback.

    Parameters
    ----------
    heston_model : HestonModel
        Instance used to simulate Heston S and v paths.
    regime_model : RegimeSwitchingModel
        Instance used to simulate regime sequences.
    hybrid_model : Optional[HybridVolModel]
        If provided, used for hybrid simulations. If None, a HybridVolModel is
        constructed from heston_model and regime_model.

    Notes
    -----
    Methods accept an optional `seed` for reproducibility. Where appropriate,
    the base seed is split into sub-seeds to ensure independent RNG streams for
    different simulators while keeping the overall experiment reproducible.
    """

    def __init__(
        self,
        heston_model: HestonModel,
        regime_model: RegimeSwitchingModel,
        hybrid_model: Optional[HybridVolModel] = None,
    ) -> None:
        self.heston = heston_model
        self.regime = regime_model
        self.hybrid = hybrid_model or HybridVolModel(heston_model=self.heston, regime_model=self.regime)

    def simulate_heston_paths(self, n_paths: int, n_steps: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Heston asset and variance paths.

        Parameters
        ----------
        n_paths : int
        n_steps : int
        seed : Optional[int]

        Returns
        -------
        S_paths, v_paths : tuple of np.ndarray
            Arrays with shape (n_paths, n_steps+1).
        """
        logger.info("Starting Heston simulation: n_paths=%d, n_steps=%d", n_paths, n_steps)
        # Directly delegate to HestonModel which supports vectorized path simulation.
        S_paths, v_paths = self.heston.simulate_paths(n_paths=n_paths, n_steps=n_steps, seed=seed)
        logger.info("Completed Heston simulation")
        return S_paths, v_paths

    def simulate_regime_switching_paths(
        self, n_paths: int, n_steps: int, start_state: Optional[int] = None, seed: Optional[int] = None
    ) -> np.ndarray:
        """Simulate multiple independent discrete-time Markov regime sequences.

        Parameters
        ----------
        n_paths : int
            Number of independent regime paths to simulate.
        n_steps : int
            Number of time steps per path (produces sequences of length n_steps).
        start_state : Optional[int]
            If provided, used as the initial regime for all paths; otherwise each
            path samples its own initial state (via the regime model's logic).
        seed : Optional[int]
            Base seed for reproducibility. Individual path seeds are derived
            from this base seed to ensure independent streams.

        Returns
        -------
        regimes_paths : np.ndarray, shape (n_paths, n_steps)
            Integer regime indices per path and time index.
        """
        logger.info(
            "Starting regime-switching simulation: n_paths=%d, n_steps=%d, start_state=%s", n_paths, n_steps, str(start_state)
        )

        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive integers")

        rng = np.random.default_rng(seed)
        # Derive seeds per path for reproducibility of individual draws
        path_seeds = rng.integers(0, 2**31 - 1, size=n_paths)

        regimes_paths = np.empty((n_paths, n_steps), dtype=np.int64)

        # Use tqdm for a progress bar across independent path simulations
        for i in tqdm(range(n_paths), desc="Simulating regimes", unit="path"):
            s = int(path_seeds[i])
            regimes_paths[i] = self.regime.simulate_regimes(n_steps=n_steps, start_state=start_state, seed=s)

        logger.info("Completed regime-switching simulation")
        return regimes_paths

    def simulate_hybrid_paths(
        self, n_paths: int, n_steps: int, start_state: Optional[int] = None, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Simulate hybrid volatility and asset paths using the HybridVolModel.

        Workflow
        - Calls HestonModel to generate base S and v paths (vectorized).
        - Simulates per-path regime sequences (with progress bar).
        - Applies regime-dependent multipliers to the Heston v paths to produce
          hybrid variance time series and returns (hybrid_v, S_paths, regimes_paths).

        Parameters
        ----------
        n_paths : int
        n_steps : int
        start_state : Optional[int]
        seed : Optional[int]

        Returns
        -------
        hybrid_v_paths : np.ndarray, shape (n_paths, n_steps+1)
        S_paths : np.ndarray, shape (n_paths, n_steps+1)
        regimes_paths : np.ndarray, shape (n_paths, n_steps)
        """
        logger.info(
            "Starting hybrid simulation: n_paths=%d, n_steps=%d, start_state=%s", n_paths, n_steps, str(start_state)
        )

        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive integers")

        # Use a deterministic RNG to derive seeds for Heston and per-path regime streams
        rng = np.random.default_rng(seed)
        seed_heston = int(rng.integers(0, 2**31 - 1))
        # Derive unique seeds for per-path regime draws
        regime_seeds = rng.integers(0, 2**31 - 1, size=n_paths)

        # 1) Simulate Heston base paths (vectorized)
        S_paths, v_paths = self.heston.simulate_paths(n_paths=n_paths, n_steps=n_steps, seed=seed_heston)

        # 2) Simulate regimes per path with a progress bar
        regimes_paths = np.empty((n_paths, n_steps), dtype=np.int64)
        for i in tqdm(range(n_paths), desc="Simulating hybrid regimes", unit="path"):
            regimes_paths[i] = self.regime.simulate_regimes(n_steps=n_steps, start_state=start_state, seed=int(regime_seeds[i]))

        # 3) Use the HybridVolModel to build hybrid variance paths
        # HybridVolModel.simulate_hybrid_paths accepts a single seed; to preserve
        # the work already done we will emulate its logic here by applying the
        # regime multipliers to the vectorized v_paths we already drew.
        regs = self.regime.vols
        if self.hybrid.vols_are_multiplicative:
            multipliers = regs.astype(float)
        else:
            baseline = self.hybrid._baseline if self.hybrid._baseline > 0.0 else 1.0
            multipliers = (regs.astype(float) / baseline)

        hybrid_v_paths = np.empty_like(v_paths)
        # Apply multipliers per path (use tqdm if many paths)
        for i in tqdm(range(n_paths), desc="Applying regime multipliers", unit="path"):
            regimes = regimes_paths[i]  # length n_steps
            regimes_ext = np.empty(n_steps + 1, dtype=np.int64)
            regimes_ext[:-1] = regimes
            regimes_ext[-1] = regimes[-1]
            time_multipliers = multipliers[regimes_ext]
            hybrid_v_paths[i] = v_paths[i] * time_multipliers

        logger.info("Completed hybrid simulation")
        return hybrid_v_paths, S_paths, regimes_paths