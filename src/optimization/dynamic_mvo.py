"""src/optimization/dynamic_mvo.py

DynamicMVO - rolling-horizon mean-variance optimizer.

This module implements a simple rolling-horizon dynamic mean-variance optimizer
that, at each rebalancing time, estimates expected returns and the covariance
matrix from a historical window and solves the constrained optimization

    maximize_w   w^T μ  -  λ * w^T Σ w
    subject to   sum(w) = 1,  w >= 0

The solver used is scipy.optimize.minimize with the SLSQP method. The primary
entry point is optimize_over_time(price_paths, window=20) which returns a time
series of portfolio weights and the resulting portfolio wealth curve when
applied to the provided price paths.

Notes on input shapes
---------------------
- price_paths: ndarray, shape (n_assets, T+1)
    Rows correspond to assets, columns correspond to times t=0..T (prices).
    The optimizer computes log-returns r_t = log(S_{t+1}/S_t) with shape
    (n_assets, T). We produce weights for each return time t (length T).

- Returned weight matrix has shape (n_assets, T) where column t corresponds
  to the weights applied to returns between time t and t+1.

This implementation is intentionally simple and readable; it is suitable for
research experiments and can be extended (transaction costs, constraints,
shorting, regularization, Bayesian estimators, etc.).
"""

from __future__ import annotations

import logging
from typing import Tuple, Optional

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DynamicMVO:
    """Rolling-horizon dynamic mean--variance optimizer.

    Parameters
    ----------
    risk_aversion : float
        Lambda parameter that trades off expected return vs. variance in the
        objective. Larger values produce more conservative portfolios.
    regularization : float, default 1e-6
        Small diagonal regularization added to the covariance matrix to ensure
        numerical stability and positive-definiteness.
    bounds : tuple(float, float) or None, default (0.0, 1.0)
        Lower and upper bounds for each weight. If None, no bounds are applied.
        Typical setting for long-only simplex: (0.0, 1.0).
    """

    def __init__(self, risk_aversion: float = 1.0, regularization: float = 1e-6, bounds: Optional[Tuple[float, float]] = (0.0, 1.0)) -> None:
        if risk_aversion < 0:
            raise ValueError("risk_aversion (lambda) must be non-negative")
        self.lambda_ = float(risk_aversion)
        self.regularization = float(regularization)
        self.bounds = bounds

    def _compute_log_returns(self, price_paths: np.ndarray) -> np.ndarray:
        """Compute log returns from price matrix.

        Parameters
        ----------
        price_paths : ndarray, shape (n_assets, T+1)

        Returns
        -------
        returns : ndarray, shape (n_assets, T)
            log returns r_t = log(S_{t+1}/S_t)
        """
        prices = np.asarray(price_paths, dtype=float)
        if prices.ndim != 2:
            raise ValueError("price_paths must be a 2D array with shape (n_assets, T+1)")
        if prices.shape[1] < 2:
            raise ValueError("price_paths must contain at least two time points (T+1 >= 2)")

        logp = np.log(prices)
        returns = logp[:, 1:] - logp[:, :-1]
        return returns

    def _opt_single_period(self, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
        """Solve the single-period mean-variance optimization.

        maximize_w   w^T mu  -  lambda * w^T cov w
        subject to   sum(w) = 1,  w_i in [lb, ub]

        We convert to a minimization for scipy.optimize.minimize by negating the objective.

        Parameters
        ----------
        mu : ndarray, shape (n_assets,)
        cov : ndarray, shape (n_assets, n_assets)

        Returns
        -------
        w_opt : ndarray, shape (n_assets,)
            Optimal weights for the period.
        """
        n = mu.shape[0]
        # objective to minimize (negative of desired objective)
        def obj(w: np.ndarray) -> float:
            # ensure w is numpy array
            w = np.asarray(w, dtype=float)
            ret = float(w @ mu)
            var = float(w @ cov @ w)
            # we minimize negative of (ret - lambda * var)
            return - (ret - self.lambda_ * var)

        # equality constraint: weights sum to 1
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)

        # bounds: (lb, ub) for each weight
        if self.bounds is None:
            bounds = tuple([(None, None) for _ in range(n)])
        else:
            lb, ub = float(self.bounds[0]), float(self.bounds[1])
            bounds = tuple([(lb, ub) for _ in range(n)])

        # initial guess: uniform allocation
        x0 = np.ones(n) / n

        # use SLSQP which supports bounds and equality constraints
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons, options={'ftol': 1e-9, 'disp': False, 'maxiter': 200})
        if not res.success:
            # If optimizer fails, fall back to uniform weights (with warning)
            logger.warning("Optimization failed (message=%s). Falling back to uniform weights.", res.message)
            w = x0
        else:
            w = res.x

        # Numerical cleanup: enforce non-negativity and normalization to avoid small infeasible values
        w = np.clip(w, 0.0, np.inf)
        s = w.sum()
        if s <= 0:
            # fallback to uniform if numerical problem
            logger.warning("Numerical issue: optimized weights sum to %g; using uniform weights instead.", s)
            w = np.ones(n) / n
        else:
            w = w / s

        return w

    def optimize_over_time(self, price_paths: np.ndarray, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """Perform rolling-horizon optimization across time.

        At each time t (for t = 0..T-1 corresponding to returns), the method
        forms an estimator for expected returns and the covariance using the
        previous `window` returns (inclusive of current time t). It then solves
        the single-period mean-variance problem to obtain weights to be applied
        for the next return step.

        Parameters
        ----------
        price_paths : ndarray, shape (n_assets, T+1)
            Price matrix where rows are assets and columns are time points t=0..T.
        window : int, default 20
            Historical window (in number of returns) used to estimate mu and Sigma.
            Must be at least 1 and at most T.

        Returns
        -------
        weights_time_series : ndarray, shape (n_assets, T)
            Column t contains the weights applied to returns between time t and t+1.
            For times t where there is insufficient history (t < window-1) we use
            a uniform weight vector by default.
        portfolio_values : ndarray, shape (T+1,)
            Portfolio wealth curve starting from 1.0 at time 0. The portfolio is
            rebalanced at each time t according to weights_time_series[:, t] and
            experiences the return implied by the asset returns at time t.
        """
        prices = np.asarray(price_paths, dtype=float)
        returns = self._compute_log_returns(prices)  # shape (n_assets, T)
        n_assets, T = returns.shape

        if not (1 <= window <= T):
            raise ValueError("window must be between 1 and the number of returns T=%d" % T)

        # allocate storage for weights and portfolio values
        weights_ts = np.zeros((n_assets, T), dtype=float)
        portfolio_values = np.empty(T + 1, dtype=float)
        portfolio_values[0] = 1.0  # initial wealth

        # For early times with insufficient history, use uniform allocation
        uniform_w = np.ones(n_assets) / n_assets

        # rolling loop over return time indices t=0..T-1
        for t in range(T):
            # Determine the history window: take up to `window` previous returns ending at t
            start_idx = max(0, t - window + 1)
            returns_hist = returns[:, start_idx : t + 1]  # shape (n_assets, window_t)

            # If there is only a single observation (window_t == 1), we cannot form a covariance
            if returns_hist.shape[1] < 2:
                # fallback: use simple estimates (mean, small diagonal covariance)
                mu = np.mean(returns_hist, axis=1)
                cov = np.eye(n_assets, dtype=float) * self.regularization
            else:
                mu = np.mean(returns_hist, axis=1)  # expected return per asset
                # np.cov expects variables in rows when rowvar=True
                cov = np.cov(returns_hist, rowvar=True, ddof=1)
                # regularize covariance for numerical stability
                cov = cov + np.eye(n_assets) * self.regularization

            # Solve single-period MVO: maximize w^T mu - lambda w^T cov w
            try:
                w_opt = self._opt_single_period(mu, cov)
            except Exception as exc:
                logger.exception("Exception during optimization at time t=%d: %s", t, str(exc))
                w_opt = uniform_w.copy()

            weights_ts[:, t] = w_opt

            # Apply weights to realized return at time t (out-of-sample one-step ahead)
            realized_return = float(weights_ts[:, t] @ returns[:, t])
            # Update portfolio value multiplicatively using simple log-return approximation:
            # if r is log-return, wealth *= exp(r)
            portfolio_values[t + 1] = portfolio_values[t] * np.exp(realized_return)

        return weights_ts, portfolio_values