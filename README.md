# Dynamic Portfolio Optimization Under Stochastic Volatility and Regime Switching

Version: 0.1.0  
Author: ayushii3101  
Repository: ayushii3101/Stochastic-Volatility-Driven-Dynamic-Portfolio-Optimization

---

## Abstract
This repository implements a research-grade framework for dynamic portfolio optimization where asset returns exhibit stochastic volatility and regime-dependent behavior. We combine the Heston stochastic volatility model with a finite-state Markov regime-switching process and solve a dynamic mean–variance (MVO) optimization problem in a rolling-horizon / simulation-based fashion. The codebase provides model definitions, Monte Carlo simulation engines, a dynamic optimizer, analysis utilities, reproducible Jupyter notebooks, and tests to reproduce and extend experiments.

## Motivation
Real-world financial returns show time-varying volatility, correlation structures, and abrupt structural changes (regime shifts). Static, single-regime portfolio methods (e.g., classic Markowitz MVO) do not capture these dynamics and can underperform or misestimate risk in stressed periods. By combining stochastic-volatility dynamics with regime switching, an investor can:

- Model continuous volatility fluctuations (heteroskedasticity) and discrete structural breaks (regimes) simultaneously.
- Adapt portfolio allocations dynamically as market conditions evolve.
- Test robust strategies in simulated environments that mimic both gradual and abrupt market changes.

This repository provides a modular, extensible platform to study such strategies.

## Mathematical models

### Heston Stochastic Volatility (per regime)
We model a single risky asset price \(S_t\) and instantaneous variance \(v_t\) by the Heston SDE:

- Variance (CIR-like):
  dv_t = κ (θ − v_t) dt + σ √(v_t) dW_t^v

- Asset price:
  dS_t = r S_t dt + √(v_t) S_t dW_t^S

with correlation
  E[dW_t^S dW_t^v] = ρ dt,  ρ ∈ [−1, 1],

and parameters per regime (i):
- κ_i: mean-reversion speed,
- θ_i: long-run variance,
- σ_i: volatility of volatility,
- ρ_i: correlation,
- v0, s0: initial variance and price,
- r: risk-free rate (can be regime-dependent in extensions).

Implementation notes
- We use an Euler–Maruyama discretization with full truncation / variance floor for numerical stability.
- Numba is used to accelerate inner loops for large Monte Carlo ensembles.

### Markov Regime Switching
The regime process r_t is modeled as a finite-state discrete-time Markov chain with transition probability matrix P:
- P_{ij} = P(r_{t+1} = j | r_t = i)
- Rows of P sum to 1.

Regimes modulate Heston parameters (e.g., θ_i, σ_i, or multiplicative volatility scalars). The chain can be simulated directly when regimes are assumed observable; when regimes are latent the framework supports filtering-based extensions.

## Portfolio optimization objective (dynamic MVO)
We focus on a dynamic mean–variance objective in a rolling-horizon setup. At each decision time t we estimate expected returns μ_t and covariance Σ_t from recent data (or simulated history) and solve a constrained single-period optimization:

maximize_w  f(w) = w^T μ_t − λ w^T Σ_t w  
subject to    sum(w) = 1,  w_i ≥ 0  (long-only simplex)

where λ > 0 is the risk-aversion parameter. The solver used is a numerical constrained optimizer (SLSQP via scipy.optimize.minimize). Across time we apply rebalancing using the computed weights and measure realized wealth.

Remarks:
- This rolling-horizon approach is a practical, simulation-driven proxy for the full dynamic programming solution (which is typically intractable under Heston + regime switching).
- The implementation supports alternative policy representations (parametric, RL-based) as extensions.

## Architecture diagram
High-level components and data flow:

Data / Calibration
       ↓
+---------------------+
|   src/models/       |  ← HestonModel, RegimeSwitchingModel, HybridVolModel
+---------------------+
       ↓
+---------------------+
|  src/simulation/    |  ← PathSimulator: Monte Carlo + RNG splitting + progress
+---------------------+
       ↓
+---------------------+       +---------------------+
| src/optimization/   | ←---- | src/utils/          |
| (DynamicMVO solver) |       | (stats, IO, plotting)|
+---------------------+       +---------------------+
       ↓
+---------------------+
|     results/        |  ← serialized outputs, figures, metrics
+---------------------+
       ↓
+---------------------+
|     notebooks/      |  ← reproducible experiments & analysis
+---------------------+  

Notes:
- `tests/` contains pytest suites for basic correctness checks.
- `data/` is intended for raw or processed calibration inputs (not always checked into VCS).

## Folder structure explanation
Recommended layout (present in this repo):

- src/
  - models/
    - heston.py            — HestonModel class (simulator)
    - regime_switching.py  — RegimeSwitchingModel (Markov chain)
    - hybrid_vol_model.py  — HybridVolModel (compose regime scaling)
  - simulation/
    - path_simulator.py    — PathSimulator convenience wrapper (tqdm + logging)
  - optimization/
    - dynamic_mvo.py       — DynamicMVO rolling-horizon optimizer
  - utils/
    - return_statistics.py — compute_log_returns, estimate_expected_returns, covariance, plotting
- notebooks/
  - 01_simulation.ipynb    — simulate and visualize paths and regimes
  - 02_portfolio_optimization.ipynb — build synthetic multi-asset, run DynamicMVO, plot results
- data/                    — (optional) calibration and raw data
- results/                 — outputs: npy files, figures, tables
- tests/                   — pytest unit tests (smoke tests & contract checks)
- README.md                — this file

Files above are examples of recommended module names; adapt to your conventions if needed.

## How to run the code

Prerequisites
- Python 3.9+ (3.10 recommended)
- Create and activate a virtual environment
- Install dependencies (example):
  pip install -r requirements.txt

Minimal dependencies (if you want to install manually):
  pip install numpy scipy pandas matplotlib numba tqdm pytest

Quick start (from repository root)
1. Clone repository
   git clone https://github.com/ayushii3101/Stochastic-Volatility-Driven-Dynamic-Portfolio-Optimization.git
   cd Stochastic-Volatility-Driven-Dynamic-Portfolio-Optimization

2. Create and activate venv:
   python -m venv .venv
   source .venv/bin/activate    # macOS / Linux
   .venv\Scripts\activate       # Windows

3. Install
   pip install -r requirements.txt

4. Run unit tests
   pytest -q

5. Run notebooks (recommended)
   jupyter lab   # or jupyter notebook
   - open notebooks/01_simulation.ipynb and 02_portfolio_optimization.ipynb to reproduce experiments and figures.

Command-line examples
- Small hybrid simulation (script-style; if provided):
  python -m src.simulation.path_simulator  # (if entrypoint implemented)

- Example: run rolling MVO in a script:
```bash
python - <<'PY'
from src.models.heston import HestonModel
from src.models.regime_switching import RegimeSwitchingModel
from src.models.hybrid_vol_model import HybridVolModel
from src.simulation.path_simulator import PathSimulator
from src.optimization.dynamic_mvo import DynamicMVO
import numpy as np

h = HestonModel(kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7, v0=0.04, s0=100.0, r=0.01, dt=1/252)
P = np.array([[0.95,0.04,0.01],[0.03,0.94,0.03],[0.02,0.08,0.90]])
reg = RegimeSwitchingModel(P=P, vols=[0.6,1.0,1.8])
hy = HybridVolModel(heston_model=h, regime_model=reg)
sim = PathSimulator(heston_model=h, regime_model=reg, hybrid_model=hy)
hy_v, S, regimes = sim.simulate_hybrid_paths(n_paths=200, n_steps=500, seed=123)
optimizer = DynamicMVO(risk_aversion=10.0)
weights, pv = optimizer.optimize_over_time(S[:5], window=20)  # treat first 5 sample paths as assets
print(weights.shape, pv.shape)
PY
```

Reproducibility tips
- Use fixed seeds when running simulations to reproduce results.
- Numba requires a first-run compilation; expect slower runtime on the first call.
- Save important inputs and outputs to `results/` for sharing and replication.

## Future extensions
Potential research and engineering directions:

- Multi-asset extension:
  - Correlated multivariate Heston-like models or factor-driven volatility processes.
- Partial observation & filtering:
  - Implement particle filters / HMM filters to infer latent regimes and variance from noisy data.
- Transaction costs and market impact:
  - Introduce turnover penalties (L1/L2), slippage models, and rebalancing constraints.
- Robust and distributionally robust optimization:
  - Protect policies against model misspecification and parameter uncertainty.
- Reinforcement learning policies:
  - Use actor-critic or policy-gradient methods to learn high-dimensional or non-parametric strategies.
- GPU-accelerated simulation:
  - Port Monte Carlo inner loops to CUDA / JAX for very large-scale experiments.
- Live or paper trading integration:
  - Add connectors to exchanges / paper-trading platforms with careful risk controls.
- Advanced calibration:
  - Implement MLE / Bayesian calibration for regime-dependent Heston parameters using historical option and returns data.

## References
- S. L. Heston (1993). "A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options." Review of Financial Studies.
- C. W. J. Granger & M. H. Pesaran (2000). "Regime Switching Models in Economics and Finance." (Survey literature)
- R. C. Merton (1971). "Optimum Consumption and Portfolio Rules in a Continuous-Time Model."
- Standard texts on stochastic calculus, dynamic programming, and numerical methods for SDEs.

---

If you use these methods for published work, please cite this repository and provide experimental configuration details (parameter values, seeds, number of Monte Carlo paths) in your methods section so results are reproducible.