"""
tests/test_markov_transition_matrix.py

Tests for RegimeSwitchingModel transition matrix behavior.

- Ensures that the transition matrix rows sum to 1 (within numerical tolerance).
- Ensures the provided vols array has the correct length.
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
    from src.models.regime_switching import RegimeSwitchingModel
except Exception as exc:  # pragma: no cover
    raise ImportError("Failed to import RegimeSwitchingModel from src.models.regime_switching") from exc


def test_transition_matrix_rows_sum_to_one():
    # Define a simple 3-state transition matrix (rows sum close to 1 but not exactly)
    P = np.array([
        [0.95, 0.04, 0.01],
        [0.03, 0.93, 0.04],
        [0.02, 0.08, 0.90],
    ], dtype=float)

    vols = [0.6, 1.0, 1.8]

    model = RegimeSwitchingModel(P=P, vols=vols, validate=True)

    # After validation/normalization, rows must sum to 1 within tolerance
    row_sums = model.P.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10), f"Row sums are not 1: {row_sums}"

    # vols length matches number of regimes
    assert model.vols.shape[0] == model.P.shape[0]