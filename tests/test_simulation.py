from __future__ import annotations

import numpy as np

from rfr.simulation import TextModel, simulate_annotations


def test_simulation_is_deterministic_for_seed():
    mu_k = np.array([0.0, 0.2], dtype=float)
    text_model = TextModel(
        type="factor",
        L=2,
        beta=np.array([[0.3, -0.1], [0.2, 0.4]], dtype=float),
        Sigma_z=np.eye(2),
        sigma_eta=np.array([0.1, 0.2], dtype=float),
    )

    first = simulate_annotations(4, 2, 3, mu_k, text_model=text_model, seed=123, return_components=True)
    second = simulate_annotations(4, 2, 3, mu_k, text_model=text_model, seed=123, return_components=True)

    assert np.array_equal(first["X"], second["X"])
    assert np.allclose(first["components"]["S"], second["components"]["S"])
