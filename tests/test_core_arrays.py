from __future__ import annotations

import numpy as np

from rfr.core import evaluate_candidate_arrays


def test_evaluate_candidate_arrays_smoke():
    humans = np.array(
        [
            [[1, 0], [0, 1]],
            [[1, 1], [0, 1]],
            [[1, 0], [1, 1]],
        ],
        dtype=np.uint8,
    )
    candidates = np.array(
        [
            [[1, 0], [0, 1]],
            [[0, 0], [0, 0]],
        ],
        dtype=np.uint8,
    )

    rows = evaluate_candidate_arrays(
        humans,
        candidates,
        candidate_ids=["ai_good", "ai_bad"],
        q=0.9,
        n_boot=20,
        seed=7,
    )

    assert len(rows) == 2
    assert rows[0].candidate_id == "ai_good"
    assert rows[0].threshold_q >= 0.0
