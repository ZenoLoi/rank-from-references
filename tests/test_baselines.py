from __future__ import annotations

import numpy as np

from rfr.baselines import majority_f1_acceptance, pairwise_f1_acceptance, strict_majority_consensus


def test_baseline_acceptance_smoke():
    humans = np.array(
        [
            [[1, 0], [0, 1]],
            [[1, 0], [1, 1]],
            [[1, 1], [0, 1]],
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

    consensus = strict_majority_consensus(humans)
    assert consensus.shape == (2, 2)

    majority = majority_f1_acceptance(humans, candidates, candidate_ids=["ai_good", "ai_bad"])
    pairwise = pairwise_f1_acceptance(humans, candidates, candidate_ids=["ai_good", "ai_bad"])

    assert [row.candidate_id for row in majority] == ["ai_good", "ai_bad"]
    assert [row.candidate_id for row in pairwise] == ["ai_good", "ai_bad"]
    assert majority[0].accepted
    assert not majority[1].accepted
