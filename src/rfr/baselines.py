from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BaselineResult:
    candidate_id: str
    score: float
    threshold: float
    accepted: bool


def micro_f1(a: np.ndarray, b: np.ndarray) -> float:
    """Micro-F1 between two binary document-by-category panels."""
    aa = a.astype(bool, copy=False).ravel()
    bb = b.astype(bool, copy=False).ravel()
    tp = int(np.sum(aa & bb))
    fp = int(np.sum((~aa) & bb))
    fn = int(np.sum(aa & (~bb)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def strict_majority_consensus(humans: np.ndarray) -> np.ndarray:
    """Strict majority vote consensus across humans."""
    if humans.ndim != 3:
        raise ValueError("humans must have shape (m, n_docs, n_cats)")
    threshold = humans.shape[0] // 2 + 1
    return (humans.sum(axis=0) >= threshold).astype(np.uint8)


def majority_f1_acceptance(
    humans: np.ndarray,
    candidates: np.ndarray,
    *,
    candidate_ids: list[str] | None = None,
) -> list[BaselineResult]:
    """Majority-consensus F1 acceptance used as a manuscript baseline."""
    if candidates.ndim == 2:
        candidates = candidates[None, :, :]
    if candidate_ids is None:
        candidate_ids = [f"candidate_{i + 1}" for i in range(candidates.shape[0])]
    if len(candidate_ids) != candidates.shape[0]:
        raise ValueError("candidate_ids length does not match number of candidates")

    consensus = strict_majority_consensus(humans)
    human_scores = np.array([micro_f1(human, consensus) for human in humans], dtype=float)
    threshold = float(human_scores.min())

    results: list[BaselineResult] = []
    for cid, cand in zip(candidate_ids, candidates):
        score = micro_f1(cand, consensus)
        results.append(
            BaselineResult(
                candidate_id=cid,
                score=score,
                threshold=threshold,
                accepted=score >= threshold,
            )
        )
    return results


def pairwise_f1_acceptance(
    humans: np.ndarray,
    candidates: np.ndarray,
    *,
    candidate_ids: list[str] | None = None,
) -> list[BaselineResult]:
    """Average pairwise F1 acceptance used as a manuscript baseline."""
    if candidates.ndim == 2:
        candidates = candidates[None, :, :]
    if candidate_ids is None:
        candidate_ids = [f"candidate_{i + 1}" for i in range(candidates.shape[0])]
    if len(candidate_ids) != candidates.shape[0]:
        raise ValueError("candidate_ids length does not match number of candidates")

    def mean_against_humans(actor: np.ndarray) -> float:
        return float(np.mean([micro_f1(actor, human) for human in humans]))

    human_scores = np.array([mean_against_humans(human) for human in humans], dtype=float)
    threshold = float(human_scores.min())

    results: list[BaselineResult] = []
    for cid, cand in zip(candidate_ids, candidates):
        score = mean_against_humans(cand)
        results.append(
            BaselineResult(
                candidate_id=cid,
                score=score,
                threshold=threshold,
                accepted=score >= threshold,
            )
        )
    return results
