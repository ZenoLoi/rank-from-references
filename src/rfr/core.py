from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io import Dataset


@dataclass
class CandidateResult:
    candidate_id: str
    rho_mean: float
    rho_tilde_mean: float
    delta_mean: float
    ci_lower: float
    ci_upper: float
    consensus_percentile: float
    threshold_q: float
    non_inferior: bool


def _ranks_and_distances_per_doc(refs: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-doc rho, rho_tilde, delta for one candidate.

    refs: (m, n_docs, n_cats)
    cand: (n_docs, n_cats)
    """
    m, n_docs, n_cats = refs.shape
    if m < 2:
        raise ValueError("Need at least two references")

    rho_d = np.zeros(n_docs, dtype=float)
    rho_tilde_d = np.zeros(n_docs, dtype=float)
    delta_d = np.zeros(n_docs, dtype=float)

    for d in range(n_docs):
        per_exp_rank = np.zeros(m, dtype=float)
        per_exp_delta = np.zeros(m, dtype=float)
        for i in range(m):
            y_i = refs[i, d]
            cand_dist = np.abs(y_i.astype(np.int16) - cand[d].astype(np.int16)).sum()
            per_exp_delta[i] = cand_dist / n_cats

            lt = 0
            eq = 0
            for j in range(m):
                if j == i:
                    continue
                peer_dist = np.abs(y_i.astype(np.int16) - refs[j, d].astype(np.int16)).sum()
                if peer_dist < cand_dist:
                    lt += 1
                elif peer_dist == cand_dist:
                    eq += 1
            per_exp_rank[i] = 1.0 + lt + 0.5 * eq

        rho = float(per_exp_rank.mean())
        rho_d[d] = rho
        rho_tilde_d[d] = (rho - 1.0) / (m - 1.0)
        delta_d[d] = float(per_exp_delta.mean())

    return rho_d, rho_tilde_d, delta_d


def _bootstrap_ci(values: np.ndarray, *, alpha: float, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    mu = float(values.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mu, lo, hi


def _human_upper_bounds(refs: np.ndarray, *, alpha: float, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    m = refs.shape[0]
    upper = np.zeros(m, dtype=float)
    for i in range(m):
        ext_refs = np.delete(refs, i, axis=0)
        _, rho_tilde_d, _ = _ranks_and_distances_per_doc(ext_refs, refs[i])
        _, _, u = _bootstrap_ci(rho_tilde_d, alpha=alpha, n_boot=n_boot, rng=rng)
        upper[i] = u
    return upper


def evaluate_candidates(
    ds: Dataset,
    *,
    alpha: float = 0.05,
    q: float = 1.0,
    n_boot: int = 1000,
    seed: int | None = None,
) -> list[CandidateResult]:
    """Evaluate all candidates in dataset with quantile-based non-inferiority."""
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1]")

    rng = np.random.default_rng(seed)

    human_upper = _human_upper_bounds(ds.humans, alpha=alpha, n_boot=n_boot, rng=rng)
    threshold_q = float(np.quantile(human_upper, q))

    results: list[CandidateResult] = []
    for cid, cand in zip(ds.candidate_ids, ds.candidates):
        rho_d, rho_tilde_d, delta_d = _ranks_and_distances_per_doc(ds.humans, cand)
        rho_mean = float(rho_d.mean())
        rho_tilde_mean = float(rho_tilde_d.mean())
        delta_mean = float(delta_d.mean())
        _, lo, hi = _bootstrap_ci(rho_tilde_d, alpha=alpha, n_boot=n_boot, rng=rng)
        non_inferior = (lo < threshold_q) and (hi < threshold_q)

        results.append(
            CandidateResult(
                candidate_id=cid,
                rho_mean=rho_mean,
                rho_tilde_mean=rho_tilde_mean,
                delta_mean=delta_mean,
                ci_lower=lo,
                ci_upper=hi,
                consensus_percentile=100.0 * rho_tilde_mean,
                threshold_q=threshold_q,
                non_inferior=non_inferior,
            )
        )

    return results
