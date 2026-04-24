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
    all_actors = np.concatenate([refs, cand[None, :, :]], axis=0).astype(np.int16, copy=False)
    pairwise = np.abs(all_actors[:, None, :, :] - all_actors[None, :, :, :]).sum(axis=3)
    cand_idx = m

    per_exp_rank = np.zeros((m, n_docs), dtype=float)
    per_exp_delta = np.zeros((m, n_docs), dtype=float)

    for i in range(m):
        cand_dist = pairwise[i, cand_idx, :].astype(float, copy=False)
        peer_idx = np.delete(np.arange(m), i)
        peer_dist = pairwise[i, peer_idx, :].astype(float, copy=False)
        lt = (peer_dist < cand_dist).sum(axis=0)
        eq = (peer_dist == cand_dist).sum(axis=0)
        per_exp_rank[i, :] = 1.0 + lt + 0.5 * eq
        per_exp_delta[i, :] = cand_dist / n_cats

    rho_d = per_exp_rank.mean(axis=0)
    rho_tilde_d = (rho_d - 1.0) / (m - 1.0)
    delta_d = per_exp_delta.mean(axis=0)
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


def evaluate_candidate_arrays(
    humans: np.ndarray,
    candidates: np.ndarray,
    *,
    candidate_ids: list[str] | None = None,
    alpha: float = 0.05,
    q: float = 1.0,
    n_boot: int = 1000,
    seed: int | None = None,
) -> list[CandidateResult]:
    """Evaluate candidate panels directly from dense arrays.

    humans: (m, n_docs, n_cats)
    candidates: (k, n_docs, n_cats) or (n_docs, n_cats)
    """
    if humans.ndim != 3:
        raise ValueError("humans must have shape (m, n_docs, n_cats)")
    if humans.shape[0] < 3:
        raise ValueError("Need at least 3 humans to compute leave-one-out human thresholds")

    if candidates.ndim == 2:
        candidates = candidates[None, :, :]
    if candidates.ndim != 3:
        raise ValueError("candidates must have shape (k, n_docs, n_cats)")
    if humans.shape[1:] != candidates.shape[1:]:
        raise ValueError("humans and candidates must share (n_docs, n_cats)")
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0,1]")

    if candidate_ids is None:
        candidate_ids = [f"candidate_{i + 1}" for i in range(candidates.shape[0])]
    if len(candidate_ids) != candidates.shape[0]:
        raise ValueError("candidate_ids length does not match number of candidates")

    rng = np.random.default_rng(seed)
    human_upper = _human_upper_bounds(humans, alpha=alpha, n_boot=n_boot, rng=rng)
    threshold_q = float(np.quantile(human_upper, q))

    results: list[CandidateResult] = []
    for cid, cand in zip(candidate_ids, candidates):
        rho_d, rho_tilde_d, delta_d = _ranks_and_distances_per_doc(humans, cand)
        rho_mean = float(rho_d.mean())
        rho_tilde_mean = float(rho_tilde_d.mean())
        delta_mean = float(delta_d.mean())
        _, lo, hi = _bootstrap_ci(rho_tilde_d, alpha=alpha, n_boot=n_boot, rng=rng)
        non_inferior = hi < threshold_q

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


def evaluate_candidates(
    ds: Dataset,
    *,
    alpha: float = 0.05,
    q: float = 1.0,
    n_boot: int = 1000,
    seed: int | None = None,
) -> list[CandidateResult]:
    """Evaluate all candidates in a long-table dataset."""
    return evaluate_candidate_arrays(
        ds.humans,
        ds.candidates,
        candidate_ids=ds.candidate_ids,
        alpha=alpha,
        q=q,
        n_boot=n_boot,
        seed=seed,
    )
