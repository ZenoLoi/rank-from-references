from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "AnnotatorConfig",
    "NoiseConfig",
    "SchoolsConfig",
    "TextModel",
    "generate_distinct_school_vectors",
    "simulate_annotations",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[neg])
    out[neg] = exp_x / (1.0 + exp_x)
    return out


def _get_rng(seed: int | None = None, rng: np.random.Generator | None = None) -> np.random.Generator:
    if rng is not None and seed is not None:
        raise ValueError("Pass either seed or rng, not both")
    return rng if rng is not None else np.random.default_rng(seed)


@dataclass
class TextModel:
    type: str = "factor"
    sigma_S: np.ndarray | None = None
    L: int | None = None
    beta: np.ndarray | None = None
    Sigma_z: np.ndarray | None = None
    sigma_eta: np.ndarray | None = None


@dataclass
class SchoolsConfig:
    R: int = 1
    g: np.ndarray | None = None
    pi: np.ndarray | None = None
    E: np.ndarray | None = None
    sigma_E: np.ndarray | None = None
    center: bool = True


@dataclass
class AnnotatorConfig:
    order: list[np.ndarray] | None = None
    sessions: list[np.ndarray] | None = None
    alpha0: np.ndarray | None = None
    phi: np.ndarray | None = None
    rw_sd_A: np.ndarray | None = None
    beta0: np.ndarray | None = None
    psi: np.ndarray | None = None
    rw_sd_B: np.ndarray | None = None


@dataclass
class NoiseConfig:
    mode: str = "annotator_category"
    scale_a: np.ndarray | None = None
    scale_ak: np.ndarray | None = None


def _build_time_profiles(
    n: int,
    m: int,
    order_per_annotator: list[np.ndarray] | None,
    sessions: list[np.ndarray] | None,
) -> dict[str, list[np.ndarray]]:
    if order_per_annotator is None and sessions is None:
        order_per_annotator = [np.arange(n, dtype=int) for _ in range(m)]
        sessions = [None] * m
    elif order_per_annotator is None:
        order_per_annotator = [np.arange(n, dtype=int) for _ in range(m)]
    elif sessions is None:
        sessions = [None] * m

    if len(order_per_annotator) != m or len(sessions) != m:
        raise ValueError("order and sessions must have length m")

    orders: list[np.ndarray] = []
    session_ids: list[np.ndarray] = []
    u_list: list[np.ndarray] = []

    for actor_idx in range(m):
        order = np.asarray(order_per_annotator[actor_idx], dtype=int)
        if order.shape != (n,) or set(order.tolist()) != set(range(n)):
            raise ValueError(f"order[{actor_idx}] is not a permutation of 0..n-1")

        session_spec = sessions[actor_idx]
        if session_spec is None:
            sid = np.ones(n, dtype=int)
        else:
            raw = np.asarray(session_spec)
            if raw.size == n and raw.ndim == 1 and np.all(raw >= 1):
                sid = raw.astype(int)
            else:
                if raw.ndim != 1 or np.any(raw <= 0) or int(raw.sum()) != n:
                    raise ValueError("sessions specification is invalid")
                sizes = raw.astype(int)
                sid = np.repeat(np.arange(1, sizes.size + 1, dtype=int), sizes)

        u = np.empty(n, dtype=float)
        for session_id in np.unique(sid):
            idx = np.where(sid == session_id)[0]
            if idx.size == 1:
                u[idx] = 0.0
            else:
                u[idx] = np.arange(idx.size, dtype=float) / (idx.size - 1)

        orders.append(order)
        session_ids.append(sid)
        u_list.append(u)

    return {"order": orders, "session_id": session_ids, "u": u_list}


def _session_random_walk(
    rng: np.random.Generator,
    n: int,
    session_id: np.ndarray,
    rw_sd: float,
) -> np.ndarray:
    if rw_sd <= 0:
        return np.zeros(n, dtype=float)

    out = np.zeros(n, dtype=float)
    for sid in np.unique(session_id):
        idx = np.where(session_id == sid)[0]
        if idx.size >= 2:
            increments = rng.normal(0.0, rw_sd, idx.size - 1)
            out[idx] = np.concatenate([[0.0], np.cumsum(increments)])
    return out


def _build_text_effect(rng: np.random.Generator, n: int, c: int, tm: TextModel) -> np.ndarray:
    if tm.type == "iid":
        sigma_s = np.ones(c, dtype=float) if tm.sigma_S is None else np.asarray(tm.sigma_S, dtype=float)
        if sigma_s.shape != (c,):
            raise ValueError("sigma_S must have length c")
        return rng.normal(0.0, sigma_s.reshape(1, c), size=(n, c))

    if tm.type != "factor":
        raise ValueError("TextModel.type must be 'iid' or 'factor'")
    if tm.L is None or tm.L < 1:
        raise ValueError("TextModel.L must be >= 1 for factor text effects")

    beta = np.asarray(tm.beta, dtype=float)
    if beta.shape != (tm.L, c):
        raise ValueError("beta must have shape (L, c)")

    sigma_z = np.eye(tm.L, dtype=float) if tm.Sigma_z is None else np.asarray(tm.Sigma_z, dtype=float)
    sigma_eta = np.zeros(c, dtype=float) if tm.sigma_eta is None else np.asarray(tm.sigma_eta, dtype=float)
    if sigma_eta.shape != (c,):
        raise ValueError("sigma_eta must have length c")

    z = rng.multivariate_normal(np.zeros(tm.L, dtype=float), sigma_z, size=n)
    eta = rng.normal(0.0, sigma_eta.reshape(1, c), size=(n, c))
    return z @ beta + eta


def _build_school_effect(
    rng: np.random.Generator,
    m: int,
    c: int,
    conf: SchoolsConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = max(conf.R, 1)
    pi = np.ones(r, dtype=float) / r if conf.pi is None else np.asarray(conf.pi, dtype=float)
    if pi.shape != (r,) or np.any(pi < 0) or not np.isclose(pi.sum(), 1.0):
        raise ValueError("pi must be a valid probability vector of length R")

    if conf.g is None:
        g = rng.choice(np.arange(1, r + 1), size=m, p=pi).astype(int)
    else:
        g = np.asarray(conf.g, dtype=int)
        if g.shape != (m,) or np.any((g < 1) | (g > r)):
            raise ValueError("g must have shape (m,) with values in 1..R")
        if conf.pi is None:
            pi = np.bincount(g, minlength=r + 1)[1:].astype(float) / m

    if conf.E is not None:
        e = np.asarray(conf.E, dtype=float)
        if e.shape != (r, c):
            raise ValueError("E must have shape (R, c)")
    else:
        sigma_e = np.zeros(c, dtype=float) if conf.sigma_E is None else np.asarray(conf.sigma_E, dtype=float)
        if sigma_e.shape != (c,):
            raise ValueError("sigma_E must have length c")
        e = rng.normal(0.0, sigma_e.reshape(1, c), size=(r, c))

    if conf.center:
        e = e - (pi.reshape(1, r) @ e).reshape(1, c)

    return e, g, pi


def _build_annotator_time_effects(
    rng: np.random.Generator,
    n: int,
    c: int,
    m: int,
    tp: dict[str, list[np.ndarray]],
    alpha0: np.ndarray,
    phi: np.ndarray,
    rw_sd_A: np.ndarray,
    beta0: np.ndarray,
    psi: np.ndarray,
    rw_sd_B: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alpha0 = alpha0 - alpha0.mean(axis=1, keepdims=True)
    a = np.zeros((n, c, m), dtype=float)
    b = np.zeros((n, m), dtype=float)

    for actor_idx in range(m):
        u = tp["u"][actor_idx]
        sid = tp["session_id"][actor_idx]
        b[:, actor_idx] = beta0[actor_idx] + psi[actor_idx] * u + _session_random_walk(
            rng, n, sid, float(rw_sd_B[actor_idx])
        )
        for cat_idx in range(c):
            a[:, cat_idx, actor_idx] = (
                alpha0[actor_idx, cat_idx]
                + phi[actor_idx, cat_idx] * u
                + _session_random_walk(rng, n, sid, float(rw_sd_A[actor_idx, cat_idx]))
            )

    return a, b


def generate_distinct_school_vectors(
    human_vectors: np.ndarray,
    n_vectors: int,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    max_cos_human: float = 0.30,
    max_cos_between_ai: float = 0.35,
    delta_min: float = 1.0,
    delta_max: float = 2.0,
    clip: float = 3.0,
) -> np.ndarray:
    """Generate vectors that stay separated from human school effects."""
    generator = _get_rng(seed=seed, rng=rng)
    human_vectors = np.asarray(human_vectors, dtype=float)
    if human_vectors.ndim != 2:
        raise ValueError("human_vectors must have shape (R, c)")

    r_h, c = human_vectors.shape
    if r_h > 0:
        _, singular_values, vt = np.linalg.svd(human_vectors, full_matrices=False)
        basis = vt[singular_values > 1e-8]
        projector = basis.T @ basis if basis.size else np.zeros((c, c), dtype=float)
    else:
        projector = np.zeros((c, c), dtype=float)

    ai_vectors = np.zeros((n_vectors, c), dtype=float)

    def max_cosine(matrix: np.ndarray, vector: np.ndarray) -> float:
        if matrix.size == 0:
            return 0.0
        denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(vector) + 1e-12) + 1e-12
        return float((np.abs(matrix @ vector) / denom).max())

    for idx in range(n_vectors):
        for _ in range(2000):
            vector = generator.normal(size=c)
            if r_h > 0:
                vector = vector - projector @ vector
                if np.linalg.norm(vector) < 1e-10:
                    continue
            vector = vector / (np.linalg.norm(vector) + 1e-12)
            if max_cosine(human_vectors, vector) > max_cos_human:
                continue
            if idx > 0 and max_cosine(ai_vectors[:idx], vector) > max_cos_between_ai:
                continue

            delta = float(generator.uniform(delta_min, delta_max))
            vector = vector / (np.max(np.abs(vector)) + 1e-12) * delta
            ai_vectors[idx] = np.clip(vector, -clip, clip)
            break
        else:
            raise RuntimeError("Unable to generate distinct AI school vectors")

    return ai_vectors


def simulate_annotations(
    n: int,
    c: int,
    m: int,
    mu_k: np.ndarray,
    text_model: TextModel | None = None,
    schools: SchoolsConfig | None = None,
    annotator: AnnotatorConfig | None = None,
    noise: NoiseConfig | None = None,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    return_components: bool = True,
    deterministic: bool = True,
    text_effect_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Simulate binary multi-label annotations under an additive logit model."""
    generator = _get_rng(seed=seed, rng=rng)

    mu_k = np.asarray(mu_k, dtype=float)
    if mu_k.shape != (c,):
        raise ValueError("mu_k must have length c")

    if text_effect_override is None:
        s = _build_text_effect(generator, n, c, text_model or TextModel())
    else:
        s = np.asarray(text_effect_override, dtype=float)
        if s.shape != (n, c):
            raise ValueError("text_effect_override must have shape (n, c)")

    schools_cfg = schools or SchoolsConfig()
    e_rc, g, pi = _build_school_effect(generator, m, c, schools_cfg)

    annot_cfg = annotator or AnnotatorConfig()
    tp = _build_time_profiles(n, m, annot_cfg.order, annot_cfg.sessions)

    def _mat(x: np.ndarray | None) -> np.ndarray:
        return np.zeros((m, c), dtype=float) if x is None else np.asarray(x, dtype=float)

    def _vec(x: np.ndarray | None) -> np.ndarray:
        return np.zeros(m, dtype=float) if x is None else np.asarray(x, dtype=float)

    a, b = _build_annotator_time_effects(
        generator,
        n,
        c,
        m,
        tp,
        _mat(annot_cfg.alpha0),
        _mat(annot_cfg.phi),
        _mat(annot_cfg.rw_sd_A),
        _vec(annot_cfg.beta0),
        _vec(annot_cfg.psi),
        _vec(annot_cfg.rw_sd_B),
    )

    noise_cfg = noise or NoiseConfig()
    if noise_cfg.mode == "annotator":
        scale_a = np.ones(m, dtype=float) if noise_cfg.scale_a is None else np.asarray(noise_cfg.scale_a, dtype=float)
        scale_ak = np.repeat(scale_a.reshape(m, 1), c, axis=1)
    elif noise_cfg.mode == "annotator_category":
        scale_ak = (
            np.ones((m, c), dtype=float)
            if noise_cfg.scale_ak is None
            else np.asarray(noise_cfg.scale_ak, dtype=float)
        )
    else:
        raise ValueError("noise.mode must be 'annotator' or 'annotator_category'")

    x = np.zeros((n, c, m), dtype=np.uint8)
    p = np.zeros((n, c, m), dtype=float)
    eta = np.zeros((n, c, m), dtype=float)
    base = s + mu_k.reshape(1, c)

    for actor_idx in range(m):
        eta_actor = base + e_rc[g[actor_idx] - 1].reshape(1, c) + a[:, :, actor_idx] + b[:, actor_idx].reshape(n, 1)
        scaled = eta_actor / scale_ak[actor_idx].reshape(1, c)
        if deterministic:
            x[:, :, actor_idx] = (scaled >= 0.0).astype(np.uint8)
            p[:, :, actor_idx] = _sigmoid(scaled)
        else:
            probs = _sigmoid(scaled)
            x[:, :, actor_idx] = (generator.random(size=(n, c)) < probs).astype(np.uint8)
            p[:, :, actor_idx] = probs
        eta[:, :, actor_idx] = eta_actor

    out: dict[str, Any] = {"X": x, "P": p, "eta": eta}
    if return_components:
        out["components"] = {
            "mu_k": mu_k,
            "S": s,
            "schools": {"E": e_rc, "g": g, "pi": pi},
            "time": tp,
            "annotator": {
                "alpha0": annot_cfg.alpha0,
                "phi": annot_cfg.phi,
                "rw_sd_A": annot_cfg.rw_sd_A,
                "beta0": annot_cfg.beta0,
                "psi": annot_cfg.psi,
                "rw_sd_B": annot_cfg.rw_sd_B,
            },
            "noise": {"mode": noise_cfg.mode, "scale_ak": scale_ak},
        }
    return out
