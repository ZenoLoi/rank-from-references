#!/usr/bin/env python3
"""Generic sensitivity analysis for RFR with respect to panel size m and NI quantile q.

Design goals:
- Single script, simulation/evaluation agnostic via provider interface.
- Built-in synthetic provider for immediate use.
- Optional custom provider import (<module>:<callable>) returning scenario batches.

Outputs:
- summary_by_m_q.csv
- replicate_metrics.csv
- config_used.json
- notes.md
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol, Tuple

import numpy as np


@dataclass
class ProviderConfig:
    n_docs: int
    n_categories: int
    n_ai_good: int
    n_ai_bad: int
    base_prevalence: float
    text_scale: float
    text_noise_sd: float
    institution_contrast: float
    annotator_bias_sd: float
    annotator_trend_sd: float
    annotator_noise_sd: float
    bad_ai_shift: float


@dataclass
class ScenarioBatch:
    humans: np.ndarray  # (m, n_docs, n_cat), binary
    ai_good: np.ndarray  # (g, n_docs, n_cat), binary
    ai_bad: np.ndarray  # (b, n_docs, n_cat), binary


class ScenarioProvider(Protocol):
    def __call__(self, *, m: int, cfg: ProviderConfig, rng: np.random.Generator) -> ScenarioBatch:
        ...


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def builtin_synthetic_provider(*, m: int, cfg: ProviderConfig, rng: np.random.Generator) -> ScenarioBatch:
    """Generate a synthetic multi-label scenario with institution and drift effects."""
    n_d, n_c = cfg.n_docs, cfg.n_categories
    base_logit = np.log(cfg.base_prevalence / (1.0 - cfg.base_prevalence))
    text_latent = rng.normal(0.0, cfg.text_scale, size=(n_d, n_c))
    text_noise = rng.normal(0.0, cfg.text_noise_sd, size=(n_d, n_c))
    anchor = base_logit + text_latent + text_noise

    group_offsets = np.linspace(-cfg.institution_contrast / 2.0, cfg.institution_contrast / 2.0, num=2)
    doc_progress = np.linspace(0.0, 1.0, n_d)

    def draw_actor(group_id: int, extra_shift: np.ndarray | None = None) -> np.ndarray:
        cat_bias = rng.normal(0.0, cfg.annotator_bias_sd, size=(n_c,))
        trend = rng.normal(0.0, cfg.annotator_trend_sd, size=(n_c,))
        trend_term = np.outer(doc_progress, trend)
        eps = rng.normal(0.0, cfg.annotator_noise_sd, size=(n_d, n_c))
        logits = anchor + group_offsets[group_id] + cat_bias + trend_term + eps
        if extra_shift is not None:
            logits = logits + extra_shift
        probs = _sigmoid(logits)
        return (rng.random(size=(n_d, n_c)) < probs).astype(np.uint8)

    humans = np.zeros((m, n_d, n_c), dtype=np.uint8)
    for i in range(m):
        humans[i] = draw_actor(group_id=i % 2)

    ai_good = np.zeros((cfg.n_ai_good, n_d, n_c), dtype=np.uint8)
    for i in range(cfg.n_ai_good):
        ai_good[i] = draw_actor(group_id=i % 2)

    ai_bad = np.zeros((cfg.n_ai_bad, n_d, n_c), dtype=np.uint8)
    sign = rng.choice(np.array([-1.0, 1.0]), size=(n_c,))
    shift = cfg.bad_ai_shift * sign[None, :]
    for i in range(cfg.n_ai_bad):
        ai_bad[i] = draw_actor(group_id=i % 2, extra_shift=shift)

    return ScenarioBatch(humans=humans, ai_good=ai_good, ai_bad=ai_bad)


def _load_custom_provider(spec: str) -> ScenarioProvider:
    module_name, func_name = spec.split(":", 1)
    mod = importlib.import_module(module_name)
    provider = getattr(mod, func_name)
    if not callable(provider):
        raise TypeError(f"Provider '{spec}' is not callable")
    return provider


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def rfr_micro_per_doc(refs: np.ndarray, cand: np.ndarray) -> np.ndarray:
    """Per-doc normalized micro-RFR in [0,1], lower is better.

    refs: (m, n_docs, n_cat)
    cand: (n_docs, n_cat)
    """
    m_ref, n_docs, _ = refs.shape
    if m_ref < 2:
        raise ValueError("Need at least 2 references for RFR rank")

    out = np.zeros(n_docs, dtype=np.float64)
    for d in range(n_docs):
        r_sum = 0.0
        for i in range(m_ref):
            y_i = refs[i, d]
            cand_dist = np.abs(y_i.astype(np.int16) - cand[d].astype(np.int16)).sum()
            lt = 0
            eq = 0
            for j in range(m_ref):
                if j == i:
                    continue
                peer_dist = np.abs(y_i.astype(np.int16) - refs[j, d].astype(np.int16)).sum()
                if peer_dist < cand_dist:
                    lt += 1
                elif peer_dist == cand_dist:
                    eq += 1
            rank_avg = 1.0 + lt + 0.5 * eq
            r_sum += rank_avg
        rho_d = r_sum / m_ref
        out[d] = (rho_d - 1.0) / (m_ref - 1.0)
    return out


def bootstrap_ci(values: np.ndarray, *, n_boot: int, alpha: float, rng: np.random.Generator) -> Tuple[float, float, float]:
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    mu = float(values.mean())
    return mu, lo, hi


def evaluate_replicate(
    scenario: ScenarioBatch,
    *,
    q: float,
    n_boot: int,
    alpha: float,
    rng: np.random.Generator,
    benchmark_stat: str,
) -> Dict[str, float]:
    humans = scenario.humans
    m = humans.shape[0]
    if m < 3:
        raise ValueError("m must be >= 3 to compute external-human benchmarks reliably")

    human_stats = []
    for i in range(m):
        refs = np.delete(humans, i, axis=0)
        per_doc = rfr_micro_per_doc(refs, humans[i])
        mu, lo, hi = bootstrap_ci(per_doc, n_boot=n_boot, alpha=alpha, rng=rng)
        human_stats.append({"mean": mu, "lower": lo, "upper": hi})

    threshold = float(np.quantile([h[benchmark_stat] for h in human_stats], q))

    tp = fp = tn = fn = 0
    good_scores = []
    bad_scores = []

    for x in scenario.ai_good:
        per_doc = rfr_micro_per_doc(humans, x)
        mu, lo, hi = bootstrap_ci(per_doc, n_boot=n_boot, alpha=alpha, rng=rng)
        score = {"mean": mu, "lower": lo, "upper": hi}[benchmark_stat]
        good_scores.append(score)
        if score <= threshold:
            tp += 1
        else:
            fn += 1

    for x in scenario.ai_bad:
        per_doc = rfr_micro_per_doc(humans, x)
        mu, lo, hi = bootstrap_ci(per_doc, n_boot=n_boot, alpha=alpha, rng=rng)
        score = {"mean": mu, "lower": lo, "upper": hi}[benchmark_stat]
        bad_scores.append(score)
        if score <= threshold:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "accuracy": accuracy,
        "threshold": threshold,
        "human_stat_mean": float(np.mean([h[benchmark_stat] for h in human_stats])),
        "good_stat_mean": float(np.mean(good_scores)),
        "bad_stat_mean": float(np.mean(bad_scores)),
        "accept_rate_good": tp / max(1, scenario.ai_good.shape[0]),
        "accept_rate_bad": fp / max(1, scenario.ai_bad.shape[0]),
    }


def summarize(rows: List[Dict[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in keys}
    out = {}
    for k, v in arr.items():
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_sd"] = float(v.std(ddof=0))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Generic RFR sensitivity analysis (m, q)")
    ap.add_argument("--provider", default="builtin", help="builtin or <module>:<callable>")
    ap.add_argument("--outdir", default="results/sensitivity_rfr", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-replicates", type=int, default=120)
    ap.add_argument("--m-grid", default="3,4,5,6,8,10,12")
    ap.add_argument("--q-grid", default="0.75,0.9,1.0")
    ap.add_argument("--benchmark-stat", default="upper", choices=["mean", "upper"], help="Stat used for threshold and candidate score")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=300)

    ap.add_argument("--n-docs", type=int, default=50)
    ap.add_argument("--n-categories", type=int, default=11)
    ap.add_argument("--n-ai-good", type=int, default=6)
    ap.add_argument("--n-ai-bad", type=int, default=6)
    ap.add_argument("--base-prevalence", type=float, default=0.57)
    ap.add_argument("--text-scale", type=float, default=1.22)
    ap.add_argument("--text-noise-sd", type=float, default=0.51)
    ap.add_argument("--institution-contrast", type=float, default=1.08)
    ap.add_argument("--annotator-bias-sd", type=float, default=0.25)
    ap.add_argument("--annotator-trend-sd", type=float, default=0.30)
    ap.add_argument("--annotator-noise-sd", type=float, default=0.20)
    ap.add_argument("--bad-ai-shift", type=float, default=1.30)

    args = ap.parse_args()

    m_grid = _parse_int_list(args.m_grid)
    q_grid = _parse_float_list(args.q_grid)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = ProviderConfig(
        n_docs=args.n_docs,
        n_categories=args.n_categories,
        n_ai_good=args.n_ai_good,
        n_ai_bad=args.n_ai_bad,
        base_prevalence=args.base_prevalence,
        text_scale=args.text_scale,
        text_noise_sd=args.text_noise_sd,
        institution_contrast=args.institution_contrast,
        annotator_bias_sd=args.annotator_bias_sd,
        annotator_trend_sd=args.annotator_trend_sd,
        annotator_noise_sd=args.annotator_noise_sd,
        bad_ai_shift=args.bad_ai_shift,
    )

    provider: ScenarioProvider
    if args.provider == "builtin":
        provider = builtin_synthetic_provider
    else:
        provider = _load_custom_provider(args.provider)

    global_rng = np.random.default_rng(args.seed)

    replicate_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []

    for m in m_grid:
        for q in q_grid:
            rows = []
            for rep in range(args.n_replicates):
                rep_rng = np.random.default_rng(int(global_rng.integers(0, 2**31 - 1)))
                scenario = provider(m=m, cfg=cfg, rng=rep_rng)
                metrics = evaluate_replicate(
                    scenario,
                    q=q,
                    n_boot=args.n_boot,
                    alpha=args.alpha,
                    rng=rep_rng,
                    benchmark_stat=args.benchmark_stat,
                )
                row = {"m": m, "q": q, "replicate": rep, **metrics}
                rows.append(row)
                replicate_rows.append(row)

            agg = summarize(
                rows,
                keys=[
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                    "accuracy",
                    "accept_rate_good",
                    "accept_rate_bad",
                    "threshold",
                    "human_stat_mean",
                    "good_stat_mean",
                    "bad_stat_mean",
                ],
            )
            summary_rows.append({"m": m, "q": q, **agg})

    cfg_dump = {
        "args": vars(args),
        "provider_config": asdict(cfg),
        "m_grid": m_grid,
        "q_grid": q_grid,
    }
    (outdir / "config_used.json").write_text(json.dumps(cfg_dump, indent=2))

    with (outdir / "replicate_metrics.csv").open("w", newline="") as f:
        fields = list(replicate_rows[0].keys()) if replicate_rows else ["m", "q", "replicate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(replicate_rows)

    with (outdir / "summary_by_m_q.csv").open("w", newline="") as f:
        fields = list(summary_rows[0].keys()) if summary_rows else ["m", "q"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary_rows)

    lines = [
        "# RFR Sensitivity Analysis",
        "",
        f"Replicates per cell: {args.n_replicates}",
        f"m grid: {m_grid}",
        f"q grid: {q_grid}",
        f"benchmark stat: {args.benchmark_stat}",
        "",
        "Interpretation guide:",
        "- accept_rate_bad_mean: permissiveness proxy (lower is better).",
        "- specificity_mean: rejection power vs biased AIs (higher is better).",
        "- recall_mean: acceptance of unbiased AIs (higher is better).",
        "",
        "See summary_by_m_q.csv for aggregated results.",
    ]
    (outdir / "notes.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
