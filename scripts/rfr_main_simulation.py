#!/usr/bin/env python3
"""Main simulation runner for manuscript-aligned reproducibility.

This script runs the baseline simulation setting (single m, single q)
across multiple replicates and stores:
- replicate_metrics.csv
- summary_main.csv
- config_used.json
- notes.md
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rfr_sensitivity_analysis import (
    ProviderConfig,
    ScenarioProvider,
    _load_custom_provider,
    builtin_synthetic_provider,
    evaluate_replicate,
    summarize,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run baseline RFR simulation (single m, single q)")
    ap.add_argument("--provider", default="builtin", help="builtin or <module>:<callable>")
    ap.add_argument("--outdir", default="results/main_simulation", help="Output directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-replicates", type=int, default=120)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--q", type=float, default=0.9)
    ap.add_argument("--benchmark-stat", default="upper", choices=["mean", "upper"])
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

    replicate_rows = []
    for rep in range(args.n_replicates):
        rep_rng = np.random.default_rng(int(global_rng.integers(0, 2**31 - 1)))
        scenario = provider(m=args.m, cfg=cfg, rng=rep_rng)
        metrics = evaluate_replicate(
            scenario,
            q=args.q,
            n_boot=args.n_boot,
            alpha=args.alpha,
            rng=rep_rng,
            benchmark_stat=args.benchmark_stat,
        )
        replicate_rows.append({"m": args.m, "q": args.q, "replicate": rep, **metrics})

    summary = summarize(
        replicate_rows,
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
    summary_row = {"m": args.m, "q": args.q, **summary}

    (outdir / "config_used.json").write_text(
        json.dumps({"args": vars(args), "provider_config": asdict(cfg)}, indent=2)
    )

    with (outdir / "replicate_metrics.csv").open("w", newline="") as f:
        fields = list(replicate_rows[0].keys()) if replicate_rows else ["m", "q", "replicate"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(replicate_rows)

    with (outdir / "summary_main.csv").open("w", newline="") as f:
        fields = list(summary_row.keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(summary_row)

    notes = [
        "# RFR Main Simulation",
        "",
        f"Replicates: {args.n_replicates}",
        f"m: {args.m}",
        f"q: {args.q}",
        f"benchmark stat: {args.benchmark_stat}",
        "",
        "Main file for manuscript-level baseline metrics: summary_main.csv",
    ]
    (outdir / "notes.md").write_text("\n".join(notes))


if __name__ == "__main__":
    main()
