from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .core import evaluate_candidates
from .io import load_long_table


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute Rank from References (RFR) on a long-table dataset")
    ap.add_argument("--input", required=True, help="Path to annotations CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument("--human-type", default="human")
    ap.add_argument("--candidate-type", default="ai")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=1.0, help="Clinical non-inferiority quantile in (0,1]")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_long_table(
        args.input,
        human_type=args.human_type,
        candidate_type=args.candidate_type,
    )
    rows = evaluate_candidates(
        ds,
        alpha=args.alpha,
        q=args.q,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "rho_mean",
                "rho_tilde_mean",
                "delta_mean",
                "ci_lower",
                "ci_upper",
                "consensus_percentile",
                "threshold_q",
                "non_inferior",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


if __name__ == "__main__":
    main()
