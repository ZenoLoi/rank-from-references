#!/usr/bin/env python3
"""Compare RFR against manuscript baselines across the primary simulation grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rfr.baselines import majority_f1_acceptance, pairwise_f1_acceptance
from rfr.core import evaluate_candidate_arrays


ALGORITHMS = ("rfr", "majority_f1", "pairwise_f1")


def _pool_metrics(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "mcc": mcc,
    }


def _save_heatmap(param: str, x_values: list[str], alg_rows: dict[str, dict[str, float]], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for --plot-heatmaps") from exc

    matrix = np.array([[alg_rows[alg].get(x, np.nan) for x in x_values] for alg in ALGORITHMS], dtype=float)
    plt.figure(figsize=(max(6, 0.4 * len(x_values)), 3.5))
    plt.imshow(matrix, aspect="auto", cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
    plt.yticks(range(len(ALGORITHMS)), ALGORITHMS)
    plt.xticks(range(len(x_values)), x_values, rotation=45, ha="right")
    plt.xlabel(param)
    plt.ylabel("Algorithm")
    plt.title(f"Mean F1 by {param}")
    cbar = plt.colorbar()
    cbar.set_label("Mean F1")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RFR against F1 baselines on the primary grid")
    parser.add_argument("--npy", default="results/main_analysis/main_grid_uint8.npy")
    parser.add_argument("--meta", default="results/main_analysis/main_grid_meta.json")
    parser.add_argument("--outdir", default="results/main_analysis/benchmark")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--q", type=float, default=0.9)
    parser.add_argument("--n-boot", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--plot-heatmaps", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    order = meta["order"]
    dimnames = meta["dimnames"]
    name_to_axis = {name: idx for idx, name in enumerate(order)}
    hp_names = [name for name in order if name not in {"agent", "text", "cat"}]
    hp_levels = [list(dimnames[name]) for name in hp_names]
    total_configs = int(np.prod([len(values) for values in hp_levels], dtype=np.int64)) if hp_levels else 1
    if args.max_configs is not None:
        total_configs = min(total_configs, args.max_configs)

    big = np.load(args.npy, mmap_mode="r")
    agent_names = list(dimnames["agent"])
    human_names = [name for name in agent_names if name.startswith("H")]
    ai_good_names = [name for name in agent_names if name.startswith("AI_good")]
    ai_bad_names = [name for name in agent_names if name.startswith("AI_bad")]
    all_ai_names = ai_good_names + ai_bad_names
    truth_map = {name: True for name in ai_good_names} | {name: False for name in ai_bad_names}

    per_config_rows: list[dict[str, Any]] = []
    correctness_rows: dict[str, list[dict[str, Any]]] = {alg: [] for alg in ALGORITHMS}
    heatmap_accumulator: dict[str, dict[str, dict[str, tuple[float, int]]]] = {
        param: {alg: {} for alg in ALGORITHMS} for param in hp_names
    }
    counts: dict[str, dict[str, int]] = {alg: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for alg in ALGORITHMS}

    from itertools import product

    for config_idx, hp_indices in enumerate(product(*[range(len(values)) for values in hp_levels]), start=1):
        if args.max_configs is not None and config_idx > args.max_configs:
            break

        index_tuple: list[Any] = [None] * len(order)
        config_labels: dict[str, str] = {}
        parts: list[str] = []
        for param, param_idx in zip(hp_names, hp_indices):
            value = hp_levels[hp_names.index(param)][param_idx]
            config_labels[param] = str(value)
            parts.append(f"{param}={value}")
            index_tuple[name_to_axis[param]] = param_idx
        index_tuple[name_to_axis["agent"]] = slice(None)
        index_tuple[name_to_axis["text"]] = slice(None)
        index_tuple[name_to_axis["cat"]] = slice(None)
        config_key = "_".join(parts)

        sl = big[tuple(index_tuple)]
        if sl.ndim != 3 or sl.shape[0] != len(agent_names):
            raise ValueError("Unexpected memory-mapped array layout")

        cube = np.transpose(sl, (0, 1, 2)).astype(np.uint8, copy=False)
        if np.any(cube == 255):
            raise ValueError(f"Found NA-coded entries in configuration {config_key}")
        humans = np.transpose(cube[[agent_names.index(name) for name in human_names], :, :], (0, 1, 2))
        ai_candidates = np.transpose(cube[[agent_names.index(name) for name in all_ai_names], :, :], (0, 1, 2))

        rfr_results = evaluate_candidate_arrays(
            humans,
            ai_candidates,
            candidate_ids=all_ai_names,
            alpha=args.alpha,
            q=args.q,
            n_boot=args.n_boot,
            seed=args.seed + config_idx,
        )
        majority_results = majority_f1_acceptance(humans, ai_candidates, candidate_ids=all_ai_names)
        pairwise_results = pairwise_f1_acceptance(humans, ai_candidates, candidate_ids=all_ai_names)

        result_sets = {
            "rfr": {row.candidate_id: row.non_inferior for row in rfr_results},
            "majority_f1": {row.candidate_id: row.accepted for row in majority_results},
            "pairwise_f1": {row.candidate_id: row.accepted for row in pairwise_results},
        }

        for alg, accepted_map in result_sets.items():
            tp = fp = tn = fn = 0
            correctness = {"config": config_key}
            for ai_name in all_ai_names:
                accepted = bool(accepted_map[ai_name])
                truth = truth_map[ai_name]
                if accepted and truth:
                    tp += 1
                elif accepted and not truth:
                    fp += 1
                elif (not accepted) and truth:
                    fn += 1
                else:
                    tn += 1
                correctness[ai_name] = int(accepted == truth)

            counts[alg]["tp"] += tp
            counts[alg]["fp"] += fp
            counts[alg]["tn"] += tn
            counts[alg]["fn"] += fn

            metrics = _pool_metrics(tp, fp, tn, fn)
            row = {
                "config": config_key,
                "algorithm": alg,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "accuracy": metrics["accuracy"],
                "f1": metrics["f1"],
                "mcc": metrics["mcc"],
                **config_labels,
            }
            per_config_rows.append(row)
            correctness_rows[alg].append(correctness)

            for param, value in config_labels.items():
                total_f1, total_n = heatmap_accumulator[param][alg].get(value, (0.0, 0))
                heatmap_accumulator[param][alg][value] = (total_f1 + metrics["f1"], total_n + 1)

        if config_idx % 10 == 0 or config_idx == total_configs:
            print(f"... processed {config_idx}/{total_configs} configurations")

    summary_rows: list[dict[str, Any]] = []
    for alg in ALGORITHMS:
        tp = counts[alg]["tp"]
        fp = counts[alg]["fp"]
        tn = counts[alg]["tn"]
        fn = counts[alg]["fn"]
        pooled = _pool_metrics(tp, fp, tn, fn)
        alg_rows = [row for row in per_config_rows if row["algorithm"] == alg]
        summary_rows.append(
            {
                "algorithm": alg,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                **pooled,
                "mean_precision_by_config": float(np.mean([row["precision"] for row in alg_rows])) if alg_rows else 0.0,
                "mean_recall_by_config": float(np.mean([row["recall"] for row in alg_rows])) if alg_rows else 0.0,
                "mean_specificity_by_config": float(np.mean([row["specificity"] for row in alg_rows])) if alg_rows else 0.0,
                "mean_accuracy_by_config": float(np.mean([row["accuracy"] for row in alg_rows])) if alg_rows else 0.0,
                "mean_f1_by_config": float(np.mean([row["f1"] for row in alg_rows])) if alg_rows else 0.0,
                "mean_mcc_by_config": float(np.mean([row["mcc"] for row in alg_rows])) if alg_rows else 0.0,
            }
        )

    _write_csv(
        outdir / "benchmark_by_config.csv",
        list(per_config_rows[0].keys()) if per_config_rows else ["config", "algorithm"],
        per_config_rows,
    )
    _write_csv(
        outdir / "benchmark_summary.csv",
        list(summary_rows[0].keys()) if summary_rows else ["algorithm"],
        summary_rows,
    )

    for alg, rows in correctness_rows.items():
        _write_csv(outdir / f"correct_{alg}.csv", ["config", *all_ai_names], rows)

    if args.plot_heatmaps:
        for param in hp_names:
            x_values = sorted(heatmap_accumulator[param]["rfr"].keys(), key=float)
            alg_rows = {
                alg: {
                    x: (total / count) if count else np.nan
                    for x, (total, count) in heatmap_accumulator[param][alg].items()
                }
                for alg in ALGORITHMS
            }
            _save_heatmap(param, x_values, alg_rows, outdir / "byparam_heatmaps" / f"heatmap_byparam_{param}.png")

    notes = {
        "alpha": args.alpha,
        "q": args.q,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_configs": total_configs,
        "n_ai_per_config": len(all_ai_names),
    }
    (outdir / "benchmark_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
