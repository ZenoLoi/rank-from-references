#!/usr/bin/env python3
"""Generate the primary manuscript simulation grid used for the main benchmark."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rfr.simulation import (
    AnnotatorConfig,
    NoiseConfig,
    SchoolsConfig,
    TextModel,
    generate_distinct_school_vectors,
    simulate_annotations,
)


GRID_BOUNDS: dict[str, tuple[float, float]] = {
    "p_base": (0.50, 0.64),
    "beta_scale": (1.17, 1.27),
    "sigma_eta": (0.49, 0.53),
    "school_contrast": (0.86, 1.30),
    "alpha0_sd": (0.20, 0.29),
    "rwA_sd": (0.036, 0.055),
    "rwB_sd": (0.046, 0.065),
    "phi_mean": (-0.50, -0.36),
    "phi_sd": (0.25, 0.35),
    "psi_mean": (-0.50, -0.36),
    "psi_sd": (0.21, 0.31),
}

STEPS_MAP = {"phi_mean": 3, "psi_mean": 3, "psi_sd": 3}
SLUG_ORDER = [
    "p_base",
    "beta_scale",
    "sigma_eta",
    "school_contrast",
    "alpha0_sd",
    "phi_sd",
    "rwA_sd",
    "rwB_sd",
    "phi_mean",
    "psi_mean",
    "psi_sd",
]


def _linspace_inclusive(lo: float, hi: float, steps: int) -> list[float]:
    if steps <= 1:
        return [lo]
    return [lo + i * (hi - lo) / (steps - 1) for i in range(steps)]


def _build_grid_values(global_steps: int) -> dict[str, list[float]]:
    return {
        key: [round(value, 6) for value in _linspace_inclusive(lo, hi, STEPS_MAP.get(key, global_steps))]
        for key, (lo, hi) in GRID_BOUNDS.items()
    }


def _format_slug(params: dict[str, float]) -> str:
    return "_".join(f"{key}={params[key]:.3f}" for key in SLUG_ORDER)


def _derive_seed(slug: str, base_seed: int) -> int:
    digest = hashlib.md5(slug.encode("utf-8")).hexdigest()
    return base_seed ^ int(digest[:8], 16)


def _build_configs(grid_values: dict[str, list[float]], base_seed: int, max_configs: int | None) -> tuple[int, Any]:
    keys = list(grid_values.keys())
    total = int(np.prod([len(grid_values[key]) for key in keys], dtype=np.int64))
    if max_configs is not None:
        total = min(total, max_configs)

    def iterator():
        for idx, values in enumerate(itertools.product(*[grid_values[key] for key in keys])):
            if max_configs is not None and idx >= max_configs:
                break
            params = dict(zip(keys, values))
            slug = _format_slug(params)
            yield params, slug, _derive_seed(slug, base_seed)

    return total, iterator()


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _human_school_assignment(
    m: int,
    n_categories: int,
    contrast: float,
    school_pi: list[float],
    rng: np.random.Generator,
) -> SchoolsConfig:
    if contrast <= 1e-12:
        return SchoolsConfig(R=1, g=np.ones(m, dtype=int), E=np.zeros((1, n_categories), dtype=float), center=True)

    counts = np.round(np.asarray(school_pi, dtype=float) * m).astype(int)
    while counts.sum() < m:
        counts[np.argmin(counts)] += 1
    while counts.sum() > m:
        counts[np.argmax(counts)] -= 1

    g = np.array([1] * counts[0] + [2] * counts[1], dtype=int)
    rng.shuffle(g)
    e = np.vstack([np.full(n_categories, +contrast / 2.0), np.full(n_categories, -contrast / 2.0)])
    return SchoolsConfig(R=2, g=g, E=e, center=True)


def _build_text_model(c: int, latent_dim: int, params: dict[str, float], rng: np.random.Generator) -> TextModel:
    return TextModel(
        type="factor",
        L=latent_dim,
        beta=rng.normal(0.0, params["beta_scale"], size=(latent_dim, c)),
        Sigma_z=np.eye(latent_dim),
        sigma_eta=np.full(c, params["sigma_eta"], dtype=float),
    )


def _generate_ai_good(
    *,
    n_docs: int,
    n_categories: int,
    n_ai: int,
    mu_k: np.ndarray,
    text_model: TextModel,
    schools: SchoolsConfig,
    shared_text: np.ndarray,
    beta0_sd: float,
    params: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    annotator = AnnotatorConfig(
        alpha0=rng.normal(0.0, params["alpha0_sd"], size=(n_ai, n_categories)),
        phi=np.zeros((n_ai, n_categories), dtype=float),
        rw_sd_A=np.zeros((n_ai, n_categories), dtype=float),
        beta0=rng.normal(0.0, beta0_sd, size=n_ai),
        psi=np.zeros(n_ai, dtype=float),
        rw_sd_B=np.zeros(n_ai, dtype=float),
    )
    noise = NoiseConfig(mode="annotator_category", scale_ak=np.ones((n_ai, n_categories), dtype=float))
    return simulate_annotations(
        n_docs,
        n_categories,
        n_ai,
        mu_k,
        text_model=text_model,
        schools=schools,
        annotator=annotator,
        noise=noise,
        rng=rng,
        return_components=False,
        text_effect_override=shared_text,
    )["X"]


def _run_one(task: tuple[dict[str, float], str, int, dict[str, Any]]) -> dict[str, Any]:
    params, slug, seed, cfg = task

    outdir = Path(cfg["outdir"]) / slug
    done_flag = outdir / "_DONE.stamp"
    npz_path = outdir / "results.npz"
    meta_path = outdir / "meta.json"

    if done_flag.exists() and npz_path.exists() and not cfg["overwrite"]:
        return {"slug": slug, "status": "skipped"}

    try:
        rng = np.random.default_rng(seed)
        n_docs = cfg["n_docs"]
        n_categories = cfg["n_categories"]
        n_annotators = cfg["n_annotators"]
        latent_dim = cfg["latent_dim"]
        beta0_sd = cfg["beta0_sd"]

        mu_k = np.full(n_categories, _logit(params["p_base"]), dtype=float)
        text_model = _build_text_model(n_categories, latent_dim, params, rng)
        shared_text = simulate_annotations(
            n_docs,
            n_categories,
            n_annotators,
            mu_k,
            text_model=text_model,
            schools=SchoolsConfig(R=1, g=np.ones(n_annotators, dtype=int), E=np.zeros((1, n_categories), dtype=float)),
            annotator=AnnotatorConfig(
                alpha0=np.zeros((n_annotators, n_categories), dtype=float),
                phi=np.zeros((n_annotators, n_categories), dtype=float),
                rw_sd_A=np.zeros((n_annotators, n_categories), dtype=float),
                beta0=np.zeros(n_annotators, dtype=float),
                psi=np.zeros(n_annotators, dtype=float),
                rw_sd_B=np.zeros(n_annotators, dtype=float),
            ),
            noise=NoiseConfig(mode="annotator_category", scale_ak=np.ones((n_annotators, n_categories), dtype=float)),
            rng=rng,
            return_components=True,
        )["components"]["S"]

        schools_h = _human_school_assignment(
            n_annotators,
            n_categories,
            params["school_contrast"],
            cfg["school_pi"],
            rng,
        )
        humans = simulate_annotations(
            n_docs,
            n_categories,
            n_annotators,
            mu_k,
            text_model=text_model,
            schools=schools_h,
            annotator=AnnotatorConfig(
                alpha0=rng.normal(0.0, params["alpha0_sd"], size=(n_annotators, n_categories)),
                phi=rng.normal(params["phi_mean"], params["phi_sd"], size=(n_annotators, n_categories)),
                rw_sd_A=np.full((n_annotators, n_categories), params["rwA_sd"], dtype=float),
                beta0=rng.normal(0.0, beta0_sd, size=n_annotators),
                psi=rng.normal(params["psi_mean"], params["psi_sd"], size=n_annotators),
                rw_sd_B=np.full(n_annotators, params["rwB_sd"], dtype=float),
            ),
            noise=NoiseConfig(mode="annotator_category", scale_ak=np.ones((n_annotators, n_categories), dtype=float)),
            rng=rng,
            return_components=False,
            text_effect_override=shared_text,
        )["X"]

        ai_good = _generate_ai_good(
            n_docs=n_docs,
            n_categories=n_categories,
            n_ai=n_annotators,
            mu_k=mu_k,
            text_model=text_model,
            schools=schools_h,
            shared_text=shared_text,
            beta0_sd=beta0_sd,
            params=params,
            rng=rng,
        )

        e_ai = generate_distinct_school_vectors(schools_h.E, n_annotators, rng=rng)
        ai_bad = simulate_annotations(
            n_docs,
            n_categories,
            n_annotators,
            mu_k,
            text_model=text_model,
            schools=SchoolsConfig(R=n_annotators, g=np.arange(1, n_annotators + 1), E=e_ai, center=False),
            annotator=AnnotatorConfig(
                alpha0=rng.normal(0.0, params["alpha0_sd"] * 0.5, size=(n_annotators, n_categories)),
                phi=np.zeros((n_annotators, n_categories), dtype=float),
                rw_sd_A=np.zeros((n_annotators, n_categories), dtype=float),
                beta0=rng.normal(0.0, beta0_sd * 0.5, size=n_annotators),
                psi=np.zeros(n_annotators, dtype=float),
                rw_sd_B=np.zeros(n_annotators, dtype=float),
            ),
            noise=NoiseConfig(mode="annotator_category", scale_ak=np.ones((n_annotators, n_categories), dtype=float)),
            rng=rng,
            return_components=False,
            text_effect_override=shared_text,
        )["X"]

        outdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            npz_path,
            X_h=humans.astype(np.uint8),
            X_ai_good=ai_good.astype(np.uint8),
            X_ai_bad=ai_bad.astype(np.uint8),
            mu_k=mu_k.astype(np.float32),
            E_h=schools_h.E.astype(np.float32),
            E_ai=e_ai.astype(np.float32),
            S=shared_text.astype(np.float32),
        )
        meta_path.write_text(
            json.dumps(
                {
                    "slug": slug,
                    "seed": seed,
                    "n_docs": n_docs,
                    "n_categories": n_categories,
                    "n_annotators": n_annotators,
                    "params": params,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        done_flag.write_text("OK\n", encoding="utf-8")
        return {"slug": slug, "status": "done"}
    except Exception as exc:
        return {"slug": slug, "status": "failed", "error": repr(exc)}


def _configure_logging(log_dir: Path, logfile: str) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / logfile
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    return log_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the primary RFR simulation grid")
    parser.add_argument("--outdir", default="results/main_analysis/classifications_high_bias_npz")
    parser.add_argument("--logdir", default="results/main_analysis/logs")
    parser.add_argument("--logfile", default="primary_grid.log")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1))
    parser.add_argument("--steps", type=int, default=3, help="Number of grid points per parameter")
    parser.add_argument("--max-configs", type=int, default=None, help="Limit the number of configurations")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--n-docs", type=int, default=50)
    parser.add_argument("--n-categories", type=int, default=12)
    parser.add_argument("--n-annotators", type=int, default=6)
    parser.add_argument("--latent-dim", type=int, default=3)
    parser.add_argument("--beta0-sd", type=float, default=0.3)
    args = parser.parse_args()

    log_path = _configure_logging(Path(args.logdir), args.logfile)
    grid_values = _build_grid_values(args.steps)
    total, task_iter = _build_configs(grid_values, args.base_seed, args.max_configs)

    worker_cfg = {
        "outdir": args.outdir,
        "overwrite": args.overwrite,
        "n_docs": args.n_docs,
        "n_categories": args.n_categories,
        "n_annotators": args.n_annotators,
        "latent_dim": args.latent_dim,
        "beta0_sd": args.beta0_sd,
        "school_pi": [0.6, 0.4],
    }

    manifest = {
        "grid_bounds": GRID_BOUNDS,
        "steps": args.steps,
        "steps_map": STEPS_MAP,
        "settings": worker_cfg,
        "max_configs": args.max_configs,
        "base_seed": args.base_seed,
        "logfile": str(log_path),
    }
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    (Path(args.outdir) / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    logging.info("Log file: %s", log_path)
    logging.info("Output directory: %s", args.outdir)
    logging.info("Workers: %d", args.workers)
    logging.info("Configurations to process: %d", total)

    done = skipped = failed = 0
    tasks = ((params, slug, seed, worker_cfg) for params, slug, seed in task_iter)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for idx, result in enumerate(executor.map(_run_one, tasks), start=1):
            status = result["status"]
            slug = result["slug"]
            if status == "done":
                done += 1
                logging.info("[DONE] %s (%d/%d)", slug, idx, total)
            elif status == "skipped":
                skipped += 1
                logging.info("[SKIP] %s (%d/%d)", slug, idx, total)
            else:
                failed += 1
                logging.error("[FAIL] %s (%d/%d): %s", slug, idx, total, result.get("error", "unknown error"))

    logging.info("Finished: %d done, %d skipped, %d failed", done, skipped, failed)


if __name__ == "__main__":
    main()
