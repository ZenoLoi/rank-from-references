# Rank from References (RFR)

Public companion repository for the **Rank from References (RFR)** preprint.

RFR is an auditable framework for evaluating whether an AI classifier remains within the disagreement envelope of a human expert panel. The repository contains:

- a reusable Python package for RFR scoring,
- a lightweight CLI and toy example,
- the manuscript source (`RFR.tex`),
- the main simulation and benchmark scripts used for the preprint,
- sensitivity-analysis scripts for panel size `m` and quantile `q`.

## Repository Layout

- `src/rfr/`: reusable package (`core`, baselines, simulation utilities, CLI)
- `scripts/`: executable workflows for the manuscript analyses
- `analyses/main/`: notes for the primary benchmark pipeline
- `analyses/sensitivity/`: notes for the sensitivity workflow
- `docs/USAGE.md`: CLI and Python API usage
- `docs/REPRODUCIBILITY.md`: end-to-end reproduction guide
- `examples/`: toy data for quick local validation
- `results/`: generated outputs, ignored by git
- `RFR.tex`: manuscript source

## Install

Core package:

```bash
python -m pip install -e .
```

Optional plotting support for benchmark heatmaps:

```bash
python -m pip install -e ".[analysis]"
```

## Input Format

The CLI expects a long-table CSV with these columns:

- `actor_id`
- `actor_type`
- `doc_id`
- `category_id`
- `label`

See `examples/toy_annotations.csv`.
At least three human annotators are required for the leave-one-out human threshold used by RFR.

## Quick Start

```bash
rfr --input examples/toy_annotations.csv --output results/toy_rfr_scores.csv --human-type human --candidate-type ai --alpha 0.05 --q 0.9 --n-boot 500
```

Equivalent direct script:

```bash
python scripts/run_rfr.py --input examples/toy_annotations.csv --output results/toy_rfr_scores.csv --q 0.9 --n-boot 500
```

## Manuscript Workflows

Primary simulation benchmark:

```bash
python scripts/rfr_generate_primary_grid.py
python scripts/rfr_build_memmap.py
python scripts/rfr_compare_benchmarks.py --q 0.9 --n-boot 200
```

Sensitivity analysis:

```bash
python scripts/rfr_main_simulation.py --outdir results/main_simulation --n-replicates 120 --m 8 --q 0.9 --n-boot 300 --benchmark-stat upper
python scripts/rfr_sensitivity_analysis.py --outdir results/sensitivity_rfr --n-replicates 60 --n-boot 200 --m-grid 3,4,5,6,8,10,12 --q-grid 0.75,0.9,1.0 --benchmark-stat upper
```

See `docs/REPRODUCIBILITY.md` for the full workflow and output locations.

## Python API

```python
from rfr.io import load_long_table
from rfr.core import evaluate_candidates

dataset = load_long_table("examples/toy_annotations.csv")
rows = evaluate_candidates(dataset, alpha=0.05, q=0.9, n_boot=500, seed=42)
```

## Historical Snapshot

`RFR_legacy/` is treated as a local archival snapshot only. The published workflow now lives in `src/rfr/` and `scripts/`, and `RFR_legacy/` is ignored by git.
