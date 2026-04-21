# Rank from References (RFR)

Public repository companion for the manuscript on **Rank from References (RFR)**.

RFR is an auditable framework to compare an AI classifier against a panel of human experts under expected disagreement, with:
- distance-based panel metrics (`Delta`, `rho`),
- normalized micro-RFR in `[0,1]` (consensus-percentile interpretation),
- quantile-based non-inferiority decision (`q`),
- bootstrap confidence intervals.

## Repository Layout

- `src/rfr/`: reusable Python module (importable API + CLI)
- `scripts/`: analysis scripts (including sensitivity analyses used in manuscript development)
- `results/`: generated outputs (ignored by git)
- `examples/`: toy datasets for quick usage
- `RFR.tex`: manuscript source

## Install

```bash
pip install -e .
```

## Input Format (long table)

The CLI expects a CSV with columns:
- `actor_id`
- `actor_type` (`human` or `ai` by default)
- `doc_id`
- `category_id`
- `label` (`0` or `1`)

See `examples/toy_annotations.csv`.

## Quick Start

```bash
rfr \
  --input examples/toy_annotations.csv \
  --output results/toy_rfr_scores.csv \
  --human-type human \
  --candidate-type ai \
  --alpha 0.05 \
  --q 0.9 \
  --n-boot 500
```

## Reproduce Simulations (Manuscript)

Main simulation:

```bash
python3 scripts/rfr_main_simulation.py \
  --outdir results/main_simulation \
  --n-replicates 120 \
  --m 8 \
  --q 0.9 \
  --n-boot 300 \
  --benchmark-stat upper
```

Sensitivity analysis:

```bash
python3 scripts/rfr_sensitivity_analysis.py \
  --outdir results/sensitivity_rfr \
  --n-replicates 60 \
  --n-boot 200 \
  --m-grid 3,4,5,6,8,10,12 \
  --q-grid 0.75,0.9,1.0 \
  --benchmark-stat upper
```

## Python API

```python
from rfr.io import load_long_table
from rfr.core import evaluate_candidates

ds = load_long_table("examples/toy_annotations.csv")
rows = evaluate_candidates(ds, alpha=0.05, q=0.9, n_boot=500, seed=42)
```

## Notes

- `q=1.0` recovers the former "worst-human" style anchor.
- Lower normalized micro-RFR means closer agreement with the expert panel.
- For clinical use, `q` should be pre-specified by domain constraints.
