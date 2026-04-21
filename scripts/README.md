# Scripts

This folder contains executable scripts for manuscript-aligned analyses and reusable RFR computation.

## `rfr_main_simulation.py`

Baseline manuscript simulation (single `m`, single `q`) with replicate aggregation.

### Usage

```bash
python3 scripts/rfr_main_simulation.py \
  --outdir results/main_simulation \
  --n-replicates 120 \
  --m 8 \
  --q 0.9 \
  --n-boot 300 \
  --benchmark-stat upper
```

### Outputs

- `summary_main.csv`
- `replicate_metrics.csv`
- `config_used.json`
- `notes.md`

## `run_rfr.py`

Thin wrapper around the `rfr` CLI (module entrypoint), useful when running directly from the repository.

### Usage

```bash
python3 scripts/run_rfr.py \
  --input examples/toy_annotations.csv \
  --output results/toy_rfr_scores.csv \
  --q 0.9 \
  --n-boot 500
```

## `rfr_sensitivity_analysis.py`

Generic sensitivity analysis script for:
- panel size `m`
- non-inferiority quantile `q`

It is simulation-agnostic via a pluggable provider interface:
- builtin provider: `--provider builtin`
- custom provider: `--provider module_name:function_name`

### Usage

```bash
python3 scripts/rfr_sensitivity_analysis.py \
  --outdir results/sensitivity_rfr \
  --n-replicates 60 \
  --n-boot 200 \
  --m-grid 3,4,5,6,8,10,12 \
  --q-grid 0.75,0.9,1.0 \
  --benchmark-stat upper
```

### Outputs

- `summary_by_m_q.csv`
- `replicate_metrics.csv`
- `config_used.json`
- `notes.md`
