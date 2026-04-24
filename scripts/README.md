# Scripts

This folder contains executable workflows for the manuscript and for local RFR usage.

## Core Usage

- `run_rfr.py`: thin launcher around the package CLI.

## Main Benchmark Workflow

### `rfr_generate_primary_grid.py`

Generates the full manuscript hyperparameter grid as one compressed NPZ archive per configuration.

Default outputs:

- `results/main_analysis/classifications_high_bias_npz/<config>/results.npz`
- `results/main_analysis/classifications_high_bias_npz/<config>/meta.json`
- `results/main_analysis/classifications_high_bias_npz/manifest.json`

### `rfr_build_memmap.py`

Consolidates the primary-grid archives into a single memory-mapped NumPy array plus metadata.

Default outputs:

- `results/main_analysis/main_grid_uint8.npy`
- `results/main_analysis/main_grid_meta.json`

### `rfr_compare_benchmarks.py`

Runs the manuscript benchmark on the consolidated array:

- RFR (quantile-based non-inferiority),
- Majority-F1,
- Average pairwise F1.

Default outputs:

- `results/main_analysis/benchmark/benchmark_summary.csv`
- `results/main_analysis/benchmark/benchmark_by_config.csv`
- `results/main_analysis/benchmark/correct_*.csv`

Optional:

- `--plot-heatmaps` to generate `byparam_heatmaps/` when `matplotlib` is installed.

## Sensitivity Workflow

### `rfr_main_simulation.py`

Baseline manuscript-style simulation for a single `(m, q)` setting with replicate aggregation.

### Usage

```bash
python scripts/rfr_main_simulation.py \
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

Thin wrapper around the package CLI entrypoint.

### Usage

```bash
python scripts/run_rfr.py \
  --input examples/toy_annotations.csv \
  --output results/toy_rfr_scores.csv \
  --q 0.9 \
  --n-boot 500
```

## `rfr_sensitivity_analysis.py`

Sensitivity analysis script for:
- panel size `m`
- non-inferiority quantile `q`

It is simulation-agnostic via a pluggable provider interface:
- builtin provider: `--provider builtin`
- custom provider: `--provider module_name:function_name`

### Usage

```bash
python scripts/rfr_sensitivity_analysis.py \
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
