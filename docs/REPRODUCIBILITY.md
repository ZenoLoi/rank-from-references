# Reproducibility

This repository exposes two reproducible analysis tracks:

1. the primary hyperparameter-grid benchmark used for the main manuscript results,
2. the lighter sensitivity workflow used to study panel size `m` and quantile `q`.

## 1. Install

Core package:

```bash
python -m pip install -e .
```

If you want benchmark heatmaps:

```bash
python -m pip install -e ".[analysis]"
```

## 2. Primary Benchmark Pipeline

### Step A: Generate the per-configuration simulation archives

```bash
python scripts/rfr_generate_primary_grid.py --outdir results/main_analysis/classifications_high_bias_npz
```

Outputs:

- `results/main_analysis/classifications_high_bias_npz/<config>/results.npz`
- `results/main_analysis/classifications_high_bias_npz/<config>/meta.json`
- `results/main_analysis/classifications_high_bias_npz/manifest.json`

### Step B: Consolidate the archives into a memory-mapped array

```bash
python scripts/rfr_build_memmap.py --npz-root results/main_analysis/classifications_high_bias_npz --out-npy results/main_analysis/main_grid_uint8.npy --meta results/main_analysis/main_grid_meta.json
```

Outputs:

- `results/main_analysis/main_grid_uint8.npy`
- `results/main_analysis/main_grid_meta.json`

### Step C: Compare RFR against manuscript baselines

```bash
python scripts/rfr_compare_benchmarks.py --npy results/main_analysis/main_grid_uint8.npy --meta results/main_analysis/main_grid_meta.json --outdir results/main_analysis/benchmark --q 0.9 --n-boot 200
```

Optional:

```bash
python scripts/rfr_compare_benchmarks.py --plot-heatmaps
```

Outputs:

- `results/main_analysis/benchmark/benchmark_summary.csv`
- `results/main_analysis/benchmark/benchmark_by_config.csv`
- `results/main_analysis/benchmark/correct_rfr.csv`
- `results/main_analysis/benchmark/correct_majority_f1.csv`
- `results/main_analysis/benchmark/correct_pairwise_f1.csv`
- `results/main_analysis/benchmark/benchmark_notes.json`

## 3. Sensitivity Workflow

Baseline single-cell run:

```bash
python scripts/rfr_main_simulation.py --outdir results/main_simulation --n-replicates 120 --m 8 --q 0.9 --n-boot 300 --benchmark-stat upper
```

Grid sensitivity:

```bash
python scripts/rfr_sensitivity_analysis.py --outdir results/sensitivity_rfr --n-replicates 60 --n-boot 200 --m-grid 3,4,5,6,8,10,12 --q-grid 0.75,0.9,1.0 --benchmark-stat upper
```

## 4. Manuscript Build

Compile the manuscript from the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error RFR.tex
pdflatex -interaction=nonstopmode -halt-on-error RFR.tex
```

## Notes

- `RFR_legacy/` is not part of the published workflow.
- Generated files are written under `results/` by default and are git-ignored.
- Use `--max-configs` on the primary-grid generator or benchmark script for smoke runs.
