# Main Analysis

This folder documents the manuscript's primary simulation benchmark.

The workflow is:

1. Generate the full hyperparameter grid with `../../scripts/rfr_generate_primary_grid.py`.
2. Consolidate the per-configuration archives into a memory-mapped array with `../../scripts/rfr_build_memmap.py`.
3. Compare RFR against the F1 baselines with `../../scripts/rfr_compare_benchmarks.py`.

Default output locations:

- `results/main_analysis/classifications_high_bias_npz/`
- `results/main_analysis/main_grid_uint8.npy`
- `results/main_analysis/main_grid_meta.json`
- `results/main_analysis/benchmark/`
