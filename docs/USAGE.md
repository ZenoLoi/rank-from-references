# Usage

## CLI

```bash
rfr --input examples/toy_annotations.csv --output results/toy_rfr_scores.csv --q 0.9 --n-boot 200
```

or from repository scripts:

```bash
PYTHONPATH=src python3 scripts/run_rfr.py --input examples/toy_annotations.csv --output results/toy_rfr_scores.csv --q 0.9 --n-boot 200
```

## Python

```python
from rfr.io import load_long_table
from rfr.core import evaluate_candidates

ds = load_long_table("examples/toy_annotations.csv")
rows = evaluate_candidates(ds, q=0.9, n_boot=200, alpha=0.05, seed=42)
for r in rows:
    print(r)
```
