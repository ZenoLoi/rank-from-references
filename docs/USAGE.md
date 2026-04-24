# Usage

## CLI

```bash
rfr --input examples/toy_annotations.csv --output results/toy_rfr_scores.csv --q 0.9 --n-boot 200
```

Repository-local launcher:

```bash
python scripts/run_rfr.py --input examples/toy_annotations.csv --output results/toy_rfr_scores.csv --q 0.9 --n-boot 200
```

## Python

```python
from rfr.io import load_long_table
from rfr.core import evaluate_candidates

dataset = load_long_table("examples/toy_annotations.csv")
rows = evaluate_candidates(dataset, q=0.9, n_boot=200, alpha=0.05, seed=42)
for row in rows:
    print(row)
```

## Dense Array API

When you already have dense `(m, n_docs, n_categories)` human panels and dense candidate panels, use:

```python
import numpy as np
from rfr.core import evaluate_candidate_arrays

humans = np.array(...)
candidates = np.array(...)
rows = evaluate_candidate_arrays(humans, candidates, candidate_ids=["ai_1", "ai_2"], q=0.9, n_boot=200)
```

RFR requires at least three human references because the human threshold is computed with leave-one-out external evaluation.

## Baselines

The manuscript baselines are also available as reusable helpers:

```python
from rfr.baselines import majority_f1_acceptance, pairwise_f1_acceptance
```

See `docs/REPRODUCIBILITY.md` for the full benchmark pipeline.
