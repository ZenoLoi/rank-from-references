from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Dataset:
    humans: np.ndarray
    candidates: np.ndarray
    human_ids: list[str]
    candidate_ids: list[str]
    doc_ids: list[str]
    category_ids: list[str]


def load_long_table(
    path: str | Path,
    *,
    human_type: str = "human",
    candidate_type: str = "ai",
) -> Dataset:
    """Load long-table annotations into dense binary tensors.

    Expected columns: actor_id, actor_type, doc_id, category_id, label
    """
    p = Path(path)
    rows: list[dict[str, str]] = []
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        req = {"actor_id", "actor_type", "doc_id", "category_id", "label"}
        if reader.fieldnames is None or not req.issubset(set(reader.fieldnames)):
            raise ValueError(f"Input CSV must contain columns {sorted(req)}")
        for r in reader:
            rows.append(r)

    actor_ids = sorted({r["actor_id"] for r in rows})
    doc_ids = sorted({r["doc_id"] for r in rows})
    category_ids = sorted({r["category_id"] for r in rows})

    actor_index = {a: i for i, a in enumerate(actor_ids)}
    doc_index = {d: i for i, d in enumerate(doc_ids)}
    cat_index = {c: i for i, c in enumerate(category_ids)}

    tensor = np.zeros((len(actor_ids), len(doc_ids), len(category_ids)), dtype=np.uint8)

    seen = set()
    for r in rows:
        key = (r["actor_id"], r["doc_id"], r["category_id"])
        if key in seen:
            raise ValueError(f"Duplicate row for {key}")
        seen.add(key)
        a = actor_index[r["actor_id"]]
        d = doc_index[r["doc_id"]]
        c = cat_index[r["category_id"]]
        label = int(r["label"])
        if label not in (0, 1):
            raise ValueError(f"label must be 0/1, got {label}")
        tensor[a, d, c] = label

    actor_type = {r["actor_id"]: r["actor_type"] for r in rows}
    human_ids = [a for a in actor_ids if actor_type.get(a) == human_type]
    candidate_ids = [a for a in actor_ids if actor_type.get(a) == candidate_type]

    if len(human_ids) < 2:
        raise ValueError("Need at least 2 humans")
    if not candidate_ids:
        raise ValueError("No candidate actors found")

    humans = tensor[[actor_index[a] for a in human_ids], :, :]
    candidates = tensor[[actor_index[a] for a in candidate_ids], :, :]

    return Dataset(
        humans=humans,
        candidates=candidates,
        human_ids=human_ids,
        candidate_ids=candidate_ids,
        doc_ids=doc_ids,
        category_ids=category_ids,
    )
