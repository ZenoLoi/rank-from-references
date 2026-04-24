#!/usr/bin/env python3
"""Build a memory-mapped NumPy array from per-configuration NPZ archives."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap


HP_ORDER = [
    "p_base",
    "beta_scale",
    "sigma_eta",
    "school_contrast",
    "alpha0_sd",
    "rwA_sd",
    "rwB_sd",
    "phi_mean",
    "phi_sd",
    "psi_mean",
    "psi_sd",
]
PAIR = re.compile(r"([A-Za-z0-9]+(?:_[A-Za-z0-9]+)?)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _fmt3(value: str | float) -> str:
    return f"{float(value):.3f}"


def _parse_slug(slug: str) -> dict[str, str]:
    return dict(PAIR.findall(slug))


def _discover_npz(root: Path) -> list[tuple[str, Path]]:
    return sorted((path.parent.name, path) for path in root.rglob("results.npz"))


def _agent_names(m: int) -> list[str]:
    return [f"H{i + 1}" for i in range(m)] + [f"AI_bad{i + 1}" for i in range(m)] + [f"AI_good{i + 1}" for i in range(m)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a memory-mapped array from primary-grid NPZ outputs")
    parser.add_argument("--npz-root", default="results/main_analysis/classifications_high_bias_npz")
    parser.add_argument("--out-npy", default="results/main_analysis/main_grid_uint8.npy")
    parser.add_argument("--meta", default="results/main_analysis/main_grid_meta.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    npz_root = Path(args.npz_root)
    out_npy = Path(args.out_npy)
    meta_path = Path(args.meta)
    pairs = _discover_npz(npz_root)
    if not pairs:
        print(f"No NPZ files found under {npz_root}", file=sys.stderr)
        raise SystemExit(2)

    levels: dict[str, set[str]] = {key: set() for key in HP_ORDER}
    n_max = c_max = 0
    m_value: int | None = None

    for slug, _ in pairs:
        parsed = _parse_slug(slug)
        if not all(key in parsed for key in HP_ORDER):
            continue
        for key in HP_ORDER:
            levels[key].add(_fmt3(parsed[key]))

    for _, npz_path in pairs[: min(50, len(pairs))]:
        with np.load(npz_path) as data:
            x_h = data["X_h"]
            n_max = max(n_max, int(x_h.shape[0]))
            c_max = max(c_max, int(x_h.shape[1]))
            m_here = int(x_h.shape[2])
            if m_value is None:
                m_value = m_here
            elif m_value != m_here:
                print("Inconsistent numbers of annotators across NPZ files", file=sys.stderr)
                raise SystemExit(3)

    if m_value is None:
        print("Unable to infer annotator count from NPZ files", file=sys.stderr)
        raise SystemExit(4)

    levels_ord = OrderedDict((key, sorted(levels[key], key=float)) for key in HP_ORDER)
    agents = _agent_names(m_value)
    texts = [f"T{i + 1}" for i in range(n_max)]
    cats = [f"C{i + 1}" for i in range(c_max)]
    dims = [len(levels_ord[key]) for key in HP_ORDER] + [len(agents), len(texts), len(cats)]

    if (out_npy.exists() or meta_path.exists()) and not args.force:
        print("Output already exists. Use --force to overwrite.", file=sys.stderr)
        raise SystemExit(5)

    out_npy.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    memmap = open_memmap(out_npy, mode="w+", dtype=np.uint8, shape=tuple(dims), fortran_order=False)
    memmap[:] = 255
    memmap.flush()

    hp_index = {key: {value: idx for idx, value in enumerate(levels_ord[key])} for key in HP_ORDER}
    agent_index = {name: idx for idx, name in enumerate(agents)}

    for count, (slug, npz_path) in enumerate(pairs, start=1):
        parsed = _parse_slug(slug)
        if not all(key in parsed for key in HP_ORDER):
            continue

        idxs = tuple(hp_index[key][_fmt3(parsed[key])] for key in HP_ORDER)
        with np.load(npz_path) as data:
            x_h = data["X_h"].astype(np.uint8, copy=False)
            x_bad = data["X_ai_bad"].astype(np.uint8, copy=False)
            x_good = data["X_ai_good"].astype(np.uint8, copy=False)

        n, c, m = x_h.shape
        for actor_idx in range(m):
            memmap[idxs + (agent_index[f"H{actor_idx + 1}"], slice(0, n), slice(0, c))] = x_h[:, :, actor_idx]
            memmap[idxs + (agent_index[f"AI_bad{actor_idx + 1}"], slice(0, n), slice(0, c))] = x_bad[:, :, actor_idx]
            memmap[idxs + (agent_index[f"AI_good{actor_idx + 1}"], slice(0, n), slice(0, c))] = x_good[:, :, actor_idx]

        if count % 500 == 0:
            print(f"... {count}/{len(pairs)}")
            memmap.flush()

    memmap.flush()
    meta = {
        "dimnames": dict(list(levels_ord.items()) + [("agent", agents), ("text", texts), ("cat", cats)]),
        "order": HP_ORDER + ["agent", "text", "cat"],
        "dtype": "uint8",
        "encoding": {"NA": 255, "TRUE": 1, "FALSE": 0},
        "shape": dims,
        "source": str(npz_root.resolve()),
        "file": str(out_npy.resolve()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Done.")
    print(f"  npy:  {out_npy.resolve()}")
    print(f"  meta: {meta_path.resolve()}")


if __name__ == "__main__":
    main()
