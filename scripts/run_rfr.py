#!/usr/bin/env python3
"""Repository-friendly launcher for RFR scoring.

This script delegates to `rfr.cli:main` so users can run RFR directly
without installing entrypoints globally.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rfr.cli import main

if __name__ == "__main__":
    main()
