#!/usr/bin/env python3
"""Repository-friendly launcher for RFR scoring.

This script delegates to `rfr.cli:main` so users can run RFR directly
without installing entrypoints globally.
"""

from rfr.cli import main

if __name__ == "__main__":
    main()
