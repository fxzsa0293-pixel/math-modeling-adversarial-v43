"""Backward-compatible wrapper for V38.

Prefer:
    python run.py --self-contained
"""
from __future__ import annotations

import sys
from run import main

if __name__ == "__main__":
    if "--self-contained" not in sys.argv and "--data-root" not in sys.argv:
        sys.argv.append("--self-contained")
    main()
