from __future__ import annotations

import os
import sys

from longmem.adapter.paths import UPSTREAM_ROOT

UPSTREAM_DIR = str(UPSTREAM_ROOT)

if UPSTREAM_DIR not in sys.path:
    sys.path.insert(0, UPSTREAM_DIR)


def run_official_harness(argv):
    import runpy

    previous = sys.argv
    try:
        sys.argv = [sys.argv[0]] + list(argv)
        runpy.run_path(
            os.path.join(UPSTREAM_DIR, "evaluation", "harness.py"), run_name="__main__"
        )
    finally:
        sys.argv = previous
