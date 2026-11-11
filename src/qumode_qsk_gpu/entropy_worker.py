"""Spawn-safe CPU entropy worker. No CUDA, no CuPy.

Imported in a fresh process (spawn). Thread-count env vars are set before
NumPy / SciPy / Numba are imported via jobs.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_key, "1")


def cpu_entropy_scan_only(spec):
    """Run the frozen CPU B* scan and write instance/scan/reference files.

    Must not call runtime.acquire(), create a GPU Model, or import cupy for
    work. Returns a host record (paths + peak fields). The parent process
    attaches GPU provenance and the optional apply_H cross-check.
    """
    from .jobs import cpu_entropy_scan_only as _impl
    return _impl(spec)
