"""Package-qualified imports of the frozen CPU oracle.

Never import `hamiltonian` or `entropy` as top-level modules. The oracle
uses relative imports, and those generic names also shadow this package.
ROOT (the qumode_qsk_gpu tree) must be on sys.path; cpu_oracle/src must not.
"""
from __future__ import annotations

import sys

from .paths import ROOT


def ensure_root_on_path():
    r = str(ROOT)
    # Remove cpu_oracle/src if a script added it; it breaks relative imports
    # and shadows package names.
    src = str(ROOT / "cpu_oracle" / "src")
    sys.path[:] = [p for p in sys.path if p != src]
    if r not in sys.path:
        sys.path.insert(0, r)
    return r


def hamiltonian():
    ensure_root_on_path()
    from cpu_oracle.src import hamiltonian as ham
    return ham


def entropy():
    ensure_root_on_path()
    from cpu_oracle.src import entropy as ent
    return ent


def spsa():
    ensure_root_on_path()
    from cpu_oracle.src import spsa as sp
    return sp


def bfgs():
    ensure_root_on_path()
    from cpu_oracle.src import bfgs as bf
    return bf


def starts():
    ensure_root_on_path()
    from cpu_oracle.src import starts as st
    return st
