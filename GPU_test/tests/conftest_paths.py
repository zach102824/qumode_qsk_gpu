"""Put the GPU package and the project ROOT on sys.path. No CUDA init.

Do not insert cpu_oracle/src — that breaks relative imports and shadows
generic module names (hamiltonian, entropy, spsa, ...).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def setup_paths():
    # Drop a stray cpu_oracle/src entry if a previous snippet added one.
    src_oracle = str(ROOT / "cpu_oracle" / "src")
    sys.path[:] = [p for p in sys.path if p != src_oracle]
    for p in (str(SRC), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    return ROOT
