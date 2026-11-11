"""Deterministic frozen starts: identical RNG recipes to the CPU campaign."""
from __future__ import annotations

import numpy as np

from .protocol import param_count
from .dtypes import REAL
from .io_util import x_sha256

N_STARTS = 16


def spsa_starts(seed, L, method):
    p = param_count(method, L)
    rng = np.random.default_rng(int(seed) * 107 + int(L) * 2003 + {"nov3": 11, "qaoa": 29}[method])
    starts = []
    for sd in (0.16, 0.6):
        x = rng.normal(0.0, sd, p)
        if method == "nov3":
            x[0] += 0.2
        else:
            x[::2] += 0.12
            x[1::2] -= 0.22
        starts.append(np.ascontiguousarray(x, dtype=REAL))
    return starts, rng


def start_rng_seed(seed, L, method, start_index):
    off = 11 if method == "nov3" else 29
    return int(int(seed) * 1000003 + int(L) * 10007 + off + int(start_index) * 97)


def make_start(seed, L, method, start_index):
    k = int(start_index)
    p = param_count(method, L)
    width = 0.16 if (k % 2) else 0.6
    width_class = "narrow" if (k % 2) else "broad"
    rng_seed = start_rng_seed(seed, L, method, k)
    rng = np.random.default_rng(rng_seed)
    x = rng.normal(0.0, width, p)
    if method == "nov3":
        x[0] += 0.2
    else:
        x[::2] += 0.12
        x[1::2] -= 0.22
    x = np.ascontiguousarray(x, dtype=REAL)
    return dict(
        start_index=k, kind="iid_fresh_cold", width=width, width_class=width_class,
        rng_seed=rng_seed, x=x, x_sha256=x_sha256(x),
    )


def all_starts(seed, L, method):
    return [make_start(seed, L, method, k) for k in range(N_STARTS)]
