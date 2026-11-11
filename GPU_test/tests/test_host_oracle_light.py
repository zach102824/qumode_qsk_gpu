"""Light host-side checks (no CUDA, no eigensolves, no state evolution)."""
from __future__ import annotations

import json
import numpy as np
from conftest_paths import setup_paths
ROOT = setup_paths()


def test_starts_and_diags():
    from qumode_qsk_gpu.starts import spsa_starts, make_start
    from qumode_qsk_gpu.model import make_J, build_diags
    from qumode_qsk_gpu import oracle

    ham = oracle.hamiltonian()
    catalog = json.loads((ROOT / "GPU_test/fixtures/validation_catalog.json").read_text())
    rows = []
    seen = set()
    for e in catalog["entries"]:
        if e["kind"] != "entropy_reference":
            continue
        key = (e["N"], e["seed"])
        if key in seen:
            continue
        seen.add(key)
        n, seed = key
        Jg = make_J(n, seed)
        Jc = ham.make_J(n, seed)
        if not np.array_equal(Jg, Jc):
            raise AssertionError(("make_J", n, seed))
        built = build_diags(Jg, n)
        cpu_m = ham.Model(n, seed, 1.0, J=Jc, compute_gs=False)
        for i in range(3):
            err = float(np.max(np.abs(built["diags"][i] - cpu_m.diags[i])))
            if err != 0.0:
                raise AssertionError(("diag", i, n, seed, err))
        err_s = float(np.max(np.abs(built["shell"] - cpu_m.shell)))
        if err_s != 0.0:
            raise AssertionError(("shell", n, seed, err_s))
        rows.append(dict(N=n, seed=seed, J_equal=True, diags_exact=True, shell_exact=True))

    n_spsa_pts = 0
    n_bfgs = 0
    for e in catalog["entries"]:
        if e["kind"] == "spsa_fixed_params":
            xs, _ = spsa_starts(e["seed"], e["L"], e["method"])
            for a, b in zip(xs, e["initial_points"]):
                if not np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)):
                    raise AssertionError(("spsa start", e["seed"], e["L"], e["method"]))
            n_spsa_pts += len(xs)
        elif e["kind"] == "bfgs_fixed_params":
            st = make_start(e["seed"], e["L"], e["method"], e["start_index"])
            if st["x_sha256"] != e["start_x_sha256"]:
                raise AssertionError(("bfgs start sha", e["result_file"]))
            if st["rng_seed"] != e["start_rng_seed"]:
                raise AssertionError(("bfgs rng seed", e["result_file"]))
            n_bfgs += 1
    n_start = n_spsa_pts + n_bfgs
    ok = n_start == 324
    if not ok:
        raise AssertionError(("start count", n_start, "expected", 324))
    return dict(ok=ok, instances=rows, n_spsa_points=n_spsa_pts, n_bfgs_starts=n_bfgs,
                n_start_vectors_checked=n_start, expected=324)


if __name__ == "__main__":
    print(json.dumps(test_starts_and_diags(), indent=2, default=str))
