"""Host-only regression contracts. No CUDA, no eigensolve, no optimizer.

Run from scripts/run_gpu_test.py before any CUDA work, or directly:
    python3 GPU_test/tests/test_host_contracts.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from conftest_paths import setup_paths

ROOT = setup_paths()


def _record(check, ok, **kw):
    rec = dict(check=check, ok=bool(ok))
    rec.update(kw)
    return rec


def test_strided_beta_contiguous():
    """R2-H1 contract on the host: a strided β view must be copied flat."""
    from qumode_qsk_gpu.kernels import prepare_beta
    # QAOA L=2, batch 2: columns are γ0, β0, γ1, β1
    xd = np.array([[0.11, 0.21, 0.31, 0.41],
                   [0.12, 0.22, 0.32, 0.42]], dtype=np.float64)
    assert not xd[:, 1].flags.c_contiguous
    beta, scalar = prepare_beta(xd[:, 1], batch=2, xp=np)
    recs = []
    recs.append(_record("beta_strided_not_scalar", scalar == 0, scalar=int(scalar)))
    recs.append(_record("beta_strided_contiguous", bool(beta.flags.c_contiguous),
                        strides=beta.strides))
    # RawKernel would have read beta[1] as xd[0,2]=0.31 (γ of next layer) if
    # the view pointer were passed through. After prepare_beta it must be 0.22.
    recs.append(_record("beta_strided_values",
                        np.array_equal(beta, np.array([0.21, 0.22])),
                        beta=beta.tolist(),
                        would_have_been_wrong=[float(xd[:, 1].ravel()[0]) if False else None,
                                               float(xd.ravel()[2])]))
    # size-1 is the scalar path
    b1, s1 = prepare_beta(xd[0, 1], batch=1, xp=np)
    recs.append(_record("beta_scalar_path", s1 == 1 and float(b1[0]) == 0.21, scalar=int(s1)))
    return recs


def test_pipeline_halt_and_continue():
    """R2-H2: injected entropy FAIL under both halt modes must terminate."""
    from qumode_qsk_gpu.pipeline import run_pipeline

    entropy = [
        dict(task="entropy", N=6, seed=1, index=0),
        dict(task="entropy", N=6, seed=2, index=1),
        dict(task="entropy", N=9, seed=3, index=2),
    ]
    opts = {
        (6, 1): [dict(task="spsa", N=6, seed=1, L=1, method="nov3")],
        (6, 2): [dict(task="spsa", N=6, seed=2, L=1, method="nov3")],
        (9, 3): [dict(task="spsa", N=9, seed=3, L=1, method="nov3")],
    }

    def entropy_fn(spec):
        if int(spec["seed"]) == 1:
            raise RuntimeError("injected entropy FAIL")
        time.sleep(0.02)
        return dict(failed=False, verdict="ok", N=spec["N"], seed=spec["seed"])

    def opt_fn(spec):
        time.sleep(0.01)
        return dict(failed=False, verdict="ok", task=spec["task"], seed=spec["seed"])

    recs = []
    for halt, name in ((True, "halt"), (False, "continue")):
        t0 = time.time()
        n_fail, n_ok, n_reused, results = run_pipeline(
            entropy, opts, halt=halt, entropy_workers=1,
            entropy_fn=entropy_fn, opt_fn=opt_fn, finalize_fn=None,
            use_processes=False, poll_s=0.02, join_timeout=2.0,
        )
        wall = time.time() - t0
        recs.append(_record(
            f"pipeline_{name}_terminates",
            wall < 2.0,
            wall_s=wall, n_fail=n_fail, n_ok=n_ok, n_results=len(results),
            halt=halt,
        ))
        recs.append(_record(
            f"pipeline_{name}_records_fail",
            n_fail >= 1,
            n_fail=n_fail,
        ))
        seeds_ok = {r.get("spec", {}).get("seed") or r.get("seed") for r in results if not r.get("failed")}
        if halt:
            # Must not run later entropy (seed 2, 3) or their GPU opts after the first FAIL
            # (seed 1 fails immediately). A concurrent start is possible only if already
            # submitted; with workers=1 and seed 1 first, later seeds must not succeed.
            recs.append(_record(
                "pipeline_halt_stops_later_entropy",
                2 not in seeds_ok and 3 not in seeds_ok,
                successful_seeds=sorted(s for s in seeds_ok if s is not None),
            ))
        else:
            recs.append(_record(
                "pipeline_continue_runs_later_entropy",
                2 in seeds_ok and 3 in seeds_ok,
                successful_seeds=sorted(s for s in seeds_ok if s is not None),
                n_ok=n_ok,
            ))
    return recs


def test_valid_for_fits_eligibility():
    """A missing instance or valid_for_fits=false must not default to True."""
    from qumode_qsk_gpu import jobs
    import shutil
    import tempfile

    recs = []
    # Missing instance
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with jobs.temporary_paths(inst=td / "instances", ent=td / "entropy",
                                  ref=td / "references", runs=td / "runs",
                                  logs=td / "logs"):
            meta = jobs.instance_meta(6, 1)
            recs.append(_record("valid_for_fits_missing_instance",
                                meta["valid_for_fits"] is False and meta.get("instance_found") is False,
                                meta=meta))
            # Explicit false + non-interior label
            inst_d = td / "instances"
            inst_d.mkdir()
            inst = dict(N=6, seed=831106000, valid_for_fits=False,
                        label="no_unique_interior_peak", b=0.5)
            (inst_d / "N6_seed831106000.json").write_text(json.dumps(inst))
            meta2 = jobs.instance_meta(6, 831106000)
            recs.append(_record("valid_for_fits_explicit_false",
                                meta2["valid_for_fits"] is False
                                and meta2.get("label") == "no_unique_interior_peak",
                                meta=meta2))
            # Key absent -> False
            inst3 = dict(N=6, seed=2, label="interior_unique_GS_entropy_peak")
            (inst_d / "N6_seed2.json").write_text(json.dumps(inst3))
            meta3 = jobs.instance_meta(6, 2)
            recs.append(_record("valid_for_fits_key_absent",
                                meta3["valid_for_fits"] is False,
                                meta=meta3))
    return recs


def test_temporary_paths_restore():
    from qumode_qsk_gpu import jobs
    from qumode_qsk_gpu import paths as pathmod
    import tempfile
    prev = jobs._path_snapshot()
    recs = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with jobs.temporary_paths(runs=td / "runs", inst=td / "inst",
                                  ent=td / "ent", ref=td / "ref", logs=td / "logs"):
            mid = jobs._path_snapshot()
            recs.append(_record("paths_redirected", Path(mid["runs"]) == td / "runs"))
        after = jobs._path_snapshot()
        recs.append(_record(
            "paths_restored",
            after == prev,
            after={k: str(v) for k, v in after.items()},
            prev={k: str(v) for k, v in prev.items()},
        ))
        recs.append(_record("logs_restored", pathmod.LOGS == prev["logs"],
                            logs=str(pathmod.LOGS), want=str(prev["logs"])))
    return recs


def test_version_pins_readable():
    """Pins exist and are complete; enforcement itself is in the gate on a real run."""
    from qumode_qsk_gpu import gate
    pins = gate.campaign_library_pins()
    host = gate.host_library_versions()
    ok = all(pins.get(k) for k in ("python", "numpy", "scipy", "numba"))
    return [_record("campaign_library_pins_complete", ok, pins=pins, host=host)]


def run_all_host():
    checks = []
    failures = []

    def absorb(fn, name):
        try:
            recs = fn()
        except Exception as ex:
            import traceback
            recs = [_record(name, False, error=repr(ex), traceback=traceback.format_exc())]
        if isinstance(recs, dict):
            recs = [recs]
        for rec in recs:
            checks.append(rec)
            if not rec.get("ok", False):
                failures.append(rec)

    absorb(test_strided_beta_contiguous, "strided_beta")
    absorb(test_pipeline_halt_and_continue, "pipeline")
    absorb(test_valid_for_fits_eligibility, "valid_for_fits")
    absorb(test_temporary_paths_restore, "paths_restore")
    absorb(test_version_pins_readable, "version_pins")
    return dict(verdict="PASS" if not failures else "FAIL",
                n_checks=len(checks), n_failures=len(failures),
                failures=failures, checks=checks, cuda=False,
                note="Host-only contracts. Not a CUDA validation.")


if __name__ == "__main__":
    import json as _json
    rep = run_all_host()
    print(_json.dumps({k: v for k, v in rep.items() if k != "checks"}, indent=2))
    print("n_checks", rep["n_checks"], "verdict", rep["verdict"])
    raise SystemExit(0 if rep["verdict"] == "PASS" else 1)
