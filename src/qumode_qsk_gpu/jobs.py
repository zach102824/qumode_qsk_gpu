"""Production job writers for the GPU campaign tree (not the CPU campaign).

Writes only under this repository. Atomic JSON/npz replace. Resume requires
matching source/stamp provenance. valid_for_fits is read from the entropy
instance record and never defaults to True.

Production is fail-closed: see gate.assert_production_allowed().
"""
from __future__ import annotations

import os, time, traceback, hashlib
import pathlib
from contextlib import contextmanager

import numpy as np

from . import protocol as P
from . import runtime, gate, memory
from .model import Model, make_J
from .entropy import scan_bstar_cpu
from .spsa import optimize as spsa_optimize
from .bfgs import make_start, run_start, select_by_energy
from .io_util import save_json, sha256_file, x_sha256, read_json, save_npz_atomic
from .paths import ROOT, RUNS, LOGS
from .dtypes import REAL

COHORT = "GPU_MAIN"
GPU_RUNS = RUNS / "main"
INST_DIR = ROOT / "data" / "instances" / "main"
ENT_DIR = ROOT / "data" / "entropy" / "main"
REF_DIR = ROOT / "data" / "references" / "main"
# Forbidden write roots are discovered portably: an explicit env var, and
# a sibling directory named qumode_qsk if it exists and is not this tree.
# The original authoring path is recorded only in provenance documents.

# Optional process-level cache of (diags, shell) per (n, seed)
_DIAG_CACHE = {}
_DIAG_CACHE_MAX = 4


def _path_snapshot():
    from . import paths
    return dict(runs=GPU_RUNS, inst=INST_DIR, ent=ENT_DIR, ref=REF_DIR, logs=paths.LOGS)


def _restore_paths(snap):
    """Replace all roots from a snapshot (None is not 'skip')."""
    global GPU_RUNS, INST_DIR, ENT_DIR, REF_DIR
    from . import paths
    GPU_RUNS = pathlib.Path(snap["runs"])
    INST_DIR = pathlib.Path(snap["inst"])
    ENT_DIR = pathlib.Path(snap["ent"])
    REF_DIR = pathlib.Path(snap["ref"])
    paths.LOGS = pathlib.Path(snap["logs"])


def configure_paths(runs=None, inst=None, ent=None, ref=None, logs=None):
    """Redirect output roots (GPU_test scratch). Returns a snapshot for restore."""
    snap = _path_snapshot()
    global GPU_RUNS, INST_DIR, ENT_DIR, REF_DIR
    if runs is not None:
        GPU_RUNS = pathlib.Path(runs)
        _assert_not_campaign(GPU_RUNS)
    if inst is not None:
        INST_DIR = pathlib.Path(inst)
        _assert_not_campaign(INST_DIR)
    if ent is not None:
        ENT_DIR = pathlib.Path(ent)
        _assert_not_campaign(ENT_DIR)
    if ref is not None:
        REF_DIR = pathlib.Path(ref)
        _assert_not_campaign(REF_DIR)
    if logs is not None:
        from . import paths
        paths.LOGS = pathlib.Path(logs)
        _assert_not_campaign(paths.LOGS)
    return snap


@contextmanager
def temporary_paths(**kw):
    snap = configure_paths(**kw)
    try:
        yield
    finally:
        _restore_paths(snap)


def _forbidden_write_roots():
    roots = []
    env = os.environ.get("QUMODE_QSK_CPU_CAMPAIGN")
    if env:
        roots.append(pathlib.Path(env).expanduser().resolve())
    sibling = (ROOT.parent / "qumode_qsk").resolve()
    here = ROOT.resolve()
    if sibling.exists() and sibling != here:
        roots.append(sibling)
    return roots


def _assert_not_campaign(path):
    path = pathlib.Path(path).resolve()
    for root in _forbidden_write_roots():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise RuntimeError(f"refusing to write inside the CPU campaign tree {root}: {path}")


def result_path(spec):
    kind = spec["task"]
    n, seed = int(spec["N"]), int(spec["seed"])
    if kind == "entropy":
        return GPU_RUNS / f"entropy_N{n}_seed{seed}.json"
    if kind == "spsa":
        return GPU_RUNS / f"spsa_N{n}_seed{seed}_L{spec['L']}_{spec['method']}.json"
    if kind == "bfgs":
        return GPU_RUNS / f"bfgs_N{n}_seed{seed}_L{spec['L']}_{spec['method']}_start{spec['start_index']}.json"
    if kind == "bfgs_pack":
        return GPU_RUNS / f"bfgs_N{n}_seed{seed}_L{spec['L']}_{spec['method']}_pack.json"
    raise ValueError(kind)


def _provenance_ok(rec):
    cur = gate.provenance_fields()
    for k in ("binding_sha256", "source_pins_sha256", "numpy_version",
              "scipy_version", "numba_version", "python_version",
              "cupy_version", "dtype_real", "dtype_cplx", "kernel_path"):
        if k not in rec:
            return False
        if rec.get(k) != cur.get(k):
            return False
    return True


def load_existing(spec):
    path = result_path(spec)
    if not path.exists():
        return None
    try:
        rec = read_json(path)
    except Exception:
        return None
    if rec.get("error") or rec.get("verdict") == "FAIL" or not rec.get("wall_s"):
        return None
    var = rec.get("coupling_variance")
    if var is not None and var != "1/N":
        return None
    if rec.get("N") != int(spec["N"]) or rec.get("seed") != int(spec["seed"]):
        return None
    if spec.get("L") is not None and rec.get("L") != int(spec["L"]):
        return None
    if spec.get("method") is not None and rec.get("method") != spec["method"]:
        return None
    if spec.get("task") == "spsa" and rec.get("calls") != 1024 * int(spec["L"]):
        return None
    if not _provenance_ok(rec):
        return None
    rec["result_file"] = str(path)
    rec["reused"] = True
    return rec


def instance_meta(n, seed):
    """Read valid_for_fits and label from the instance file. Missing => invalid."""
    inst = INST_DIR / f"N{n}_seed{seed}.json"
    if not inst.exists():
        return dict(valid_for_fits=False, label=None, b=None, instance_found=False)
    rec = read_json(inst)
    # Never default a missing flag to True.
    if "valid_for_fits" not in rec:
        valid = False
    else:
        valid = bool(rec.get("valid_for_fits"))
    return dict(
        valid_for_fits=valid,
        label=rec.get("label"),
        b=rec.get("b"),
        instance_found=True,
    )


def _attach(out, spec, extra=None):
    out = dict(out)
    out.update(gate.provenance_fields())
    if extra:
        out.update(extra)
    return out


def _assert_memory(n, L, method, batch):
    if not runtime.is_acquired():
        return None
    info = runtime.info()
    plan = memory.plan(n, L, method, info.total_bytes, requested_batch=batch)
    if plan["max_batch"] < int(batch):
        raise MemoryError(
            f"device cannot fit N={n} L={L} {method} batch={batch}: {plan}"
        )
    return plan


def _host_scan_provenance():
    """Library versions that do not require CUDA (worker-safe)."""
    from . import gate
    host = gate.host_library_versions()
    return dict(
        numpy_version=host.get("numpy"),
        scipy_version=host.get("scipy"),
        numba_version=host.get("numba"),
        python_version=host.get("python"),
        eigensolver="cpu_oracle_scan_bstar",
        host_scan_only=True,
    )


def _save_instance_and_scan(n, seed, report, J, gs, attach_gpu=True):
    for d in (INST_DIR, ENT_DIR, REF_DIR):
        d.mkdir(parents=True, exist_ok=True)
        _assert_not_campaign(d)
    iu = np.triu_indices(n, 1)
    inst = dict(
        N=n, seed=int(seed), cohort=COHORT,
        coupling_variance="1/N",
        hamiltonian=P.HAMILTONIAN,
        J_upper=J[iu].tolist(),
        b=report["peak"]["b"],
        E0=report["peak"]["E0"],
        gap=report["peak"]["gap"],
        even_gap=report["peak"]["even_gap"],
        S=report["peak"]["S"],
        label=report["label"],
        valid_for_fits=bool(report["valid_for_fits"]),
        gs_sha256=hashlib.sha256(np.ascontiguousarray(gs).tobytes()).hexdigest(),
        eigensolver="cpu_oracle_scan_bstar",
    )
    if attach_gpu:
        inst.update(gate.provenance_fields())
    else:
        inst.update(_host_scan_provenance())
    inst_path = INST_DIR / f"N{n}_seed{seed}.json"
    scan_path = ENT_DIR / f"N{n}_seed{seed}_scan.json"
    ref_path = REF_DIR / f"N{n}_seed{seed}_reference.npz"
    for pth in (inst_path, scan_path, ref_path):
        _assert_not_campaign(pth)
    save_json(inst_path, inst)
    save_json(scan_path, report)
    save_npz_atomic(ref_path, J=J, gs=gs, E0=report["peak"]["E0"], b=report["peak"]["b"],
                    gap=report["peak"]["gap"], even_gap=report["peak"]["even_gap"],
                    S=report["peak"]["S"], seed=seed, N=n)
    return dict(instance=str(inst_path), scan=str(scan_path), reference=str(ref_path),
                reference_sha256=sha256_file(ref_path), reference_bytes=ref_path.stat().st_size)


def load_model(n, seed):
    ref = REF_DIR / f"N{n}_seed{seed}_reference.npz"
    z = np.load(ref)
    J = z["J"]
    meta = instance_meta(n, seed)
    gs = z["gs"]
    gs_sha = hashlib.sha256(np.ascontiguousarray(gs).tobytes()).hexdigest()
    if meta.get("gs_sha256") and gs_sha != meta["gs_sha256"]:
        raise AssertionError(
            f"reference gs sha {gs_sha} != instance gs_sha256 {meta.get('gs_sha256')}"
        )
    # Reuse host diags/shell for this (N, seed) to avoid rebuilding 3 N(N-1)/2 loops.
    cache_key = (int(n), int(seed), hashlib.sha256(np.ascontiguousarray(J).tobytes()).hexdigest())
    built = _DIAG_CACHE.get(cache_key)
    m = Model(n, seed, float(z["b"]), J=J, compute_gs=False, built=built)
    if built is None:
        _DIAG_CACHE[cache_key] = dict(diags=m.diags_host, shell=m.shell_host,
                                      groups=getattr(m, "groups", None), jrms=getattr(m, "jrms", None))
        if len(_DIAG_CACHE) > _DIAG_CACHE_MAX:
            # drop an arbitrary old entry (not this one)
            for k in list(_DIAG_CACHE):
                if k != cache_key:
                    _DIAG_CACHE.pop(k, None)
                    break
    if not np.array_equal(m.J, J):
        raise AssertionError("J mismatch")
    m.load_gs(gs, float(z["E0"]), gap=float(z["gap"]))
    return m, z, meta


def cpu_entropy_scan_only(spec):
    """Host-only B* scan. No CUDA, no GPU Model, no cupy work.

    Writes instance/scan/reference without GPU provenance. The parent
    must call finalize_entropy_gpu to attach the gate binding and the
    optional apply_H cross-check.
    """
    n, seed = int(spec["N"]), int(spec["seed"])
    t0 = time.time()
    J = make_J(n, seed)
    from .paths import LOGS
    progress_path = LOGS / f"entropy_scan_gpu_N{n}_seed{seed}.jsonl"
    _assert_not_campaign(progress_path)
    report, gs = scan_bstar_cpu(n, seed, J, progress_path=progress_path)
    paths = _save_instance_and_scan(n, seed, report, J, gs, attach_gpu=False)
    return dict(
        task="entropy_host_scan",
        N=n, seed=seed, index=spec.get("index"),
        b=report["peak"]["b"], S=report["peak"]["S"], E0=report["peak"]["E0"],
        gap=report["peak"]["gap"], even_gap=report["peak"]["even_gap"],
        label=report["label"], valid_for_fits=bool(report["valid_for_fits"]),
        residual=report["peak"]["residual"], parity=report["peak"]["parity"],
        scan_resource=report["scan_resource"],
        eigensolver="cpu_oracle_scan_bstar",
        paths=paths,
        wall_s_scan=time.time() - t0,
        pid=os.getpid(),
        failed=False,
        used_cuda=False,
    )


def finalize_entropy_gpu(spec, rec):
    """On the GPU/main thread: apply_H cross-check, GPU provenance, result JSON.

    Rewrites the instance file with the full binding once acquire() has run.
    """
    n, seed = int(spec["N"]), int(spec["seed"])
    t0 = time.time()
    paths = rec["paths"]
    z = np.load(paths["reference"])
    J, gs = z["J"], z["gs"]
    Echk = None
    residual_note = "cpu_scan_residual"
    if runtime.is_acquired():
        m2 = Model(n, seed, float(rec["b"]), J=J, compute_gs=False)
        m2.load_gs(gs, rec["E0"])
        Echk = m2.energy(m2.gs)
        residual_note = "gpu_apply_H_vs_cpu_E0"
        # Refresh instance with GPU provenance (binding_sha256 etc.).
        inst_path = pathlib.Path(paths["instance"])
        inst = read_json(inst_path)
        inst.update(gate.provenance_fields())
        inst.pop("host_scan_only", None)
        save_json(inst_path, inst)
    out = dict(
        task="entropy", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
        b=rec["b"], S=rec["S"], E0=rec["E0"],
        gap=rec["gap"], even_gap=rec["even_gap"],
        label=rec["label"], valid_for_fits=bool(rec["valid_for_fits"]),
        residual=rec["residual"], parity=rec["parity"],
        E_apply_H_abs_err=None if Echk is None else abs(float(Echk) - rec["E0"]),
        residual_check=residual_note,
        eigensolver="cpu_oracle_scan_bstar",
        gpu_model_only_for_energy_crosscheck=True,
        scan_resource=rec.get("scan_resource"),
        wall_s=float(rec.get("wall_s_scan") or 0.0) + (time.time() - t0),
        pid=os.getpid(), paths=paths,
        coupling_variance="1/N",
    )
    if runtime.is_acquired():
        out = _attach(out, spec)
    else:
        out.update(_host_scan_provenance())
    dest = result_path(spec)
    _assert_not_campaign(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json(dest, out)
    out["result_file"] = str(dest)
    out["failed"] = False
    return out


def task_entropy(spec):
    """Sequential host scan + optional GPU finalize (same-thread, tests/dispatch)."""
    rec = cpu_entropy_scan_only(spec)
    return finalize_entropy_gpu(spec, rec)


def task_spsa(spec):
    n, seed, L, method = int(spec["N"]), int(spec["seed"]), int(spec["L"]), spec["method"]
    t0 = time.time()
    plan = _assert_memory(n, L, method, batch=2)
    m, z, meta = load_model(n, seed)
    plus_E = m.energy(m.plus.reshape(1, -1))
    plus_F = m.fidelity(m.plus.reshape(1, -1))
    out = spsa_optimize(m, method, L)
    compact_log = [(q["query"], q["kind"], q["E"]) for q in out["query_log"]]
    out.update(
        task="spsa", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
        b=float(z["b"]), E0=float(z["E0"]), plus_E=plus_E, plus_F=plus_F,
        energy_error=out["best_E"] - float(z["E0"]),
        energy_error_over_N=(out["best_E"] - float(z["E0"])) / n,
        wall_s=time.time() - t0, pid=os.getpid(),
        query_log_sha256=hashlib.sha256(repr(compact_log).encode()).hexdigest(),
        coupling_variance="1/N",
        valid_for_fits=bool(meta["valid_for_fits"]),
        entropy_label=meta.get("label"),
        memory_plan=plan,
        device_name=runtime.device_name() if runtime.is_acquired() else None,
    )
    out = _attach(out, spec)
    dest = result_path(spec)
    _assert_not_campaign(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stored = {k: out[k] for k in out if k not in ("query_log", "readout_candidates", "trace")}
    save_json(dest, stored)
    summary = {k: out[k] for k in out if k not in ("query_log", "readout_candidates", "trace", "initial_points", "x")}
    summary.update(result_file=str(dest), best_E=out["best_E"], F=out["F"], calls=out["calls"],
                   x_sha256=x_sha256(out["x"]))
    return summary


def task_bfgs_pack(spec):
    n, seed, L, method = int(spec["N"]), int(spec["seed"]), int(spec["L"]), spec["method"]
    t0 = time.time()
    plan = _assert_memory(n, L, method, batch=1)
    m, z, meta = load_model(n, seed)
    rows = []
    n_reused = 0
    n_ran = 0
    for k in range(P.BFGS_STARTS):
        sub = dict(task="bfgs", N=n, seed=seed, index=spec.get("index"),
                   L=L, method=method, start_index=k)
        ex = load_existing(sub)
        if ex is not None:
            rows.append(ex)
            n_reused += 1
            continue
        st = make_start(seed, L, method, k)
        t1 = time.time()
        out = run_start(m, method, L, st)
        out.update(
            task="bfgs", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
            L=L, method=method, b=float(z["b"]),
            wall_s=time.time() - t1, pid=os.getpid(),
            coupling_variance="1/N",
            valid_for_fits=bool(meta["valid_for_fits"]),
            entropy_label=meta.get("label"),
            device_name=runtime.device_name() if runtime.is_acquired() else None,
        )
        out = _attach(out, sub)
        dest = result_path(sub)
        _assert_not_campaign(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        stored = {k2: out[k2] for k2 in out if k2 not in ("final_grad",)}
        save_json(dest, stored)
        summary = {k2: out[k2] for k2 in out if k2 not in ("x", "final_grad")}
        summary.update(result_file=str(dest), x_sha256=out["x_sha256"], reused=False)
        rows.append(summary)
        n_ran += 1
    sel = select_by_energy(rows)
    pack = dict(
        task="bfgs_pack", N=n, seed=seed, cohort=COHORT, index=spec.get("index"),
        L=L, method=method, b=float(z["b"]), E0=float(z["E0"]),
        n_starts=len(rows), n_reused=n_reused, n_ran=n_ran,
        selected=sel, selected_by="energy_not_F",
        start_files=[r.get("result_file") for r in rows],
        wall_s=time.time() - t0, pid=os.getpid(),
        coupling_variance="1/N",
        valid_for_fits=bool(meta["valid_for_fits"]),
        entropy_label=meta.get("label"),
        memory_plan=plan,
        device_name=runtime.device_name() if runtime.is_acquired() else None,
    )
    pack = _attach(pack, spec)
    dest = result_path(spec)
    _assert_not_campaign(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_json(dest, pack)
    pack["result_file"] = str(dest)
    return pack


def dispatch(spec, require_gate=True):
    kind = spec["task"]
    try:
        if require_gate:
            gate.assert_production_allowed()
        # Entropy host scan itself uses no CUDA; still enter the device
        # context only for optimizer tasks / finalize on this thread.
        with runtime.device_context() if runtime.is_acquired() else _nullctx():
            if kind == "entropy":
                return task_entropy(spec)
            if kind == "spsa":
                return task_spsa(spec)
            if kind == "bfgs_pack":
                return task_bfgs_pack(spec)
            if kind == "bfgs":
                raise ValueError("run bfgs_pack, not a bare bfgs task, in production")
            raise ValueError(kind)
    except Exception as ex:
        rec = dict(task=kind, spec=spec, error=repr(ex), traceback=traceback.format_exc(),
                   pid=os.getpid(), wall_s=None, verdict="FAIL")
        if runtime.is_acquired():
            try:
                rec.update(gate.provenance_fields())
            except Exception:
                rec["provenance_error"] = "failed to attach provenance on FAIL"
        try:
            dest = result_path(spec)
            _assert_not_campaign(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            save_json(dest.with_name(dest.stem + "_FAIL.json"), rec)
        except Exception:
            pass
        return rec


class _nullctx:
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def plan_round(round_id):
    idxs = P.round_indices(round_id)
    entropy, spsa, bfgs_pack = [], [], []
    for n in P.NS:
        for i in idxs:
            seed = P.main_seed(n, i)
            entropy.append(dict(task="entropy", N=n, seed=seed, index=i, round=round_id))
            for L in P.DEPTHS:
                for method in P.METHODS:
                    spsa.append(dict(task="spsa", N=n, seed=seed, index=i, L=L, method=method, round=round_id))
                    bfgs_pack.append(dict(task="bfgs_pack", N=n, seed=seed, index=i, L=L, method=method, round=round_id))
    return dict(round=round_id, indices=idxs, entropy=entropy, spsa=spsa, bfgs_pack=bfgs_pack,
                n_instances_cap=P.N_INSTANCES, round_size=P.ROUND_SIZE)


def official_complete(round_id, runs_dir=None):
    """True if every official cell of this round has a reusable result."""
    prev_runs = GPU_RUNS
    if runs_dir is not None:
        configure_paths(runs=runs_dir)
    try:
        plan = plan_round(round_id)
        missing = []
        for spec in plan["entropy"] + plan["spsa"]:
            if load_existing(spec) is None:
                missing.append(spec)
        for spec in plan["bfgs_pack"]:
            # require all 16 starts
            for k in range(P.BFGS_STARTS):
                sub = dict(task="bfgs", N=spec["N"], seed=spec["seed"], L=spec["L"],
                           method=spec["method"], start_index=k, index=spec.get("index"))
                if load_existing(sub) is None:
                    missing.append(sub)
                    break
        return dict(complete=not missing, n_missing=len(missing), missing=missing[:20],
                    round=round_id)
    finally:
        if runs_dir is not None:
            configure_paths(runs=prev_runs)
