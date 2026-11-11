"""GPU_test suite — run on a CUDA host. Not executed during code-only authoring.

Covers fixed-parameter E/F/state/grad (oracle + FD), entropy of stored GS,
fixture integrity, start recipes, batch>=2, odd-N synthetic checks, and the
SPSA / BFGS / jobs drivers (H7). Exceptions are recorded; a report is always
produced. CUDA correctness cannot be proved by AST.
"""
from __future__ import annotations

import json
import traceback
import shutil
from pathlib import Path

import numpy as np

from conftest_paths import setup_paths
ROOT = setup_paths()

from qumode_qsk_gpu.io_util import now_shanghai, x_sha256
from qumode_qsk_gpu.dtypes import REAL, CPLX


def load_tolerances():
    return json.loads((ROOT / "GPU_test/tolerances.json").read_text())


def load_catalog():
    return json.loads((ROOT / "GPU_test/fixtures/validation_catalog.json").read_text())


def _tol(tols, key, n):
    v = tols[key]
    if isinstance(v, dict) and "N_le_15" in v:
        return float(v["N_le_15"] if int(n) <= 15 else v["N_ge_18"])
    return float(v)


def _phase_aligned_err(a, b):
    """max|a - e^{iφ} b| with φ from the overlap, plus the strict max|a-b|."""
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    strict = float(np.max(np.abs(a - b)))
    ov = np.vdot(b, a)
    if abs(ov) == 0:
        return strict, strict
    phase = ov / abs(ov)
    aligned = float(np.max(np.abs(a - phase * b)))
    return strict, aligned


def _record(check, ok, **kw):
    rec = dict(check=check, ok=bool(ok))
    rec.update(kw)
    return rec


def _safe(fn, check, *a, **k):
    try:
        out = fn(*a, **k)
        if isinstance(out, list):
            return out
        if isinstance(out, dict):
            out.setdefault("check", check)
            out.setdefault("ok", True)
            return [out]
        return [_record(check, True, value=out)]
    except Exception as ex:
        return [_record(check, False, error=repr(ex), traceback=traceback.format_exc())]


def _models(n, seed):
    from qumode_qsk_gpu.model import Model
    z = np.load(ROOT / "GPU_test/fixtures/references" / f"N{n}_seed{seed}_reference.npz")
    m = Model(n, seed, float(z["b"]), J=z["J"], compute_gs=False)
    m.load_gs(z["gs"], float(z["E0"]), gap=float(z["gap"]))
    return m, z


def _cpu_model(n, seed, z):
    from qumode_qsk_gpu import oracle
    ham = oracle.hamiltonian()
    m = ham.Model(n, seed, float(z["b"]), J=z["J"], compute_gs=False)
    m.load_gs(z["gs"], float(z["E0"]), gap=float(z["gap"]))
    return m


def test_fixture_integrity():
    from qumode_qsk_gpu import gate
    from qumode_qsk_gpu.model import make_J
    from qumode_qsk_gpu.io_util import sha256_file, x_sha256
    ver = gate.verify_fixture_manifest()
    rows = [_record("fixture_manifest_sha", ver["ok"], **{k: ver[k] for k in ver if k != "ok"})]
    man = json.loads((ROOT / "GPU_test/fixtures/MANIFEST.json").read_text())
    cat = load_catalog()
    for e in cat["entries"]:
        if e["kind"] != "entropy_reference":
            continue
        n, seed = int(e["N"]), int(e["seed"])
        z = np.load(ROOT / "GPU_test/fixtures/references" / f"N{n}_seed{seed}_reference.npz")
        gs_sha = __import__("hashlib").sha256(np.ascontiguousarray(z["gs"]).tobytes()).hexdigest()
        inst = json.loads((ROOT / e["instance_file"]).read_text())
        ok_gs = gs_sha == inst["gs_sha256"]
        J = make_J(n, seed)
        ok_J = np.array_equal(J, z["J"])
        iu = np.triu_indices(n, 1)
        ok_upper = np.allclose(J[iu], np.asarray(inst["J_upper"], dtype=np.float64), rtol=0, atol=0) or np.array_equal(J[iu], np.asarray(inst["J_upper"], dtype=np.float64))
        rows.append(_record("fixture_gs_and_J", ok_gs and ok_J and ok_upper,
                            N=n, seed=seed, gs_sha_ok=ok_gs, J_ok=ok_J, J_upper_ok=ok_upper))
    return rows


def test_start_recipes():
    from qumode_qsk_gpu.starts import spsa_starts, make_start
    cat = load_catalog()
    n_pts = 0
    rows = []
    for e in cat["entries"]:
        if e["kind"] == "spsa_fixed_params":
            xs, _ = spsa_starts(e["seed"], e["L"], e["method"])
            for a, b in zip(xs, e["initial_points"]):
                if not np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)):
                    return [_record("start_recipes", False, detail="spsa mismatch", seed=e["seed"])]
            n_pts += len(xs)
        elif e["kind"] == "bfgs_fixed_params":
            st = make_start(e["seed"], e["L"], e["method"], e["start_index"])
            if st["x_sha256"] != e["start_x_sha256"] or st["rng_seed"] != e["start_rng_seed"]:
                return [_record("start_recipes", False, detail="bfgs mismatch", file=e["result_file"])]
            n_pts += 1
    return [_record("start_recipes", n_pts == 324, n_start_vectors_checked=n_pts, expected=324)]


def test_stored_gs_entropy_residual(m, z, entry, tols):
    from qumode_qsk_gpu.entropy import entropy_host, entropy_device
    n = int(entry["N"])
    rows = []
    res = m.residual_on_device()
    rows.append(_record("E0_residual", res < tols["E0_residual"], residual=res, limit=tols["E0_residual"], N=n, seed=entry["seed"]))
    E = m.energy(m.gs)
    dE = abs(float(E) - float(entry["E0"]))
    # M6: energy tolerance alone; residual is a separate check.
    rows.append(_record("E_gs_vs_E0", dE <= _tol(tols, "energy_abs_vs_cpu_stored", n),
                        dE=dE, N=n, seed=entry["seed"]))
    px = m.parity_px(m.gs)
    if entry.get("parity") is not None:
        dp = abs(float(px) - float(entry["parity"]))
        rows.append(_record("parity", dp <= tols["parity_abs_vs_stored"], stored=entry["parity"], got=float(px), d=dp))
    S_host = entropy_host(z["gs"], n)
    dS = abs(S_host - float(entry["S"]))
    rows.append(_record("entropy_host_of_stored_gs", dS <= tols["entropy_abs_vs_stored_S"],
                        S=S_host, stored=entry["S"], dS=dS))
    try:
        S_dev = entropy_device(m.gs, n)
        dSd = abs(S_dev - float(entry["S"]))
        rows.append(_record("entropy_device_of_stored_gs", dSd <= tols["entropy_abs_vs_stored_S"], S=S_dev, dS=dSd))
    except Exception as ex:
        rows.append(_record("entropy_device_of_stored_gs", False, error=repr(ex),
                            note="device SVD failed; host entropy is the official comparison"))
    scan = json.loads((ROOT / "GPU_test/fixtures/entropy" / f"N{n}_seed{z['seed']}_scan.json").read_text())
    curve = scan["scan"]
    peak = max(curve, key=lambda r: r["S"])
    ok_ord = abs(peak["b"] - float(entry["b"])) < 1e-12 and abs(peak["S"] - float(entry["S"])) < 1e-12
    ordered = sorted(curve, key=lambda r: r["b"])
    bs = [r["b"] for r in ordered]
    k = bs.index(peak["b"])
    interior = 0 < k < len(ordered) - 1
    if interior:
        ok_ord = ok_ord and peak["S"] > ordered[k-1]["S"] + 1e-12 and peak["S"] > ordered[k+1]["S"] + 1e-12
    rows.append(_record("entropy_scan_peak_ordering", ok_ord, peak_b=peak["b"],
                        label=scan.get("label"), interior=interior))
    return rows


def test_fixed_params(m, cpu_m, entry, tols, with_fd=True):
    from qumode_qsk_gpu.gradients import energy_grad, central_diff_grad
    from qumode_qsk_gpu import runtime, kernels
    from qumode_qsk_gpu import oracle
    ham = oracle.hamiltonian()
    n = int(entry["N"])
    L = int(entry["L"])
    method = entry["method"]
    rows = []

    def one(x, stored_E=None, stored_F=None, tag="", do_fd=with_fd):
        x = np.ascontiguousarray(x, dtype=REAL)
        E, g, v = energy_grad(m, x, method, L)
        F = m.fidelity(v) if m.gs is not None else None
        nrm = float(runtime.to_host(kernels.norms(v if v.ndim == 2 else v.reshape(1, -1))).reshape(-1)[0])
        rec = dict(tag=tag, E=float(E), F=F, norm=nrm, ok=True, check=tag)
        if stored_E is not None:
            rec["dE_vs_stored"] = abs(float(E) - float(stored_E))
            rec["ok"] = rec["ok"] and rec["dE_vs_stored"] <= _tol(tols, "energy_abs_vs_cpu_stored", n)
        if stored_F is not None and F is not None:
            rec["dF_vs_stored"] = abs(float(F) - float(stored_F))
            rec["ok"] = rec["ok"] and rec["dF_vs_stored"] <= _tol(tols, "fidelity_abs_vs_cpu_stored", n)
        rec["ok"] = rec["ok"] and abs(nrm - 1.0) <= tols["norm_abs_vs_one"]
        Ec, gc, vc = ham.energy_grad(cpu_m, x, method, L)
        psi = runtime.to_host(v if v.ndim == 2 else v.reshape(1, -1))[0]
        strict, aligned = _phase_aligned_err(psi, vc)
        rec["state_max_abs_vs_oracle"] = strict
        rec["state_max_abs_vs_oracle_phase_aligned"] = aligned
        rec["grad_max_abs_vs_oracle"] = float(np.max(np.abs(np.asarray(g) - np.asarray(gc))))
        rec["dE_vs_oracle"] = abs(float(E) - float(Ec))
        rec["ok"] = rec["ok"] and strict <= _tol(tols, "state_max_abs_vs_cpu_oracle", n)
        rec["ok"] = rec["ok"] and rec["grad_max_abs_vs_oracle"] <= _tol(tols, "grad_max_abs_vs_cpu_oracle", n)
        rec["ok"] = rec["ok"] and rec["dE_vs_oracle"] <= _tol(tols, "energy_abs_vs_cpu_stored", n)
        if do_fd:
            g_fd = central_diff_grad(m, x, method, L, eps=1e-7)
            rec["grad_max_abs_vs_fd"] = float(np.max(np.abs(np.asarray(g) - g_fd)))
            rec["ok"] = rec["ok"] and rec["grad_max_abs_vs_fd"] <= tols["grad_max_abs_vs_central_fd_eps_1e-7"]
        return rec

    if entry["kind"] == "spsa_fixed_params":
        rows.append(one(entry["x"], entry["best_E"], entry["F"], tag="spsa_best_x"))
        plus_E = m.energy(m.plus.reshape(1, -1))
        plus_F = m.fidelity(m.plus.reshape(1, -1))
        rec = dict(check="plus", tag="plus", E=float(plus_E), F=float(plus_F),
                   dE=abs(float(plus_E) - float(entry["plus_E"])),
                   dF=abs(float(plus_F) - float(entry["plus_F"])))
        rec["ok"] = rec["dE"] <= tols["plus_energy_abs"] and rec["dF"] <= _tol(tols, "fidelity_abs_vs_cpu_stored", n)
        rows.append(rec)
        for i, x0 in enumerate(entry["initial_points"]):
            rows.append(one(x0, tag=f"spsa_start_{i}"))
    elif entry["kind"] == "bfgs_fixed_params":
        rows.append(one(entry["start_x"], entry["E_initial"], entry["F_initial"], tag="bfgs_start_x"))
        rows.append(one(entry["x"], entry["best_E"], entry["F"], tag="bfgs_final_x"))
    return rows


def test_batch_ge2(m, cpu_m, entry, tols):
    """Exercise batch=2 energy and gradients (SPSA ± path, H1 RX broadcast)."""
    from qumode_qsk_gpu.gradients import energy_grad
    from qumode_qsk_gpu import oracle
    ham = oracle.hamiltonian()
    if entry["kind"] != "spsa_fixed_params":
        return []
    xs = np.stack([np.asarray(entry["initial_points"][0], dtype=REAL),
                   np.asarray(entry["initial_points"][1], dtype=REAL)], axis=0)
    E, v = m.energy_of_params(xs, entry["L"], entry["method"])
    E = np.asarray(E, dtype=REAL).reshape(2)
    Eg, gg, vg = energy_grad(m, xs, entry["method"], entry["L"])
    rows = []
    n = entry["N"]
    for i in range(2):
        Ec, gc, vc = ham.energy_grad(cpu_m, xs[i], entry["method"], entry["L"])
        dE = abs(float(E[i]) - float(Ec))
        dEb = abs(float(np.asarray(Eg).reshape(2)[i]) - float(Ec))
        gerr = float(np.max(np.abs(np.asarray(gg)[i] - np.asarray(gc))))
        ok = dE <= _tol(tols, "energy_abs_vs_cpu_stored", n) and gerr <= _tol(tols, "grad_max_abs_vs_cpu_oracle", n)
        rows.append(_record("batch2_energy_grad", ok, N=n, method=entry["method"], L=entry["L"],
                            i=i, dE_energy=dE, dE_grad=dEb, grad_max_abs=gerr))
    return rows


def test_va_not_clobbered(m, entry):
    """Copy vA, apply Walsh/RX with preserve_src, confirm source is unchanged."""
    from qumode_qsk_gpu import runtime, kernels
    # use plus as a stand-in input
    v = m.plus.reshape(1, -1).copy()
    src = v.copy()
    work = runtime.xp().empty_like(src)
    out = kernels.walsh(src, m.n, work=work, preserve_src=True)
    src_h = runtime.to_host(src)
    orig = runtime.to_host(v)
    ok_w = np.array_equal(src_h, orig) or np.max(np.abs(src_h - orig)) == 0
    src2 = v.copy()
    out2 = kernels.rx(src2, 0.3, m.n, work=work, preserve_src=True)
    ok_r = float(np.max(np.abs(runtime.to_host(src2) - orig))) == 0
    return [_record("preserve_src_walsh_rx", ok_w and ok_r, walsh_src_intact=ok_w, rx_src_intact=ok_r, N=m.n)]


def test_spsa_driver(m, cpu_m, entry, tols):
    """Run the real SPSA driver against the CPU oracle.

    PASS: calls, direction hashes, query roles, starts, and the first K
    query-log energies including plus and minus separately (not a swap).
    best_E / x remain report-only for the full trajectory.
    """
    from qumode_qsk_gpu.spsa import optimize
    from qumode_qsk_gpu import oracle
    if entry["kind"] != "spsa_fixed_params":
        return []
    out = optimize(m, entry["method"], entry["L"])
    rec = dict(
        check="spsa_optimize_driver",
        N=entry["N"], seed=entry["seed"], L=entry["L"], method=entry["method"],
        calls=out["calls"], expected_calls=entry["calls"],
        direction_hashes=out["direction_hashes"],
        expected_direction_hashes=entry["direction_hashes"],
        query_roles=out["query_roles"],
        expected_query_roles=entry["query_roles"],
        initial_points_match=all(
            np.array_equal(np.asarray(a, dtype=REAL), np.asarray(b, dtype=REAL))
            for a, b in zip(out["initial_points"], entry["initial_points"])
        ),
        dE_best_vs_fixture=abs(float(out["best_E"]) - float(entry["best_E"])),
        x_sha=x_sha256(out["x"]),
        note="hashes/roles/calls/starts and prefix ledger energies are PASS; full-trajectory best_E is report-only.",
    )
    rec["ok"] = (
        out["calls"] == entry["calls"]
        and out["direction_hashes"] == entry["direction_hashes"]
        and out["query_roles"] == entry["query_roles"]
        and rec["initial_points_match"]
    )
    # CPU oracle same algorithm, compare plus/minus energy order (R2-M11).
    sp = oracle.spsa()
    cpu = sp.optimize(cpu_m, entry["method"], entry["L"])
    K = int(tols.get("spsa_ledger_prefix_queries") or 20)
    glog = list(out.get("query_log") or [])[:K]
    clog = list(cpu.get("query_log") or [])[:K]
    e_tol = _tol(tols, "spsa_ledger_energy_abs", entry["N"]) if "spsa_ledger_energy_abs" in tols else _tol(tols, "energy_abs_vs_cpu_stored", entry["N"])
    pair_ok = True
    max_dE = 0.0
    swapped = 0
    n_pairs = 0
    if len(glog) != len(clog):
        pair_ok = False
    for a, b in zip(glog, clog):
        dE = abs(float(a.get("E")) - float(b.get("E")))
        max_dE = max(max_dE, dE)
        if a.get("kind") != b.get("kind"):
            pair_ok = False
        if dE > e_tol:
            pair_ok = False
    # Explicit plus/minus pairing: the minus energy must match the CPU minus,
    # not the CPU plus (a swap would still pass a bag-of-energies check).
    for i in range(len(glog) - 1):
        if str(glog[i].get("kind", "")).endswith("_plus") and str(glog[i + 1].get("kind", "")).endswith("_minus"):
            n_pairs += 1
            if i + 1 >= len(clog):
                pair_ok = False
                continue
            d_pp = abs(float(glog[i]["E"]) - float(clog[i]["E"]))
            d_mm = abs(float(glog[i + 1]["E"]) - float(clog[i + 1]["E"]))
            d_pm = abs(float(glog[i]["E"]) - float(clog[i + 1]["E"]))
            d_mp = abs(float(glog[i + 1]["E"]) - float(clog[i]["E"]))
            if d_pp > e_tol or d_mm > e_tol:
                pair_ok = False
            # If plus matches CPU minus better than CPU plus, flag a swap.
            if d_pm + d_mp + 1e-18 < d_pp + d_mm:
                swapped += 1
                pair_ok = False
    rec["ledger_prefix_K"] = K
    rec["ledger_prefix_max_dE"] = max_dE
    rec["ledger_plus_minus_pairs"] = n_pairs
    rec["ledger_swapped_pairs"] = swapped
    rec["ledger_cpu_calls"] = cpu.get("calls")
    rec["ok"] = bool(rec["ok"] and pair_ok and swapped == 0)
    rec["dE_best_vs_oracle"] = abs(float(out["best_E"]) - float(cpu["best_E"]))
    rec["best_E_report_only"] = rec["dE_best_vs_oracle"]
    return [rec]


def _bfgs_capped_record(m, cpu_m, entry, tols, maxiter, tag):
    from qumode_qsk_gpu.bfgs import run_start, make_start
    from qumode_qsk_gpu import oracle
    bf = oracle.bfgs()
    st = make_start(entry["seed"], entry["L"], entry["method"], entry["start_index"])
    gpu = run_start(m, entry["method"], entry["L"], st, maxiter=maxiter)
    cpu = bf.run_start(cpu_m, entry["method"], entry["L"], st, maxiter=maxiter)
    xg = np.asarray(gpu["x"], dtype=REAL)
    xc = np.asarray(cpu["x"], dtype=REAL)
    gg = np.asarray(gpu["final_grad"], dtype=REAL)
    gc = np.asarray(cpu["final_grad"], dtype=REAL)
    dE = abs(float(gpu["best_E"]) - float(cpu["best_E"]))
    d_grad_l2 = abs(float(gpu["final_grad_l2"]) - float(cpu["final_grad_l2"]))
    d_x = float(np.max(np.abs(xg - xc)))
    finite = all(np.isfinite(np.r_[xg, xc, gg, gc, gpu["best_E"], cpu["best_E"], gpu["E_initial"]]))
    e_tol = float(tols.get("bfgs_capped_energy_abs", 1e-9))
    g_tol = float(tols.get("bfgs_capped_grad_l2_abs", 1e-7))
    x_tol = float(tols.get("bfgs_capped_x_max_abs", 1e-7))
    inc_tol = float(tols.get("bfgs_capped_no_increase", 1e-9))
    nit_ok = int(gpu["nit"]) == int(cpu["nit"]) and int(gpu["nfev"]) == int(cpu["nfev"])
    rec = dict(
        check=tag,
        N=entry["N"], seed=entry["seed"], L=entry["L"], method=entry["method"],
        start_index=entry.get("start_index"), maxiter=maxiter,
        gpu_nit=gpu["nit"], cpu_nit=cpu["nit"],
        gpu_nfev=gpu["nfev"], cpu_nfev=cpu["nfev"],
        dE=dE, d_grad_l2=d_grad_l2, d_x_max=d_x,
        grad_l2_gpu=gpu["final_grad_l2"], grad_l2_cpu=cpu["final_grad_l2"],
        E_initial=gpu["E_initial"], E_final=gpu["best_E"],
        E_improved=float(gpu["best_E"]) <= float(gpu["E_initial"]) + inc_tol,
        cpu_E_improved=float(cpu["best_E"]) <= float(cpu["E_initial"]) + inc_tol,
        start_x_sha_match=gpu["start_x_sha256"] == entry.get("start_x_sha256", gpu["start_x_sha256"]),
        finite=finite,
        nit_nfev_match=nit_ok,
    )
    if not nit_ok:
        rec["line_search_divergence"] = True
        rec["ok"] = False
        rec["failure"] = "line-search divergence: nit/nfev differ from the CPU oracle under the same cap"
        return rec
    # Objective: GPU must not increase when the CPU did not; always require finiteness
    # and numerical agreement with the oracle.
    obj_ok = (float(gpu["best_E"]) <= float(gpu["E_initial"]) + inc_tol) or (
        float(cpu["best_E"]) > float(cpu["E_initial"]) + inc_tol
    )
    rec["ok"] = bool(
        rec["start_x_sha_match"] and finite
        and dE <= e_tol and d_grad_l2 <= g_tol and d_x <= x_tol
        and obj_ok
    )
    if not rec["ok"]:
        rec["failure"] = "capped BFGS disagreed with the CPU oracle beyond documented tolerances"
    return rec


def test_bfgs_capped(m, cpu_m, entry, tols):
    """Meaningful capped BFGS vs the CPU oracle (R2-H4). Same SciPy settings."""
    if entry["kind"] != "bfgs_fixed_params":
        return []
    rows = []
    rows.append(_bfgs_capped_record(m, cpu_m, entry, tols, maxiter=3, tag="bfgs_capped_maxiter3"))
    rows.append(_bfgs_capped_record(m, cpu_m, entry, tols, maxiter=1, tag="bfgs_one_optimizer_step"))
    return rows


def _scratch_tree(name="scratch_jobs"):
    scratch = ROOT / "GPU_test" / name
    if scratch.exists():
        shutil.rmtree(scratch)
    dirs = {k: scratch / k for k in ("instances", "entropy", "references", "runs", "logs")}
    for d in dirs.values():
        d.mkdir(parents=True)
    dirs["root"] = scratch
    return dirs


def test_jobs_spsa_scratch(entry):
    """Run jobs.task_spsa on a scratch tree pointed at a fixture reference (require_gate=False)."""
    from qumode_qsk_gpu import jobs
    if entry["kind"] != "spsa_fixed_params":
        return []
    n, seed = int(entry["N"]), int(entry["seed"])
    d = _scratch_tree("scratch_jobs")
    shutil.copy2(ROOT / f"GPU_test/fixtures/instances/N{n}_seed{seed}.json", d["instances"] / f"N{n}_seed{seed}.json")
    shutil.copy2(ROOT / f"GPU_test/fixtures/references/N{n}_seed{seed}_reference.npz", d["references"] / f"N{n}_seed{seed}_reference.npz")
    rows = []
    with jobs.temporary_paths(runs=d["runs"], inst=d["instances"], ent=d["entropy"],
                              ref=d["references"], logs=d["logs"]):
        spec = dict(task="spsa", N=n, seed=seed, L=entry["L"], method=entry["method"], index=entry.get("index"))
        rec = jobs.dispatch(spec, require_gate=False)
        ok = rec.get("verdict") != "FAIL" and rec.get("error") is None and rec.get("calls") == entry["calls"]
        inst = json.loads((d["instances"] / f"N{n}_seed{seed}.json").read_text())
        fit_ok = rec.get("valid_for_fits") == bool(inst.get("valid_for_fits"))
        rows.append(_record("jobs_spsa_scratch", bool(ok and fit_ok),
                            calls=rec.get("calls"), valid_for_fits=rec.get("valid_for_fits"),
                            expected_valid=inst.get("valid_for_fits"), result_file=rec.get("result_file"),
                            error=rec.get("error")))
        # Resume: a second dispatch must report reused=True and not rerun.
        rec2 = jobs.dispatch(spec, require_gate=False)
        # load_existing short-circuits dispatch only if we call it; dispatch itself
        # re-runs unless the caller uses load_existing. Check load_existing here.
        ex = jobs.load_existing(spec)
        rows.append(_record("jobs_spsa_resume",
                            ex is not None and ex.get("reused") is True,
                            reused=None if ex is None else ex.get("reused"),
                            result_file=None if ex is None else ex.get("result_file")))
    return rows


def test_jobs_entropy_and_pack(entry, tols):
    """N=6 task_entropy vs the fixture scan, then 16-start BFGS pack, then resume."""
    from qumode_qsk_gpu import jobs
    if int(entry["N"]) != 6 or entry["kind"] != "entropy_reference":
        return []
    n, seed = int(entry["N"]), int(entry["seed"])
    d = _scratch_tree("scratch_entropy_pack")
    scan = json.loads((ROOT / entry["scan_file"]).read_text()) if entry.get("scan_file") else {}
    rows = []
    with jobs.temporary_paths(runs=d["runs"], inst=d["instances"], ent=d["entropy"],
                              ref=d["references"], logs=d["logs"]):
        espec = dict(task="entropy", N=n, seed=seed, index=entry.get("index"))
        rec = jobs.dispatch(espec, require_gate=False)
        failed = rec.get("verdict") == "FAIL" or rec.get("error")
        # Same CPU oracle and libraries: b*, S, label must match the fixture scan.
        b_tol = float(tols.get("entropy_scan_b_abs", 1e-15))
        s_tol = float(tols.get("entropy_scan_S_abs", 1e-12))
        peak = (scan.get("peak") or scan)
        db = abs(float(rec.get("b")) - float(entry["b"])) if rec.get("b") is not None else np.inf
        dS = abs(float(rec.get("S")) - float(entry["S"])) if rec.get("S") is not None else np.inf
        lab_ok = rec.get("label") == entry.get("label")
        rows.append(_record("jobs_entropy_scratch",
                            (not failed) and db <= b_tol and dS <= s_tol and lab_ok,
                            d_b=db, d_S=dS, label=rec.get("label"),
                            expected_label=entry.get("label"),
                            valid_for_fits=rec.get("valid_for_fits"),
                            error=rec.get("error")))
        # BFGS pack, 16 starts, L=1 nov3 (and the caller may also request qaoa).
        pspec = dict(task="bfgs_pack", N=n, seed=seed, L=1, method="nov3", index=entry.get("index"))
        pack = jobs.dispatch(pspec, require_gate=False)
        n_starts = pack.get("n_starts")
        rows.append(_record("jobs_bfgs_pack_scratch",
                            pack.get("verdict") != "FAIL" and pack.get("error") is None
                            and n_starts == 16,
                            n_starts=n_starts, n_reused=pack.get("n_reused"),
                            error=pack.get("error")))
        ex = jobs.load_existing(pspec)
        rows.append(_record("jobs_bfgs_pack_resume",
                            ex is not None and ex.get("reused") is True,
                            reused=None if ex is None else ex.get("reused")))
        # official_complete on this tiny custom tree is not a full official round;
        # still load_existing for all 16 starts.
        missing = 0
        from qumode_qsk_gpu import protocol as _P
        for k in range(_P.BFGS_STARTS):
            sub = dict(task="bfgs", N=n, seed=seed, L=1, method="nov3", start_index=k)
            if jobs.load_existing(sub) is None:
                missing += 1
        rows.append(_record("jobs_bfgs_starts_resumable", missing == 0, n_missing=missing))
    return rows


def test_jobs_invalid_eligibility(entry):
    """SPSA and BFGS records must be valid_for_fits=False when the instance is."""
    from qumode_qsk_gpu import jobs
    if int(entry["N"]) != 6 or entry["kind"] != "spsa_fixed_params" or entry["method"] != "nov3" or int(entry["L"]) != 1:
        return []
    n, seed = int(entry["N"]), int(entry["seed"])
    d = _scratch_tree("scratch_invalid")
    inst = json.loads((ROOT / f"GPU_test/fixtures/instances/N{n}_seed{seed}.json").read_text())
    inst["valid_for_fits"] = False
    inst["label"] = "no_unique_interior_peak"
    (d["instances"] / f"N{n}_seed{seed}.json").write_text(json.dumps(inst))
    shutil.copy2(ROOT / f"GPU_test/fixtures/references/N{n}_seed{seed}_reference.npz",
                 d["references"] / f"N{n}_seed{seed}_reference.npz")
    rows = []
    with jobs.temporary_paths(runs=d["runs"], inst=d["instances"], ent=d["entropy"],
                              ref=d["references"], logs=d["logs"]):
        sspec = dict(task="spsa", N=n, seed=seed, L=1, method="nov3")
        rec = jobs.dispatch(sspec, require_gate=False)
        rows.append(_record("jobs_spsa_invalid_fits",
                            rec.get("valid_for_fits") is False
                            and rec.get("error") is None,
                            valid_for_fits=rec.get("valid_for_fits"),
                            entropy_label=rec.get("entropy_label"),
                            error=rec.get("error")))
        pspec = dict(task="bfgs_pack", N=n, seed=seed, L=1, method="nov3")
        pack = jobs.dispatch(pspec, require_gate=False)
        rows.append(_record("jobs_bfgs_pack_invalid_fits",
                            pack.get("valid_for_fits") is False
                            and pack.get("error") is None,
                            valid_for_fits=pack.get("valid_for_fits"),
                            entropy_label=pack.get("entropy_label"),
                            n_starts=pack.get("n_starts"),
                            error=pack.get("error")))
        # missing instance
        d2 = _scratch_tree("scratch_missing_inst")
        shutil.copy2(ROOT / f"GPU_test/fixtures/references/N{n}_seed{seed}_reference.npz",
                     d2["references"] / f"N{n}_seed{seed}_reference.npz")
        with jobs.temporary_paths(runs=d2["runs"], inst=d2["instances"], ent=d2["entropy"],
                                  ref=d2["references"], logs=d2["logs"]):
            recm = jobs.dispatch(dict(task="spsa", N=n, seed=seed, L=1, method="nov3"),
                                 require_gate=False)
            rows.append(_record("jobs_spsa_missing_instance_fits",
                                recm.get("valid_for_fits") is not True,
                                valid_for_fits=recm.get("valid_for_fits"),
                                error=recm.get("error"),
                                note="missing instance must not default True; the job may still fail to start"))
    return rows


def test_synthetic_other_N(tols, Ns=(6, 9, 12, 15, 18, 21)):
    """One frozen-start energy/grad vs the CPU oracle at every official N, including odd N and N=21."""
    from qumode_qsk_gpu.model import Model, make_J
    from qumode_qsk_gpu.starts import make_start, spsa_starts
    from qumode_qsk_gpu.gradients import energy_grad
    from qumode_qsk_gpu import oracle, memory, runtime
    ham = oracle.hamiltonian()
    rows = []
    info = runtime.info()
    for n in Ns:
        seed = 831100000 + n * 1000  # official main index 0, no fixtures required
        plan = memory.plan(n, 1, "nov3", info.total_bytes, requested_batch=2)
        rows.append(_record("memory_plan_synthetic", plan["max_batch"] >= 1,
                            N=n, plan=plan))
        J = make_J(n, seed)
        b = 0.5
        mg = Model(n, seed, b, J=J, compute_gs=False)
        mc = ham.Model(n, seed, b, J=J, compute_gs=False)
        for method in ("nov3", "qaoa"):
            x = make_start(seed, 1, method, 0)["x"]
            E, g, v = energy_grad(mg, x, method, 1)
            Ec, gc, vc = ham.energy_grad(mc, x, method, 1)
            from qumode_qsk_gpu import runtime as rt
            psi = rt.to_host(v if v.ndim == 2 else v.reshape(1, -1))[0]
            strict, aligned = _phase_aligned_err(psi, vc)
            gerr = float(np.max(np.abs(np.asarray(g) - np.asarray(gc))))
            dE = abs(float(E) - float(Ec))
            ok = (strict <= _tol(tols, "state_max_abs_vs_cpu_oracle", n)
                  and gerr <= _tol(tols, "grad_max_abs_vs_cpu_oracle", n)
                  and dE <= _tol(tols, "energy_abs_vs_cpu_stored", n))
            rows.append(_record("synthetic_N_start", ok, N=n, method=method, L=1,
                                dE=dE, grad_max_abs=gerr, state_strict=strict,
                                state_phase_aligned=aligned, seed=seed))
            # batch=2 on small/medium N only if memory allows
            if plan["max_batch"] >= 2 and n <= 18:
                xs, _ = spsa_starts(seed, 1, method)
                Eb, gb, _ = energy_grad(mg, np.stack(xs, axis=0), method, 1)
                E0c, g0c, _ = ham.energy_grad(mc, xs[0], method, 1)
                E1c, g1c, _ = ham.energy_grad(mc, xs[1], method, 1)
                d = max(abs(float(np.asarray(Eb)[0]) - float(E0c)),
                        abs(float(np.asarray(Eb)[1]) - float(E1c)))
                ge = max(float(np.max(np.abs(np.asarray(gb)[0] - g0c))),
                         float(np.max(np.abs(np.asarray(gb)[1] - g1c))))
                okb = d <= _tol(tols, "energy_abs_vs_cpu_stored", n) and ge <= _tol(tols, "grad_max_abs_vs_cpu_oracle", n)
                rows.append(_record("synthetic_N_batch2", okb, N=n, method=method, dE=d, grad_max_abs=ge))
    return rows


def run_all(device_id=None, max_bfgs_starts_per_cell=2, run_full_spsa=True):
    from qumode_qsk_gpu import runtime, gate, memory

    tols = load_tolerances()
    catalog = load_catalog()
    info = runtime.acquire(device_id=device_id, activate=True)
    checks = []
    failures = []

    def absorb(recs):
        for rec in recs:
            checks.append(rec)
            if not rec.get("ok", False):
                failures.append(rec)

    absorb(_safe(test_fixture_integrity, "fixture_integrity"))
    absorb(_safe(test_start_recipes, "start_recipes"))

    refs = [e for e in catalog["entries"] if e["kind"] == "entropy_reference"]
    # Drivers are expensive; run them once per (N, method) at L=1 for the first seed of each N.
    spsa_driver_done = set()
    jobs_done = set()
    bfgs_cap_done = set()
    va_done = set()
    batch_done = set()

    for ref in refs:
        n, seed = int(ref["N"]), int(ref["seed"])
        try:
            m, z = _models(n, seed)
            cpu_m = _cpu_model(n, seed, z)
        except Exception as ex:
            absorb([_record("build_models", False, N=n, seed=seed, error=repr(ex),
                            traceback=traceback.format_exc())])
            continue
        absorb(_safe(test_stored_gs_entropy_residual, "stored_gs", m, z, ref, tols))
        if n not in va_done:
            absorb(_safe(test_va_not_clobbered, "preserve_src", m, ref))
            va_done.add(n)

        related = [e for e in catalog["entries"]
                   if e["kind"] in ("spsa_fixed_params", "bfgs_fixed_params")
                   and e["N"] == n and e["seed"] == seed]
        for e in related:
            if e["kind"] == "bfgs_fixed_params" and int(e["start_index"]) >= int(max_bfgs_starts_per_cell):
                # lite final-x check (no FD)
                try:
                    recs = test_fixed_params(m, cpu_m, dict(e, kind="bfgs_fixed_params"), tols, with_fd=False)
                    # drop start_x if we only wanted final? keep both but skip extra starts' start+fd
                    recs = [r for r in recs if r.get("tag") == "bfgs_final_x"]
                    for rec in recs:
                        rec.update(N=n, seed=seed, L=e["L"], method=e["method"], start_index=e["start_index"])
                    absorb(recs)
                except Exception as ex:
                    absorb([_record("bfgs_final_x_lite", False, error=repr(ex), traceback=traceback.format_exc(),
                                    N=n, seed=seed, L=e.get("L"))])
                continue
            absorb(_safe(test_fixed_params, e.get("kind"), m, cpu_m, e, tols))
            keyb = (n, e["method"], e.get("L"))
            if e["kind"] == "spsa_fixed_params" and keyb not in batch_done:
                absorb(_safe(test_batch_ge2, "batch2", m, cpu_m, e, tols))
                batch_done.add(keyb)
            if run_full_spsa and e["kind"] == "spsa_fixed_params" and e["L"] == 1 and (n, e["method"]) not in spsa_driver_done:
                absorb(_safe(test_spsa_driver, "spsa_driver", m, cpu_m, e, tols))
                spsa_driver_done.add((n, e["method"]))
                if (n, e["method"]) not in jobs_done and n == 6:
                    absorb(_safe(test_jobs_spsa_scratch, "jobs_spsa", e))
                    jobs_done.add((n, e["method"]))
            if (e["kind"] == "bfgs_fixed_params" and int(e.get("start_index", -1)) == 0
                    and int(e["L"]) in (1, 3) and (n, e["method"], int(e["L"])) not in bfgs_cap_done):
                absorb(_safe(test_bfgs_capped, "bfgs_capped", m, cpu_m, e, tols))
                bfgs_cap_done.add((n, e["method"], int(e["L"])))

    # Production-path coverage on N=6 scratch (entropy, 16-start pack, resume, eligibility).
    n6_ent = next((e for e in refs if int(e["N"]) == 6), None)
    if n6_ent is not None:
        absorb(_safe(test_jobs_entropy_and_pack, "jobs_entropy_pack", n6_ent, tols))
    n6_spsa = next((e for e in catalog["entries"]
                    if e["kind"] == "spsa_fixed_params" and int(e["N"]) == 6
                    and e["method"] == "nov3" and int(e["L"]) == 1), None)
    if n6_spsa is not None:
        absorb(_safe(test_jobs_invalid_eligibility, "jobs_invalid_fits", n6_spsa))

    absorb(_safe(test_synthetic_other_N, "synthetic_N", tols))

    # Record the kernel path that actually ran (stamp binds it).
    try:
        from qumode_qsk_gpu import kernels
        kinfo = kernels.kernel_path_info()
    except Exception as ex:
        kinfo = dict(kernel_path=None, raw_kernels_error=repr(ex))

    verdict = "PASS" if not failures else "FAIL"
    report = dict(
        verdict=verdict,
        shanghai=now_shanghai(),
        device=dict(name=info.device_name, id=info.device_id, activated=getattr(info, "activated", None),
                    cupy=info.cupy_version, cuda=info.cuda_version,
                    cc=info.compute_capability, total_bytes=info.total_bytes,
                    numpy=getattr(info, "numpy_version", None),
                    scipy=getattr(info, "scipy_version", None)),
        dtype=dict(real=str(REAL), cplx=str(CPLX)),
        kernel_path=kinfo,
        n_checks=len(checks), n_failures=len(failures),
        failures=failures,
        checks=checks,
        tolerances=tols,
        cuda_validated=True,
        note=(
            "Executed on a CUDA device. A PASS is necessary for production but "
            "does not prove bitwise identity with the CPU campaign, nor any "
            "speedup. AST/static checks cannot prove CUDA correctness."
        ),
    )
    if verdict == "PASS":
        brief = {k: report[k] for k in report if k != "checks"}
        gate.write_pass_stamp(brief)
        report["stamp"] = str(gate.stamp_path())
    return report


if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=None)
    ap.add_argument("--max-bfgs-starts-per-cell", type=int, default=2)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()
    rep = run_all(device_id=args.device_id, max_bfgs_starts_per_cell=args.max_bfgs_starts_per_cell)
    text = _json.dumps({k: v for k, v in rep.items() if k != "checks"}, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(_json.dumps(rep, indent=2) + "\n")
    raise SystemExit(0 if rep["verdict"] == "PASS" else 1)
