"""Official 10-per-N round summaries for the GPU campaign tree."""
from __future__ import annotations

import math
from collections import defaultdict

from . import protocol as P
from . import jobs, runtime
from .bfgs import select_by_energy
from .io_util import save_json, read_json, now_shanghai
from .paths import SUMMARIES, ROOT


def _mean_sem(xs):
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    n = len(xs)
    if n == 0:
        return dict(n=0, mean=None, sem=None)
    mu = sum(xs) / n
    if n == 1:
        return dict(n=1, mean=mu, sem=None)
    var = sum((x - mu) ** 2 for x in xs) / (n - 1)
    return dict(n=n, mean=mu, sem=math.sqrt(var / n))


def _load_raw(path):
    try:
        return read_json(path)
    except Exception:
        return None


def summarize_round(round_id, out_path=None):
    """Official per-round summary: indices of this round only.

    Only provenance-checked records (jobs.load_existing) enter official
    means. Means are reported for all verified records and for
    valid_for_fits=True records separately. Incomplete BFGS cells are listed,
    not silently dropped. Unverified on-disk files are counted but excluded.
    Requires runtime.acquire() so resume provenance is well-defined.
    """
    if not runtime.is_acquired():
        raise RuntimeError(
            "summarize_round requires runtime.acquire() so records can be "
            "provenance-checked (unverified files are otherwise excluded)"
        )
    plan = jobs.plan_round(round_id)
    idxs = set(plan["indices"])
    entropy = defaultdict(list)
    unresolved = defaultdict(list)
    unverified = []
    e0 = {}
    for n in P.NS:
        for i in sorted(idxs):
            seed = P.main_seed(n, i)
            spec = dict(task="entropy", N=n, seed=seed, index=i, round=round_id)
            rec = jobs.load_existing(spec)
            if rec is None:
                raw = _load_raw(jobs.result_path(spec))
                if raw is not None:
                    unverified.append(dict(task="entropy", N=n, seed=seed,
                                           reason="missing_or_mismatched_provenance"))
                continue
            rec["index"] = i
            entropy[n].append(rec)
            e0[(n, seed)] = rec.get("E0")
            if not rec.get("valid_for_fits", False) or rec.get("label") != "interior_unique_GS_entropy_peak":
                unresolved[n].append(dict(index=i, seed=seed, label=rec.get("label"),
                                          valid_for_fits=rec.get("valid_for_fits")))

    def _panel(rows, ekey, fkey, e0_map=None):
        all_rows = rows
        valid_rows = [r for r in rows if r.get("valid_for_fits")]
        def _one(subset):
            ev = [r.get(ekey) for r in subset]
            fv = [r.get(fkey) for r in subset]
            out = dict(n=len(subset), energy=_mean_sem(ev), fidelity=_mean_sem(fv))
            if e0_map is not None:
                dE = []
                for r in subset:
                    e0v = e0_map.get((r.get("N"), r.get("seed")))
                    evv = r.get(ekey)
                    if e0v is not None and evv is not None:
                        dE.append(float(evv) - float(e0v))
                out["energy_minus_E0"] = _mean_sem(dE)
            return out
        return dict(all=_one(all_rows), valid_only=_one(valid_rows),
                    n=len(all_rows), n_valid_for_fits=len(valid_rows))

    panels = {}
    incomplete_bfgs = []
    for n in P.NS:
        panels[n] = {}
        for L in P.DEPTHS:
            panels[n][L] = {}
            for method in P.METHODS:
                spsa, bfgs_rows = [], []
                for i in sorted(idxs):
                    seed = P.main_seed(n, i)
                    sp = jobs.load_existing(dict(task="spsa", N=n, seed=seed, index=i,
                                                 L=L, method=method, round=round_id))
                    if sp is not None:
                        spsa.append(sp)
                    else:
                        if _load_raw(jobs.result_path(dict(task="spsa", N=n, seed=seed, L=L, method=method))) is not None:
                            unverified.append(dict(task="spsa", N=n, seed=seed, L=L, method=method))
                    starts, missing_starts = [], []
                    for k in range(P.BFGS_STARTS):
                        sub = dict(task="bfgs", N=n, seed=seed, L=L, method=method,
                                   start_index=k, index=i)
                        bf = jobs.load_existing(sub)
                        if bf is not None:
                            starts.append(bf)
                        else:
                            missing_starts.append(k)
                    if missing_starts:
                        incomplete_bfgs.append(dict(N=n, seed=seed, L=L, method=method, index=i,
                                                    n_starts=len(starts), missing_start_indices=missing_starts[:8],
                                                    n_missing=len(missing_starts)))
                    if len(starts) == P.BFGS_STARTS:
                        sel = select_by_energy(starts)
                        bfgs_rows.append(dict(N=n, seed=seed, index=i, **sel,
                                              valid_for_fits=starts[0].get("valid_for_fits")))
                e0_map = e0
                panels[n][L][method] = dict(
                    spsa=_panel(spsa, "best_E", "F", e0_map),
                    bfgs=_panel([{**r, "best_E": r.get("selected_E"), "F": r.get("selected_F")}
                                 for r in bfgs_rows], "best_E", "F", e0_map),
                    selection="lowest_energy",
                )

    status = jobs.official_complete(round_id)
    doc = dict(
        round=int(round_id),
        indices=plan["indices"],
        shanghai=now_shanghai(),
        cohort=jobs.COHORT,
        official_complete=status["complete"],
        n_missing=status["n_missing"],
        n_unverified_on_disk=len(unverified),
        unverified_on_disk=unverified[:50],
        n_incomplete_bfgs_cells=len(incomplete_bfgs),
        incomplete_bfgs=incomplete_bfgs[:50],
        entropy={str(n): dict(n=len(entropy[n]),
                              n_valid=sum(1 for r in entropy[n] if r.get("valid_for_fits")),
                              unresolved=unresolved[n],
                              S_all=_mean_sem([r.get("S") for r in entropy[n]]),
                              S_valid_only=_mean_sem([r.get("S") for r in entropy[n] if r.get("valid_for_fits")]),
                              b_all=_mean_sem([r.get("b") for r in entropy[n]]),
                              b_valid_only=_mean_sem([r.get("b") for r in entropy[n] if r.get("valid_for_fits")]))
                 for n in P.NS},
        panels={str(n): panels[n] for n in P.NS},
        note=(
            "Official 10-per-N summary. Only provenance-checked records are included. "
            "Means are reported for all verified records and for valid_for_fits=True "
            "separately. Selection is by energy, not F. Invalid peaks are flagged, not replaced. "
            "energy_minus_E0 is a per-instance offset (raw E means across different J have little meaning)."
        ),
    )
    out_path = out_path or (SUMMARIES / f"round_{int(round_id):02d}.json")
    save_json(out_path, doc)
    doc["summary_file"] = str(out_path)
    return doc


def write_round_manifest(round_id, binding, path=None, extra=None):
    plan = jobs.plan_round(round_id)
    doc = dict(
        round=int(round_id),
        shanghai=now_shanghai(),
        indices=plan["indices"],
        n_entropy=len(plan["entropy"]),
        n_spsa=len(plan["spsa"]),
        n_bfgs_pack=len(plan["bfgs_pack"]),
        binding_sha256=(binding or {}).get("binding_sha256"),
        device_name=(binding or {}).get("device_name"),
        tasks_brief=dict(
            entropy=[(t["N"], t["seed"]) for t in plan["entropy"]],
        ),
        status="planned",
    )
    if extra:
        doc.update(extra)
    path = path or (ROOT / "runs" / f"round_{int(round_id):02d}_manifest.json")
    save_json(path, doc)
    return path, doc
