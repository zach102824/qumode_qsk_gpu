# ROUND 2 fix log

Source report: `reviews/ROUND2.md` (verbatim copy of the Opus 5.5 medium
review asset `01M74BNS5Y1J2Z1S25F6JMB9CY.md`, 35088 bytes, SHA-256
`fb3c2ff23352b07e5a3f2215554257b42a9a43d7ed03c123462d043cbda2a505`).

The original verdict was **CHANGES NEEDED** (4 High, no Critical). That
verdict is not being restated as a PASS. This tree was **edited after
review 2**; there is no third review. CUDA was not executed here. No new
numerical experiments were run in the authoring environment. The live CPU
campaign was not written. Reviewers were not launched.

`protocol/PROTOCOL_FROZEN.json` was not edited (file SHA-256
`8253fae8dcfc4f8242e980f446d9e8073364580e9fe9901b17060a7b69ffba96`).
Fixture files were not rewritten. `cpu_oracle/src` was not edited.
Line numbers below refer to the tree after these edits.

## High

### R2-H1 — Strided β passed to the RX RawKernel

**Issue.** `xd[:, 2*ell+1]` is a non-contiguous view. RawKernels ignore
strides and read `beta[b]` as a flat pointer, so the QAOA SPSA minus probe
(and batched row 1) used the wrong angle.

**Fix.**
- `kernels.py` `prepare_beta` (line 160) and `_beta_device` (line 182):
  `xp.ascontiguousarray(...reshape(-1))`.
- `_require_contiguous` / `_require_states` (lines 61–) assert
  C-contiguity on every RawKernel buffer (states, β, apply_X out).
- Host regression: `GPU_test/tests/test_host_contracts.py`
  `test_strided_beta_contiguous` (line 24) checks that a strided
  `xd[:, 1]` becomes `[β0, β1]` and not the next-layer γ.

### R2-H2 — Pipeline livelock when the entropy thread stops early

**Issue.** On `--halt-on-fail`, or on any exception escaping dispatch
(including a mid-round gate mismatch), the entropy thread died while the
main loop required `ent_i >= len(order)` and spun forever. `_finish` was
skipped.

**Fix.**
- New `pipeline.py` `run_pipeline` (line 47): liveness-based exit, `stop`
  flag, `try/except` around each runner, join in-flight work (including an
  N=21 scan) without a short timeout in production, no 100% busy-spin
  (`wait`/`sleep` when idle).
- `jobs.dispatch` (line 470): `assert_production_allowed()` is inside the
  `try`, so a gate exception is a recorded FAIL.
- `run_production.py` always calls `_finish` (manifest/summary) if the
  pipeline aborts.
- Host regression: `test_pipeline_halt_and_continue` (line 49) injects an
  entropy FAIL under both halt modes with a 2 s wall-clock bound. The exit
  condition is `(not in_flight) and (queue drained or stop)` so a halt
  cannot spin on unsubmitted later scans (this was regression-tested here
  on the host; a first draft of the loop still hung and was corrected).

### R2-H3 — Round-ordering guard unsatisfiable before `acquire`

**Issue.** `official_complete(r-1)` ran before `runtime.acquire()`.
`provenance_fields()` omitted `binding_sha256`, so every prior result
looked non-resumable and rounds 2–20 could not start without
`--force-unordered`.

**Fix.**
- `run_production.py` lines 79–89: `acquire(activate=True)` and
  `assert_production_allowed()` **before** `official_complete`. Dry-run
  still returns before any CUDA acquire (line 74).
- `gate.provenance_fields()` (line 371) **raises** if not acquired.

### R2-H4 — Capped BFGS check could not fail on numerics

**Issue.** `ok` was only `start_x_sha_match`. Energy/grad agreement was
report-only. Only `start_index==0`, `L==1`.

**Fix.** `test_gpu_validation.py` `_bfgs_capped_record` / `test_bfgs_capped`
(line 410) PASS criteria, documented in `GPU_test/tolerances.json`:
- finite E, g, x
- `dE ≤ 1e-9`, `|grad_l2_gpu−grad_l2_cpu| ≤ 1e-7`, `max|x| ≤ 1e-7`
- `nit`/`nfev` equality, otherwise fail as `"line-search divergence"`
- objective does not increase unless the CPU oracle also increased
- `maxiter=3` and a `maxiter=1` one-optimizer-step check
- runs for L=1 and L=3, start 0, both ansatz (when present in the catalog)

These still have to be executed on a GPU. This revision does not claim they passed.

## Medium

### R2-M1 — Walsh multiply-by-reciprocal vs CPU divide by √2

`kernels_cuda.py` WALSH now uses `const double s = sqrt(2.0)` and
`(a±d)/s`, matching Numba `(a+b)/s`. `--fmad=false` remains. Docs and
LIMITATIONS **qualify** that this is not a bitwise-identity claim (libm,
CuPy ufuncs/reductions, and other ULP sources remain). The Python
fallback already divided by `SQRT2`.

### R2-M2 — Lazy RawKernel compile; stamp must bind the path

`kernels._ensure_raw` calls `k.compile()` inside the try (line 128
`ensure_compiled`). The stamp/binding records `kernel_path` (`raw` or
`python`) and `raw_kernel_options`. `assert_production_allowed` rejects a
mixed path.

### R2-M3 — Entropy thread must not use CUDA

`jobs.cpu_entropy_scan_only` (line 286) is host-only (no Model, no
acquire). `finalize_entropy_gpu` (line 318) runs apply_H and attaches
GPU provenance on the **main** thread. The spawn worker
(`entropy_worker.py`) sets BLAS/Numba thread counts before importing
NumPy and never uses CUDA.

### R2-M4 — `--entropy-workers` is now truthful

`run_production.py --entropy-workers N` (line 44): `N>1` is a spawn
`ProcessPoolExecutor` (BLAS threads 1, at most one N≥21 scan). `N=1`
(default) is a host thread calling the same host-only function, so it can
overlap GPU finalize/optimize if SciPy releases the GIL. Help text
matches the implementation. Dead `--continue-on-fail` was removed (R2-L3);
default remains continue, `--halt-on-fail` is the opt-in.

### R2-M5 — Small-N first

Entropy is ordered by increasing N, then seed
(`run_production.py` around the `entropy_order` sort). GPU work can start
after the first N=6 scan instead of waiting for N=21.

### R2-M6 — Cache the binding once per process

`gate.current_binding` (line 258) caches after acquire.
`verify_cached_binding` (line 266) re-hashes at stamp write, round start,
and round finish. `_provenance_ok` / resume compare the cached SHA.

### R2-M7 — Enforce campaign library pins

`check_library_pins` (line 178) refuses unless Python / NumPy / SciPy /
Numba equal
`PROVENANCE_ADDENDUM.campaign_library_versions_on_authoring_box`
(3.13.5 / 2.5.4 / 1.18.1 / 0.68.0). Binding and results record those
versions plus CuPy, CUDA runtime, device name, and kernel path.
`--allow-version-drift` on GPU_test and production marks the stamp and
results unofficial; production refuses an unofficial stamp unless the
flag is passed again.

### R2-M8 — Summaries exclude invalid / unverified records

`summaries.py` uses `jobs.load_existing` (provenance-checked), reports
`all` and `valid_only` means, lists incomplete BFGS cells and unverified
on-disk files, and adds `energy_minus_E0`. Requires acquire.

### R2-M9 / R2-M10 / R2-M11 — Test coverage

- Host: missing instance, `valid_for_fits=false`, key absent
  (`test_host_contracts.py`); paths restore via `temporary_paths`.
- GPU_test (user-run, not executed here):
  - `test_jobs_invalid_eligibility`: false label and missing instance
    must not yield `valid_for_fits=True` on SPSA and BFGS-pack records.
  - `test_jobs_entropy_and_pack`: N=6 `task_entropy` vs fixture b*/S/label,
    16-start `task_bfgs_pack` at L=1, `load_existing` resume.
  - `test_spsa_driver` (line 272): CPU oracle `spsa.optimize` at the same
    cell; first K query-log energies compared, plus/minus pairs checked
    for a swap, not only direction hashes. Full-trajectory `best_E` remains
    report-only.
  - Pipeline halt/continue covered on the host (R2-H2).

## Performance (feasible without CUDA; no new unreviewed kernels)

- In-place RawKernels and force compile remain; Walsh op-order parity (M1).
- Host entropy overlapping GPU finalize/opt; process pool for workers>1;
  small-N first.
- Binding cache; host diag/shell cache per (N, seed) in `load_model`.
- `runtime.free_memory_pool()` when N changes on the optimizer thread.
- **Not done:** shared-memory tiled Walsh/RX (new high-risk kernel, no
  device test in this environment). Correctness above that optimization.

## Low items that were cheap and safe

- R2-L1: `load_model` checks `gs` SHA against the instance record.
- R2-L2: host diag/shell cache (see above).
- R2-L3: removed dead `--continue-on-fail`.
- R2-L5: removed the dead `_safe(lambda: _models)` in `run_all`.
- R2-L6: host start-recipe test asserts `n_start == 324`.
- R2-L7: `jobs.temporary_paths` saves and restores `paths.LOGS`.
- R2-L8: removed the no-op `download` argument of `entropy_device`.

## Left open (not blockers for a careful GPU_test, still unresolved)

- **R2-L9:** stock CuPy ufunc FMA on diagonal phases / reductions (documented).
- **R2-L10:** resume uses the full gate binding (docs typo or a different
  GPU id invalidates long results). Not split in this revision.
- **CUDA unexecuted.** None of the GPU numerical tests, the process-pool
  entropy path, or production on hardware have been run here. AST cannot
  prove those.
- **Pin installability** of numpy 2.5.4 / scipy 1.18.1 / numba 0.68.0 with
  the GPU host's Python and CuPy wheel was not verified here.
- Shared-memory tiling (review §6 optional) was deliberately not added.

## Portable paths

Runtime/import code no longer hard-codes `/workspace/qumode_qsk`. Writers
refuse a sibling directory named `qumode_qsk` (if it exists and is not
this tree) and `$QUMODE_QSK_CPU_CAMPAIGN`. Provenance documents (the frozen
protocol, the addendum, cpu_oracle snapshot notes) still record the
original campaign path, as required.
