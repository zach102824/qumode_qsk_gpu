# ROUND 1 fix log

Source report: `reviews/ROUND1.md` (verbatim copy of the Opus 5.5 medium
review asset `01M74AC9Z1X30CRXY4S1S8K9NA.md`, 34606 bytes, SHA-256
`8f866a5cac37d2c87ca44c82ee4705ef9e69a89b68c1dcb58e50296765f0944e`).
The frozen protocol file was not edited. No CUDA was executed in this
environment. No new numerical experiments were run. The live CPU campaign
at `/workspace/qumode_qsk` was not written. Reviewers were not launched
from this revision.

Line numbers below refer to the revised tree after these edits.

## Critical

### C1 — `lam = lam * conj(...)` left an untracked ping-pong buffer

**Issue.** Adjoint phase was `lam = lam * xp.conj(...)`, which allocates a
new array and can leave the previous `lam` as the forgotten half of a
ping-pong pair (and can crash or silently reuse stale storage).

**Fix.** `src/qumode_qsk_gpu/gradients.py`:
- `_multiply_inplace` at lines 58–66 writes `dst *= factor` (or copies into
  `dst` if the runtime cannot multiply in place).
- Nov3 backward: `lam = Hv.copy()` at line 104, then
  `_multiply_inplace(lam, xp.conj(sph[...]))` / `_multiply_inplace(lam, xp.conj(phase))`
  so `lam` stays in one allocation.
- QAOA backward: same pattern at lines 159–176
  (`_multiply_inplace(lam, xp.conj(phase))` before `rx_inplace(lam, -beta, n)`).

### C2 — ping-pong clobbered `vA` used for shared Nov3 and QAOA γ grads

**Issue.** After `vA = vin * phase`, Walsh/RX on a source that aliases `vA`
destroys the pre-transform vector needed for shared-parameter / γ grads,
silently wrong.

**Fix.**
- `src/qumode_qsk_gpu/kernels.py` lines 13–14, 116–135, 179–192, 198–215,
  272–279: default in-place RawKernels (source is the dest; each thread
  owns a pair). Python ping-pong fallback **documents and enforces** that
  `src` is clobbered unless `preserve_src=True` (copy into a distinct `dst`
  first).
- `gradients.py` lines 18–19, 112–117 (Nov3) and 170–176 (QAOA): `vA` is a
  dedicated buffer and is never passed as a Walsh/RX source. A `fwd`
  buffer is filled with `fwd[...] = vA` and transformed instead. `vA` is
  then used in `imag(conj(lam)*vA)` at lines 123 and 178.
- `GPU_test/tests/test_gpu_validation.py` `test_va_not_clobbered` at
  lines 256–269 checks `preserve_src` leaves the source intact.

## High

### H1 — batched RX reshape `(B, 1, 1, 1)` vs 3-D slices

**Issue.** Coefficients were reshaped one dimension too tall and did not
broadcast against `(batch, pairs, 2)` slices, so batched RX (SPSA ±, any
B≥2) was wrong.

**Fix.** `kernels.py` lines 236–237:
`c` and `s` are `.reshape(batch, 1, 1)`.
`test_gpu_validation.py` `test_batch_ge2` (lines 230–249) and the official-N
loop (lines 366–408) exercise B=2.

### H2 — `np.asarray(CuPy)` in SPSA (and energy) can crash / copy badly

**Issue.** `np.asarray` on a CuPy ndarray does not produce a host float64
the way the ledger needs, and can raise.

**Fix.**
- `runtime.py` `as_host_array` lines 230– (uses `is_device_array` then
  `.get()`), `pack_to_host` line 289, `as_scalar` line 286 only after a
  host conversion.
- `spsa.py` line 102–104: `E = np.asarray(runtime.as_host_array(E, dtype=REAL), dtype=REAL).reshape(2)`.
- `model.py` `energy` / `energy_of_params` / `_host_energy` lines 217–236:
  always return host float64 (Python float if batch==1).
- `gradients.py` `_as_batch_x` line 47 and the public wrappers around
  line 206: `runtime.as_host_array`. Final `(E, g)` come from one
  `pack_to_host` (lines 129, 182).

### H3 / H4 — wrong oracle imports (`from hamiltonian`, `cpu_oracle/src` on `sys.path`)

**Issue.** Bare `from hamiltonian import ...` and putting `cpu_oracle/src`
on `sys.path` break the oracle's relative imports and can shadow the
production package. Tests and production entropy were at risk.

**Fix.**
- `oracle.py` lines 1–52: `ensure_root_on_path()` strips `cpu_oracle/src`
  and inserts ROOT only; every loader is
  `from cpu_oracle.src import hamiltonian|entropy|spsa|bfgs|starts`.
- `scripts/run_production.py` lines 26–28, `scripts/run_gpu_test.py` lines
  20–22, `scripts/benchmark.py` lines 20–21, `scripts/static_check.py`
  lines 80 and 95: ROOT and/or `src/` only.
- `GPU_test/tests/conftest_paths.py` lines 1–19: same strip-and-insert.
- `entropy.py` no longer has a dead `cpu_oracle_import` path.

### H5 — NumPy 2 `.device` / `hasattr(..., "device")` is a false type check

**Issue.** NumPy ≥2 ndarrays have a `.device` attribute, so
`hasattr(x, "device")` treats host arrays as device arrays (and can skip
an upload or take the CuPy branch on NumPy).

**Fix.** `runtime.py` `is_device_array` lines 223–224:
`isinstance(x, cupy.ndarray)` only, never `hasattr`. `model.py` uploads
via `as_host_array` / `to_device` (line 145 and the constructor path).
The only remaining mentions of `hasattr(..., "device")` are comments
warning against it (`runtime.py:224`, `model.py:166`).

### H6 — `Device(k)` used only as a query context; allocations landed on another device

**Issue.** Opening `Device(k)` as a context for property queries does not
make `k` the current device, so later `xp.empty` / `to_device` can land
on device 0 (or whatever was current).

**Fix.** `runtime.py`:
- `acquire(..., activate=False)` default at line 84.
- `activate_acquired_device()` lines 128–135 calls
  `cp.cuda.Device(int(_STATE["device_id"])).use()`.
- Production (`run_production.py:77`), GPU_test
  (`test_gpu_validation.py:418`), and the benchmark (`benchmark.py:31,82`)
  all call `acquire(..., activate=True)`.
- Import still does not touch CUDA.

### H7 — test harness did not run SPSA / BFGS / jobs, B≥2, other N, or the direction-hash claim

**Issue.** The first GPU_test suite was mostly fixed-parameter E/F. It did
not actually execute `spsa.optimize`, SciPy BFGS, `jobs.task_spsa`,
batch≥2, odd/other official N, or compare direction hashes.

**Fix.** `GPU_test/tests/test_gpu_validation.py` and
`scripts/run_gpu_test.py`:
- Fixture SHA / J / GS (lines 97–110).
- Start recipes for all 324 catalog starts (lines 113–130).
- Stored GS residual and energy (lines 133–143); see M6.
- Fixed-param E/F, norm, oracle state/grad, FD (lines 174–227).
- Batch≥2 (lines 230–249).
- `preserve_src` / vA (lines 256–269).
- Full `spsa.optimize`: `direction_hashes`, `initial_points`,
  `query_roles`, `calls` are PASS; `best_E` / `x` are report-only
  (lines 283–297).
- Capped BFGS `maxiter=3` vs the CPU oracle (line 304+).
- `jobs.task_spsa` on a scratch tree, including `valid_for_fits` from
  the instance (lines 354–359).
- Structural tiny tests at N∈{6,9,12,15,18,21} (odd N included),
  code-only, no stored-campaign recompute (lines 366–408).
- `run_gpu_test.py` runs host oracle light first, then `run_all`; any
  exception is written as `verdict=FAIL`. No fabricated PASS.

These still have to be executed on a GPU host. This revision cannot and
does not claim they passed.

## Medium

### Stamp must hash actual fixture files, tolerances, and library versions

**Fix.** `gate.py`:
- `fixture_file_pins` / `verify_fixture_manifest` lines 69–109 hash every
  file under `GPU_test/fixtures` plus `tolerances.json`.
- `current_binding` lines 121–148 records `fixture_files_sha256`,
  `tolerances_sha256`, `numpy_version`, `scipy_version`, `cupy_version`,
  `cuda_version`, `compute_capability`.
- `assert_production_allowed` / `_COMPARE_KEYS` lines 169–211 compare all
  of those.
- `_iter_source_files` lines 25–48 includes the package, cpu_oracle,
  protocol (including `PROVENANCE_ADDENDUM.json`), scripts, tests, docs,
  README, requirements, tolerances. **`reviews/` is excluded** so
  appending review notes does not invalidate a PASS stamp.

### `valid_for_fits` must propagate entropy labels (no `True` fallback)

**Fix.** `jobs.py` `instance_meta` lines 121–136: if the instance file is
missing or the key is absent, `valid_for_fits=False`. SPSA / BFGS / pack
writers copy `bool(meta["valid_for_fits"])` and `entropy_label` (lines
263–264, 304–305, 327–328). Summaries (`summaries.py` lines 47–77) do not
invent True.

### Result resume must pin source / libraries / N,seed,L,method

**Fix.** `jobs.py` `load_existing` / `_provenance_ok` lines 82–121:
require `binding_sha256`, `source_pins_sha256`, numpy/scipy/cupy, dtypes,
plus N/seed and L/method when present, and `calls == 1024 L` for SPSA.
`gate.provenance_fields()` (lines 216–235) is stored on every result.

### Atomic reference NPZ

**Fix.** `io_util.py` `save_npz_atomic` lines 56–71 (tmp, fsync, replace,
best-effort directory fsync). `jobs.py` line 188 uses it for reference
NPZ. JSON writers already used sibling-tmp replace (`save_json` lines 5–20).

### Round manifest and summaries

**Fix.** `summaries.py` `summarize_round` (line 33) and
`write_round_manifest` (line 110). `run_production.py` writes both at the
end of a run (`_finish`).

### Pin NumPy / SciPy (and Numba) to the campaign-box actual versions

**Fix.** `requirements.txt` pins `numpy==2.5.4`, `scipy==1.18.1`,
`numba==0.68.0`, documented as the versions imported on the same machine
that was running the CPU campaign (Python 3.13.5). The addendum repeats
this. README install instructions use `-r requirements.txt` instead of
loose lower bounds. The PASS stamp binds the live numpy/scipy versions.

### M6 — GS energy tolerance was OR-ed with residual, which is the wrong predicate

**Fix.** `test_gpu_validation.py` lines 137–143: `E0_residual` is one
check (`res < tols["E0_residual"]`); `E_gs_vs_E0` is a separate check
(`dE <= energy_abs_vs_cpu_stored`). They are not OR-ed.

## Provenance addendum (frozen protocol file never edited)

New file: `protocol/PROVENANCE_ADDENDUM.json`. File SHA of
`protocol/PROTOCOL_FROZEN.json` is unchanged:
`8253fae8dcfc4f8242e980f446d9e8073364580e9fe9901b17060a7b69ffba96`
(116020 bytes). Embedded `protocol_sha256` remains
`f32a96d4af337525edbbeea903b848d9b8c6163af3efa214f4ca71cd143cbb81`.
The addendum distinguishes those two hashes, records both SHAs of the
write-guarded `starts.py` copy (campaign `0a39d54a…` / 2864 B vs bundled
`03c1d355…` / 3095 B), documents that protocol `source_pins` predate the
live audited `hamiltonian.py` / `bfgs.py` / `entropy.py` (and never
pinned `kernels_numba.py`), and labels the fixture producer as not
proven. Fixture history was not silently repaired.

## Efficiency (feasible without a GPU, same calculation)

- Pair-kernel fusion when safely reviewable: default Walsh/RX are
  in-place RawKernels, one launch per qubit, `--fmad=false`
  (`kernels.py` lines 31, 44, 159, 250). `apply_X` is a fused kernel
  (ascending q). Not a science change.
- CPU entropy / GPU overlap, memory-safe:
  `run_production.py` `_pipeline` lines 123+ runs one entropy thread
  (N≥21 never concurrent with another scan) while the main thread runs
  GPU optimizer tasks as soon as that instance's reference exists.
  Unexecuted here.
- Avoid gradient temps / sync: in-place λ, dedicated vA + one fwd copy,
  cached `sph`, shared/shell grads via a contracted `Im(conj(λ)v)`
  reduction, **one** `pack_to_host` of (norm, E, g) per eval
  (`gradients.py` lines 123–129, 178–182). Benchmarks use
  `runtime.elapsed_time_ms` (`runtime.py:251`, `benchmark.py:50–56`) and
  are labelled end-to-end (they still include the host syncs inside
  `energy_grad`).
- Concurrency does not change cold-start / serial algorithm: SciPy BFGS
  remains host, one start at a time. SPSA ledger, host RNG, and float64
  math are unchanged; only the ± pair is batched (batch 2).
- No silent routing of all small N to CPU. An opt-in flag is not the
  default and was not added as a hidden policy. Small-N GPU benchmarks
  may be used as correctness-only runs; no speedup is printed
  (`benchmark.py` `speedup_vs_cpu="not_claimed"`, fail-closed unless
  `--allow-unstamped`).
- `memory.grad_vector_count` line 48; `jobs.py` lines 152–153 refuse a
  cell whose required batch (1 BFGS / 2 SPSA) does not fit.
- `zz_diag` is a host pair-loop matching Numba (the einsum form was 1 ULP
  off on fixture J; this was already corrected before the review and is
  kept). `shell_ops` loops groups (no `(dim, n/3, 3)` spike at N=21).

## Explicitly out of scope / not claimed

- CUDA correctness is **not** proven by this fix log, by AST, or by
  `scripts/static_check.py`.
- No PASS stamp was written here. No speedup was measured or advertised.
- No new numerical experiments, no CUDA install, no campaign writes.
- A second review round is for the parent to launch; this revision does
  not launch reviewers.
