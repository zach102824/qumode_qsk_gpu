# qumode_qsk_gpu — Review Round 1 (efficiency / correctness)

Reviewer: Opus 5.5, medium effort, static inspection only.
Input: `review_bundle.tar_812b.gz`, extracted to a scratch directory outside any repo (391 files).

## Verdict: CHANGES NEEDED

The port cannot produce a GPU_test PASS or run production as shipped. There are
**two critical gradient defects** (one makes the Nov3 gradient crash; the other,
hidden behind it, silently corrupts the shared-parameter and QAOA γ gradients)
and **six high-severity blockers**: the batched QAOA RX broadcast, the SPSA
batched energy download, two oracle-import failures (tests and production
entropy), NumPy ≥ 2 handling of host arrays, and device selection. There is
also a test-coverage gap: even after those fixes, a PASS would not cover the
SPSA, BFGS or production code paths.

The physics formulas, conventions and protocol constants match the CPU oracle
line for line where they are implemented. The defects are in buffer
management, array plumbing, imports and the test/gate harness. Fixing them
does not change the science. Every fix below keeps the CPU math.

### Scope and limits of this review

- **Static inspection only.** No CUDA was installed or executed. No GPU or CPU
  numerical experiment was run, nothing was recomputed, no job or production
  run was started, and nothing in the live CPU campaign was touched.
- The only commands run were: extracting the archive, SHA-256 hashing of
  bundled files against their manifests, `ast.parse` of the 35 Python files,
  text diffs between the GPU and oracle sources, reading JSON fixture
  metadata, one integer-formula check of `SEED_MAP.json`, and one
  `hasattr(np.zeros(1), "device")` API probe (local NumPy is 1.26.4).
- **CUDA is untested. No speedup is measured or claimed.** Statements about
  CuPy behaviour come from its documented and known semantics and are marked
  where they are uncertain. Memory figures are arithmetic estimates.

---

## 1. Findings summary (priority order)

| ID | Sev | Location | One-line |
|----|-----|----------|----------|
| C1 | Critical | `src/qumode_qsk_gpu/gradients.py:129-130, 134→126`; `:187→185` | Adjoint `lam` is rebound to a fresh array, so `_other()` raises. Nov3 gradient always crashes; QAOA crashes for L≥2 |
| C2 | Critical (silent) | `gradients.py:121-123 → 131-133`; `:181-182 → 186` | `walsh_pingpong`/`rx_pingpong` use **src as scratch**, so `vA` is clobbered. Shared γ0/α1/α2 and QAOA γ gradients are wrong |
| H1 | High | `kernels.py:116-117` (called from `model.py:225-227`, `gradients.py:157-159,179-185`) | Per-batch RX coefficients reshaped to `(B,1,1,1)` against 3-D slices. Broadcasts to `(B,B,…)`; raises for B≥2 |
| H2 | High | `spsa.py:100-101` with `model.py:244-245` | Batch-2 energies are a CuPy array, so `np.asarray(cupy)` raises TypeError. Every SPSA run crashes |
| H3 | High | `GPU_test/tests/test_gpu_validation.py:64,124,188,242`; `test_host_oracle_light.py:26-27` | `from hamiltonian import …` loads oracle as top-level module; its relative import fails. GPU_test cannot run |
| H4 | High | `entropy.py:118-123` with `scripts/run_production.py:24-26` | Same import failure; fallback `cpu_oracle.src.*` needs ROOT on `sys.path`, which production omits. Every entropy task FAILs |
| H5 | High (env) | `model.py:176`, `gradients.py:78,145`; `requirements.txt` | `hasattr(x,"device")` is True for NumPy ≥ 2 ndarrays, so host x is never uploaded. NumPy/CuPy mixing TypeError |
| H6 | High | `runtime.py:86-97,186-208`; no device context in scripts/jobs | `--device-id k` only scopes queries and uploads. All other allocations go to the *current* device (usually 0) |
| H7 | High (test credibility) | `test_gpu_validation.py` (whole) | GPU_test never runs `spsa.optimize`, `bfgs.run_start`, `jobs`, batch>1, or N∉{6,15}. H1/H2 would pass the gate |
| M1 | Med-High | `gate.py:25-27, 123-124`; tests | Stamp does not pin `tolerances.json` or individual fixture files; CuPy/CUDA/NumPy/SciPy versions not enforced |
| M2 | Med-High | `jobs.py:184, 207/221, 241`; `plan_round` | `valid_for_fits` of SPSA/BFGS results defaults to True and is never read from the entropy label |
| M3 | Medium | `jobs.py:65-82, 112`; `run_production.py` | Resume ignores code/stamp provenance; reference `.npz` not atomic; no run manifest; no summary writer |
| M4 | Medium | `MANIFEST.json` / `PROTOCOL_FROZEN.json` / `SOURCE_SNAPSHOT.json` | Provenance mismatches (starts.py pin; protocol pins predate reverse-mode; kernels_numba unpinned) need a documented addendum |
| M5 | Medium | `requirements.txt` | NumPy/SciPy unpinned. SciPy BFGS line search and NumPy RNG stream must equal the campaign's |
| M6 | Medium | `test_gpu_validation.py:82` | `dE <= 1e-12 or dE <= 1e-8` makes the tight GS-energy tolerance meaningless |
| L1 | Low-Med | `scripts/benchmark.py:73` | `Event.elapsed_time` does not exist in CuPy; use `cupy.cuda.get_elapsed_time(start, end)` |
| L2 | Low | `memory.py:49-52`; README | Planner undercounts temporaries (~L+15, not L+9 vectors) and nothing in production uses it to cap batch |
| L3 | Low | `entropy.py:89, 102-109`; `jobs.py:37`; `kernels.py:113-114`; `run_production.py:37` | Dead or wrong code: dead import of nonexistent module, dead expression, unreachable check, wrong `--limit` help |
| L4 | Low | `test_gpu_validation.py:181-202` | "one optimizer step" takes no step; duplicates the start-x check |
| L5 | Low | `run_production.py:76-78` | Halts the whole round on the first FAIL (fail-closed, but one bad instance stops 1,500 tasks) |
| E1–E10 | Efficiency | see §4 | Walsh/RX/apply_X pass structure, small-N launch latency, serialized CPU entropy, gradient temporaries, syncs |

---

## 2. Correctness blockers in detail

### C1 — Adjoint buffer leaves the ping-pong pair → guaranteed exception (Critical)

`gradients.py` Nov3 backward loop:

```
125 lam = kernels.walsh_pingpong(lam, _other(lam_a, lam_b, lam), n)   # ok: lam ∈ {lam_a, lam_b}
129 lam = lam * xp.conj(sph)                                          # NEW array, not lam_a/lam_b
130 lam = kernels.walsh_pingpong(lam, _other(lam_a, lam_b, lam), n)   # _other raises ValueError
134 lam = lam * xp.conj(phase)                                        # same problem for next layer's line 125
```

`_other()` (lines 57-63) raises `"current buffer is not a member of the ping-pong pair"`.
So `nov3_energy_grad` **raises on every call, for every L≥1**. QAOA (`:185`, `:187`)
raises at the second backward layer, so L≥2 fails. QAOA L=1 does not hit this
(see C2).

This fails closed. However, it blocks all Nov3 BFGS work and every GPU_test
gradient check.

**Fix (same math):** multiply in place, `xp.multiply(lam, xp.conj(sph), out=lam)`
and `xp.multiply(lam, xp.conj(phase), out=lam)`. A diagonal multiply is
alias-safe. `lam` then stays a member of `{lam_a, lam_b}`.

### C2 — Ping-pong destroys its *source*; `vA` reused after being overwritten (Critical, silent)

`walsh_pingpong(src, dst)` (`kernels.py:61-81`) alternates `cur/other` each qubit.
After qubit 1 it writes back into `src`, so **`src` is scratch and does not
survive**. For even N the result *is* `src`. For odd N the result is `dst` and
`src` holds the state after N−1 butterflies. `rx_pingpong` (`:93-126`) behaves
the same way.

- Nov3 (`gradients.py:121-123`): `vA = vin*phase`, then `vB = walsh_pingpong(vA, fwd_a, n)`.
  Lines 131-133 then use `vA` for g[0], g[1], g[2] (γ0, α1, α2). That `vA` is
  garbage. The CPU oracle (`hamiltonian.py:243-255`) keeps `vA` intact because
  `walsh_c` copies.
- QAOA (`gradients.py:181-182`): `vB = rx_pingpong(vA, fwd_a, beta, n)`, then
  `:186` uses `vA` for the γ gradient. That value is wrong.

Once C1 is fixed, this corrupts exactly the **shared-parameter accumulation**
the review was asked to stress. For QAOA L=1, batch=1, it is *already* silent
today, unless H1 raises first. GPU_test's oracle and finite-difference checks
would catch it, but only once H3 is fixed.

The shell-angle gradients (lines 126-128) and β gradients (`:183-184`) use
`vC`/`vB` and are fine.

**Fix:** keep `vA` in its own buffer. Copy it into a dedicated scratch pair
(`fwd_a`, `fwd_b`), run the Walsh/RX on that pair, and leave `vA` untouched.
Add a docstring and assertion to `walsh_pingpong`/`rx_pingpong` stating
"src is clobbered". Add a unit test (GPU host) that checks `vA` before and
after.

Model forward (`model.py:199-213, 222-229`) and the gradient forward
(`gradients.py:93-102`) are **correct**: the input buffer is never needed after
the transform, and `layer_in` copies are taken first.

### H1 — Per-batch RX coefficients broadcast to the wrong rank (High)

`kernels.py:116-117` reshapes `c`/`s` to `(batch,1,1,1)`. However, `a = x[:, :, 0, :]`
is 3-D, `(batch, n_blocks, step)`. So `c*a` broadcasts to `(batch, batch, n_blocks, step)`:

- **batch ≥ 2:** the assignment into `y[:, :, 0, :]` cannot broadcast and raises.
  SPSA always evaluates the ± pair as batch 2, so QAOA SPSA always fails.
- **batch = 1:** a 4-D `(1,1,nb,step)` value is assigned into a 3-D view.
  NumPy drops leading unit dimensions in that case. Whether CuPy's
  `__setitem__`/`elementwise_copy` accepts it is **not verified**; treat it as
  luck at best.

Every QAOA path takes this branch, because `model.py:225` and
`gradients.py:157/179` pass `xd[:, k]` (1-D, not 0-d).

**Fix:** `reshape(batch, 1, 1)`. Remove the unreachable check at `kernels.py:113-114`.

### H2 — SPSA batched energies never reach the host (High)

`Model.energy_of_params` returns a **device** `E` when batch>1 (`model.py:244-245`).
`spsa.py:101` then does `np.asarray(E, dtype=REAL)`. CuPy deliberately raises
`TypeError` on implicit device-to-host conversion. As a result, **every SPSA cell
(both ansatz) crashes at the first calibration pair**.

**Fix:** `runtime.to_host(E)`. Better, make `energy_of_params` always return host
float64 for the energy and document it.

The ledger logic itself is equivalent to the CPU oracle. I diffed it: same
roles, order (+ then −), marks, padding, readouts, RNG consumption and
direction hashing.

### H3 — GPU_test cannot import the CPU oracle (High)

Every test imports `hamiltonian` (and `starts`) as **top-level** modules from
`cpu_oracle/src`. `hamiltonian.py:21` and `starts.py:4-6` use relative imports
(`from .kernels_numba import …`). Loaded top-level, they raise
`ImportError: attempted relative import with no known parent package`.

`run_all` therefore dies in `_cpu_model` before any check runs. No report and
no stamp are produced (fail-closed, but the suite is unrunnable). The "cheap"
`test_host_oracle_light.py` is broken the same way.

**Fix:** `from cpu_oracle.src.hamiltonian import …`. `setup_paths()` already puts
ROOT on `sys.path`. Stop inserting `cpu_oracle/src` itself, which also removes
module-name shadowing (`entropy`, `protocol`, `spsa`, `bfgs`, `starts`, `io_util`
are generic names).

### H4 — Production entropy scan cannot import the oracle (High)

`entropy.scan_bstar_cpu` (`entropy.py:118-123`) tries `import hamiltonian`, which
fails as in H3. It then falls back to `cpu_oracle.src.hamiltonian`. That needs
ROOT on `sys.path`, but `run_production.py:24-26` inserts only `src`,
`cpu_oracle` and `cpu_oracle/src`. When the script runs as
`python3 scripts/run_production.py`, `sys.path[0]` is `scripts/`.

The import therefore fails. The first task of the round (entropy N=6, index 0)
FAILs and the round stops (`run_production.py:76-78`). The bare
`import_module("entropy")` is also a shadowing hazard: any module or package
named `entropy` on the path would be used silently.

**Fix:** insert ROOT in `run_production.py` (and in benchmark/test scripts) and
import only `cpu_oracle.src.hamiltonian` / `cpu_oracle.src.entropy`.

### H5 — NumPy ≥ 2 host arrays are mistaken for device arrays (High, environment-conditional)

`model.py:176` and `gradients.py:78,145` choose "already on device" with
`hasattr(x, "device")`. NumPy 2.0 added `ndarray.device` (Array-API). Under
NumPy ≥ 2, host `x` is never uploaded, and `a1 * D1` mixes NumPy and CuPy,
which raises TypeError.

`requirements.txt` allows any `numpy>=1.24`, so a fresh GPU host very likely
gets NumPy 2. The "device x" branch is dead anyway, because `_prep_x` and
`_as_batch_x` call `np.ascontiguousarray` first, which CuPy rejects.

**Fix:** always upload, or test `isinstance(x, cupy.ndarray)`. Pin NumPy (M5).

### H6 — Device selection is incomplete (High)

`runtime.acquire(device_id=k)` enters `Device(k)` only for the property query.
`to_device` enters it only for uploads. Every `xp.empty/exp/matmul/…` in
`kernels`/`model`/`gradients` runs on the **process-current device**, typically 0.
With k≠current, this either raises a CuPy device-mismatch error or does work
and allocations on GPU 0. GPU 0 may be a device the user meant to avoid.

**Fix:** after `acquire`, the *scripts* (not import-time library code) should
call `cp.cuda.Device(k).use()`. Alternatively, wrap `run_all` and each `dispatch`
in `with runtime.device_context():`. Record the device actually used in each
result.

### H7 — GPU_test coverage is not credible for production (High)

What the suite checks (once H3 is fixed): GPU E, F and norm at stored CPU x
(SPSA best x, both SPSA initial points, BFGS start and final x); GPU state and
gradient against the oracle at the same x; GPU central-difference gradient;
stored-GS residual, energy, parity and entropy (host and device SVD); and
fixture self-consistency of the scan peak.

That covers the requested N6/N15 fixed-params E, F, state, gradient
(oracle + FD), entropy and reference checks.

Missing:

1. **No optimizer driver is exercised.** `spsa.optimize`, `bfgs.run_start` and
   `jobs.*` never run. H1 and H2 are invisible to the gate.
2. **batch>1 is never exercised** (the SPSA ± path, and the batched RX/phase
   branches).
3. **SPSA ledger reproducibility is not tested.** Directions depend only on the
   host RNG, not on energies, so a GPU rerun must reproduce the fixture's
   `direction_hashes`, `initial_points`, `query_roles` and `calls`
   **bitwise**. `tolerances.json` says `direction_hashes_must_match_if_rerun:
   true`, but nothing implements it. `gain_multiplier`, `best_E` and `x` may
   drift and should be report-only.
4. **BFGS:** no capped run (for example `maxiter=1..3`) comparing `nit/nfev`,
   first-iterate E and g to a CPU-oracle run with the same cap. This should be
   report-only with an energy tolerance.
5. **N=9, 12, 18, 21 are never touched.** The `N_ge_18` tolerances are dead.
   Add one synthetic-seed single evaluation per N (E, g, state against the
   oracle at a frozen start). That is cheap on the GPU host and exercises odd
   N=9/21 and the N=21 memory path.
6. **Fixture integrity is not checked.** Tests never verify fixture SHA-256s
   against `MANIFEST.json`, `reference.npz gs` against the instance
   `gs_sha256`, or `J` against `make_J(seed)` and `J_upper`.
7. The **start-recipe check** (`static_check.check_starts_against_catalog`) is
   not part of `run_gpu_test`, so the PASS stamp does not certify starts on
   the GPU host's NumPy.
8. **Failure handling:** exceptions propagate (crash with no report). Wrap
   each check and record `ok=False` plus the traceback, so the report is
   always written and the verdict is FAIL.

**State comparison and global phase:** the test compares states exactly
(`max|ψ_gpu − ψ_cpu|`, no phase alignment). That is *stricter* than "up to
global phase" and is correct here, because both sides apply identical phase
conventions to the same |+⟩. Recommendation: also log the phase-aligned error.
If strict fails while aligned passes, that diagnoses a conjugation or sign
convention bug.

**Tolerance realism (estimate):** `state 1e-12`, `E 1e-12`, `grad 1e-10`, and
`FD 5e-7` at ε=1e-7 are plausible for complex128 at N≤15. FD round-off is
about 1e-16·|E|/ε ≈ 1e-8 for |E|≈13, and truncation is about ε² ≈ 1e-14. This
has not been verified on hardware.

### M1 — Gate binding gaps (Medium-High)

`gate._iter_source_files` pins `src/`, `cpu_oracle/`, `protocol/`, `scripts/` and
`GPU_test/tests/`. It does **not** pin `GPU_test/tolerances.json`, so tolerances
can be loosened without invalidating the stamp. Fixture files are pinned only
through the `MANIFEST.json` digest, and nothing re-verifies the files against
it (H7 item 6).

`assert_production_allowed` (lines 123-124) compares source, fixtures, device
name, dtype and backend, but **not** `cupy_version`, `cuda_version` or
`compute_capability`, even though they are recorded. NumPy and SciPy versions
are not recorded at all.

**Fix:** pin tolerances, verify every fixture SHA in `run_all`, and add the
library versions to the binding and its comparison.

### M2 — `valid_for_fits` never comes from the entropy label (Medium-High)

`plan_round` never sets `valid_for_fits`. `task_spsa`/`task_bfgs_pack` use
`spec.get("valid_for_fits", True)` (`jobs.py:184, 207, 221, 241`). An instance
whose peak is `boundary_or_plateau_UNRESOLVED` or near-degenerate is invalid
in `data/instances/…json`, yet every optimizer result for it says
`valid_for_fits: true`. That violates the frozen "flag invalid, do not
replace" policy in downstream fits.

**Fix:** read it from the instance or entropy record in `load_model`.

### M3 — Resume, atomicity, manifest, summaries (Medium)

- `load_existing` (`jobs.py:65-82`) accepts any record with `wall_s` and no
  error. It does not check the source or stamp binding, the start
  `x_sha256`/`rng_seed`, or N/seed/L/method inside the record. A run resumed
  after a code change would silently mix code versions.
  **Fix:** store `binding_sha256` (from the stamp), `source_pins_sha256` and the
  library versions in every record, and require them to match on reuse.
- `np.savez_compressed(ref_path, …)` (`jobs.py:112`) writes in place. A crash
  can leave a truncated reference while the instance and scan JSON already
  exist. **Fix:** write to a temp file, `os.replace`, then fsync. `save_json`
  also lacks fsync (acceptable, but note it).
- **No round manifest** records the planned task list, the binding, and
  start/finish/fail per task. The README's "manifest" is the fixture manifest
  only.
- **No summary writer** exists, although the README promises "official
  summaries in balanced 10-per-N rounds". `summaries/` is empty and no code
  writes it.
- **Round balance is not enforced.** Nothing prevents running round r+1 before
  round r completes.

The plan itself is correct. `plan_round` yields 60 entropy tasks
(6 N × 10 instances), 720 SPSA cells and 720 BFGS packs (11,520 starts) per
round: all N, L=1..6, four arms (SPSA/BFGS × Nov3/QAOA), with round_id 1..20
giving the 200 cap. Over 20 rounds that totals 1,200 instances, 14,400 SPSA
cells and 230,400 BFGS starts, which matches `PROTOCOL_FROZEN.workloads`.
Seeds `SEED_BASE + N·1000 + i` match `SEED_MAP.json` (checked for all
6 × 200).

### M4 — Provenance: document, do not re-hash the science (Medium)

What I verified by hashing:

- All 322 copied fixture files and 20 start slices match their
  `MANIFEST.json` SHA-256s.
- `validation_catalog_sha256` matches.
- The `manifest_sha256` self-hash matches (sorted, compact JSON without that
  key).
- `protocol/` and `GPU_test/fixtures/` copies of `PROTOCOL_FROZEN.json` and
  `SEED_MAP.json` are byte-identical.
- All 11 `SOURCE_SNAPSHOT.json` entries match the bundled oracle.

Mismatches that must be **documented, not "fixed" by editing
`PROTOCOL_FROZEN.json`**:

1. **Two protocol hashes are in circulation.** The file SHA-256 is `8253fae8…`.
   The embedded `protocol_sha256` is `f32a96d4…`, which I verified to be the
   SHA-256 of the sorted, compact JSON with that key removed.
   `docs/LIMITATIONS.md` calls `f32a96d4…` "SHA-256" without saying which one.
   State both.
2. **`starts.py`:** the MANIFEST `cpu_oracle_source_pins` entry gives the
   campaign hash `0a39d54a…` (2864 B) for `bundle_path cpu_oracle/src/starts.py`.
   The bundled file is `03c1d354…` (3095 B), the write-guarded edit. The
   oracle is therefore not byte-identical to the campaign for this file.
   Record `original_sha256` and `bundle_sha256` separately, plus a diff
   summary. Only `pack_panel_starts` changed; `spsa_starts` and `all_starts`
   are unchanged.
3. **Protocol `source_pins` predate the audited code:**

   | File | Protocol pin | Oracle/live |
   |------|--------------|-------------|
   | `hamiltonian.py` | `66efb69c…`, 9915 B | `769bd98a…`, 12078 B |
   | `bfgs.py` | `8abd2b4b…` | `f6fc989a…` |
   | `entropy.py` | `b157eae8…` | `3f6440c0…` |
   | `kernels_numba.py` | not pinned at all | `b3e7a4c6…` |

   `spsa`, `protocol` and `io_util` match. The protocol's
   `bfgs.jacobian = "analytic forward-mode, pre-fit FD audit"` describes the
   pre-reverse-mode code.
4. **Fixture scope:** fixture run JSONs carry no source pins or timestamps, so
   which `hamiltonian.py` version produced each BFGS/SPSA fixture is not
   provable from the bundle. Scientifically this is bounded. E and F at a
   stored x depend only on the state map, which is the same in forward and
   reverse code. The gradient implementation affects only the trajectory that
   produced x. GPU gradients are compared against the *bundled* oracle, not
   against fixtures.

**Recommendation:** add `protocol/PROVENANCE_ADDENDUM.json` (new file, pinned by
the gate) listing items 1–4, the audited-reverse-mode statement, and fixture
scope. Leave `PROTOCOL_FROZEN.json` and both of its hashes untouched.

### M5 — Library pins are part of "same optimizer" (Medium)

SciPy's `_minimize_bfgs` and line search have changed across releases.
`Generator.normal` streams are not guaranteed stable across NumPy versions
(NEP 19). To call this "same SciPy BFGS, same frozen starts", pin NumPy and
SciPy to the campaign's exact versions, and record them in the stamp and in
each result. Starts are still checked bitwise, but only if H7 item 7 is added.

### M6 — Ineffective GS-energy tolerance (Medium)

`test_gpu_validation.py:82`: `ok_E = dE <= tol_E(1e-12) or dE <= E0_residual(1e-8)`
is effectively 1e-8. Use the energy tolerance alone. Keep the residual as a
separate check, as it already is.

### Low items

- **L1:** `benchmark.py:73` calls `start.elapsed_time(end)`, which does not
  exist on CuPy Events, so the benchmark crashes. The lambda at `:21-23` is
  also dead. Timings include the per-eval host syncs (§4 E5), so label them
  "end-to-end per eval".
- **L2:** `memory.grad_vector_count = L+9` undercounts. Counting live
  `(batch, 2^N)` complex arrays at the backward peak: `layer_in`×L, `buf_a`,
  `buf_b`, `psi`, `Hv`, `lam_a`, `lam_b`, `fwd_a`, `phase`, `sph`, `vA`, `vC`,
  the product temporary, the `conj·` temporary, and the new `lam`, plus real
  `gen`/`ang`. That is about L+15. The safety factors (×1.2, ×0.4) absorb it at
  batch 1. Nothing in production consults the planner (BFGS is fixed at
  batch 1, SPSA at 2), so README's "the planner will cap the batch" is
  inaccurate.
- **L3:** Dead code: `entropy.py:102-109` imports a nonexistent
  `cpu_oracle_import`; `entropy.py:89` is a dead expression; `jobs.py:37` is
  `_CPU_CAMPAIGN = Path if False else None`; `kernels.py:113-114` is
  unreachable. `run_production.py:37` says `--limit` applies "after entropy",
  but it applies to the whole list.
- **L4:** `test_one_optimizer_step` does not take a SciPy step. Either rename it
  or implement the capped-iteration comparison from H7 item 4.
- **L5:** `run_production` returns on the first FAIL. Prefer: record the FAIL,
  continue with independent tasks, and exit nonzero at the end. Never replace
  a seed.
- `test_host_oracle_light` counts one per SPSA entry (306 in total) while
  `static_check` counts 324. Harmonize the counting.

---

## 3. Verification of the requested science points

| Point | Status | Notes |
|-------|--------|-------|
| FP64/complex128 | OK | `dtypes.py` float64/complex128; `check_cplx` on all kernel entries; Python complex scalars keep complex128. No float32 path found |
| Batched FWHT/RX ping-pong aliasing | **Partly wrong** | Within-butterfly aliasing is avoided. The *source clobber* is not handled in the backward pass (C2). RX batch rank is wrong (H1) |
| Odd-N, LSB qubit 0 | OK | `_reshape_qubit` → `(B, dim/2^{q+1}, 2, 2^q)` selects bit q, matching the Numba stride. Odd-N result-buffer parity is handled by `_other`/rebinding in forward passes |
| Final H / Nov3 layer | OK | phase(shared) → Walsh → shell phase → Walsh per layer; same as `hamiltonian.py:150-160` |
| Nov3 3+3L params, shared accumulation | Formula OK, values wrong | The `+=` accumulation over layers of g[0..2] matches the CPU, but it uses a clobbered `vA` (C2) and crashes first (C1) |
| QAOA 2L | OK forward (modulo H1) | (γ,β) pairs; RX c=cosβ, s=−i sinβ (textbook RX(2β)) as frozen |
| Adjoint convention | OK | g = 2 Re⟨λ\|(−i G) ξ⟩ with λ=H\|ψ⟩; Walsh self-adjoint; RX(−β); conj phases; ordering identical to the CPU oracle |
| Reduced density matrix / entropy | OK | `reshape(2^{N−na}, 2^{na}).T` gives `[a,b] = gs[a + (b<<na)]`, equal to the CPU `schmidt`. Correct for odd N; σ>1e-15 cut; natural log |
| Norm | OK | `|‖ψ‖²−1| ≥ 1e-12` assertion as in the CPU; no mid-circuit renormalization |
| zz_diag / shell / J / K / M | OK (by construction) | Same pair order and integer-exact shell sums, so bitwise-equal values are expected. `test_host_oracle_light` would confirm this but is broken (H3) |
| SPSA 1024L inclusive ledger, 2 frozen starts, RNG | OK logic; broken I/O | Diffed against the oracle: identical apart from ev_pair batching (H2) |
| BFGS 16 cold starts, 8/8 widths, settings, energy selection | OK | `make_start`/`start_rng_seed`/options/`select_by_energy` identical; SciPy on host; batch 1 |
| Frozen protocol, var 1/N, seeds, no warm start | OK | Constants identical; SEED_MAP formula verified |
| Entropy eigensolve | CPU fallback (documented) | See §5 |

**First-stage claims, checked logically:**

- *"35 AST files clean":* I confirmed 35 `.py` files parse (`ast.parse`). The
  claim means py_compile plus no top-level `Device()`/`.use()` calls. It is
  true, but it says nothing about any defect above.
- *"324 starts bitwise match":* 324 = 18 cells × 2 SPSA points (36, compared
  with `array_equal` against CPU-run JSON; JSON float repr round-trips
  exactly) + 288 BFGS starts (SHA-256 of float64 bytes against
  `start_x_sha256`). The GPU start code is textually identical to the oracle
  (diffed). The claim is logically sound **for the NumPy that ran it**. It
  compares against CPU *outputs*, not against the frozen `starts/*.npz`
  slices or `starts_archive_x`, and it is not repeated inside GPU_test on the
  GPU host. I did not execute it.

---

## 4. Efficiency issues (no measurements; static analysis)

| ID | Prio | Location | Issue → same-math recommendation |
|----|------|----------|----------------------------------|
| E1 | High | `kernels.py:73-80, 118-125, 147-152` | Each qubit of Walsh/RX does about 4–6 elementwise launches with half-size temporaries: `a+b`, `/s`, copy, `a−b`, `/s`, copy. So ~6N launches and several full-state memory passes per qubit, and apply_X does 2N `+=` passes. **Fix:** one RawKernel per qubit that reads (a,b) and writes both in place (a thread owns its pair, so there is no aliasing and no ping-pong is needed). Better, a shared-memory tile kernel that applies ~10 low qubits per tile, then the high qubits. Keep the per-element operation order `(a+b)/√2` and qubit order 0..N−1. Fused apply_X: `out[s] = Σ_{q=0..N−1} v[s ^ 2^q]` in ascending q gives the same accumulation order as Numba. Compile with `--fmad=false` (see E9) |
| E2 | High | design (batch 1 BFGS, batch 2 SPSA) | At N=6–12 (dim 64–4096) the GPU is launch- and sync-latency bound, likely slower than the Numba oracle (unmeasured). **Fix:** route small N to the bundled CPU oracle kernels on the GPU host (identical math), and/or run several independent cells concurrently (multiple processes or streams; CUDA MPS). Do not batch inside SciPy |
| E3 | High | `run_production.py:62`, `jobs.task_entropy` | All 60 entropy scans (single-thread CPU SciPy) of a round run **before** any GPU work, so the GPU idles. **Fix:** a CPU process pool for scans (one thread each, preserving the campaign's BLAS-thread setting) pipelined with GPU optimizer tasks for instances whose reference exists |
| E4 | Med | `gradients.py:126-133, 184-186` | Each `_re_inner` allocates `conj(a)*b` plus a weight×vector temporary: about 12 full-size temporaries per Nov3 layer. Since Re(conj(λ)(−i w)v) = w·Im(conj(λ)v) for real w: compute `t = Im(conj(λ)·vC)` once, then `t @ shell`, giving 3 values in one matmul. Similarly use `Im(conj(λ)·vA)` against `[gen, g0·D1, g0·D2]`. This changes only reduction rounding, which already differs from the CPU. Alternatively, a ReductionKernel evaluating the exact per-element CPU expression |
| E5 | Med | `gradients.py:105, 192-203`; `model.py:239` | Three blocking device-to-host syncs per `energy_grad` (norm, E, g) plus a pageable H2D of x. **Fix:** pack `[norm, E, g…]` into one device vector and make one `.get()`; use pinned host buffers |
| E6 | Med | `gradients.py:99, 122, 129, 134` | `sph` is recomputed in the backward pass, and `conj(sph)`/`conj(phase)` are allocated each layer. Cache `sph` per layer (L×dim complex) or multiply by conj in place. Preallocate a per-(N,L,batch) workspace to avoid allocator churn |
| E7 | Low-Med | `bfgs.py:61-63` | The end-of-start check downloads two full statevectors (2×32 MiB at N=21) to compute `state_err`. Compute `max|Δ|` on the device |
| E8 | Low | `jobs.load_model`, `task_entropy:141-145` | The model is rebuilt per task. `zz_diag_host` does 3·N(N−1)/2 full-dim passes; `shell_ops_host` allocates a (dim, N/3, 3) int64 temporary, about 350 MB at N=21. `task_entropy` builds and uploads a GPU Model only for an energy cross-check. **Fix:** cache diags and shell per instance (or store them in the reference npz with SHA); group tasks per instance |
| E9 | Info | all CuPy elementwise | NVRTC defaults to FMA contraction, while Numba (fastmath off) does not. Complex multiplies therefore round differently, so bitwise CPU parity is impossible with stock CuPy ops. Custom kernels with `--fmad=false` plus identical op order could make Walsh, RX and diagonal multiplies bitwise-equal. `exp`/`cos`/`sin` (libm) and reductions will still differ at ULP level |
| E10 | Info | `spsa.py:95-115` | Batch-2 and batch-1 energy evaluations of the same x can differ at ULP level (reduction shape). This is harmless, but document it beside the LIMITATIONS note on SPSA drift |

**Hidden host copies and syncs, enumerated:** per `energy_grad`, x H2D plus
norm, E and g D2H. Per SPSA query, x H2D and E D2H. Per BFGS start, two
full-state D2H. `entropy_device` boolean masking syncs. `p[p>cut]` and
`as_scalar` each sync. The gate re-hashes the source tree per task (cheap).
There are no per-layer transfers. README's "only a few scalars per eval" is
true for `energy_grad`/`energy_of_params`.

**RAM/VRAM at N=21 (arithmetic estimate):**

- One state is 32 MiB.
- Device constants: 3 diags (48 MiB) + shell (48 MiB) + |+⟩ (32 MiB) + GS (32 MiB) ≈ 160 MiB.
- Nov3 gradient at L=6, batch 1: about (6+15)×32 MiB ≈ 0.7 GiB, plus the CuPy pool slack.
- SPSA energy-only at batch 2: about 0.5 GiB.

These fit any ≥4 GB GPU. The 0.4 safety factor is conservative.

Host RAM per N=21 entropy scan: two parity-sector CSR matrices with about
21·2²⁰ nnz each (~0.5 GB), a rebuilt `Hp`/`Hm` per field, `_parity_ops`
transients (~1 GB), Lanczos vectors (~160 MB) and the GS (32 MiB). Plan for
about 2–4 GB per concurrent scan process, plus the ~0.5 GB `shell_ops_host`
spike.

---

## 5. Entropy eigensolve on the CPU, so the pipeline is not fully GPU

The README, LIMITATIONS and the `entropy.py` docstring all say correctly that
the B* scan, the ground-state extraction (parity-sector `eigsh`/`eigh`), the
gauge, degeneracy and peak rules run on the **CPU oracle**. Only S(ψ) of a
given state may use the device SVD. No "all-GPU" wording was found.

Impact:

1. **Science:** this is positive for parity. The stored GS, E0, b* and labels
   come from the identical CPU code path, so F and energy-error references
   match campaign semantics.
2. **Wall time:** the scan is single-threaded (the oracle forces thread env
   vars to "1" on import). The N=15 fixture recorded 174 eigensolves in about
   9 s (CPU-side metadata). At N=18/21 the cost grows with the 2^{N−1} sector
   dimension. It is a per-instance fixed cost that the GPU cannot accelerate,
   and with the current scheduler it is fully serialized ahead of GPU work
   (E3).
3. **Documentation to add:** a statement that end-to-end per-instance time
   includes a CPU scan; the host-RAM figures above; and that the GPU Model in
   `task_entropy` is only an energy cross-check.

---

## 6. Recommended fix order (no science changes)

1. C1 and C2: gradient buffer handling. Add a src-clobber assertion and docs to `walsh_pingpong`/`rx_pingpong`.
2. H1: RX reshape. H2: SPSA host download. H5: always upload / isinstance check.
3. H3 and H4: package-qualified oracle imports; ROOT on `sys.path`; stop adding `cpu_oracle/src`.
4. H6: explicit device `.use()` in scripts, or contexts around all work.
5. H7 and M1: driver, batch and all-N tests; fixture SHA verification; pin tolerances; versions in the binding; never crash without writing a report.
6. M2, M3: `valid_for_fits` propagation; provenance-checked resume; atomic npz; round manifest; summary writer; round ordering guard.
7. M4, M5: `PROVENANCE_ADDENDUM.json`; pin NumPy and SciPy.
8. Efficiency: E1 → E3 → E2 → E4/E5 → the rest. Re-run GPU_test after each change, because the source pins will change.

## 7. CUDA-untested limitations (restated)

- Nothing here has been executed on a GPU. All CuPy behaviour statements are
  from documented semantics. The batch-1 4-D-into-3-D assignment in H1 is
  explicitly uncertain.
- No performance figures were measured. The efficiency items are structural
  (launch counts, temporaries, passes, syncs).
- C1, H1 (batch≥2), H2, H3 and H4 are certain from the code paths. C2 is
  certain once C1 is fixed. H5 is certain under NumPy ≥ 2. H6 is certain for
  `device_id` ≠ the current device.
- A future GPU_test PASS remains necessary but **not sufficient** until H7 is
  addressed.
