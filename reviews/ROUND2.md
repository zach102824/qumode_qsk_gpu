# qumode_qsk_gpu — Review Round 2 (revised GPU implementation)

Archive reviewed: `review_bundle.tar_4690.gz`, SHA-256
`5331474c0549b4a82634402a7f49144790a571d978176a7f1c1760c40a440fe6`. It was
extracted into a separate folder (`/tmp/rb2/qumode_qsk_gpu`). Round 1 code was
used only for `diff`/`cmp`, never as the source under review.

## Verdict: CHANGES NEEDED (4 High, no Critical)

**This is a static review only. It is not CUDA approval.** No CUDA, no
numerical experiments, no installs, no production runs. Only SHA-256 hashing
of bundle files was executed.

All Round 1 Critical and High items in the gradient, kernel, import and device
code are fixed as written. Four new or remaining High issues block a credible
production run:

| ID | Severity | Summary |
|----|----------|---------|
| R2-H1 | High (new regression) | The RX RawKernel receives a **strided β view**. For batch ≥ 2 the QAOA β of row 1 is read from the wrong memory: γ of the next layer, or row 1's γ₀. GPU_test should catch it (fail-closed), so it blocks the PASS stamp rather than corrupting data silently. |
| R2-H2 | High | The production pipeline **livelocks** (busy-spins forever) if the entropy thread stops early. That happens on `--halt-on-fail`, or on any exception escaping `jobs.dispatch`, for example a gate mismatch after a pinned doc is edited mid-round. |
| R2-H3 | High | The round-ordering guard runs **before `runtime.acquire()`**. `provenance_fields()` then has no `binding_sha256`, so every result looks non-resumable and round r−1 always reports "incomplete". Rounds 2–20 cannot start without `--force-unordered`, which defeats the balanced-round guard. |
| R2-H4 | High (test credibility) | The capped BFGS check is **masked**: `ok = start_x_sha_match` only, and the energy agreement with the CPU oracle is report-only. The only BFGS-driver test can never fail on numerics. |

The Medium items cover:

- Walsh RawKernel op-order parity: it multiplies by 1/√2 where the CPU divides by √2.
- The silent effect of lazy RawKernel compilation on `_ensure_raw`.
- GPU work on the entropy thread.
- `--entropy-workers` and `--continue-on-fail` that do nothing.
- Per-call re-hashing of the whole tree.
- Library-version pins that are documented but not enforced.
- Summary means that include invalid instances and non-provenance-checked records.
- Weak tests for `valid_for_fits`, entropy jobs, resume and SPSA energies.

Details are in §2–§4. §5 maps every Round 1 item to resolved or unresolved.

### Scope and limits

- I read `reviews/ROUND1_FIXLOG.md` and every new or changed source file:
  - `src/qumode_qsk_gpu/*`;
  - `scripts/*`;
  - `GPU_test/tests/*`, `GPU_test/tolerances.json`, `GPU_test/README.md`;
  - `protocol/PROVENANCE_ADDENDUM.json`;
  - `requirements.txt`, `README.md`, `docs/*`.
- Unchanged relative to Round 1, verified by `cmp`/`diff -rq`:
  - all of `cpu_oracle/src/*.py`;
  - `GPU_test/fixtures/` (346 files);
  - `protocol/PROTOCOL_FROZEN.json`, `protocol/SEED_MAP.json`;
  - `src/qumode_qsk_gpu/{starts,protocol,dtypes}.py`.
- I did not change the implementation.
- CUDA semantics are taken from the documented CuPy and NVRTC behaviour:
  - RawKernel ndarray arguments are passed as raw device pointers, and strides are ignored;
  - RawKernel compilation is lazy;
  - CuPy's current device is thread-local.
  None of it was executed.

---

## 1. Verification of the Round 1 fixes requested in this round

| Requested check | Result | Evidence |
|---|---|---|
| vA kept independent | **OK** | `gradients.py` (Nov3 and QAOA): `vA = vin * phase` is a dedicated array, `fwd[...] = vA` is copied, and Walsh/RX run on `fwd` only. `vA` is read afterwards only in `Im(conj(λ)·vA)`. |
| λ tracked in place | **OK** | `lam = Hv.copy()`. After that, λ is changed only by `walsh_inplace(lam)`, `rx_inplace(lam, −β)` (which returns the same array) and `_multiply_inplace(lam, conj(·))`. It is never rebound to a fresh temporary. |
| Odd N, pair ownership | **OK (static)** | WALSH/RX kernels: `n_pairs = dim>>1`, `blk = p/step`, `off = p − blk·step`, `i0 = b·dim + blk·2step + off`, `i1 = i0 + step`. This is a bijection from thread id to (b, pair) for any `step = 2^q < dim`, independent of N parity. Each thread reads and writes only its own two elements, so `src == dst` (in place) has no race. Batched indexing uses `b·dim` with `long long`. |
| Batched Walsh/RX | **Walsh OK; RX FAILS for batch ≥ 2** (R2-H1) | The β pointer is not made contiguous. |
| complex128 ABI | **OK** | `double2*` matches CuPy complex128 layout (re, im, 16 B, 16-B aligned allocations). Scalars are passed as `np.int64` for `long long` and `np.int32` for `int beta_scalar`; the width and order match the kernel signatures. |
| Grid bounds | **OK** | `_grid = ceil(total/256)` (1-D) with a `tid >= total` guard, and `long long tid`. At N=21, batch 2, `total = 2^21` gives 8192 blocks, far below the gridDim.x limit. |
| Shared adjoint gradient signs | **OK** | Shell gradient: `2·(Im(conj(λ)vC) @ shell)`. Shared parameters (γ₀, α₁, α₂): `tA = Im(conj(λ)vA)` against `gen`, `g0·D1` and `g0·D2`, summed over layers. This uses the identity `Re(conj(λ)(−i w)v) = w·Im(conj(λ)v)` for real w, the same sign as the CPU `_re_inner`. QAOA: β gradient is `2 Re Σ conj(λ)(−i X vB)` before `RX(−β)` is applied to λ; γ gradient is `2 D·Im(conj(λ)vA)` after it. Both are in the oracle's order. |
| CuPy imports, device activation | **OK** | `oracle.py` uses package-qualified `from cpu_oracle.src import …`, strips `cpu_oracle/src` from `sys.path` and inserts ROOT. `acquire(activate=True)` calls `Device(k).use()`. `dispatch` wraps tasks in `device_context()`. That matters because the entropy thread does not inherit the main thread's current device; CuPy's current device is per thread. |
| CPU parameter type | **OK** | `is_device_array` uses `isinstance(x, cupy.ndarray)`. `_prep_x` goes through `as_host_array`, and `state()` always uploads. |
| One packed transfer per gradient | **OK** | One `runtime.pack_to_host(nrm, E, g)` per `energy_grad` (concatenate on device, then a single `.get()`). The norm assertion runs on the packed host copy. The x upload (H2D) remains. |

---

## 2. High findings with fixes and minimal static scenarios

### R2-H1 — Strided β passed to the RX RawKernel (High, new regression from the H1 fix)

**Where.**

- `model.py:211-213` (`_state_qaoa`) and `gradients.py:151-153` (QAOA forward): `beta = xd[:, 2*ell+1]`, followed by `kernels.rx_inplace(cur, beta, n)`.
- `gradients.py:168-172` (backward recompute): `beta = xd[:, ib]`, followed by `rx_inplace(fwd, beta, n)`.
- `kernels.py:104-113` (`_beta_device`): `xp.asarray(beta, dtype=REAL)` and `.reshape(-1)`. Both return the same non-contiguous view; neither copies.
- RX kernel: `double ang = beta_scalar ? beta[0] : beta[b];`.

**Minimal static scenario.**

1. Take QAOA with L=1 and batch 2, which is exactly the SPSA ± pair in `ev_pair`.
2. `xd` is C-contiguous with shape (2, 2): `[[γa, βa], [γb, βb]]`.
3. `xd[:, 1]` is a view starting at element 1 with a stride of 2 elements.
4. The kernel reads `beta[0] = βa` (correct) and `beta[1] = *(ptr+1) = xd[1,0] = γb` (wrong).

For L ≥ 2 and ℓ < L−1, row 1 reads γ of layer ℓ+1 of row 0.

Consequences:

- The minus probe of every QAOA SPSA iteration is evolved with the wrong angle.
- In the batched QAOA gradient, the backward recompute of `vB` for row 1 uses the wrong β, while `−β` (line 176, a fresh contiguous array) is correct. Row 1's gradient is therefore internally inconsistent.
- Batch 1 is unaffected: the size-1 view's pointer already includes the offset.

**Detection.** `test_batch_ge2` (`test_gpu_validation.py:230-252`) and `synthetic_N_batch2` (`:399-409`) compare batch-2 QAOA rows against the CPU oracle at 1e-12/1e-10, so GPU_test should FAIL and no stamp is written. The bug therefore blocks PASS; it is not a silent production corruption. It still must be fixed before any QAOA SPSA can run.

**Fix.**

- In `_beta_device`, use `beta = xp.ascontiguousarray(xp.asarray(beta, dtype=REAL).reshape(-1))`.
- In the raw path (`walsh_inplace`, `rx_inplace`, `apply_X`), assert `states.flags.c_contiguous` and `beta.flags.c_contiguous`. Put the same assertion in `_require_states`, because every RawKernel argument is pointer-only.
- Optionally pre-extract `betas = xp.ascontiguousarray(xd[:, 1::2].T)` once, with shape (L, batch).

### R2-H2 — Pipeline livelock when the entropy thread stops early (High)

**Where.** `scripts/run_production.py:153-190` (entropy thread) and `:194-220` (main loop).

The main loop's exit condition is:

```python
finished = (ent_i >= len(entropy_order) and in_flight_entropy == 0 and not gpu_q)
if spec is None:
    if finished and not th.is_alive(): break
    th.join(timeout=0.2); continue
```

**Scenario A (`--halt-on-fail`).** An entropy scan FAILs, so `:188-190` sets `done` and returns while `ent_i < len(entropy_order)`.

1. The thread is dead, so `th.join(0.2)` returns immediately.
2. `finished` stays False forever.
3. Once `gpu_q` drains, the main thread busy-spins at 100% CPU and never reaches `_finish`.

**Scenario B (default flags).** Any exception escaping `_dispatch_record` kills the thread with `in_flight_entropy == 1`, which produces the same infinite spin. Concrete triggers:

- `jobs.dispatch:342-343` calls `gate.assert_production_allowed()` outside its try/except. If anyone edits a pinned file during a round, the next entropy dispatch raises and the run hangs. Pinned files include `README.md`, `docs/*.md`, `GPU_test/README.md` and `requirements.txt`.
- Re-running `run_gpu_test.py`, which rewrites the stamp, has the same effect.
- So does an `OSError` while hashing.

In the main thread, the same gate exception propagates out of `_pipeline`. `_finish` is skipped, the manifest status is never updated, and the daemon thread is killed mid-scan at interpreter exit.

**Fix.**

1. Wrap the entropy-thread body in `try/except/finally`. The `finally` must always decrement `in_flight_entropy` and `heavy_entropy` and record a FAIL row. Set a shared `stop` flag on halt or fatal error.
2. Make the main-loop exit `if not th.is_alive() and not gpu_q: break`, so it no longer depends on counters.
3. On halt, set `stop` and join the thread without a timeout. Better, check `stop` before starting each scan, so a half-finished N=21 scan is never orphaned.
4. Move `assert_production_allowed()` inside the try in `dispatch`, or catch it in `_dispatch_record`. A mid-round gate mismatch should then become a recorded FAIL plus an orderly halt.

### R2-H3 — Round-ordering guard is unsatisfiable before `acquire` (High)

**Where.**

- `run_production.py:70-75` calls `jobs.official_complete(r−1)`. `runtime.acquire()` is only called at `:77`.
- `official_complete` → `load_existing` → `_provenance_ok` (`jobs.py:82-91`) → `gate.provenance_fields()` (`gate.py:217-237`). When `runtime.is_acquired()` is False, `provenance_fields()` omits `binding_sha256` and `source_pins_sha256`, and `cupy_version` is None.
- `_provenance_ok` requires `rec[k] == cur[k]` for those keys.

**Scenario.**

1. Round 1 completes with valid records, each carrying `binding_sha256 = H`.
2. Start round 2. In `_provenance_ok`, `cur.get("binding_sha256")` is None, which is not equal to H. So `load_existing` returns None for every spec.
3. The guard reports round 1 as incomplete with `n_missing = 60+…`, and the script exits 2.
4. The only way forward is `--force-unordered`, which disables the guard entirely.

**Fix.** Call `runtime.acquire(activate=True)` and `gate.assert_production_allowed()` before the round-r−1 check. Also make `provenance_fields()` raise when not acquired instead of returning a partial dict, so this class of bug fails loudly.

### R2-H4 — The capped BFGS check cannot fail on numerics (High, test credibility)

**Where.** `test_gpu_validation.py:304-327`. It runs `run_start(…, maxiter=3)` on GPU and CPU and computes `dE` and `dE_within_tol`, but `rec["ok"] = rec["start_x_sha_match"]`. That value is already guaranteed by `test_start_recipes` and `test_host_oracle_light`. The check also runs only for `start_index == 0` and `L == 1`.

**Scenario.** Suppose the GPU `run_start` hands SciPy a gradient with a wrong scale, or a stale cached gradient: for example a `jac` that returns g from the previous x, or the λ/E pairing swapped between `fun` and `jac`.

- E and g at fixed x still match, so `test_fixed_params` passes.
- After three iterations, GPU and CPU `best_E` differ by orders of magnitude more than 1e-10, yet the check reports `ok=True` and the stamp is written.

The FIXLOG claims meaningful BFGS sanity; this is the masked-failure pattern the user asked to eliminate.

**Fix.**

1. Make the following PASS criteria for the capped run (identical SciPy settings; three iterations is short enough to be insensitive to ULP drift in practice):
   - `dE ≤ 1e-9`, a documented tolerance added to `tolerances.json`;
   - `|grad_l2_gpu − grad_l2_cpu| ≤ 1e-7`;
   - `max|x_gpu − x_cpu| ≤ 1e-7`.
2. Record `nit`/`nfev` equality. If it fails, fail with a clear "line-search divergence" label instead of skipping it.
3. Also run L=3 and one QAOA start, which would also exercise R2-H1 in batch-1 form.

---

## 3. Medium and Low findings

| ID | Sev | Location | Finding | Fix |
|---|---|---|---|---|
| R2-M1 | Med | `kernels_cuda.py` WALSH: `inv = 1.0/sqrt(2.0); e.x=(a.x+d.x)*inv` vs `cpu_oracle/src/kernels_numba.py:25,34-35` `(a+b)/s` | The FIXLOG and LIMITATIONS claim the same op order as Numba. The kernel multiplies by a rounded reciprocal; Numba divides by `sqrt(2)`. These differ by 1 ULP on a fraction of elements, so the claimed bitwise-equal Walsh does not hold. This is not a physics error. The Python fallback (`/ SQRT2`) and the RawKernel also disagree with each other. | Use `e.x = (a.x + d.x) / s` with `const double s = sqrt(2.0);`. `--fmad=false` already prevents contraction. |
| R2-M2 | Med | `kernels.py:74-89` `_ensure_raw` | `cp.RawKernel(...)` is lazy: NVRTC compiles at the first launch. The try/except therefore almost never sets `ok=False`; compile errors surface at the first call as an exception. That is fail-closed, so OK. But the `ok=False` fallback branch and `raw_kernels_available()` are effectively untested, and the stamp does not record which path ran. | Call `walsh.compile(); rx.compile(); ax.compile()` inside the try, or launch a 1-element warm-up. Add `kernel_path` (raw or python) plus the `_RAW_OPTIONS` string to the gate binding and to every result. |
| R2-M3 | Med | `jobs.py:217-220` (task_entropy), called from the entropy thread | The entropy thread runs CUDA: a GPU `Model` upload plus `energy` cross-check at N=21, about 0.3 GiB. It runs concurrently with the main thread on the same device and the legacy default stream, sharing the memory pool. Results are correct, because the null stream serializes and each thread syncs on its own `.get()`. But there are side effects: <ul><li>`_assert_memory` in the main thread does not count these allocations (covered only by the 0.6 slack);</li><li>`_ensure_raw` can race on first use;</li><li>the docs say entropy workers "do not need the GPU".</li></ul> | Return `gs`/E0 from the thread and run the `apply_H` cross-check on the main GPU thread. Alternatively, warm the kernels before starting the thread and document the shared-device use. |
| R2-M4 | Med | `run_production.py:41-42, 145-151, 192` | `--entropy-workers N > 1` does nothing: there is exactly one thread, so `in_flight_entropy ≤ 1`. The help text says "CPU processes". The overlap relies on SciPy releasing the GIL. The sparse matvec and ARPACK reverse-communication loop return to Python each iteration, and the GPU thread is Python-launch-heavy, so the effective overlap is unknown. | Use a `ProcessPoolExecutor` (spawn) for `scan_bstar_cpu` only. Keep BLAS threads at 1 as in the campaign, cap concurrent N=21 scans at 1, and have no CUDA in the workers. Keep GPU work in the parent. Fix the help text. |
| R2-M5 | Med | `run_production.py:92` | Entropy runs in −N order, so the GPU idles for the whole first N=21 scan before any optimizer task exists. Round 1's E3 is only partly addressed. | Start with small-N scans (in seconds, small-N GPU work is queued), or interleave N. This does not affect science. |
| R2-M6 | Med | `gate.py:116-146, 217-237`; `jobs.py:82-91, 341-344`; `summaries.py:87` | `current_binding()` re-hashes every pinned source plus all 346 fixture files (about 3.4 MB, about 420 file opens) on every call: twice or more per task (gate plus `provenance_fields`) and once per existing record in `load_existing`. A full `official_complete` (about 60 + 720 + 720×16 lookups) re-hashes about 12k times, which is tens of GB of SHA-256 and millions of file opens per resume scan or summary. | Compute the binding once per process after `acquire`. Re-verify cheaply at round start and finish, and before writing the stamp. Compare cached `binding_sha256` in `_provenance_ok`. |
| R2-M7 | Med | `gate.py:_COMPARE_KEYS`, `current_binding`; `requirements.txt`; `PROVENANCE_ADDENDUM.json` | The pins numpy 2.5.4, scipy 1.18.1, numba 0.68.0 and Python 3.13.5 are documented, but the gate only checks that versions equal the stamp's. A GPU host with NumPy 2.3 and SciPy 1.15 would PASS GPU_test and stamp, even though SciPy BFGS line search and the entropy scan (Numba oracle) would no longer be "the same CPU calculation". `numba` and `python` are not in the binding at all, yet Numba runs the production entropy scan. | Add `numba_version` and `python_version` to the binding and results. Make the gate refuse unless versions equal `PROVENANCE_ADDENDUM.campaign_library_versions_on_authoring_box`, with an explicit, recorded `--allow-version-drift` override that also marks results unofficial. |
| R2-M8 | Med | `summaries.py:26-31, 42, 60-85` | The summary uses `_load` with no provenance check, so stale records from older code or devices are counted. Panel means include `valid_for_fits=False` instances: `n_valid` is counted but not used to filter. Cells with fewer than 16 BFGS starts are silently dropped from the BFGS panel and not counted as missing. | Use `jobs.load_existing` (provenance-checked), report `valid_only` and `all` means separately, and list incomplete BFGS cells. Optionally add an energy-error column (`E − E0`, per-N normalized), since raw E means across different J have little meaning. |
| R2-M9 | Med | `test_gpu_validation.py:330-362` | `jobs_spsa_scratch` checks `valid_for_fits == instance.valid_for_fits`, but the fixture instance is valid, so a regression back to "default True" still passes. The invalid-entropy eligibility the user asked about is not tested. | Add a scratch case with a copy of the instance where `valid_for_fits=false` and the label is non-interior, and one with the instance file missing. Both must yield `valid_for_fits=False` in the SPSA and BFGS records. |
| R2-M10 | Med | GPU_test overall | These production paths are never exercised before the stamp: <ul><li>`task_entropy` (scan + reference npz + instance write + GPU cross-check);</li><li>`task_bfgs_pack`;</li><li>`load_existing` / `_provenance_ok` resume;</li><li>`official_complete`;</li><li>`summarize_round`;</li><li>the pipeline thread.</li></ul> R2-H2 and R2-H3 would therefore not be caught. | On the scratch tree at N=6, add: <ol><li>`task_entropy`, comparing b*, S and label to the fixture scan exactly (same CPU code and libraries);</li><li>`task_bfgs_pack` with 16 starts at L=1;</li><li>a second dispatch that must report `reused=True`;</li><li>`_pipeline` over two N=6 instances with an injected entropy FAIL, under both halt modes, with a wall-clock timeout.</li></ol> |
| R2-M11 | Med | `test_gpu_validation.py:272-301` | The SPSA driver PASS criteria (calls, direction hashes, query roles, starts) depend only on the host RNG, so they cannot detect a ± swap or a probe-energy error in `ev_pair`. `best_E` is report-only. | At N=6 L=1, also run the CPU oracle `spsa.optimize` on the GPU host (cheap) and compare the first K ledger energies (for example the calibration block) at the energy tolerance. Report full-trajectory drift separately. |
| R2-L1 | Low | `jobs.py:195-203` | `load_model` does not verify `sha256(gs)` against the instance's `gs_sha256`, or the reference npz against the entropy record. | Check both before optimizing. |
| R2-L2 | Low | `jobs.py:34-35` | `_DIAG_CACHE` is defined but unused, so the model (3·N(N−1)/2 diag passes) is rebuilt per task. Round 1's E8 is open. | Cache diags/shell per (N, seed) or store them in the reference npz with a SHA. |
| R2-L3 | Low | `run_production.py:45-46` | `--continue-on-fail` has `store_true` with `default=True`, so it is a dead flag. | Remove it; keep `--halt-on-fail`. |
| R2-L4 | Low | `run_production.py:218-220` | On halt, `join(timeout=5)` returns while the daemon scan continues, and `_finish` summarizes concurrently. Atomic writes prevent corruption, but `.tmp` leftovers and a half-run N=21 scan are lost. | See the R2-H2 fix. |
| R2-L5 | Low | `test_gpu_validation.py:441-444` | A dead `_safe(lambda: _models(n, seed))` builds and discards an extra GPU model per reference: wasted VRAM and time at N=21. | Delete it. |
| R2-L6 | Low | `GPU_test/tests/test_host_oracle_light.py:59` | Returns `ok=True` without asserting `n_start == 324`. `run_all` re-checks it, so this is only a redundancy gap. | Assert the count. |
| R2-L7 | Low | `jobs.configure_paths` and `test_gpu_validation.py:349,362` | `logs=` is redirected, but `finally` does not restore it. Later logs in the same process go to `scratch_jobs`. | Save and restore `paths.LOGS`. |
| R2-L8 | Low | `entropy.py` `entropy_device(download=…)` | The `download` parameter has no effect; p is always downloaded. | Remove it or honour it. |
| R2-L9 | Low | `gradients.py`, `model.py` (CuPy ufuncs) | Diagonal phase multiplies (`cur *= exp(…)`, `*= sph`), `exp` and reductions are still stock CuPy with default FMA contraction. This is documented in the addendum (`fma_and_reductions`) and LIMITATIONS. It is ULP-level only, so not a physics change. | Optional: an `ElementwiseKernel` with `--fmad=false` for complex multiplies, if closer parity is wanted. |
| R2-L10 | Low (ops) | `gate.current_binding` | `binding_sha256` covers docs/README/tests and `device_id`. A doc typo fix or moving to GPU 1 invalidates every resumable result, including multi-hour N=21 scans. | Split the binding into a numerics binding (src, cpu_oracle, protocol, fixtures, tolerances, versions, device name/CC, dtypes, kernel path) used by resume, and a full gate binding used by the stamp. |

---

## 4. Requested topic checks

**GPU_test coverage of what production needs.**

Present and meaningful:

- fixture SHA manifest and per-file pins;
- J, diags and shell exact against Numba;
- 324 start vectors bitwise;
- E, F, state (strict, non-phase-aligned), grad vs the oracle, and FD at 5e-7;
- the M6 energy tolerance is now independent;
- parity, residual, and host/device entropy of the stored GS;
- batch-2 E/g vs the oracle;
- synthetic N ∈ {6, 9, 12, 15, 18, 21} with both methods at L=1 (odd N and N=21 covered);
- the full SPSA driver ledger;
- the jobs SPSA path.

Exceptions become FAIL rows, and any FAIL means no stamp. Tolerances are unchanged from Round 1 and none are permissive.

Missing or masked: R2-H4 (BFGS), R2-M9 (invalid-instance eligibility), R2-M10 (entropy, BFGS-pack, resume and pipeline paths), and R2-M11 (SPSA energy alignment).

**Production gate.**

- The binding covers:
  - source pins (gpu package, cpu_oracle, protocol + addendum, scripts, tests, docs, READMEs, requirements, tolerances);
  - every fixture file and the manifest;
  - device name, compute capability and `device_id`;
  - CuPy, CUDA, NumPy and SciPy versions;
  - dtypes.
- `reviews/` is excluded on purpose. That is acceptable, since nothing there is executed.
- Gaps:
  - versions are not enforced against the campaign (R2-M7);
  - numba and python are absent (R2-M7);
  - the kernel path is absent (R2-M2);
  - the per-call cost (R2-M6).
- Integrity and device checks are otherwise sound. `assert_production_allowed` requires `acquire`, float64/complex128 and a PASS verdict.

**New RawKernels.**

- FP64 with `double2` throughout. No float literals or single-precision intrinsics in the kernel sources; `sqrt(2.0)` is double.
- `--fmad=false` is passed.
- APPLYX is fused with ascending q, which gives the same accumulation order as Numba.
- RX component formulas match the CPU (`c·a + s·(−i)d` expanded correctly).
- Walsh parity: R2-M1.
- Stride safety: R2-H1.

**Entropy thread pipeline.**

- Stream safety: correct but implicit, through the legacy null stream (R2-M3).
- Deadlock and progress: R2-H2. The `done.wait/clear` race only adds ≤0.2 s polling delays; it is not a deadlock.
- `--entropy-workers` is a no-op (R2-M4).
- Throughput: the ordering penalty in R2-M5.
- The N=21 exclusive lock (`heavy_entropy`) is correct, but with one worker it is trivially satisfied.

**Memory accounting.** `grad_vector_count = 2L + 14` counts layer inputs, the sph cache, and the psi, Hv, λ, fwd, vA, vC, buffer, phase and temporary vectors. It is consistent with the revised gradients; QAOA allocates fewer.

`_assert_memory` runs before model load and refuses cells that do not fit at batch 2 for SPSA or batch 1 for BFGS. Arithmetic estimate, not measured: at N=21, L=6, batch 2 that is 26 × 32 MiB × 2 × 1.2 ≈ 2.0 GiB plus about 160 MiB of constants.

Not counted:

- concurrent entropy-thread allocations (R2-M3);
- CuPy pool caching across tasks of different N. This is covered by the 0.4 safety factor, but `free_all_blocks()` between N changes is advisable.

**Resume, rounds and summaries.**

- Atomic JSON and npz writes: OK.
- Provenance-checked resume: OK, apart from R2-H3 and R2-L10.
- `official_complete` checks entropy, SPSA and all 16 BFGS starts: OK.
- The round manifest is written and finished: OK, but not on crash (R2-H2).
- The FAIL files use a distinct `_FAIL.json` name, so they are not mislabelled as results.
- Summaries: R2-M8.

**No ansatz, budget or seed changes.**

- `starts.py`, `protocol.py` and `cpu_oracle/src/*` are byte-identical to Round 1.
- SPSA keeps the 1024·L inclusive ledger and host RNG.
- BFGS keeps the same SciPy settings, with `maxiter` only in tests.
- `plan_round` keeps the frozen indices, seeds and depths.

No science change was found.

**Documentation and provenance.** The addendum is honest and checks out:

- The file SHA of `PROTOCOL_FROZEN.json` is `8253fae8…ffba96`.
- Recomputing the embedded hash (sorted compact JSON without the key) gives `f32a96d4…3cbb81`, which matches.
- The `oracle_live` SHAs for hamiltonian, bfgs, entropy and kernels_numba match the bundled files.
- The spsa, protocol and io_util pins match `PROTOCOL_FROZEN.source_pins`.
- The original `starts.py` SHA `0a39d54a…` appears in `PROTOCOL_FROZEN.json` and `GPU_test/fixtures/MANIFEST.json`, and the bundled `03c1d355…` is recorded separately.
- The fixture producer is labelled "not proven", not silently repaired.
- `PROTOCOL_FROZEN.json` is unchanged.

Library pins:

- The claim that these are the CPU campaign's actual versions cannot be verified from the bundle. This review VM is not the campaign machine, and no fixture JSON records versions independently.
- Portability: `numba==0.68.0` and the NumPy/SciPy pins assume a Python matching 3.13. The CuPy wheel must support NumPy 2.5.
- Neither Python nor CuPy is pinned, and nothing enforces the pins (R2-M7). Treat availability on the GPU host as an open risk.

---

## 5. Round 1 items: resolved or unresolved

| R1 ID | Status | Notes |
|---|---|---|
| C1 adjoint λ rebinding | **Resolved** | λ is in place throughout (§1). |
| C2 vA clobbered | **Resolved** | Dedicated vA plus a `fwd` copy. The fallback still clobbers its src, but it is documented and unused by the gradients. |
| H1 RX batch reshape | **Partly resolved** | The Python fallback `(batch,1,1)` is correct. The RawKernel path introduces R2-H1. |
| H2 SPSA host download | **Resolved** | `as_host_array` in `ev_pair`, and host float64 from `energy*`. |
| H3 GPU_test oracle imports | **Resolved** | `oracle.py` with package imports, and `conftest_paths` strips `cpu_oracle/src`. |
| H4 production entropy imports | **Resolved** | ROOT is on `sys.path` in `run_production.py`, with package-qualified imports. |
| H5 NumPy 2 `hasattr(device)` | **Resolved** | `isinstance(cupy.ndarray)`. |
| H6 device selection | **Resolved** | `activate=True` plus `device_context` in dispatch, including the entropy thread. |
| H7 GPU_test coverage | **Partly resolved** | SPSA driver, batch 2, all N and jobs SPSA are added. The BFGS check is masked (R2-H4), and entropy, BFGS-pack, resume and pipeline are untested (R2-M10). |
| M1 gate binding gaps | **Mostly resolved** | Tolerances, fixture files and versions are bound. Numba, Python and the kernel path are missing (R2-M2, R2-M7). |
| M2 valid_for_fits | **Resolved in code** | `instance_meta` defaults to False. The test cannot detect a regression (R2-M9). |
| M3 resume, atomic writes, manifest, summaries | **Partly resolved** | Implemented, but see R2-H3, R2-M6, R2-M8 and R2-L10. |
| M4 provenance addendum | **Resolved** | Verified by hashing (§4). |
| M5 library pins | **Partly resolved** | Pinned in `requirements.txt`, not enforced (R2-M7). |
| M6 GS energy tolerance | **Resolved** | `E_gs_vs_E0` uses only the energy tolerance; the residual is checked separately. |
| L1 benchmark timing API | **Resolved** | `cupy.cuda.get_elapsed_time` via `runtime.elapsed_time_ms`. |
| L2 memory planner | **Resolved** | 2L+14 vectors, enforced by `_assert_memory`. |
| L3 dead code | **Mostly resolved** | New dead items: R2-L3, R2-L5, R2-L8, and the unused `_DIAG_CACHE`. |
| L4 "one optimizer step" | **Resolved** | Replaced by the capped BFGS test, but see R2-H4. The `one_optimizer_step` key in `tolerances.json` is now orphaned. |
| L5 halt on first FAIL | **Resolved** | Continue is now the default. The new halt path hangs (R2-H2). |
| E1 fused per-qubit kernels | **Resolved** | In-place Walsh/RX and fused apply_X. A shared-memory tiling of the low qubits is not done (optional, §6). |
| E2 small-N latency | **Unresolved (by design)** | No explicit opt-in to the identical-math CPU oracle for small N, and no multi-cell concurrency. |
| E3 entropy serialization | **Partly resolved** | One thread, GIL-bound, −N order (R2-M4, R2-M5). |
| E4 `Im(conj(λ)v)` reductions | **Resolved** | |
| E5 sync packing | **Resolved** | One D2H per gradient. Pinned host buffers are not used (minor). |
| E6 sph cache and in-place conj | **Resolved** | `sph_cache` is (L, batch, dim) and `_multiply_inplace` is used. |
| E7 state_err on device | **Resolved** | `kernels.max_abs_diff`. |
| E8 model rebuild and shell temporary | **Partly resolved** | The shell temporary is fixed. The per-task rebuild remains (R2-L2). |
| E9 FMA parity | **Partly resolved** | RawKernels use `--fmad=false`, but the Walsh reciprocal (R2-M1) and CuPy ufunc multiplies (R2-L9) remain. |
| E10 batch-2 vs batch-1 ULP | **Resolved (documented)** | |

---

## 6. Performance assessment (static; no measured claims)

Nothing below was timed. These are operation counts and qualitative expectations only.

- **Per `energy_grad` launch count.**
  - Nov3: about 2·L·N Walsh launches, forward plus backward-recompute plus λ adjoint, so about 3·L·N, plus roughly 10–15 CuPy ufunc and reduction launches per layer.
  - QAOA: about 3·L·N RX launches plus L·(apply_X + ufuncs).
  - At N = 6–12 (dim ≤ 4096), each launch does only a few kilobytes of work, so the cost is dominated by launch and Python overhead and the single sync. These sizes are likely no faster than the Numba oracle and possibly slower (unmeasured).
  - At N = 18–21, each per-qubit pass streams the full state (32 MiB for N=21, read and write). Walsh/RX/apply_X are memory-bandwidth-bound rather than FP64-bound: about 2 flops per 32 B, so weak consumer FP64 matters less here than in the `exp`/`cos`/`sin` diagonal phases.
- **Largest same-math wins still available.** None of these changes the per-element operation sequence, so results stay bitwise identical to the current kernels.
  1. A shared-memory tiled Walsh/RX that applies the ~10 lowest qubits per tile in ascending order, then the high qubits per pass. This cuts full-state passes per transform from N to about N − 9.
  2. Run small-N cells on the identical CPU oracle through an explicit, recorded flag, or run several independent cells concurrently (processes or streams).
  3. A process pool for entropy scans, with ascending-N ordering (R2-M4, R2-M5).
  4. A per-process binding cache (R2-M6).
  5. A diag/shell cache per instance (R2-L2).
- **BFGS** remains serial and batch 1 on the host. That is inherent to "same SciPy optimizer". Concurrency across starts or cells is the only lever that does not change semantics.
- **Transfers** are minimal and well placed: x H2D, one packed D2H per gradient, and one per SPSA batch-2 energy. There are no per-layer transfers.

---

## 7. Remaining blockers before production (explicit)

1. Fix R2-H1 (contiguous β plus contiguity assertions), R2-H2 (thread `finally`, liveness-based exit, gate inside try), R2-H3 (acquire before the ordering guard) and R2-H4 (real BFGS criteria).
2. Close the gaps in R2-M7 (version enforcement), R2-M9 and R2-M10 (eligibility, entropy, pack, resume and pipeline tests) and R2-M1 (Walsh division). Each changes either the gate's meaning or a PASS-relevant parity claim.
3. Run `scripts/run_gpu_test.py` on the target GPU with the pinned stack and obtain PASS. Every fix above changes source pins, so the stamp must be regenerated after the last edit.
4. Confirm that numpy 2.5.4, scipy 1.18.1 and numba 0.68.0 are installable with the GPU host's Python and the CuPy wheel. This review could not verify it.
5. Smoke-test the production pipeline on a scratch round at N=6 with `--limit`, including one injected entropy FAIL under both halt modes. This must be done on hardware; it has never been executed.
6. Unvalidated assumptions that the hardware must confirm:
   - FP64 performance of the target device;
   - norm stability below 1e-12 at N=21, L=6;
   - NVRTC honouring `--fmad=false` for these kernels;
   - CuPy RawKernel behaviour with the pinned CuPy version.

## 8. CUDA-untested limitations (restated)

- Nothing in this review executed CUDA, CuPy, NVRTC, Numba or SciPy.
- The correctness statements are static derivations from source plus documented library semantics.
- The kernel index bijection, adjoint signs and op order were checked by reading, not by running.
- Any of them can still be wrong in ways only a device run reveals: compiler behaviour, CuPy argument marshalling, pool behaviour under threads.
- No speedup or timing is claimed.
- **Static review only, not CUDA approval.**
