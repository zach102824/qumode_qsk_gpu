# Implementation notes (after review rounds 1 and 2)

See `reviews/ROUND1.md` (verbatim Opus report) and `reviews/ROUND1_FIXLOG.md`
for the issue-by-issue map. CUDA has still not been executed here.

## Why CuPy

The CPU campaign is NumPy + Numba with explicit complex128 butterflies and
a hand-written reverse-mode adjoint, plus SciPy BFGS and a NumPy RNG. CuPy
is the closest array API: we can keep the control flow, stay on float64 /
complex128, and upload only constants and a few scalars per SciPy
evaluation. PyTorch was not chosen because autocast, default dtypes, and
complex autograd conventions are easy to get scientifically wrong, and we
do not want a custom autograd graph to replace the audited adjoint.

## Mapping from CPU kernels

| CPU (Numba) | GPU |
|-------------|-----|
| `walsh_inplace_c` / `walsh_c` | default: `walsh_inplace` RawKernel (one launch / qubit, thread owns a pair, `--fmad=false`); fallback `walsh_pingpong` (src clobbered unless `preserve_src`) |
| `rx_inplace_c` / `rx_c` | same pattern; batched `c,s` reshaped `(batch, 1, 1)` |
| `apply_X_{c,r}` | fused RawKernel, or Python loop into a separate accumulator |
| `apply_H_*`, `energy_c` | `D0*v - B*apply_X(v)`, `Re(sum conj(v)*Hv)` |
| `zz_diag`, `shell_ops` | host pair-loop / group-loop NumPy, uploaded once |
| `nov3_energy_grad`, `qaoa_energy_grad` | `gradients.py`: in-place `λ ← λ * conj(phase)`; dedicated `vA`; one `pack_to_host` of (norm, E, g) |
| `spsa.optimize` | same ledger; energy on GPU; ± probes batched; host via `as_host_array` |
| `bfgs.run_start` | same SciPy options; `fun` downloads E,g |
| `entropy.scan_bstar` | delegated to cpu_oracle (default) |

## Device / dtype hygiene (ROUND 1)

- `runtime.acquire(..., activate=False)` by default; production and tests
  pass `activate=True` → `Device(k).use()`.
- `is_device_array` is `isinstance(..., cupy.ndarray)`, not `hasattr(..., "device")`.
- `as_host_array` is the only CuPy→NumPy bridge used by SPSA / energy / uploads.
- Oracle imports are `from cpu_oracle.src import ...`. `cpu_oracle/src` is
  stripped from `sys.path`.

## GPU_test fixtures

Copied, not recomputed, from the live MAIN campaign (round-1 indices).
See `GPU_test/README.md` and `GPU_test/fixtures/MANIFEST.json`. Producer
source is not proven from the result JSONs; see
`protocol/PROVENANCE_ADDENDUM.json`.

## Production gate

`gate.assert_production_allowed()` checks a PASS stamp against current
source pins, every fixture file, tolerances, device name, dtypes, and
NumPy/SciPy/CuPy/CUDA versions. Jobs write only under this repository,
atomically (JSON and reference NPZ), and resume only with matching
provenance plus N/seed/L/method. `reviews/` is not part of the source pin
set.

## What reviewers should still stress (round 2, on the revised tree)

1. In-place RawKernel pair ownership vs the Python ping-pong fallback,
   especially odd N (9, 15, 21) in the adjoint.
2. That `vA` is never used as a Walsh/RX source, and `λ` stays in one
   allocation after the conjugate-phase multiply.
3. Shared Nov3 parameter accumulation vs a single-layer check.
4. Energy-only vs energy-grad consistency, including batch≥2 vs batch=1
   ULP differences.
5. Memory plan vs actual allocations at N=18/21, including the entropy
   thread overlapping a GPU cell.
6. Whether SciPy BFGS + GPU `fun` should cache-avoid or re-evaluate more
   conservatively (current code matches the CPU recompute-at-the-end).
7. That GPU_test actually exercises SPSA / capped BFGS / jobs, and that
   a FAIL cannot be stamped as PASS. AST review still cannot prove CUDA.


## After review round 2 (not itself reviewed)

Code in this tree was edited after the Round 2 CHANGES NEEDED report. There
is no third review. CUDA has still not been executed here.

- `_beta_device` / `prepare_beta` copy strided β to C-contiguous storage;
  RawKernel launches assert contiguity (R2-H1).
- `pipeline.run_pipeline` is liveness-based, host-only entropy, optional
  spawn pool, small-N first, finalize/apply_H on the main thread (R2-H2,
  R2-M3, R2-M4, R2-M5).
- Production acquires and binds before `official_complete` (R2-H3).
  `provenance_fields()` raises if not acquired.
- GPU_test capped BFGS uses documented numerical tolerances (R2-H4).
- Binding is cached; kernel path and Python/Numba versions are bound;
  pins are enforced (R2-M2, R2-M6, R2-M7).
- Shared-memory tiling was not added.
