# Known limitations and correctness concerns

**Review status.** Round 1 and Round 2 both returned CHANGES NEEDED (static
review only, not CUDA approval). The code in this tree was edited after
review 2 and has **not** had a third review. Static analysis still cannot
prove CUDA correctness. No PASS stamp is shipped.

This document is part of the review packet. Nothing here is hidden in the
code comments alone. GPU numerics have **not** been validated in the
authoring environment (no CUDA). Static analysis and AST review cannot
prove CUDA correctness. Do not treat a source-tree SHA, a fixture SHA, or
a clean `scripts/static_check.py` as a PASS.

## Not CUDA-validated

All GPU kernels, reverse-mode gradients, and optimizer drivers were written
to match the audited CPU source by inspection and were revised after the
ROUND 1 and ROUND 2 reviews (see `reviews/ROUND1_FIXLOG.md` and `reviews/ROUND2_FIXLOG.md`). Until
`scripts/run_gpu_test.py` PASSes on a real device, treat the port as
unproven. This tree never writes a PASS stamp in the authoring environment
and never fabricates a performance number.

## Hardware FP64

The implementation is float64 / complex128 only. Many consumer NVIDIA parts
have weak FP64 throughput. Correctness may still hold while performance is
poor. No speedup is claimed. If a device is so weak that norms drift above
the campaign assertion `||ψ||−1 < 1e-12`, GPU_test must FAIL — the production
path does not silently loosen that bound.

## Kernels: in-place RawKernel default, ping-pong fallback

The default Walsh / RX path (when RawKernels compile) is **in-place**: one
launch per qubit, each thread owns a unique pair, so there is no butterfly
aliasing inside a launch. `--fmad=false` is passed at compile time.

If RawKernels are unavailable, a Python ping-pong fallback is used. In that
path **the source buffer is clobbered** unless `preserve_src=True` (which
copies first). Adjoint code never uses the pre-transform vector `vA` as a
Walsh/RX source; it copies into a `fwd` buffer first. `apply_X` still
accumulates into a third buffer (it is a sum over qubits, not a swap).

Stock CuPy elementwise / reductions may still FMA-contract. libm
exp/cos/sin and reductions can differ from Numba at ULP level. Batch-2 vs
batch-1 energy of the same `x` can also differ at ULP level. Bitwise CPU
parity is not promised.

## Endian, Walsh normalization, RX angle

- Qubit 0 is the least-significant bit (stride 1), same as the Numba kernels.
- Each Walsh qubit applies `(a±b)/√2` (unitary, factor 1/√2 per qubit).
- RX(β) uses `cos(β)` and `-i sin(β)`, i.e. textbook RX(2β). This is frozen;
  do not “fix” the factor of two.
- Batched RX coefficients are reshaped `(batch, 1, 1)` to match the 3-D
  slice `(batch, pairs, 2)`. A `(B, 1, 1, 1)` reshape was a ROUND 1 bug.

## Gradient convention

Reverse-mode Euclidean gradient `2 Re(λ† ∂_θ U ξ)` with `λ` back-propagated
by the adjoint (Walsh is self-adjoint; RX(β)† = RX(−β); diagonal phases use
the conjugate, applied **in place** on `λ` so the ping-pong pair is not
abandoned). Shared Nov3 parameters (γ0, α1, α2) are **summed** over
layers. States are not renormalized mid-circuit; a norm assertion is applied
at the end. GPU_test compares against the cpu_oracle reverse-mode and
against central differences (ε=1e−7), and also runs a structural batch≥2
and odd-N check. Those checks still have to be executed on a GPU.

## BFGS trajectories are not bitwise

SciPy BFGS runs on the host and sees GPU energies/gradients. After tiny
(1e−14…1e−12) differences, line search and Wolfe tests can take a different
path than the CPU campaign. GPU_test compares **E and g at stored CPU
parameter vectors** and at frozen starts. A **capped** BFGS run
(`maxiter=3`) is executed against the CPU oracle for the jobs/driver path;
full 16-start replay identity (`nit`, `nfev`, final `x`) is **not** a PASS
condition.

## SPSA ledger vs energy noise

The host RNG and the inclusive 1024L ledger are identical. The ± pair of
each update is batched (two states). If GPU energies differ from CPU
energies even slightly, the *update path* and the selected `best_x` can
drift even when direction hashes still match. GPU_test runs a full
`spsa.optimize` on fixture cells: `direction_hashes`, `initial_points`,
`query_roles`, and `calls` are PASS criteria; `best_E` / `x` are
report-only.

## Entropy eigensolver is not all-GPU

Default B* scan and ground-state extraction use
`cpu_oracle.src.entropy.scan_bstar` (SciPy). Degeneracy, even-branch gauge,
residual, and the official peak rule stay on that path. Do not claim the
scan runs on the GPU. Optional device SVD of the Schmidt matrix is only
for \(S(\psi)\) given a state. Eigenvectors from any future opt-in GPU
solver will not, in general, match CPU vectors (phase, degeneracy).

0 log 0 is avoided by dropping Schmidt weights `≤ 1e-15`, same as the CPU
code. The partition is logical low bits, not a cavity cut, and not a
C-order `reshape` that is wrong for odd N.

Production can overlap **one** CPU entropy thread with GPU optimizer tasks
on the main thread (`scripts/run_production.py`). N≥21 scans never run
concurrently with another scan. This overlap is memory-safety limited and
has not been executed here.

There is **no silent “send all small N to CPU” routing**. An explicit
future flag could opt in; it is not the default.

## Memory and micro-batching

N=21: one complex128 vector is 32 MiB. Reverse mode stores L layer inputs
plus adjoints and workspace. `memory.max_batch` applies a 0.4 safety
factor; production consults it (SPSA requested batch 2, BFGS batch 1) and
refuses to start a cell that does not fit. The SciPy BFGS driver still
evaluates **one** start at a time.

## Import-time and device policy

`runtime.acquire()` is explicit and defaults to `activate=False`.
Production, GPU_test, and the benchmark pass `activate=True`, which calls
`Device(k).use()` so later allocations land on k. The library does not
pick device 0 at import. Device arrays are detected with
`isinstance(..., cupy.ndarray)`, never `hasattr(x, "device")` (true for
NumPy 2 ndarrays). Host conversions go through `runtime.as_host_array`
(never `np.asarray` on a CuPy array).

## Host constructors

`zz_diag` is a pair-loop over `i<j`, matching the Numba loop (an einsum
`0.5 zᵀJz` form differed by 1 ULP on fixture J). `shell_ops` loops the
frozen triplet groups (no `(dim, n/3, 3)` temporary).
`scripts/static_check.py` does **not** re-run that comparison against
Numba JIT on the campaign machine. `GPU_test/tests/test_host_oracle_light.py`
does, and should be run on the GPU host as a cheap extra check (it is
invoked by `scripts/run_gpu_test.py` before the CUDA tests).

## Protocol hashes and provenance (do not edit PROTOCOL_FROZEN.json)

Two hashes must not be conflated:

- **File SHA-256** of `protocol/PROTOCOL_FROZEN.json` (exact bytes):
  `8253fae8dcfc4f8242e980f446d9e8073364580e9fe9901b17060a7b69ffba96`
  (116020 bytes).
- **Embedded** `protocol_sha256` (SHA-256 of the sorted compact JSON with
  that key removed):
  `f32a96d4af337525edbbeea903b848d9b8c6163af3efa214f4ca71cd143cbb81`.

`cpu_oracle/src/starts.py` is a write-guarded copy, not byte-identical to
the campaign file (only `pack_panel_starts` gained a required `out_dir`).
Both SHAs are recorded in `protocol/PROVENANCE_ADDENDUM.json`.

`PROTOCOL_FROZEN.json` `source_pins` predate later audited campaign edits
(`hamiltonian.py`, `bfgs.py`, `entropy.py`; `kernels_numba.py` was never
pinned). The gpu `cpu_oracle` snapshot is the **live** audited CPU code as
of the tree build. That mismatch is documented; the frozen protocol file
is not rewritten.

Fixture run JSONs do not prove which `hamiltonian.py` produced each
trajectory. The producer is labelled unknown in the addendum; fixture
history is not silently repaired. Stored E/F at a stored `x` still depend
only on the state map.

Campaign library versions on the authoring box (pin these on the GPU
host): Python 3.13.5, numpy 2.5.4, scipy 1.18.1, numba 0.68.0.

## Production stamp

`GPU_TEST_PASS` binds source pins (gpu package, cpu_oracle, protocol
including the addendum, scripts, tests, docs, README, requirements,
tolerances — **not** `reviews/`), SHA of every fixture file, the fixture
manifest, tolerances SHA, device name / compute capability, dtypes, and
NumPy / SciPy / CuPy / CUDA versions. Resume of a result file also
requires those provenance fields plus N/seed/L/method (and SPSA
`calls == 1024 L`). Reference NPZ files are written atomically
(tmp + fsync + replace).

`scripts/benchmark.py` is fail-closed on the same stamp unless
`--allow-unstamped` is passed; unstamped output is labelled unvalidated.
Times are end-to-end CUDA-event averages (they include host syncs in
`energy_grad`). No speedup versus the CPU campaign is printed.

## Things this port deliberately does not do

- No GNN, no custom optimizer, no SPSA warm start, no layer transfer, no
  half-variance, no seed replacement, no mid-job SIGSTOP of other work.
- No writes into `/workspace/qumode_qsk`.
- No claimed measured speedup and no fabricated PASS.
- No automatic production launch.
- No new numerical experiments in the authoring environment.


## After review round 2 (code edited after the review)

- Batched QAOA β is `ascontiguousarray`'d and RawKernel launches assert
  C-contiguity (R2-H1). Untested on a device here.
- The production pipeline exits on liveness, joins in-flight entropy
  (including N=21) on halt, and records gate exceptions as FAIL (R2-H2).
  Host-only tests cover injected FAIL under both halt modes; the real
  production loop has not been executed.
- `runtime.acquire()` and the production gate run **before**
  `official_complete(r-1)` (R2-H3). Dry-run still does not acquire CUDA.
- Capped BFGS (`maxiter=3` and `maxiter=1`) is a real numerical PASS check
  against the CPU oracle (R2-H4). Untested on a device here.
- Walsh RawKernels divide by `sqrt(2.0)` like Numba. Other ULP sources
  remain (libm, CuPy ufuncs/reductions). Do not claim bitwise Walsh.
- RawKernels are force-compiled inside the acquire guard. The stamp binds
  `kernel_path` and rejects a mixed raw/python path.
- Entropy workers are host-only. `--entropy-workers N` is a spawn process
  pool for `N>1` (BLAS threads 1, at most one N=21 scan). Default is 1
  host thread overlapping GPU finalize/optimize. Small-N scans start first.
- Library pins (Python 3.13.5, numpy 2.5.4, scipy 1.18.1, numba 0.68.0)
  are enforced; `--allow-version-drift` marks the stamp and results unofficial.
- The binding is cached per process and re-verified at stamp write and at
  round start/finish.
- Summaries use provenance-checked records only, report `valid_only` and
  `all` means, and list incomplete BFGS cells.
- Shared-memory tiled Walsh/RX was **not** added (new high-risk kernel,
  no GPU test in this environment).
- Resume still uses the full gate binding (a docs typo or a move to GPU 1
  invalidates long N=21 results). Splitting a numerics-only resume binding
  was left as a low item (R2-L10).
- CuPy wheel installability with NumPy 2.5.4 on the target host was not
  verified here.
