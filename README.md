# qumode_qsk_gpu

A self-contained GPU port of the quantum-SK Nov3 vs QAOA campaign. It must
not write to, signal, or reschedule the original CPU campaign tree.

**This snapshot is code plus immutable CPU fixtures. CUDA has not been
executed in the authoring environment.** Treat every numerical claim as
unvalidated until `scripts/run_gpu_test.py` PASSes on your GPU. Static
analysis cannot prove CUDA correctness. Both review rounds returned
CHANGES NEEDED; this tree was edited after review 2 and has not had a
third review.

## Quick start (GPU host)

Do this first, before any production run.

```bash
# 1. Inspect the toolkit
nvidia-smi
nvcc --version   # if present

# 2. Python 3.13.5 plus the pinned scientific stack
python3 --version    # expect 3.13.5
python3 -m pip install -r requirements.txt
# pins: numpy==2.5.4 scipy==1.18.1 numba==0.68.0

# 3. Exactly one CuPy wheel, matching the toolkit
python3 -m pip install cupy-cuda11x   # CUDA 11.x
python3 -m pip install cupy-cuda12x   # CUDA 12.x
python3 -m pip install cupy-cuda13x   # CUDA 13.x (if that wheel exists for your CuPy)
```

If a wheel name is rejected, use the current CuPy install table rather than
forcing an unmatched build. Confirm `import cupy; cupy.cuda.runtime.runtimeGetVersion()`.

These pip commands were not run in the authoring environment (no CUDA here).

### First tests (required)

From the `qumode_qsk_gpu` root, after you have copied this folder onto the
GPU host:

```bash
# Syntax / import safety (also works without a GPU)
python3 scripts/static_check.py
python3 scripts/run_gpu_test.py --import-only

# Host contracts only (no CUDA): strided β, pipeline halt, eligibility
python3 scripts/run_gpu_test.py --host-only

# Full GPU_test vs the copied N=6 and N=15 CPU fixtures
python3 scripts/run_gpu_test.py --device-id 0
```

A PASS writes `stamps/GPU_TEST_PASS.json`, bound to:

- SHA-256 of GPU sources, scripts, tests, protocol (including
  `PROVENANCE_ADDENDUM.json`), docs, README, requirements, and the
  cpu_oracle snapshot (`reviews/` is excluded)
- SHA-256 of `GPU_test/fixtures/MANIFEST.json` **and of every fixture file**
- SHA-256 of `GPU_test/tolerances.json`
- the live kernel path (`raw` or `python` fallback) and `--fmad=false`
- the CUDA device **name** and compute capability
- dtypes `float64` / `complex128`
- Python, NumPy, SciPy, Numba, CuPy, and CUDA runtime versions

If the host Python/NumPy/SciPy/Numba versions are not exactly the campaign
pins, GPU_test refuses unless you pass `--allow-version-drift` (the stamp
and all results are then unofficial).

If you change code, fixtures, tolerances, libraries, the kernel path, or
move to another GPU, the stamp will not match and production stays closed
until you re-run GPU_test.

A PASS is necessary for production. It does **not** prove bitwise identity
with the CPU campaign, and it is not a speedup claim. AST/static checks
cannot prove CUDA correctness. This tree does not ship a PASS stamp.

### Production (only after GPU_TEST_PASS on this machine)

```bash
# Plan official round 1. No CUDA, no writes.
python3 scripts/run_production.py --dry-run --round 1

# Only after GPU_test PASS on this machine, start work deliberately:
python3 scripts/run_production.py --run --round 1 --device-id 0
```

There is no auto-start, no implicit `--run`, and no background scheduler.
Results go to `runs/`, `data/`, and `logs/` with atomic JSON/NPZ replace
and provenance-checked resume. Official rounds stay 10-per-N until you
later add the next 10, up to the cap of 200. Entropy eigensolves stay on
the CPU oracle (process pool with `--entropy-workers`, default 1; workers
never use CUDA). Ansatz energy and gradients use the GPU.

Do not run production on a machine that is already using its cores for a
live CPU campaign. This package is meant to be used after you transfer the
folder to a GPU host.

To refuse writes into a CPU campaign tree that is not a sibling named
`qumode_qsk`, set `QUMODE_QSK_CPU_CAMPAIGN` to that tree's path.

## What is frozen (unchanged physics)

- Hamiltonian \(H=\sum_{i<j}J_{ij}Z_iZ_j - B\sum_i X_i\) with \(J_{ij}\sim\mathcal N(0,1/N)\) for \(i<j\), then symmetrized (variance **1/N**, not 1/(2N)).
- Per-instance finite-N ground-state von Neumann entropy peak on \(A=\) the lowest \(\lfloor N/2\rfloor\) logical spins (natural log).
- Initial state \(|+\rangle^{\otimes N}\). Exact statevector energy, no shots.
- Nov3: \(3+3L\) parameters. Shared \((\gamma_0,\alpha_1,\alpha_2)\) generate a diagonal from \((J,J^2,M)\) (frozen jrms normalization); each layer is that shared phase, Walsh (Hadamard), a 3-parameter shell phase, and a final Walsh. Shared parameters are **summed** over layers in the reverse-mode gradient.
- QAOA: \(p=L\), \(2L\) parameters, cost phase then RX(\(\beta\)) per layer. RX uses the frozen CPU convention \(c=\cos\beta\), \(s=-i\sin\beta\) (textbook RX(\(2\beta\))).
- SPSA: two cold starts (narrow 0.16, broad 0.6), inclusive budget \(Q=1024L\), R9 query ledger, host `numpy.random.default_rng`, energy selection, no warm start, no layer transfer.
- BFGS: 16 cold starts (8 broad even index, 8 narrow odd), SciPy `minimize(..., method="BFGS", jac=True)` on the **host**, `maxiter=2000`, `gtol=1e-10`, infinite-norm, energy selection.
- Official work plan: N = 6,9,12,15,18,21, depths L=1..6, cap 200 instances, official summaries in balanced **10-per-N rounds**.

The audited CPU source (Numba kernels + reverse-mode gradients) is snapshotted
in `cpu_oracle/` for user-side comparisons.

## Backend

CuPy, states as **`[batch, 2**N]` complex128**, float64 everywhere else.
No autocast, no implicit float32, no dense \(2^N\times 2^N\) operators.
Default Walsh / RX are in-place RawKernels (`--fmad=false`), compiled
eagerly after `acquire()`, with a Python ping-pong fallback. Batched QAOA
β is made C-contiguous before the RawKernel (strided `xd[:, column]`
views are copied). Walsh uses division by \(\sqrt{2}\), matching the
Numba loops; this is **not** a bitwise-identity claim (other ULP sources
remain).

The default **entropy-peak eigensolver is still the CPU SciPy path** in the
oracle. A GPU eigensolver is **not** claimed. Given a statevector, the
Schmidt SVD for \(S(\psi)\) *may* run on the device.

## Benchmarks (optional, no speedup claim)

```bash
python3 scripts/benchmark.py --N 15 --L 3 --method nov3 --batch 1 --device-id 0
```

Fail-closed without a matching `GPU_TEST_PASS` unless `--allow-unstamped`.
Times are CUDA-event **end-to-end** averages per eval. This repository does
**not** advertise a measured speedup.

## Layout

```
qumode_qsk_gpu/
  src/qumode_qsk_gpu/     GPU library (lazy CUDA, no import-time allocation)
  cpu_oracle/             frozen independent copy of the audited CPU physics
  protocol/               PROTOCOL_FROZEN.json, SEED_MAP.json, PROVENANCE_ADDENDUM.json
                          (the frozen protocol file is never edited)
  GPU_test/               fixtures, tolerances, tests
  scripts/                run_gpu_test, benchmark, run_production, static_check
  docs/LIMITATIONS.md     known correctness and performance caveats
  reviews/                ROUND 1 and ROUND 2 reports (verbatim) and fix logs
  stamps/                 GPU_TEST_PASS (created on the GPU host; not shipped)
  runs/, data/, logs/     GPU production outputs only
```

## Safety

- Importing `qumode_qsk_gpu` does not initialize CUDA, allocate device memory, or change the current device. Call `runtime.acquire()` explicitly.
- Writers refuse paths under a sibling `qumode_qsk` tree or `$QUMODE_QSK_CPU_CAMPAIGN`.
- Production is fail-closed until GPU_test PASSes for this source, fixture files, tolerances, kernel path, device, dtype, and library versions.

---

## Published snapshot (2026-11-11 12:09:53 +08 Asia/Shanghai, UTC+8)

This GitHub repository is a portable copy of the local `qumode_qsk_gpu` tree,
including `GPU_test` fixtures for **N=6 and N=15**, protocol files, source,
and both review reports plus fix logs (`reviews/ROUND1.md`, `ROUND2.md`,
`ROUND1_FIXLOG.md`, `ROUND2_FIXLOG.md`).

**CUDA has not been validated in the authoring environment.** Both review
rounds (Opus 5.5 medium) returned **CHANGES NEEDED**. Code was edited after
review 2; there was no third review. No `GPU_TEST_PASS` stamp is shipped.
A clean static check is not a substitute for `scripts/run_gpu_test.py` on a
real GPU.

See `PUBLISH_SNAPSHOT.json` for the snapshot inventory. Empty `logs/`,
`runs/`, `stamps/`, and `summaries/` directories are placeholders only.

Follow the quick start above: install the pinned stack, run GPU_test first,
and start production only after a PASS on that machine.

## Local approved GPU task queue

See [Local GPU runner](docs/LOCAL_GPU_RUNNER.md) for owner approval, start/stop
commands, isolated artifacts, and the offline one-task safety test. The runner
supports validation, production planning, and gate verification only.
