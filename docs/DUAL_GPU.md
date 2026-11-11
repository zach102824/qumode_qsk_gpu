# Dual RTX 5090 production

New entropy scans use exactly 61 fields for every N: 0.01, 0.06, ..., 2.96,
plus 3.00. There is no fine scan or boundary extension. Boundary peaks remain
unresolved under the existing eligibility rules. Reports record `initial_61_only`.
Historical refined scans and fixtures retain their original provenance.
After this policy change, revalidate both devices; old physics bindings cannot
resume into the new policy. To rerun N=12 and continue larger sizes while leaving
historical N=6 and N=9 results in place, launch with `--min-n 12`.

The scheduler has no automatic production start. Inspect a round without CUDA:

```sh
.venv/bin/python scripts/dual_gpu.py plan --round 1
```

Validate both physical devices separately (sequential validation, no production):

```sh
.venv/bin/python scripts/dual_gpu.py validate
```

Each validation uses new scratch directories and session logs/reports. Device stamps
live under `runs/dual_gpu/validation/gpu0` and `gpu1`; replaced stamps are archived
in their `history` directories. The legacy stamp and validation scratch are untouched.
Any source change requires fresh validation. A failed or unofficial validation
cannot authorize official production. Device IDs are explicitly compared.

Launch both GPUs for one official round, in the foreground:

```sh
.venv/bin/python scripts/dual_gpu.py launch --round 1
```

Monitor from another terminal:

```sh
.venv/bin/python scripts/dual_gpu.py monitor
```

Monitor reports current instance, completed instances, failures, process liveness,
and child exit codes. Detailed stdout/stderr lives in each session's `gpu0.log` and
`gpu1.log`. A nonzero child exit makes the launcher fail. Interrupted workers can
be retried with the same launch command. Workers retain device locks if the
coordinator dies; another worker cannot take the same GPU until it exits.

Instance index modulo two owns the instance, for all six N values. Each GPU runs
all L=1–6 and both methods for its assigned instances using the original job
implementations, seeds, optimizers and float64/complex128 arithmetic. One global
CPU scan lock serializes entropy eigensolvers (including N=21); BLAS, OpenMP,
Numba, BLIS and NumExpr thread counts are set to one before worker imports.
There are no per-GPU CPU pools. GPU optimization can overlap the other GPU's scan.

Outputs are under `runs/dual_gpu/instances/N*_seed*/attempt-*`. Retries create new
attempts, preserve old files, and copy hash-verified checkpoints. Completed attempts
are reused only with the same gate provenance and all recorded file hashes intact.
BFGS starts checkpoint individually through the existing pack writer and are sealed
also on a reported pack failure. An abrupt kill during an unsealed task may cause
that task to be recomputed; no completed sealed instance is recomputed. A host scan
receipt permits recovery after scan completion but before GPU finalization.

Legacy results are imported by copying only results accepted by the existing
provenance validator, with reference SHA and ground-state SHA checks. Legacy resume additionally accepts a preserved official validation stamp when
all physics source hashes (including optimizer, entropy, protocol, runtime, kernels
and CPU oracle), libraries, fixtures, tolerances, precision, and kernel settings
match. Only orchestration, gate, tests and documentation may differ. Device IDs
may differ after both GPUs pass separate validation. Reused records retain their
original provenance; accepted bindings are recorded per session. Changes to physics
source invalidate resume. Existing data and the legacy single-GPU
manifest/summary files are never updated by this scheduler. Per-attempt receipts
are the dual-GPU completion manifests. An instance count of 30 per device means a
complete official round. The dual scheduler accepts round selection explicitly;
it does not require earlier rounds to be complete.

The legacy CLI and dual coordinator share a campaign lock. Stop any already-running
legacy runner launched before this implementation before using the scheduler.
Do not invoke job writers directly against scheduler attempt directories. Locks
use Linux flock on this host; directories must remain on a filesystem supporting it.

Host scheduler tests:

```sh
.venv/bin/python -m unittest discover -s tests -p test_dual_gpu.py -v
```
