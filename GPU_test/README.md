# GPU_test

Immutable copies of a few **current CPU-campaign** N=6 and N=15 MAIN results,
plus the tests that compare a GPU run against them.

Nothing in this folder was recomputed when the GPU tree was created. Ground
states, entropy curves, SPSA/BFGS parameter vectors, energies and fidelities
are copies of the original CPU-campaign files (paths recorded in the manifest). Original paths and SHA-256
digests are in `fixtures/MANIFEST.json`.

## What is in the fixtures

| N | seeds (MAIN index) | depths | content |
|---|--------------------|--------|---------|
| 6 | 831106000 (idx 0) | L=1,3,6 | instance, entropy scan, GS npz, SPSA + 16 BFGS starts, both ansatz |
| 6 | 831106001 (idx 1) | L=1,6 | same |
| 15 | 831115000 (idx 0) | L=1,5 | same |
| 15 | 831115005 (idx 5) | L=1,5 | same |

`fixtures/validation_catalog.json` lists **explicit parameter vectors** (not
seed-only): SPSA best `x`, SPSA initial points, BFGS start `x` and final `x`,
with the CPU-stored E, F, norms, start SHAs and RNG seeds.

## First command on the GPU host

```bash
# from the qumode_qsk_gpu root, after installing a matching CuPy wheel
python3 scripts/run_gpu_test.py --import-only
python3 scripts/run_gpu_test.py --host-only
python3 scripts/run_gpu_test.py --device-id 0
```

The runner first checks host start recipes / oracle light tests, then the
CUDA suite: fixture SHA/J/GS, stored E/F and residual (energy tolerance is
not OR-ed with residual), oracle + FD gradients, batch≥2, preserve_src /
vA, a full SPSA optimize (direction hashes / initial points / query roles
/ calls are PASS; best_E and x are report-only), capped BFGS (maxiter=3)
versus the CPU oracle, a jobs-path SPSA write on scratch, and a structural
check at official N including odd N (9, 15, 21). AST review cannot prove
these. A PASS writes `stamps/GPU_TEST_PASS.json`, bound to source pins,
every fixture file, tolerances, device name, dtypes, and library versions.
Production refuses to start without that stamp.

Import-only (no device):

```bash
python3 scripts/run_gpu_test.py --import-only
python3 scripts/static_check.py
```

See the root README for tolerances, benchmarks, and the production dry-run.
