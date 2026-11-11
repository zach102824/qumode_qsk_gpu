# CPU oracle (frozen independent snapshot)

This directory is a **read-only scientific snapshot** of the audited CPU
implementation in `/workspace/qumode_qsk/src`, copied when
`qumode_qsk_gpu` was created. It exists so GPU tests can compare against the
original Numba / reverse-mode CPU kernels **without importing or writing to
the live CPU campaign tree**.

Included modules only (physics and optimizer recipes):

- `src/hamiltonian.py`
- `src/kernels_numba.py`
- `src/entropy.py`
- `src/spsa.py`
- `src/bfgs.py`
- `src/starts.py` (write-guarded: `pack_panel_starts` requires an explicit
  `out_dir` and will not default to the campaign `data/` tree)
- `src/protocol.py`
- `src/io_util.py`

Not included: scheduler, rounds, main_jobs, resources, campaign `paths.py`.
Those modules could touch the live campaign; they are intentionally absent.

SHAs and original absolute paths are recorded in
`SOURCE_SNAPSHOT.json` and `GPU_test/fixtures/MANIFEST.json`.

The live campaign may still be running. Do not copy this oracle back over
`/workspace/qumode_qsk`, and do not point this package at campaign output
directories.
