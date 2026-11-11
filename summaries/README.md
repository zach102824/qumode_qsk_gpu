# Production result snapshots

Latest snapshot: [20261112_075219](progress_20261112_075219/snapshot.json), local China time.

| N | Fully completed instances |
|---:|---:|
| 6 | 30 |
| 9 | 30 |
| 12 | 30 |
| 15 | 30 |
| 18 | 25 |
| 21 | 20 |

- [Means, sample standard deviations and standard errors](progress_20261112_075219/depth_summary_uncertainty.csv)
- [Per-instance selected optimizer results](progress_20261112_075219/results.csv)
- [Entropy peaks and reference metadata](progress_20261112_075219/instances.csv)
- [Completed N=24 CPU entropy references](progress_20261112_075219/N24_entropy_references.csv)

The comparison uses Nov3 and QAOA, depths 1 through 6, SPSA with the existing
query budgets, and lowest-energy selection among 16 cold BFGS starts. Fidelity
is measured after energy selection. Only fully completed instances enter these
tables; unfinished instances are excluded. SEM = sample SD / sqrt(instance count).
Errors do not include field-grid systematic uncertainty.

New scans use 0.01, 0.06, ..., 2.96, 3.00 (61 fields), without refinement or
boundary extension. The first 10 N=6 and N=9 instances retain the original
refined scans, so those sizes mix two field-selection policies. All new N=24
work is CPU-only precomputation for future comparisons, with no N=24 GPU
optimization claim. 1 N=24 reference is complete in this snapshot.

Dual-device GPU validation reports are included with the latest snapshot.
They describe the tested source hashes, libraries and physical devices; a
checkout on another machine must pass its own validation before production.
See [dual-GPU operation](../docs/DUAL_GPU.md) for scheduler and resume details.

Raw runs, statevectors, checkpoints, logs, stamps and validation scratch stay
on the production host. Historical result snapshots remain in dated folders.
Production work was active when this snapshot was exported.
