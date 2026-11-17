# Nov17 folder audit

The current baseline is internally consistent after the fixes below. This audit checks numerical implementation and retained artifacts; it does not establish experimentally calibrated device parameters or globally optimal pulses.

## Fixes made

- Bundled the 20 reference-state snapshots in this folder, with original run paths and hashes, so a GitHub checkout does not depend on ignored local run outputs.

- Fixed division by zero when the off-diagonal J-squared template K vanishes. Zero-coupling and single-edge circuits now agree with the CPU oracle.
- Added `validation_cutoff: 24` to the coherent-H pulse records. Their `cutoff: 16` identifies the optimization cutoff; their stored metrics reproduce exactly at 24 levels.
- Recovered missing `local_chi_time` metadata in older hardware records from their recorded H-stage durations. Fidelity values and waveforms were preserved.
- Made the trajectory default jump step use the configured 0.005 us value. Full-circuit replay uses the configured cutoff/step and now defaults to the shorter tested H, chi-time 48.
- Added early rejection of mismatched segment/duration arrays, nonpositive durations or timesteps, invalid sample counts, insufficient cutoffs, nonnormalized targets, complex drive quadratures, NaN rates and unknown parameter names.
- Removed raw-result rewriting from the report generator. Added effective-model and device metadata to future trajectory results.
- Replaced the fixed PyTorch path with normal package loading and a documented optional sibling-environment/`NOV17_TORCH_SITE` fallback. Small trajectory checks can also run on CPU.

## Verification

| Check | Result |
| --- | --- |
| Unit and regression checks | 20 CPU checks pass; 3 additional physics checks pass on GPU 1 |
| Nov3/Nov17 circuit equivalence | 140 saved/random cases pass; maximum observable discrepancy below 4e-14 |
| Standard-Pauli Hamiltonian and full Lindblad RHS | Agree with the numerical model after the stated global energy offset |
| Joint cost phases | All four saved pulses reproduce their target tables and recorded metrics |
| Local full-shell phases | All six saved pulses reproduce their target tables and recorded metrics |
| Coherent H pulses | All four recorded metrics reproduce at the documented validation cutoff |
| Initial preparation | Selected coherent and density-matrix noisy fidelities reproduce exactly |
| Readout | Both recorded coherent cutoffs and the waveform CSV reproduce |
| Readout calibration | All 20 pulse responses reproduce in each of coherent, loss-only and ancilla-noise scenarios: 60 response replays |
| Measurement banks | All nine NPZ banks are finite, positive and complete; saved weights and residual metadata agree |
| Tensor ordering | Complex entangled-state measurement probabilities match an independent Kronecker-product calculation |
| 8192-shot example | Energy, susceptibility estimator, uncertainty and allocation reproduce |
| Hardware schedules | All ten historical coherent/noisy schedule records match stage order and durations |
| Pulse gradients | Coherent/measurement GRAPE, preparation and joint phase adjoints match finite differences |
| Noise physics on GPU 1 | Cavity loss and dephasing agree with analytic decay; a driven bus agrees with an independent density-matrix ODE |
| Documentation | LaTeX rebuild succeeds; no overfull equations or unresolved references; standard Pauli conventions are retained |

The 8192-shot example allocation is 179, 1243, 1291 and 5479 shots. Its energy is -2.6550414426245714 and its equal-time spin-glass indicator estimate is 1.485427172257352. These are synthetic example estimates, not a new noisy hardware campaign.

The selected H pulses change average fidelity by about 3.8e-8 (chi-time 48) and 2.3e-6 (chi-time 72) between cavity cutoffs 24 and 32. One unselected 72 candidate changes by about 2.1e-4. This is a numerical truncation check, not a dispersive-model validity test.

## Practical limits

- The measurement-only pulse and coherent H are different controls. The former has output-phase freedom and must not replace an interior coherent H. Its roughly 99% measurement fidelity is not a 99% coherent-unitary fidelity.
- Some retained optimization records have `success: false`. Their reported pulse fidelities were independently reproduced; the audit does not claim optimizer convergence or global optimality.
- The useful full-system configuration supports separate cavity/transmon lifetimes and optional cavity dephasing/residuals. Legacy local preparation/readout helpers retain their documented 1-MHz, zero-detuning model and implement T1/T2 noise only. Extending those helpers or using asymmetric readout requires a corresponding new calibration model.
- Only the cavity lifetime was user supplied. Other device values are assumptions. Calibration banks and existing pulses must be revalidated when the physical model changes.
- The historical full N=24 noisy schedules were not rerun during this audit. Their files, schedules and summary were checked; independent small-system GPU checks validate the engine. Full-system dense density-matrix integration and a complete 8192-shot energy/indicator campaign for the noisy schedules remain future work.
- The project's chi_SG is the squared equal-time correlation indicator defined in the methods, not a separately computed dynamical response susceptibility. Finite N=6 results alone do not establish a thermodynamic phase transition.

## Reproduce

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nov17_circuit.scripts.audit
.venv/bin/python -m nov17_circuit.scripts.verify
NOV17_RUN_GPU_TESTS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest nov17_circuit.tests.test_trajectories -v
```

`folder_audit.json` records the CPU/artifact check results and input hashes. GPU checks are opt-in; their test definitions are retained in `tests/test_trajectories.py` and their synthetic stronger-noise parameters do not replace the baseline.
