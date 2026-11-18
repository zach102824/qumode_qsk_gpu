# Ground-state fidelity verification

Instance: N=6, L=1, seed 831106000, B=0.183; short coherent H, preparation time 21.5492965855 us.

An independent explicit 64-by-64 Pauli Hamiltonian gives E0=-2.7968862145021784 and a ground-state gap of 0.03225699424969264. The eigenvector residual is 2.74e-15; its overlap with the oracle ground state is unity to floating-point accuracy. Transforming both state and ground-state target into the final-H-omitted frame preserves the overlap.

| Calculation | Cavity ground-state fidelity | All transmons g as well |
| --- | ---: | ---: |
| Ideal ansatz | 76.48685% | 76.48685% with ideal auxiliaries |
| Independent sparse complex128 pulse replay | 71.84396% | 71.79669% |
| Original 512 trajectories, 5 ns | 34.71% ± 1.30 pp | 32.45% ± 1.34 pp |
| New 1024 trajectories, 5 ns | 35.13% ± 0.90 pp | 32.70% ± 0.92 pp |
| New 1024 trajectories, 2.5 ns | 34.66% ± 0.90 pp | 32.26% ± 0.93 pp |

Uncertainties are one Monte Carlo standard error in percentage points. The new timestep results differ by 0.47 percentage points, or 0.37 combined standard errors. They are consistent with approximately 35% noisy cavity ground-state fidelity. This comparison does not certify a sub-percent timestep error.

The double-precision sparse solver independently reconstructs the physical Hamiltonian and propagates full pulse bins with SciPy expm_multiply. It agrees with the prior complex64 GPU result within 5.8e-6 absolute fidelity. Its final state norm is 1+2.9e-13. The reported overlaps include leakage as lost overlap and do not renormalize or postselect.

The PDF Pauli convention, cavity half-shifts, complex-drive signs and Lindblad T1/T2 factors pass the independent matrix/RHS tests. The PDF now explicitly defines both ground-state fidelities and the output-frame target. The cited SNAP paper supports number-selective phases; the bilinear-coupling paper describes a different exchange mechanism, as already distinguished in the text. The two-mode extension and assumed device values remain modeling assumptions.

All results refer to fixed parameters optimized for the ideal ansatz. The physical final Hadamard/readout pulse is excluded. These are model-based simulation results for one disorder instance, not calibrated experimental fidelities.

Reproduce the independent calculation and timestep checks on GPU 1:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nov17_circuit.scripts.verify_ground_state --noise
```
