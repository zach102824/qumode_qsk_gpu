# Nov17 circuit and noisy simulation workspace

This is the continuing workspace for the Nov17 hardware implementation of the unchanged Nov3 six-spin ansatz. Two cavities encode three binary spins each; two local control transmons and one shared bus provide physical control. GPU runs default to **GPU 1**.

- [Methods PDF](docs/nov17_methods.pdf): LaTeX/REVTeX 4.2 PRB format with standard transmon Pauli notation; total physical Hamiltonian, active terms for joint SNAP/local GRAPE, explicit K/M ansatz and state evolution, Lindblad noise bullets, and seven primary references.
- [Noise strengths](noise_strengths.txt): readable parameter values, units, provenance, and omitted channels.
- [Machine-readable noise configuration](config/noise_model.json): baseline source for the full-system Hamiltonian, collapse operators, and trajectories.
- [Existing hardware comparisons](results/hardware_summary.md): coherent and noisy state fidelities, with pulse durations and uncertainties.

## Circuit and measurement

Start from cavity vacuum with all three transmons in g. Prepare the encoded |+> states locally, then repeat the all-15 ZZ cost phase and the Hadamard–full local diagonal shell–Hadamard mixer. Preserve the complete Hamming-weight shell, including its induced ZZ and ZZZ contributions. The final Hadamard can be assigned to the measurement boundary: stopping before it yields original-X readout; calibrated basis conversion yields original-Z correlations. Interior Hadamards require coherent gates.

The variational SK Hamiltonian is an objective, not an extra physical drift during microwave pulses. See the PDF for both definitions and their relation.

## Organization

| Directory | Contents |
| --- | --- |
| `models/` | Physical Hamiltonian, Lindblad generator, quantum-jump propagation, local GRAPE and readout models |
| `scripts/` | Compilation, verification, replay, calibration, estimation, and reports |
| `pulses/` | Retained preparation, coherent Hadamard, phase, and measurement pulses |
| `calibration/` | Measurement-response banks, including optional full banks |
| `results/` | Retained verification, coherent/noisy replays, readout results and optimization records |
| `examples/` | Synthetic 8192-shot samples, expected estimate, and acquisition schedules |
| `tests/` | Circuit equivalence, Hamiltonian units, trace preservation, and T1/T2 conventions |
| `docs/` | LaTeX source, BibTeX references, generated PDF and build helper |

The obsolete Nov13 workspace and unrelated direct-XX searches have been removed. Useful GRAPE calibration and noise results were moved here. No recovery archive is required. The package uses the repository's tracked `cpu_oracle` as its logical reference. The 20 saved state-parameter snapshots are bundled under `results/reference_states/`, with original run paths and SHA-256 provenance, so tests do not depend on untracked local run files.

## Noise baseline and what was already simulated

Cavity T1 is 600 us (user supplied, assigned to both cavities). Each transmon has assumed T1=50 us and T2=30 us, not device measurements. The TXT gives the corresponding rates and all explicit assumptions.

Existing local preparation runs propagate a density matrix with symmetric unitary/noise splitting. Existing full-circuit runs use finite-step quantum-jump trajectories, an unraveling of the stated Lindblad equation. They are not a full-system dense density-matrix integration. The new `models/system.py` exposes the common Hamiltonian, named collapse operators, and a density-matrix RHS for future integration. Results retained here were generated with the baseline parameters; changing configuration does not regenerate saved results or recalibrate pulse/readout banks.

The local preparation/readout helpers implement cavity T1 and local-transmon T1/T2 using local-A values and the historical 1 MHz, zero-detuning local control model. Optional cavity number dephasing and residual couplings are implemented in the full-system model, not these legacy local helpers. Independent identical-cavity product estimates assume identical local devices. The full-system model supports separate A/B noise and dispersive parameters. Existing pulses must be recompiled if their control model changes. The noise simulation excludes finite-temperature excitation, higher transmon levels, finite detector/reset duration, and uncalibrated residual interactions. See the PDF and TXT for details.

## Commands (from the repository root)

```bash
.venv/bin/python -m unittest discover -s nov17_circuit/tests -t . -v
.venv/bin/python -m nov17_circuit.scripts.verify
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m nov17_circuit.scripts.audit
.venv/bin/python -m nov17_circuit.scripts.schedule --seed 831106000 --B 0.183 --shots 8192
.venv/bin/python -m nov17_circuit.scripts.estimate --samples nov17_circuit/examples/samples.csv --seed 831106000 --B 0.183
.venv/bin/python -m nov17_circuit.scripts.noise_spec
.venv/bin/python nov17_circuit/docs/build.py
```

The last command needs Tectonic, or TeX Live with REVTeX 4.2 and BibTeX. A first Tectonic build downloads its TeX bundle. LaTeX source and bibliography are included for rebuilding independently.

Optional full-circuit replays (expensive; use GPU 1):

```bash
.venv/bin/python -m nov17_circuit.scripts.hardware_test --device 1 --coherent --local-chi-time 48 --no-reference
.venv/bin/python -m nov17_circuit.scripts.hardware_test --device 1 --samples 512 --local-chi-time 48 --no-reference
.venv/bin/python -m nov17_circuit.scripts.report
```

The shorter tested coherent Hadamard (chi-time 48) gave the best noisy point estimates among the retained comparisons. This is a finite candidate comparison, not a claim of globally optimal control. The stored phase waveforms are representative parameter-specific compilations, not an already calibrated waveform family for every optimizer step.

## Folder audit

[Audit findings and limits](results/folder_audit.md) and [machine-readable checks](results/folder_audit.json) record the review. The audit replays all joint/shell phase targets, both coherent-H candidates at each duration, the selected noisy preparation, and all 20 readout pulses in each of the three calibration scenarios. It also checks stored hardware schedules and runs the regression tests. It does not repeat the expensive historical full hardware simulations.

Independent GPU-1 noise checks can be run explicitly:

```bash
NOV17_RUN_GPU_TESTS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m unittest nov17_circuit.tests.test_trajectories -v
```

The full-circuit replay now defaults to the shorter tested H (chi-time 48), the configured cavity cutoff and jump step, and GPU 1. New runs record their effective model parameters. The trajectory engine also accepts `device='cpu'` for small checks. PyTorch must be installed in the active environment; this workstation can alternatively use its sibling `pytorch_gpu` environment, or an explicit site-packages path supplied through `NOV17_TORCH_SITE`. The CPU model, pulse and readout code requires NumPy and SciPy.
