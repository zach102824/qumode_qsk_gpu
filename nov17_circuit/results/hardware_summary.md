# Nov17 hardware comparison

State fidelity relative to the same ideal Nov3 ansatz state, in the common final-H-omitted frame. All 15 cost terms and the full local shell are included. No trajectories are postselected. The strict target includes all transmons in g; cavity-only values trace them out. These are not ground-state fidelities or average gate fidelities.

|L|Initial stage|Interior H duration|Preparation time|Noiseless state fidelity|Noisy state fidelity +/- SEM|Cavity-only noisy fidelity|
|---:|---|---:|---:|---:|---:|---:|
|1|state_preparation|7.64 us|21.55 us|94.68%|43.0% +/- 1.8%|46.2%|
|2|state_preparation|7.64 us|48.83 us|89.68%|15.6% +/- 1.2%|18.1%|
|1|state_preparation|11.46 us|25.37 us|96.65%|35.9% +/- 1.7%|39.3%|
|1|general_H|11.46 us|34.92 us|96.44%|25.6% +/- 1.5%|28.3%|
|2|state_preparation|11.46 us|60.29 us|94.95%|11.9% +/- 1.0%|13.9%|

The same saved seed 831106000 was used at both depths, with its separately optimized parameters. Noise uses 512 trajectories, a 5-ns jump timestep, and 24 levels per cavity. The earlier entangler study independently checked this jump simulator and cutoff behavior. A wider disorder ensemble, different pulse starts, and hardware bandwidth/noise-aware optimization can change the ranking.

The best point estimates among the tested reusable-H schedules favor the shorter interior H. Do not call the highest noiseless-fidelity pulse the best noisy implementation automatically.

Readout is additional: native Fock readout estimates original-X observables in this common frame. A calibrated measurement-GRAPE pulse on both cavities supplies the original-Z histogram/correlations. The previous nominal readout pulse takes 7.64 us at the assumed chi. Preparation leakage and hardware reset/detection effects need to be included in a real detector model; the existing eight-level calibration is not a guarantee for arbitrary leaked inputs.

For an online variational optimizer, the cost and shell phase tables vary with parameters. The static pulse compilations here validate representative targets; they do not establish an instantaneous, calibrated parametric waveform family. Fixed coherent H pulses and the initial state-preparation pulse can be reused.

Noisy performance remains limited even after the state-preparation saving. No complete energy/chi measurement campaign with 8192 shots was rerun for these new noisy schedules. Logical equivalence, component compilation, and whole-state preparation were tested separately.
