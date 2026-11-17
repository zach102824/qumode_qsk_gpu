"""Generate the human-readable noise baseline and LaTeX parameter macros."""
import math,json,hashlib
from ..models.system import ROOT,DEFAULT_CONFIG,parameters

def main():
    p=parameters();lines=['NOV17 OPEN-SYSTEM BASELINE','==========================','',
        'Architecture: two cavities + local transmons q_A,q_B + shared bus q_b.',
        'Encoded spins: 3 per cavity, 6 total, binary Fock levels 0..7.',
        'Tensor order: (q_b, q_A, cavity_A, q_B, cavity_B).',
        'GPU policy: GPU 1 only by default; GPU 0 is reserved for other work.','',
        'STATUS: cavity T1=600 us was supplied by the user. All other device',
        'numbers below are assumptions, not calibrated measurements.',
        'The user accepted unequal bus shifts; 1.0/1.3 MHz are illustrative values.','',
        'Units: time us; h=H/hbar in rad/us; quoted MHz = cycles/us.',
        'Multiply every quoted detuning, chi, and drive quadrature by 2*pi',
        'when inserting it in h. Dissipative rates are in 1/us, NOT rad/us;',
        'do NOT multiply 1/T1, 1/T2, gamma_phi by 2*pi.','',
        'Dissipator: D[L]rho=L*rho*L^dagger - {L^dagger*L,rho}/2.',
        'Master equation: d(rho)/dt=-i[h(t),rho] + sum_mu D[L_mu]rho.','',
        'Operator convention in the methods: sigma_z=|e><e|-|g><g|;',
        'sigma_minus=|g><e|; sigma_plus=|e><g|.',
        'Pauli dispersive term: -sum_rj chi_rj*n_j*sigma_z,r/2.',
        'Pauli cavity detuning: Delta_tilde_c,j=Delta_c,j-sum_r chi_rj/2.',
        'The configured Delta_c,j is the ground-transmon cavity detuning.',
        'Omitted global energy: sum_r Delta_q,r/2; no number shift is dropped.','',
        'CAVITIES']
    for name in ('A','B'):
        T=p[f'T1_cavity_{name}_us'];rate=1/T
        lines += [f'  Cavity {name}: T1={T:g} us, kappa={rate:.12g} /us = {rate*1e6:.12g} /s.',
            f'    L_loss=sqrt(kappa)*a_{name}; sqrt(kappa)={math.sqrt(rate):.12g} us^(-1/2).',
            f"    Number-dephasing coefficient={p[f'cavity_{name}_number_dephasing_per_us']:g} /us (baseline zero)."]
    lines+=['','TRANSMONS (relaxation plus PURE dephasing, independently for each)']
    for name in ('local_A','local_B','bus'):
        T1=p[f'T1_{name}_us'];T2=p[f'T2_{name}_us'];g1=1/T1;g2=1/T2;gp=g2-g1/2
        lines += [f'  {name}: T1={T1:g} us, T2={T2:g} us.',
            f'    gamma1={g1:.12g} /us; gamma2={g2:.12g} /us.',
            f'    gamma_phi=1/T2-1/(2*T1)={gp:.12g} /us.',
            f'    T_phi={1/gp:.12g} us.' if gp else '    T_phi=infinite.',
            f'    L_relax=sqrt(gamma1)*sigma_minus; coefficient={math.sqrt(g1):.12g} us^(-1/2).',
            f'    L_phi=sqrt(gamma_phi/2)*sigma_z; coefficient={math.sqrt(gp/2):.12g} us^(-1/2).',
            f'    Backend uses the equivalent sqrt(2*gamma_phi)*|e><e| dissipator.']
    lines+=['','HAMILTONIAN PARAMETERS (MHz, multiply by 2*pi in h)']
    for key in ('chi_local_A_MHz','chi_local_B_MHz','chi_bus_A_MHz','chi_bus_B_MHz',
                'detuning_cavity_A_MHz','detuning_cavity_B_MHz','detuning_local_A_MHz',
                'detuning_local_B_MHz','detuning_bus_MHz','Kerr_A_MHz','Kerr_B_MHz','cross_Kerr_MHz','exchange_MHz'):
        lines.append(f'  {key}={p[key]:g}')
    lines += ['',f"Drive bounds, per quadrature: cavity {p['cavity_drive_quadrature_bound_MHz']:g} MHz, transmon {p['transmon_drive_quadrature_bound_MHz']:g} MHz.",
        'No factor 1/2 in Omega_I*sigma_x-Omega_Q*sigma_y; resonant angular',
        'Rabi frequency is 2*sqrt(Omega_I^2+Omega_Q^2).','',
        'EXCLUDED/UNKNOWN IN THE NOMINAL MODEL',
        '  Thermal excitation and finite-temperature baths: zero.',
        '  Cavity self-Kerr, cross-Kerr, direct exchange: zero in baseline.',
        '  No correlated baths, transmon f level/anharmonicity, bandwidth/filter',
        '  response, drive calibration drift, measurement/reset overhead, or',
        '  experimental detector errors have been supplied.',
        '  Dispersive critical-photon-number validity needs actual device g/Delta.',
        '  With the bus in g, its Pauli dispersive term cancels the associated',
        '  half-shift in the Pauli cavity detuning. Residual physical couplings',
        '  are not automatically eliminated by that state.','',
        'SIMULATION CONVENTIONS',
        f"  Default cavity cutoff N={p['default_cavity_cutoff']}; Hilbert d=8*N^2={8*p['default_cavity_cutoff']**2}.",
        f"  Quantum-jump timestep <= {p['default_jump_step_us']:g} us (5 ns in baseline).",
        '  Quantum jumps are one unraveling of the stated Lindblad equation.',
        '  Full circuits used 512 trajectories in the current Nov17 comparisons.',
        '  Local preparation was also evolved as a density matrix with symmetric',
        '  unitary/noise splitting. Dense full-circuit Liouvillian integration',
        '  was NOT performed. No noisy trajectories were postselected.',
        '  600 us is photon ENERGY lifetime. First loss from |n> has mean',
        '  waiting time T1/n; |7> gives 85.7 us, |7,7> gives 42.9 us.',
        '  Parameter changes invalidate old compiled pulses/calibration unless',
        '  they are revalidated; historical results retain their original values.','',
        'Source of truth: config/noise_model.json; models/system.py implements',
        'the same Hamiltonian and collapse operators used by the trajectory code.',
        'Regenerate: python -m nov17_circuit.scripts.noise_spec','']
    (ROOT/'noise_strengths.txt').write_text('\n'.join(lines))
    names={'ChiLA':'chi_local_A_MHz','ChiLB':'chi_local_B_MHz','ChiBA':'chi_bus_A_MHz',
        'ChiBB':'chi_bus_B_MHz','DeltaBus':'detuning_bus_MHz','CavBound':'cavity_drive_quadrature_bound_MHz',
        'QBound':'transmon_drive_quadrature_bound_MHz','CavityLife':'T1_cavity_A_us',
        'QOne':'T1_bus_us','QTwo':'T2_bus_us','Cutoff':'default_cavity_cutoff','JumpStep':'default_jump_step_us'}
    macros=['% Generated from config/noise_model.json; do not edit.']
    macros += [f'\\newcommand{{\\{key}}}{{{p[value]:g}}}' for key,value in names.items()]
    (ROOT/'docs/config_values.tex').write_text('\n'.join(macros)+'\n')
    print(ROOT/'noise_strengths.txt')

if __name__=='__main__':main()
