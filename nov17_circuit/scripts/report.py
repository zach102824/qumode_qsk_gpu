"""Collect pulse and state-specific comparisons for the Nov17 schedule."""
from ..sequence import *
import csv,math

def main():
    rows=[]
    for T in (48,72):
        suffix=f'_H{T}' if T!=72 else ''
        path=P/f'results/replays/hardware_coherent_N24_bus10{suffix}.json'
        coherent=json.loads(path.read_text())['records']
        # Coherent one-vector runs have no Monte Carlo uncertainty.
        for r in coherent:
            for key in ('standard_error','cavity_only_standard_error'):
                if not math.isfinite(r[key]):r[key]=0.
        noisy=json.loads((P/f'results/replays/hardware_noisy_N24_bus10{suffix}.json').read_text())['records']
        for n in noisy:
            c=next(r for r in coherent if r['L']==n['L'] and r['initial_general_H']==n['initial_general_H'])
            rows.append(dict(L=n['L'],initial='general_H' if n['initial_general_H'] else 'state_preparation',
                local_chi_time=T,duration_us=n['duration_us'],coherent_state_fidelity=c['state_fidelity'],
                noisy_state_fidelity=n['state_fidelity'],standard_error=n['standard_error'],
                cavity_only_noisy_fidelity=n['cavity_only_state_fidelity'],samples=n['samples'],cutoff=n['cutoff']))
    verification=json.loads((P/'results/verification.json').read_text())
    selected={str(L):max((r for r in rows if r['L']==L and r['initial']=='state_preparation'),key=lambda r:r['noisy_state_fidelity']) for L in (1,2)}
    summary=dict(verification={k:v for k,v in verification.items() if k!='records'},
        assumptions='cavity T1=600us; all three transmons T1/T2=50/30us; bus shifts 1/1.3MHz; no bandwidth or residual couplings',
        GPU=1,rows=rows,selected_by_noisy_point_estimate=selected,
        scope='One saved instance, one ideal parameter vector per depth. Readout noise excluded. Not an overall optimum certificate.')
    (P/'results/hardware_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (P/'results/hardware_summary.csv').open('w') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    lines=['# Nov17 hardware comparison','',
        'State fidelity relative to the same ideal Nov3 ansatz state, in the common final-H-omitted frame. All 15 cost terms and the full local shell are included. No trajectories are postselected. The strict target includes all transmons in g; cavity-only values trace them out. These are not ground-state fidelities or average gate fidelities.','',
        '|L|Initial stage|Interior H duration|Preparation time|Noiseless state fidelity|Noisy state fidelity +/- SEM|Cavity-only noisy fidelity|',
        '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"|{r['L']}|{r['initial']}|{r['local_chi_time']/(2*np.pi):.2f} us|{r['duration_us']:.2f} us|{100*r['coherent_state_fidelity']:.2f}%|{100*r['noisy_state_fidelity']:.1f}% +/- {100*r['standard_error']:.1f}%|{100*r['cavity_only_noisy_fidelity']:.1f}%|")
    lines+=['','The same saved seed 831106000 was used at both depths, with its separately optimized parameters. Noise uses 512 trajectories, a 5-ns jump timestep, and 24 levels per cavity. The earlier entangler study independently checked this jump simulator and cutoff behavior. A wider disorder ensemble, different pulse starts, and hardware bandwidth/noise-aware optimization can change the ranking.','',
        'The best point estimates among the tested reusable-H schedules favor the shorter interior H. Do not call the highest noiseless-fidelity pulse the best noisy implementation automatically.','',
        'Readout is additional: native Fock readout estimates original-X observables in this common frame. A calibrated measurement-GRAPE pulse on both cavities supplies the original-Z histogram/correlations. The previous nominal readout pulse takes 7.64 us at the assumed chi. Preparation leakage and hardware reset/detection effects need to be included in a real detector model; the existing eight-level calibration is not a guarantee for arbitrary leaked inputs.','',
        'For an online variational optimizer, the cost and shell phase tables vary with parameters. The static pulse compilations here validate representative targets; they do not establish an instantaneous, calibrated parametric waveform family. Fixed coherent H pulses and the initial state-preparation pulse can be reused.','',
        'Noisy performance remains limited even after the state-preparation saving. No complete energy/chi measurement campaign with 8192 shots was rerun for these new noisy schedules. Logical equivalence, component compilation, and whole-state preparation were tested separately.']
    (P/'results/hardware_summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
