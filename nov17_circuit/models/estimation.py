"""Fixed-batch CSV estimator, including quadratic sampling-bias correction."""
import numpy as np
from .observables import PAIRS
def estimate(samples,design,J,B):
    f=design['f'];y=design['y'];settings=[tuple(s) for s in design['settings']]
    groups={s:[] for s in settings}
    for row in samples:
        s=(int(row['setting_A']),int(row['setting_B']))
        if s not in groups:raise ValueError(f'Unscheduled setting pair {s}')
        a,b=int(row['outcome_A']),int(row['outcome_B'])
        if min(a,b)<0:raise ValueError('Negative Fock outcome')
        groups[s].append(min(a,8)*9+min(b,8))
    mean=np.zeros(16);V=np.zeros((16,16));shots=0
    for g,s in enumerate(settings):
        outcomes=groups[s];n=len(outcomes);shots+=n
        if n<2:raise ValueError(f'At least two shots required in setting pair {s}')
        values=y[g,outcomes];m=values.mean(axis=0)
        cov=np.cov(values,rowvar=False,ddof=1)
        mean+=f[g]*m;V+=f[g]**2/n*cov
    h=np.r_[[J[i,j] for i,j in PAIRS],-B]
    naive=1+np.dot(mean[:15],mean[:15])/3
    correction=np.trace(V[:15,:15])/3
    return dict(shots=shots,energy=float(h@mean),energy_per_spin=float(h@mean/6),
                energy_estimated_sem_per_spin=float(np.sqrt(max(0,h@V@h))/6),
                chi_sg=float(naive-correction),chi_sg_naive=float(naive),
                chi_sg_sampling_bias_correction=float(correction),
                Czz={f'{i},{j}':float(mean[k]) for k,(i,j) in enumerate(PAIRS)},
                calibration_note='Quadratic sampling bias removed; residual measurement-model bias remains.')
