"""Compile one joint bus pulse for all 15 ZZ terms, with unequal bus shifts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,time
import numpy as np
from scipy.optimize import minimize
from ..sequence import P,ROOT,load_saved,cost_phases,ORDER
from ..models.phase_control import bus_detunings,gate_metrics
from ..models.phase_control import propagate
from ..models.phase_control import noisy_bus

def main():
    prior=json.loads((P/'pulses/all15_phase_search.json').read_text())['records']
    rows=[]
    for model,x,L,_ in (r for r in load_saved() if r[0].seed==831106000):
        full=cost_phases(model,x)[0][ORDER].reshape(8,8)
        for T in (10.,20.):
            original=max((r for r in prior if r['seed']==model.seed and r['L']==L and r['duration_us']==T),key=lambda r:r['metrics']['average_gate_fidelity'])
            initial=np.array(original['parameters']);delta=bus_detunings().ravel()
            fun=lambda x:propagate(x,T,delta,True,full.ravel())
            start=time.monotonic()
            fit=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=[(-1,1)]*len(initial),
                options={'maxiter':650,'ftol':1e-12,'gtol':1e-7,'maxls':30})
            v=propagate(fit.x,T,delta);target=dict(phase=full,D=np.diag(np.exp(-1j*full.ravel())))
            rec=dict(seed=model.seed,L=L,duration_us=T,parameters=fit.x.tolist(),steps=len(initial)//2,
                iterations=int(fit.nit),success=bool(fit.success),seconds=time.monotonic()-start,
                metrics=gate_metrics(np.diag(v[:,0]),target['D']),phase_table=full.tolist(),
                model_note='One joint bus pulse, all 15 pairs, 1/1.3 MHz shifts, same bounds as prior benchmark')
            rec['noisy_bus_only']=noisy_bus(rec,target)
            rows.append(rec);(P/'pulses/all15_phase_search.json').write_text(json.dumps(dict(records=rows),indent=2)+'\n')
            print('ALL15',L,T,rec['metrics'],rec['noisy_bus_only']['average_gate_fidelity'],flush=True)

if __name__=='__main__':main()
