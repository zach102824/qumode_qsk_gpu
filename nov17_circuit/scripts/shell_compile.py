"""Physical local-transmon pulses for the complete Nov3 Hamming-weight shell."""
from ..sequence import *
from ..models.phase_control import propagate
from ..models.phase_control import gate_metrics
from scipy.optimize import minimize
import time

def main():
    rng=np.random.default_rng(171119);results=[]
    delta=2*np.pi*np.arange(8)
    for model,x,L,_ in (r for r in load_saved() if r[0].seed==831106000):
        for layer,b in enumerate(x[3:].reshape(L,3)):
            k=np.array([n.bit_count() for n in range(8)])
            phase=b[0]*k+b[1]*k*(k-1)/2+b[2]*k*(k-1)*(k-2)/6
            previous=None
            for T in (1.,2.):
                steps=int(np.ceil(T/.02))
                if previous is None:initial=rng.normal(0,.1,steps*2)
                else:
                    old=np.array(previous['parameters']).reshape(-1,2)
                    initial=np.array([np.interp(np.linspace(0,1,steps),np.linspace(0,1,len(old)),old[:,i]) for i in range(2)]).T.ravel()
                fun=lambda x:propagate(x,T,delta,True,phase)
                start=time.monotonic()
                fit=minimize(fun,initial,jac=True,method='L-BFGS-B',bounds=[(-1,1)]*len(initial),
                    options={'maxiter':1000,'ftol':1e-12,'gtol':1e-7,'maxls':30})
                v=propagate(fit.x,T,delta)
                row=dict(seed=model.seed,L=L,layer=layer,duration_us=T,steps=steps,
                    parameters=fit.x.tolist(),phase_table=phase.tolist(),shell_parameters=b.tolist(),
                    iterations=int(fit.nit),success=bool(fit.success),seconds=time.monotonic()-start,
                    metrics=gate_metrics(np.diag(v[:,0]),np.diag(np.exp(-1j*phase))),
                    note='Both local transmons run the same pulse in parallel; no cavity drive; bus parked in g.')
                results.append(row);previous=row
                (P/'pulses/shell_phase_search.json').write_text(json.dumps(dict(records=results),indent=2)+'\n')
                print('SHELL',L,layer,T,row['metrics'],flush=True)

if __name__=='__main__':main()
