"""Replay the retained N=6, L=1, B=0.183 schedule against the actual SK ground state.

Uses GPU 1, cutoff 24 and 512 noisy trajectories. Output includes both the
cavity-only overlap and the stricter overlap requiring all transmons in g.
This is one saved optimized instance; measurement noise is excluded.
"""
import json,time
import numpy as np
from nov17_circuit.sequence import P,load_saved,W,ORDER,state
from nov17_circuit.scripts.hardware_test import schedule
from nov17_circuit.models.trajectories import simulate

def main():
    model,x,L,_=next(r for r in load_saved() if r[0].seed==831106000 and r[2]==1)
    model.compute_gs_fullspace()
    target=(W@model.gs)[ORDER];vac=np.zeros(64,complex);vac[0]=1
    ideal=state(model,x,1,'Z');F=float(abs(np.vdot(model.gs,ideal))**2)
    rows,durations,stages=schedule(1,local_time=48.)
    out={'N':6,'L':1,'instance_seed':model.seed,'B':model.b,'E0':model.e0,'ideal_ansatz_ground_state_fidelity':F,'duration_us':float(durations.sum()),'target':'Actual SK ground state in final-H-omitted X coordinates','readout_noise_included':False,'postselection':False,'results':{}}
    print('IDEAL_GS',F,flush=True)
    for noise in (False,True):
     r=simulate(rows,durations,None,N=24,samples=512 if noise else 1,dt_max=.005,noise=noise,device=1,initial=vac,desired=target,seed=161119)
     r['ground_state_fidelity_all_transmons_g']=r.pop('average_gate_fidelity')
     r['ground_state_fidelity_cavities']=r.pop('cavity_only_average_fidelity')
     r['fidelity_kind']='ground-state preparation fidelity'
     out['results']['with_noise' if noise else 'without_noise']=r
     (P/'results/ground_state_fidelity_N6_L1_H48.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
     print('GROUND_STATE_RESULT',noise,r['ground_state_fidelity_all_transmons_g'],r['standard_error'],r['ground_state_fidelity_cavities'],r['cavity_only_standard_error'],flush=True)

if __name__=='__main__':main()
