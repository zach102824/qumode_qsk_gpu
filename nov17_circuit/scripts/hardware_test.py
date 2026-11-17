"""State-specific coherent/noisy replay of full 15-term Nov17 hardware schedules.

Common prepared state stops before the last H (original-X frame). The final
measurement H is calibrated readout, not an additional preparation gate.
"""
import argparse,json
from ..sequence import *
from ..models.trajectories import simulate
from ..models.system import DEFAULT_CONFIG

def local_controls(rec):
    x=np.array(rec['parameters']).reshape(-1,4);rows=np.zeros((len(x),10))
    rows[:,:2]=x[:,:2];rows[:,2:4]=x[:,:2];rows[:,4:6]=x[:,2:];rows[:,6:8]=x[:,2:]
    return rows,np.full(len(x),rec['duration_us']/len(x))

def schedule(L,initial_general_H=False,bus_duration=10.,local_time=48.):
    all_local=json.loads((P/'pulses/coherent_h_search.json').read_text())['records']
    local=max((r for r in all_local if r['value']==local_time),key=lambda r:r['metrics']['average_gate_fidelity'])
    prep=json.loads((P/'pulses/selected_initial_prep.json').read_text())
    initial=local if initial_general_H else prep
    rows,durations=local_controls(initial);stages=[dict(kind='general_H_initial' if initial_general_H else 'state_preparation',duration_us=float(durations.sum()))]
    chunks=[rows];times=[durations]
    busrecords=json.loads((P/'pulses/all15_phase_search.json').read_text())['records']
    shells=json.loads((P/'pulses/shell_phase_search.json').read_text())['records']
    for layer in range(L):
        bus=next(r for r in busrecords if r['L']==L and r['duration_us']==bus_duration)
        bx=np.array(bus['parameters']).reshape(-1,2);br=np.zeros((len(bx),10));br[:,8:]=bx
        chunks.append(br);times.append(np.full(len(bx),bus_duration/len(bx)))
        stages.append(dict(kind='all_15_ZZ_bus_phase',layer=layer,duration_us=bus_duration))
        hr,ht=local_controls(local);chunks.append(hr);times.append(ht)
        stages.append(dict(kind='coherent_H',layer=layer,duration_us=float(ht.sum())))
        # Prefer a robust high coherent score; noise-optimal local phases were not searched.
        shell=max((r for r in shells if r['L']==L and r['layer']==layer),key=lambda r:r['metrics']['average_gate_fidelity'])
        sx=np.array(shell['parameters']).reshape(-1,2);sr=np.zeros((len(sx),10));sr[:,4:6]=sx;sr[:,6:8]=sx
        chunks.append(sr);times.append(np.full(len(sx),shell['duration_us']/len(sx)))
        stages.append(dict(kind='full_local_shell_Z',layer=layer,duration_us=shell['duration_us']))
        if layer<L-1:
            chunks.append(hr);times.append(ht);stages.append(dict(kind='coherent_H',layer=layer,duration_us=float(ht.sum())))
    return np.concatenate(chunks),np.concatenate(times),stages

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',type=int,default=1)
    ap.add_argument('--samples',type=int,default=512);ap.add_argument('--cutoff',type=int,default=DEFAULT_CONFIG['default_cavity_cutoff'])
    ap.add_argument('--coherent',action='store_true');ap.add_argument('--bus-duration',type=float,default=10.)
    ap.add_argument('--local-chi-time',type=float,default=48.,choices=[48.,72.])
    ap.add_argument('--no-reference',action='store_true')
    args=ap.parse_args();results=[]
    cases=[(1,False),(2,False)] if args.no_reference else [(1,False),(1,True),(2,False)]
    for L,old in cases:
        model,x,_,_=next(r for r in load_saved() if r[0].seed==831106000 and r[2]==L)
        desired=state(model,x,L,'X')[ORDER];vacuum=np.zeros(64,complex);vacuum[0]=1
        rows,durations,stages=schedule(L,old,args.bus_duration,args.local_chi_time)
        result=simulate(rows,durations,dict(U=np.eye(64)),N=args.cutoff,
            samples=1 if args.coherent else args.samples,dt_max=DEFAULT_CONFIG['default_jump_step_us'],noise=not args.coherent,
            device=args.device,initial=vacuum,desired=desired)
        result['state_fidelity']=result.pop('average_gate_fidelity')
        result['cavity_only_state_fidelity']=result.pop('cavity_only_average_fidelity')
        result.update(L=L,seed=831106000,initial_general_H=old,duration_us=float(durations.sum()),
                      local_chi_time=args.local_chi_time,
                      stages=stages,readout='Common original-X frame; final H omitted. Final readout noise excluded.')
        results.append(result)
        suffix=f'_H{args.local_chi_time:g}' if args.local_chi_time!=72 else ''
        name=f"hardware_{'coherent' if args.coherent else 'noisy'}_N{args.cutoff}_bus{args.bus_duration:g}{suffix}.json"
        (P/'results/replays'/name).write_text(json.dumps(dict(records=results),indent=2)+'\n')
        print('NOV17_RESULT',result,flush=True)

if __name__=='__main__':main()
