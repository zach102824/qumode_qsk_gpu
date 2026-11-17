"""Rebuild GRAPE-only model effects and compact/full estimators."""
from ..models.system import DEFAULT_CONFIG as NOISE
import argparse
import json
import numpy as np
from nov17_circuit.models.readout_effects import native,effects
from nov17_circuit.models.reconstruction import calibration
from nov17_circuit.models.observables import P,design_frame

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario',choices=['coherent','loss','ancilla'],default='ancilla')
    p.add_argument('--rebuild-effects',action='store_true')
    args=p.parse_args();path=P/'calibration'/f'bank_{args.scenario}.npz'
    records=json.loads((P/'pulses/correction_bank.json').read_text())['records']
    if args.rebuild_effects:
        bank=[native()];durations=[0.]
        for rec in records:
            E,t=effects(rec,T1=None if args.scenario=='coherent' else NOISE['T1_cavity_A_us'],ancilla_noise=args.scenario=='ancilla')
            bank.append(E);durations.append(t)
        np.savez_compressed(path,effects=bank,durations=durations)
    E=np.load(path)['effects']
    # These ridge choices are the winners in the retained historical benchmark.
    ridge={'coherent':0.,'loss':.01,'ancilla':.001}[args.scenario]
    for full in (False,True):
        bank=E if full else E[:2]
        c=calibration(bank,0. if full else ridge,optimize=full)
        d=design_frame(bank,c)
        name=f"{'full' if full else 'compact'}_{args.scenario}"
        np.savez_compressed(P/'calibration'/f'{name}.npz',effects=bank,weights=c['weights'],
            local_f=c['f'],y=d['y'],f=d['f'],settings=d['settings'])
        summary={k:v for k,v in c.items() if k not in ('weights','f')}
        (P/'calibration'/f'{name}.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(name,summary,flush=True)

if __name__=='__main__':main()
