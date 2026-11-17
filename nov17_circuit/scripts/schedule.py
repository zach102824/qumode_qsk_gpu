"""Generate a fixed-budget acquisition schedule from known J and B."""
import argparse
import json
from cpu_oracle.src.hamiltonian import make_J
from nov17_circuit.models.observables import allocate,load_design

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--B',type=float,required=True)
    p.add_argument('--shots',type=int,default=8192)
    p.add_argument('--scenario',choices=['coherent','loss','ancilla'],default='ancilla')
    p.add_argument('--full',action='store_true')
    args=p.parse_args();d=load_design(args.scenario,args.full)
    counts=allocate(d,dict(J=make_J(6,args.seed),B=args.B),args.shots)
    print(json.dumps(dict(seed=args.seed,B=args.B,shots=int(counts.sum()),scenario=args.scenario,
        settings=d['settings'],counts=counts.tolist()),indent=2))

if __name__=='__main__':main()
