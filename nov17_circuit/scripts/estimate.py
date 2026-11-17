"""Estimate original-spin energy and susceptibility from fixed-batch readout."""
import argparse
import csv
import json
from pathlib import Path
from cpu_oracle.src.hamiltonian import make_J
from nov17_circuit.models.observables import load_design
from nov17_circuit.models.estimation import estimate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True)
    p.add_argument('--scenario',choices=['coherent','loss','ancilla'],default='ancilla')
    p.add_argument('--full',action='store_true',help='Use optional zero-ridge full bank')
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--B',type=float,required=True)
    args=p.parse_args()
    with args.samples.open() as h:samples=list(csv.DictReader(h))
    result=estimate(samples,load_design(args.scenario,args.full),make_J(6,args.seed),args.B)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
