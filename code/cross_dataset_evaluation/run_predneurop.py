#!/usr/bin/env python3
"""Run released PredNeuroP models on a cross-dataset manifest."""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    repo=a.repo.resolve(); frame=pd.read_csv(a.manifest,sep='\t'); a.output.parent.mkdir(parents=True,exist_ok=True)
    fasta=a.output.with_suffix('.input.fasta')
    with fasta.open('w') as h:
        for row in frame.itertuples(index=False): h.write(f'>{row.seq_id}\n{row.sequence}\n')
    os.chdir(repo); sys.path.insert(0,str(repo))
    from PredNeuroP import meta_pred
    prob=np.asarray(meta_pred(str(fasta)))[:,1]
    if len(prob)!=len(frame): raise RuntimeError('Prediction count mismatch')
    frame['prob_np']=prob; frame['pred']=(prob>=0.5).astype(int); frame.to_csv(a.output,sep='\t',index=False)
    fasta.unlink()
if __name__=='__main__': main()
