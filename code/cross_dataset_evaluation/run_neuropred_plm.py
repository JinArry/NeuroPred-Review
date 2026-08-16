#!/usr/bin/env python3
"""Run released NeuroPred-PLM on a cross-dataset manifest."""
import argparse, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
MODEL_URL='https://zenodo.org/record/7042286/files/model.pth'
def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    repo=a.repo.resolve(); frame=pd.read_csv(a.manifest,sep='\t'); a.output.parent.mkdir(parents=True,exist_ok=True); os.chdir(repo); sys.path.insert(0,str(repo))
    old=torch.load
    def compat(*args,**kwargs): kwargs.setdefault('weights_only',False); return old(*args,**kwargs)
    torch.load=compat
    from NeuroPredPLM.model import EsmModel
    from NeuroPredPLM.utils import load_hub_workaround
    model=EsmModel(); model.load_state_dict(load_hub_workaround(MODEL_URL)); model=model.to(a.device).eval(); probs=[]
    peptides=list(zip(frame.seq_id.astype(str),frame.sequence.astype(str)))
    with torch.no_grad():
        for start in range(0,len(peptides),a.batch_size):
            logits,_=model(peptides[start:start+a.batch_size],a.device); probs.append(torch.softmax(logits,dim=-1)[:,1].cpu().numpy())
    prob=np.concatenate(probs); frame['prob_np']=prob; frame['pred']=(prob>=0.5).astype(int); frame.to_csv(a.output,sep='\t',index=False)
if __name__=='__main__': main()
