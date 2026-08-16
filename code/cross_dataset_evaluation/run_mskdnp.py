#!/usr/bin/env python3
"""Run released MSKDNP student and MLP head on a cross-dataset manifest."""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,TensorDataset
def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--reproduction-script-dir',type=Path,required=True); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--device',default='cuda:0'); a=p.parse_args()
    sys.path.insert(0,str(a.reproduction_script_dir.resolve())); from run_released_student_model_test import build_models
    frame=pd.read_csv(a.manifest,sep='\t'); a.output.parent.mkdir(parents=True,exist_ok=True); device=torch.device(a.device); tok,student,head=build_models(a.repo.resolve(),device)
    enc=tok(frame.sequence.astype(str).tolist(),padding='max_length',truncation=True,max_length=100,return_tensors='pt'); loader=DataLoader(TensorDataset(enc['input_ids'],enc['attention_mask']),batch_size=a.batch_size,shuffle=False); base=[]; mlp=[]
    with torch.inference_mode():
        for ids,mask in loader:
            logits,mapped=student(ids.to(device),mask.to(device)); base.append(torch.softmax(logits,1)[:,1].cpu().numpy()); mlp.append(torch.softmax(head(mapped),1)[:,1].cpu().numpy())
    frame['prob_np_base']=np.concatenate(base); frame['prob_np']=np.concatenate(mlp); frame['pred']=(frame.prob_np>=0.5).astype(int); frame.to_csv(a.output,sep='\t',index=False)
if __name__=='__main__': main()
