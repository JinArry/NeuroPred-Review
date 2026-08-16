#!/usr/bin/env python3
"""Run released NeuroPred-CLQ assets on a cross-dataset manifest."""
import argparse, os, sys
from pathlib import Path
import gensim.models
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--repo',type=Path,required=True); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    repo=a.repo.resolve(); frame=pd.read_csv(a.manifest,sep='\t'); a.output.parent.mkdir(parents=True,exist_ok=True)
    os.chdir(repo); sys.path.insert(0,str(repo)); from model import ourmodel
    w2v=gensim.models.Word2Vec.load(str(repo/'CLQ_model/NPs4'))
    features=[]
    for seq in frame.sequence.astype(str):
        padded=seq[:100]+'X'*max(0,100-len(seq)); vectors=[]
        for i in range(97):
            word=padded[i:i+4]; vectors.append(w2v.wv[word] if word in w2v.wv else np.zeros(150,dtype=np.float32))
        features.append(vectors)
    model=ourmodel(); model.load_weights(str(repo/'CLQ_model/FinModel.h5'))
    prob=model.predict(np.asarray(features),verbose=0)[:,1]
    frame['prob_np']=prob; frame['pred']=(prob>=0.5).astype(int); frame.to_csv(a.output,sep='\t',index=False)
if __name__=='__main__': main()
