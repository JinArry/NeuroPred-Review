#!/usr/bin/env python3
"""Unified-data reimplementation of NeuroPred-ResSE (OH+DDE+NV, ResNet+SE)."""

import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

AA = "ACDEFGHIKLMNPQRSTVWY"; AA_INDEX = {a:i for i,a in enumerate(AA)}
CODONS = dict(zip(AA, [4,2,2,2,2,4,2,3,2,6,1,2,4,2,6,6,4,3,1,2]))


def encode(sequence):
    s = sequence.upper(); terminal = (s[:5] + s[-5:])
    if len(terminal) < 10: terminal = (terminal + "X" * 10)[:10]
    oh = np.zeros(200, np.float32)
    for pos, a in enumerate(terminal[:10]):
        if a in AA_INDEX: oh[pos * 20 + AA_INDEX[a]] = 1
    counts = np.zeros(400, np.float64)
    for a,b in zip(s[:-1], s[1:]):
        if a in AA_INDEX and b in AA_INDEX: counts[AA_INDEX[a]*20 + AA_INDEX[b]] += 1
    if counts.sum(): counts /= counts.sum()
    tm = np.array([(CODONS[a]/61)*(CODONS[b]/61) for a in AA for b in AA])
    tv = tm * (1-tm) / max(len(s)-1, 1); dde = (counts-tm) / np.sqrt(np.maximum(tv, 1e-12))
    nv=[]
    for a in AA:
        positions=np.array([i+1 for i,x in enumerate(s) if x==a], dtype=float); n=len(positions)
        if n: u=positions.mean(); d=(((positions-u)**2).sum()/(n*len(s)))
        else: u=d=0.0
        nv.extend([n,u,d])
    return np.concatenate([oh, dde.astype(np.float32), np.asarray(nv,np.float32)]).astype(np.float32)


class ResSE(nn.Module):
    def __init__(self):
        super().__init__(); self.conv1=nn.Conv1d(660,64,3,padding=1); self.bn1=nn.BatchNorm1d(64)
        self.conv2=nn.Conv1d(64,64,3,padding=1); self.bn2=nn.BatchNorm1d(64); self.skip=nn.Conv1d(660,64,1)
        self.se1=nn.Linear(64,16,bias=False); self.se2=nn.Linear(16,64,bias=False); self.drop=nn.Dropout(.5)
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,2))
    def forward(self,x):
        x=x.unsqueeze(-1); residual=self.skip(x); z=torch.relu(self.bn1(self.conv1(x))); z=torch.relu(self.bn2(self.conv2(z)))
        w=z.mean(-1); w=torch.sigmoid(self.se2(torch.relu(self.se1(w)))).unsqueeze(-1); z=self.drop(residual+z*w)
        return self.head(z.flatten(1))


def load_metadata(path):
    rows=list(csv.DictReader(open(path),delimiter="\t")); return rows


def score(y,prob):
    y=np.asarray(y,int); pred=(np.asarray(prob)>=.5).astype(int); tn=int(((y==0)&(pred==0)).sum()); fp=int(((y==0)&(pred==1)).sum()); fn=int(((y==1)&(pred==0)).sum()); tp=int(((y==1)&(pred==1)).sum())
    return {"n":len(y),"acc":accuracy_score(y,pred),"sn":recall_score(y,pred,zero_division=0),"sp":tn/(tn+fp),"precision":precision_score(y,pred,zero_division=0),"f1":f1_score(y,pred,zero_division=0),"mcc":matthews_corrcoef(y,pred),"auroc":roc_auc_score(y,prob),"auprc":average_precision_score(y,prob),"tn":tn,"fp":fp,"fn":fn,"tp":tp}


@torch.no_grad()
def evaluate(model, X, y, device, batch=256):
    model.eval(); probs=[]
    for (xb,) in DataLoader(TensorDataset(torch.tensor(X)),batch_size=batch): probs.extend(torch.softmax(model(xb.to(device)),1)[:,1].cpu().tolist())
    return score(y,probs),np.asarray(probs)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--metadata",required=True); ap.add_argument("--split-manifest",required=True); ap.add_argument("--output-dir",required=True)
    ap.add_argument("--external-manifest",default=None,help="Optional labelled external target set used only after source-model selection")
    ap.add_argument("--seed",type=int,required=True); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--patience",type=int,default=8); ap.add_argument("--batch-size",type=int,default=32); ap.add_argument("--learning-rate",type=float,default=1e-3); args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    rows=load_metadata(args.metadata); train_all=[r for r in rows if r["split"]=="train"]; test=[r for r in rows if r["split"]=="test"]
    assign={r["seq_id"]:r["partition"] for r in csv.DictReader(open(args.split_manifest),delimiter="\t")}; train=[r for r in train_all if assign[r["seq_id"]]=="train"]; val=[r for r in train_all if assign[r["seq_id"]]=="validation"]
    if {r["sequence"] for r in train}&{r["sequence"] for r in val}: raise ValueError("exact sequence crosses train/validation")
    Xtr=np.stack([encode(r["sequence"]) for r in train]); Xv=np.stack([encode(r["sequence"]) for r in val]); Xt=np.stack([encode(r["sequence"]) for r in test])
    scaler=StandardScaler().fit(Xtr); Xtr=scaler.transform(Xtr).astype(np.float32); Xv=scaler.transform(Xv).astype(np.float32); Xt=scaler.transform(Xt).astype(np.float32)
    ytr=np.array([int(r["label"]) for r in train]); yv=np.array([int(r["label"]) for r in val]); yt=np.array([int(r["label"]) for r in test])
    gen=torch.Generator().manual_seed(args.seed); loader=DataLoader(TensorDataset(torch.tensor(Xtr),torch.tensor(ytr)),batch_size=args.batch_size,shuffle=True,generator=gen)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=ResSE().to(device); opt=torch.optim.Adam(model.parameters(),lr=args.learning_rate); loss_fn=nn.CrossEntropyLoss(); best=-2.; stale=0; history=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device); opt.zero_grad(set_to_none=True); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb)
        vm,_=evaluate(model,Xv,yv,device); rec={"epoch":epoch,"train_loss":total/len(ytr),**{f"val_{k}":v for k,v in vm.items()}}; history.append(rec); print(json.dumps(rec),flush=True)
        if vm["mcc"]>best+1e-8:
            best=vm["mcc"]; stale=0; torch.save({"state":model.state_dict(),"seed":args.seed,"epoch":epoch,"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_},out/"best_model.pt")
        else:
            stale+=1
            if stale>=args.patience: break
    ck=torch.load(out/"best_model.pt",map_location=device,weights_only=False); model.load_state_dict(ck["state"]); tm,prob=evaluate(model,Xt,yt,device)
    with open(out/"test_predictions.tsv","w",newline="") as f:
        w=csv.writer(f,delimiter="\t"); w.writerow(["seq_id","sequence","label","prob_np","pred_label"])
        for r,p in zip(test,prob): w.writerow([r["seq_id"],r["sequence"],r["label"],f"{p:.10g}",int(p>=.5)])
    result={"status":"NeuroPred-ResSE reimplementation","config":vars(args),"train_n":len(train),"validation_n":len(val),"test_n":len(test),"selected_epoch":ck["epoch"],"test_metrics":tm,"elapsed_seconds":time.time()-started}; (out/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    if args.external_manifest:
        external=load_metadata(args.external_manifest)
        Xe=np.stack([encode(r["sequence"]) for r in external]); Xe=scaler.transform(Xe).astype(np.float32)
        ye=np.array([int(r["label"]) for r in external]); em,eprob=evaluate(model,Xe,ye,device)
        with open(out/"external_predictions.tsv","w",newline="") as f:
            fields=list(external[0].keys())+["prob_np","pred"]
            w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader()
            for r,p in zip(external,eprob): w.writerow({**r,"prob_np":f"{p:.10g}","pred":int(p>=.5)})
        (out/"external_metrics.json").write_text(json.dumps(em,indent=2)+"\n")
    with open(out/"history.tsv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=history[0].keys(),delimiter="\t"); w.writeheader(); w.writerows(history)
    print(json.dumps(result),flush=True)


if __name__=="__main__": main()
