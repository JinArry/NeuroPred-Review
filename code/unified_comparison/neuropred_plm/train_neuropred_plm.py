#!/usr/bin/env python3
"""Unified retraining adapter for the published NeuroPred-PLM architecture."""
import argparse, copy, csv, json, random, sys, time
from pathlib import Path
import esm, numpy as np, torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

def read(path): return list(csv.DictReader(open(path),delimiter="\t"))
class DS(Dataset):
 def __init__(self,r): self.r=r
 def __len__(self): return len(self.r)
 def __getitem__(self,i): return self.r[i]
def collate(x): return x
def metrics(y,p):
 y=np.asarray(y,int); p=np.asarray(p); q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return {"acc":accuracy_score(y,q),"sn":recall_score(y,q),"sp":tn/(tn+fp),"precision":precision_score(y,q),"f1":f1_score(y,q),"mcc":matthews_corrcoef(y,q),"auroc":roc_auc_score(y,p),"auprc":average_precision_score(y,p)}
@torch.no_grad()
def evaluate(model,loader,device):
 model.eval(); ys=[]; ps=[]; ids=[]
 for b in loader:
  pairs=[(r["seq_id"],r["sequence"]) for r in b]; logits,_=model(pairs,device); p=torch.softmax(logits,1)[:,1].cpu().numpy(); ps.extend(p); ys.extend(int(r["label"]) for r in b); ids.extend(r["seq_id"] for r in b)
 return metrics(ys,ps),ids,np.asarray(ys),np.asarray(ps)
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--metadata",required=True); ap.add_argument("--split",required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--batch-size",type=int,default=16); a=ap.parse_args(); started=time.time()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); torch.serialization.add_safe_globals([argparse.Namespace]); sys.path.insert(0,str(a.repo.resolve())); from NeuroPredPLM.model import EsmModel
 rows=read(a.metadata); assign={r["seq_id"]:r["partition"] for r in read(a.split)}; train=[r for r in rows if r["split"]=="train" and assign[r["seq_id"]]=="train"]
 val=[r for r in rows if r["split"]=="train" and assign[r["seq_id"]]=="validation"]; test=[r for r in rows if r["split"]=="test"]
 model=EsmModel(); pretrained,_=esm.pretrained.esm1_t12_85M_UR50S(); model.esm=pretrained
 for p in model.esm.parameters(): p.requires_grad=False
 for layer in model.esm.layers[-3:]:
  for p in layer.parameters(): p.requires_grad=True
 device=torch.device("cuda"); model.to(device); head=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith("esm.")]; back=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith("esm.")]; opt=torch.optim.Adam([{"params":head,"lr":5e-4},{"params":back,"lr":2e-6}]); lossfn=nn.CrossEntropyLoss(label_smoothing=.1)
 gen=torch.Generator().manual_seed(a.seed); tl=DataLoader(DS(train),batch_size=a.batch_size,shuffle=True,generator=gen,collate_fn=collate); vl=DataLoader(DS(val),batch_size=a.batch_size,collate_fn=collate); el=DataLoader(DS(test),batch_size=a.batch_size,collate_fn=collate)
 best=-2.; state=None; stale=0; hist=[]
 for epoch in range(1,a.epochs+1):
  model.train(); total=0
  for b in tl:
   pairs=[(r["seq_id"],r["sequence"]) for r in b]; y=torch.tensor([int(r["label"]) for r in b],device=device); logits,_=model(pairs,device); loss=lossfn(logits,y); opt.zero_grad(); loss.backward(); opt.step(); total+=loss.item()*len(b)
  vm,_,_,_=evaluate(model,vl,device); hist.append({"epoch":epoch,"loss":total/len(train),**{"val_"+k:v for k,v in vm.items()}}); print(json.dumps(hist[-1]),flush=True)
  if vm["mcc"]>best: best=vm["mcc"]; state=copy.deepcopy(model.state_dict()); stale=0
  else:
   stale+=1
   if stale>=8: break
 model.load_state_dict(state); tm,ids,y,p=evaluate(model,el,device); a.output.mkdir(parents=True,exist_ok=True); (a.output/"metrics.json").write_text(json.dumps({"seed":a.seed,"selected_val_mcc":best,"test_metrics":tm,"elapsed_seconds":time.time()-started},indent=2));
 with open(a.output/"test_predictions.tsv","w",newline="") as f:
  w=csv.writer(f,delimiter="\t"); w.writerow(["seq_id","label","prob_np","pred_label"]); w.writerows((i,int(t),float(x),int(x>=.5)) for i,t,x in zip(ids,y,p))
 with open(a.output/"history.tsv","w",newline="") as f: w=csv.DictWriter(f,fieldnames=hist[0].keys(),delimiter="\t"); w.writeheader(); w.writerows(hist)
 torch.save(state,a.output/"best_model.pt"); print(json.dumps(tm),flush=True)
if __name__=="__main__": main()
