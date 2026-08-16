#!/usr/bin/env python3
"""Paper-described reconstruction of the missing MSKDNP ESM2-650M teacher."""
import argparse, copy, csv, json, random, time
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

class SeqDS(Dataset):
 def __init__(self,df): self.df=df.reset_index(drop=True)
 def __len__(self): return len(self.df)
 def __getitem__(self,i): return self.df.iloc[i].to_dict()

class Teacher(nn.Module):
 def __init__(self,name,pooling="masked"):
  super().__init__(); self.encoder=AutoModel.from_pretrained(name); self.pooling=pooling; h=self.encoder.config.hidden_size; self.classifier=nn.Sequential(nn.Linear(h,512),nn.ReLU(),nn.Linear(512,256),nn.ReLU(),nn.Linear(256,2))
 def forward(self,tokens):
  h=self.encoder(**tokens).last_hidden_state
  if self.pooling=="paper_unmasked": pooled=h.mean(1)
  else:
   ids=tokens["input_ids"]; mask=tokens["attention_mask"].bool()
   for special in (0,1,2,3): mask &= ids.ne(special)
   pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
  return self.classifier(pooled),pooled

def grouped_split(df,seed,fraction=.1):
 groups=defaultdict(list)
 for i,r in df.iterrows(): groups[(str(r.seq),int(r.label))].append(i)
 val=set()
 for label in (0,1):
  gs=[v for (s,y),v in groups.items() if y==label]; random.Random(seed*1009+label).shuffle(gs); target=round((df.label==label).sum()*fraction); n=0
  for g in gs:
   if n+len(g)<=target: val.update(g); n+=len(g)
   if n==target: break
 return df.loc[~df.index.isin(val)].copy(),df.loc[df.index.isin(val)].copy()

def collator(tokenizer,maxlen,pooling):
 def f(batch):
  padding="max_length" if pooling=="paper_unmasked" else True
  tok=tokenizer([str(x["seq"]) for x in batch],padding=padding,truncation=True,max_length=maxlen,return_tensors="pt"); y=torch.tensor([int(x["label"]) for x in batch]); return tok,y,batch
 return f
def metric(y,p):
 y=np.asarray(y);p=np.asarray(p);q=(p>=.5).astype(int);tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel();return {"ACC":accuracy_score(y,q),"Precision":precision_score(y,q),"Recall":recall_score(y,q),"SP":tn/(tn+fp),"F1":f1_score(y,q),"MCC":matthews_corrcoef(y,q),"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p),"tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}
@torch.no_grad()
def evaluate(model,loader,device):
 model.eval();ys=[];ps=[];ids=[]
 for tok,y,b in loader:
  tok={k:v.to(device) for k,v in tok.items()};logits,_=model(tok);ps.extend(torch.softmax(logits,1)[:,1].cpu().tolist());ys.extend(y.tolist());ids.extend(str(x.get("seq_id",x.get("id",""))) for x in b)
 return metric(ys,ps),ids,np.asarray(ys),np.asarray(ps)

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--train",required=True);ap.add_argument("--validation");ap.add_argument("--test",required=True);ap.add_argument("--test-extra",required=True);ap.add_argument("--output",type=Path,required=True);ap.add_argument("--seed",type=int,default=42);ap.add_argument("--model",default="facebook/esm2_t33_650M_UR50D");ap.add_argument("--pooling",choices=("masked","paper_unmasked"),default="masked");ap.add_argument("--epochs",type=int,default=15);ap.add_argument("--batch-size",type=int,default=1);ap.add_argument("--accumulation",type=int,default=16);ap.add_argument("--lr",type=float,default=1e-5);ap.add_argument("--max-length",type=int,default=102);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 df=pd.read_csv(a.train)
 if a.validation: tr,va=df,pd.read_csv(a.validation)
 else: tr,va=grouped_split(df,a.seed)
 tests={"setmain":pd.read_csv(a.test),"setextra":pd.read_csv(a.test_extra)};tokenizer=AutoTokenizer.from_pretrained(a.model);model=Teacher(a.model,a.pooling);model.encoder.gradient_checkpointing_enable();device=torch.device("cuda");model.to(device);coll=collator(tokenizer,a.max_length,a.pooling);gen=torch.Generator().manual_seed(a.seed);tl=DataLoader(SeqDS(tr),batch_size=a.batch_size,shuffle=True,generator=gen,collate_fn=coll);vl=DataLoader(SeqDS(va),batch_size=16,collate_fn=coll);opt=torch.optim.AdamW(model.parameters(),lr=a.lr);lossfn=nn.CrossEntropyLoss();scaler=None;best=-2.;stale=0;history=[];started=time.time()
 for epoch in range(1,a.epochs+1):
  model.train();opt.zero_grad(set_to_none=True);total=0
  for step,(tok,y,b) in enumerate(tl,1):
   tok={k:v.to(device) for k,v in tok.items()};y=y.to(device)
   with torch.autocast("cuda",dtype=torch.bfloat16): logits,_=model(tok);loss=lossfn(logits,y)/a.accumulation
   loss.backward();total+=loss.item()*a.accumulation*len(y)
   if step%a.accumulation==0 or step==len(tl): torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad(set_to_none=True)
  vm,_,_,_=evaluate(model,vl,device);rec={"epoch":epoch,"train_loss":total/len(tr),**{"val_"+k:v for k,v in vm.items()}};history.append(rec);print(json.dumps(rec),flush=True)
  if vm["MCC"]>best:best=vm["MCC"];stale=0;torch.save({"state":model.state_dict(),"epoch":epoch,"seed":a.seed},a.output/"best_teacher.pt")
  else:
   stale+=1
   if stale>=3:break
 ck=torch.load(a.output/"best_teacher.pt",map_location=device,weights_only=False);model.load_state_dict(ck["state"]);result={"selected_epoch":ck["epoch"],"best_val_mcc":best,"seed":a.seed,"train_n":len(tr),"validation_n":len(va),"elapsed_seconds":time.time()-started,"tests":{}}
 for name,d in tests.items():
  met,ids,y,p=evaluate(model,DataLoader(SeqDS(d),batch_size=16,collate_fn=coll),device);result["tests"][name]=met;pd.DataFrame({"seq_id":d.get("seq_id",pd.Series(ids)).astype(str).tolist(),"row_index":np.arange(len(d)),"label":y,"probability":p,"prediction":(p>=.5).astype(int)}).to_csv(a.output/f"{name}_predictions.tsv",sep="\t",index=False)
 pd.DataFrame(history).to_csv(a.output/"history.tsv",sep="\t",index=False);(a.output/"metrics.json").write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=="__main__":main()
