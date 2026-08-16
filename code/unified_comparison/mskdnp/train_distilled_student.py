#!/usr/bin/env python3
"""Validation-selected reconstruction of the MSKDNP distilled ESM2-8M student."""
import argparse, json, random, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def grouped_indices(frame, seed, fraction=.1):
    groups = defaultdict(list)
    for i, row in frame.iterrows(): groups[(str(row.seq), int(row.label))].append(i)
    validation = set()
    for label in (0, 1):
        candidates = [v for (seq, y), v in groups.items() if y == label]
        random.Random(seed * 1009 + label).shuffle(candidates)
        target, count = round((frame.label == label).sum() * fraction), 0
        for group in candidates:
            if count + len(group) <= target: validation.update(group); count += len(group)
            if count == target: break
    train = [i for i in range(len(frame)) if i not in validation]
    return train, sorted(validation)


class DistillDS(Dataset):
    def __init__(self, frame, logits=None, features=None, indices=None):
        self.frame = frame.reset_index(drop=True); self.logits = logits; self.features = features
        self.indices = list(range(len(frame))) if indices is None else indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, j):
        i = self.indices[j]; row = self.frame.iloc[i]
        item = {"seq": str(row.seq), "label": int(row.label), "row_index": i}
        if self.logits is not None: item.update(teacher_logits=self.logits[i], teacher_features=self.features[i])
        return item


class Student(nn.Module):
    def __init__(self, name):
        super().__init__(); self.encoder = AutoModel.from_pretrained(name); h = self.encoder.config.hidden_size
        self.mapping = nn.Linear(h, 1280)
        self.classifier = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))
    def forward(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        # This deliberately follows the authors' released training code.
        mapped = self.mapping(hidden.mean(dim=1))
        return self.classifier(mapped), mapped


class BaseHeadStudent(nn.Module):
    """Architecture used by the model actually released by the authors."""
    def __init__(self, name, released_dir=None):
        super().__init__()
        source = released_dir or name
        self.student_model = AutoModelForSequenceClassification.from_pretrained(source, num_labels=2)
        self.mapping = nn.Linear(self.student_model.config.hidden_size, 1280)
        if released_dir:
            state = torch.load(Path(released_dir) / "student_model_with_mapping.pt", map_location="cpu", weights_only=False)
            self.load_state_dict(state)
    def forward(self, tokens):
        outputs = self.student_model(**tokens, output_hidden_states=True)
        mapped = self.mapping(outputs.hidden_states[-1].mean(dim=1))
        return outputs.logits, mapped


def metric(y, p):
    y, p = np.asarray(y), np.asarray(p); q = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return {"ACC": accuracy_score(y,q), "Precision": precision_score(y,q,zero_division=0), "Recall": recall_score(y,q,zero_division=0), "SP": tn/(tn+fp), "F1": f1_score(y,q,zero_division=0), "MCC": matthews_corrcoef(y,q), "AUROC": roc_auc_score(y,p), "AUPRC": average_precision_score(y,p), "tn":int(tn), "fp":int(fp), "fn":int(fn), "tp":int(tp)}


def make_collator(tokenizer, max_length):
    def collate(batch):
        tokens = tokenizer([x["seq"] for x in batch], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        labels = torch.tensor([x["label"] for x in batch])
        teacher_logits = None if "teacher_logits" not in batch[0] else torch.tensor(np.stack([x["teacher_logits"] for x in batch]))
        teacher_features = None if "teacher_features" not in batch[0] else torch.tensor(np.stack([x["teacher_features"] for x in batch]))
        return tokens, labels, teacher_logits, teacher_features, [x["row_index"] for x in batch]
    return collate


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); labels=[]; probabilities=[]; rows=[]
    for tokens, y, _, _, idx in loader:
        tokens={k:v.to(device) for k,v in tokens.items()}; logits,_=model(tokens)
        labels.extend(y.tolist()); probabilities.extend(torch.softmax(logits,1)[:,1].cpu().tolist()); rows.extend(idx)
    return metric(labels, probabilities), np.asarray(rows), np.asarray(labels), np.asarray(probabilities)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--train",type=Path,required=True); p.add_argument("--validation",type=Path); p.add_argument("--test",type=Path,required=True); p.add_argument("--targets",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--model",default="facebook/esm2_t6_8M_UR50D"); p.add_argument("--architecture",choices=("mapped_mlp","base_head"),default="mapped_mlp"); p.add_argument("--released-student-dir",type=Path); p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=15); p.add_argument("--patience",type=int,default=3); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--lr",type=float,default=5e-5); p.add_argument("--alpha",type=float,default=.5); p.add_argument("--beta",type=float,default=.5); p.add_argument("--temperature",type=float,default=5.); p.add_argument("--max-length",type=int,default=100); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    frame=pd.read_csv(a.train); test=pd.read_csv(a.test); teacher_logits=np.load(a.targets/"teacher_logits.npy"); teacher_features=np.load(a.targets/"teacher_features.npy")
    assert len(frame)==len(teacher_logits)==len(teacher_features)
    if a.validation: tr=list(range(len(frame))); val_frame=pd.read_csv(a.validation); va=list(range(len(val_frame)))
    else: tr,va=grouped_indices(frame,a.seed); val_frame=frame
    tokenizer=AutoTokenizer.from_pretrained(a.model); collate=make_collator(tokenizer,a.max_length); generator=torch.Generator().manual_seed(a.seed)
    train_loader=DataLoader(DistillDS(frame,teacher_logits,teacher_features,tr),batch_size=a.batch_size,shuffle=True,generator=generator,collate_fn=collate)
    val_loader=DataLoader(DistillDS(val_frame,indices=va),batch_size=64,collate_fn=collate); test_loader=DataLoader(DistillDS(test),batch_size=64,collate_fn=collate)
    device=torch.device("cuda"); model=(Student(a.model) if a.architecture=="mapped_mlp" else BaseHeadStudent(a.model,a.released_student_dir)).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr); best=-2.; stale=0; history=[]; started=time.time()
    # A released checkpoint is itself a meaningful calibration baseline. Allow
    # validation selection to retain epoch 0 if reconstructed-teacher updates hurt.
    if a.released_student_dir:
        vm,_,_,_=evaluate(model,val_loader,device); best=vm["MCC"]; history.append({"epoch":0,"train_loss":None,**{"val_"+k:v for k,v in vm.items()}}); torch.save({"state":model.state_dict(),"epoch":0,"seed":a.seed},a.output/"best_student.pt"); print(json.dumps(history[-1]),flush=True)
    for epoch in range(1,a.epochs+1):
        model.train(); total=0.
        for tokens,y,tlogits,tfeatures,_ in train_loader:
            tokens={k:v.to(device) for k,v in tokens.items()}; y=y.to(device); tlogits=tlogits.to(device); tfeatures=tfeatures.to(device); optimizer.zero_grad(set_to_none=True)
            logits,features=model(tokens); hard=F.cross_entropy(logits,y); soft=F.kl_div(F.log_softmax(logits/a.temperature,dim=-1),F.softmax(tlogits/a.temperature,dim=-1),reduction="batchmean")*(a.temperature**2); feature=F.mse_loss(features,tfeatures); loss=a.alpha*hard+(1-a.alpha)*soft+a.beta*feature; loss.backward(); optimizer.step(); total+=loss.item()*len(y)
        vm,_,_,_=evaluate(model,val_loader,device); record={"epoch":epoch,"train_loss":total/len(tr),**{"val_"+k:v for k,v in vm.items()}}; history.append(record); print(json.dumps(record),flush=True)
        if vm["MCC"]>best: best=vm["MCC"]; stale=0; torch.save({"state":model.state_dict(),"epoch":epoch,"seed":a.seed},a.output/"best_student.pt")
        else:
            stale+=1
            if stale>=a.patience: break
    checkpoint=torch.load(a.output/"best_student.pt",map_location=device,weights_only=False); model.load_state_dict(checkpoint["state"]); tm,rows,y,pred=evaluate(model,test_loader,device)
    pd.DataFrame({"seq_id":test.iloc[rows].get("seq_id",pd.Series(rows)).astype(str).tolist(),"row_index":rows,"label":y,"probability":pred,"prediction":(pred>=.5).astype(int)}).to_csv(a.output/"test_predictions.tsv",sep="\t",index=False)
    pd.DataFrame(history).to_csv(a.output/"history.tsv",sep="\t",index=False)
    result={"selected_epoch":checkpoint["epoch"],"best_val_mcc":best,"seed":a.seed,"train_n":len(tr),"validation_n":len(va),"elapsed_seconds":time.time()-started,"test":tm,"hyperparameters":{"architecture":a.architecture,"released_student_initialization":str(a.released_student_dir) if a.released_student_dir else None,"alpha":a.alpha,"beta":a.beta,"temperature":a.temperature,"lr":a.lr}}
    (a.output/"metrics.json").write_text(json.dumps(result,indent=2)); print(json.dumps(result),flush=True)


if __name__=="__main__": main()
