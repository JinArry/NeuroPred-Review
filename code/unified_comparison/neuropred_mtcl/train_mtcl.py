#!/usr/bin/env python3
"""PyTorch execution adapter for the released NeuroPred-MTCL architecture."""
import argparse, copy, json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, matthews_corrcoef, precision_score, recall_score, roc_auc_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReleasedMTCL(nn.Module):
    def __init__(self, hidden=160, heads=8, projection=96):
        super().__init__()
        self.encoder = nn.LSTM(768, hidden, batch_first=True, bidirectional=True)
        # Keras MHA uses key_dim=hidden per head, not hidden/heads.
        width = heads * hidden
        self.heads, self.hidden = heads, hidden
        self.q = nn.Linear(hidden * 2, width); self.k = nn.Linear(hidden * 2, width)
        self.v = nn.Linear(hidden * 2, width); self.out = nn.Linear(width, hidden * 2)
        self.pool_score = nn.Linear(hidden * 2, 1)
        self.main = nn.Linear(hidden * 2, 2); self.aux = nn.Linear(hidden * 2, 2)
        self.projector = nn.Sequential(nn.Linear(hidden * 2, hidden * 2), nn.ReLU(), nn.Linear(hidden * 2, projection))

    def forward(self, x):
        x, _ = self.encoder(x)
        b, n, _ = x.shape
        def split(v): return v.reshape(b, n, self.heads, self.hidden).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.hidden ** .5, dim=-1)
        x = self.out((attn @ v).transpose(1, 2).reshape(b, n, -1))
        pooled = (x * torch.softmax(torch.tanh(self.pool_score(x)), dim=1)).sum(1)
        return self.main(pooled), self.aux(pooled), self.projector(pooled)


def released_kld(raw_main, raw_aux):
    # tf.keras.losses.KLDivergence clips both raw tensors to [epsilon, 1].
    eps = 1e-7
    p = raw_main.detach().clamp(eps, 1.0); q = raw_aux.clamp(eps, 1.0)
    return (p * (p.log() - q.log())).sum(1).mean()


def released_nt_xent(z, temperature=.5):
    z = nn.functional.normalize(z, dim=1)
    logits = z @ z.T / temperature
    return nn.functional.cross_entropy(logits, torch.arange(len(z), device=z.device))


def metrics(y, p):
    pred = (p >= .5).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"ACC": accuracy_score(y,pred), "SN": recall_score(y,pred), "SP": tn/(tn+fp),
            "Precision": precision_score(y,pred), "F1": f1_score(y,pred), "MCC": matthews_corrcoef(y,pred),
            "AUROC": roc_auc_score(y,p), "AUPRC": average_precision_score(y,p)}


@torch.no_grad()
def predict(model, x, batch, device):
    model.eval(); out=[]
    for (xb,) in DataLoader(TensorDataset(x), batch_size=batch):
        out.append(torch.softmax(model(xb.to(device))[0], 1)[:,1].cpu())
    return torch.cat(out).numpy()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--split",type=Path,required=True)
    ap.add_argument("--metadata",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--epochs",type=int,default=50); ap.add_argument("--batch-size",type=int,default=64); a=ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic=True; device=torch.device("cuda")
    # The feature archive is generated locally by extract_esm1_t6.py; seq_id is
    # stored as a NumPy object string array, hence trusted pickle loading here.
    z=np.load(a.features, allow_pickle=True); ids=z["seq_id"].astype(str); X=z["features"]
    pos={v:i for i,v in enumerate(ids)}; split=pd.read_csv(a.split,sep="\t"); meta=pd.read_csv(a.metadata,sep="\t")
    tr=split[split.partition=="train"].seq_id.map(pos).to_numpy(); va=split[split.partition=="validation"].seq_id.map(pos).to_numpy()
    te=meta[meta.split=="test"].seq_id.map(pos).to_numpy(); y=meta.set_index("seq_id").label
    yt=torch.tensor(split[split.partition=="train"].label.to_numpy(),dtype=torch.long); yv=split[split.partition=="validation"].label.to_numpy(); ytest=meta[meta.split=="test"].label.to_numpy()
    xt=torch.from_numpy(X[tr].astype(np.float32)); xv=torch.from_numpy(X[va].astype(np.float32)); xte=torch.from_numpy(X[te].astype(np.float32))
    loader=DataLoader(TensorDataset(xt,yt),batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed))
    model=ReleasedMTCL().to(device); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    best=-1.; state=None; stale=0; lr_stale=0; history=[]
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for xb,yb in loader:
            xb,yb=xb.to(device),yb.to(device); main,aux,proj=model(xb)
            loss=nn.functional.cross_entropy(main,yb)+.6*released_kld(main,aux)+.03*released_nt_xent(proj)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        pv=predict(model,xv,a.batch_size,device); acc=accuracy_score(yv,pv>=.5); history.append({"epoch":epoch,"loss":np.mean(losses),"val_accuracy":acc,"lr":opt.param_groups[0]["lr"]})
        print(f"seed={a.seed} epoch={epoch} loss={np.mean(losses):.5f} val_acc={acc:.5f}",flush=True)
        if acc>best:
            best=acc; state=copy.deepcopy(model.state_dict()); stale=0; lr_stale=0
        else:
            stale+=1; lr_stale+=1
            if lr_stale>=3: opt.param_groups[0]["lr"]=max(opt.param_groups[0]["lr"]*.5,1e-6); lr_stale=0
            if stale>=8: break
    model.load_state_dict(state); pt=predict(model,xte,a.batch_size,device); result=metrics(ytest,pt)
    a.output.mkdir(parents=True,exist_ok=True); pd.DataFrame(history).to_csv(a.output/"history.tsv",sep="\t",index=False)
    pd.DataFrame({"seq_id":meta[meta.split=="test"].seq_id,"label":ytest,"probability":pt}).to_csv(a.output/"test_predictions.tsv",sep="\t",index=False)
    (a.output/"metrics.json").write_text(json.dumps({**result,"seed":a.seed,"best_val_accuracy":best,"epochs_ran":len(history)},indent=2))
    torch.save(state,a.output/"best_model.pt"); print(json.dumps(result),flush=True)


if __name__=="__main__": main()
