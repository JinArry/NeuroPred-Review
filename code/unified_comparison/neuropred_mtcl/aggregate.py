#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

METRICS = ["ACC", "SN", "SP", "Precision", "F1", "MCC", "AUROC", "AUPRC"]

def score(y, p):
    q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {"ACC":accuracy_score(y,q),"SN":recall_score(y,q),"SP":tn/(tn+fp),"Precision":precision_score(y,q),"F1":f1_score(y,q),"MCC":matthews_corrcoef(y,q),"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("root",type=Path); a=ap.parse_args()
    rows=[]; preds=[]
    for d in sorted(a.root.glob("seed_*")):
        if not (d/"metrics.json").exists(): continue
        m=json.loads((d/"metrics.json").read_text()); rows.append(m); preds.append(pd.read_csv(d/"test_predictions.tsv",sep="\t"))
    frame=pd.DataFrame(rows); frame.to_csv(a.root/"per_seed_metrics.tsv",sep="\t",index=False)
    summary={k:{"mean":float(frame[k].mean()),"sd":float(frame[k].std(ddof=1))} for k in METRICS}
    ens=preds[0][["seq_id","label"]].copy(); ens["probability"]=np.mean([x.probability.to_numpy() for x in preds],axis=0)
    ens.to_csv(a.root/"ensemble_predictions.tsv",sep="\t",index=False); ensemble=score(ens.label.to_numpy(),ens.probability.to_numpy())
    (a.root/"summary.json").write_text(json.dumps({"n_seeds":len(rows),"mean_sd":summary,"ensemble":ensemble},indent=2))
    pd.DataFrame([{"metric":k,**v} for k,v in summary.items()]).to_csv(a.root/"summary_metrics.tsv",sep="\t",index=False)
    table="| Method | "+" | ".join(METRICS)+" |\n|---|"+"---:|"*len(METRICS)+"\n| NeuroPred-MTCL | "+" | ".join(f'{summary[k]["mean"]:.4f} ± {summary[k]["sd"]:.4f}' for k in METRICS)+" |\n"
    (a.root/"paper_table_row.md").write_text(table)
    (a.root/"COMPLETE").write_text("five seeds complete\n")
    print(table); print("Ensemble",json.dumps(ensemble))
if __name__=="__main__": main()
