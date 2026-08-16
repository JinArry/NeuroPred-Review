#!/usr/bin/env python3
"""Audit and aggregate five unified MSKDNP reimplementation seeds."""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score

KEYS=["ACC","SN","SP","Precision","F1","MCC","AUROC","AUPRC"]
def score(y,p):
    q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
    return {"ACC":accuracy_score(y,q),"SN":recall_score(y,q),"SP":tn/(tn+fp),"Precision":precision_score(y,q),"F1":f1_score(y,q),"MCC":matthews_corrcoef(y,q),"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("root",type=Path); p.add_argument("--expected-test",type=Path,required=True); a=p.parse_args()
    expected=pd.read_csv(a.expected_test)[["seq_id","label"]]; expected_ids=expected.seq_id.tolist(); records=[]; predictions=[]; teacher_records=[]
    for seed_dir in sorted(a.root.glob("seed_*")):
        seed=int(seed_dir.name.split("_")[-1]); pred=pd.read_csv(seed_dir/"student"/"test_predictions.tsv",sep="\t")
        assert len(pred)==871 and pred.seq_id.tolist()==expected_ids and pred.label.tolist()==expected.label.tolist()
        assert pred.seq_id.nunique()==871 and pred.probability.between(0,1).all()
        calculated=score(pred.label.to_numpy(),pred.probability.to_numpy()); payload=json.loads((seed_dir/"student"/"metrics.json").read_text())
        for key, source in (("ACC","ACC"),("SN","Recall"),("SP","SP"),("Precision","Precision"),("F1","F1"),("MCC","MCC"),("AUROC","AUROC"),("AUPRC","AUPRC")): assert abs(calculated[key]-payload["test"][source])<1e-12
        records.append({"seed":seed,**calculated,"selected_epoch":payload["selected_epoch"],"best_val_mcc":payload["best_val_mcc"]}); predictions.append(pred)
        teacher=json.loads((seed_dir/"teacher"/"metrics.json").read_text()); tm=teacher["tests"]["setmain"]
        teacher_records.append({"seed":seed,"ACC":tm["ACC"],"SN":tm["Recall"],"SP":tm["SP"],"Precision":tm["Precision"],"F1":tm["F1"],"MCC":tm["MCC"],"AUROC":tm["AUROC"],"AUPRC":tm["AUPRC"]})
    assert len(records)==5
    frame=pd.DataFrame(records); frame.to_csv(a.root/"per_seed_metrics.tsv",sep="\t",index=False); pd.DataFrame(teacher_records).to_csv(a.root/"teacher_per_seed_metrics.tsv",sep="\t",index=False)
    summary={k:{"mean":float(frame[k].mean()),"sd":float(frame[k].std(ddof=1))} for k in KEYS}
    ensemble=expected.copy(); ensemble["probability"]=np.mean([x.probability.to_numpy() for x in predictions],axis=0); ensemble["prediction"]=(ensemble.probability>=.5).astype(int); ensemble.to_csv(a.root/"ensemble_predictions.tsv",sep="\t",index=False); ensemble_metrics=score(ensemble.label.to_numpy(),ensemble.probability.to_numpy())
    payload={"method":"MSKDNP (reimplemented)","n_seeds":5,"audited_test_n":871,"mean_sd":summary,"ensemble":ensemble_metrics}; (a.root/"summary.json").write_text(json.dumps(payload,indent=2)+"\n")
    pd.DataFrame([{"metric":k,**v} for k,v in summary.items()]).to_csv(a.root/"summary_metrics.tsv",sep="\t",index=False)
    row="| MSKDNP (reimplemented) | "+" | ".join(f'{summary[k]["mean"]:.4f} ± {summary[k]["sd"]:.4f}' for k in KEYS)+" |\n"; (a.root/"paper_table_row.md").write_text(row); (a.root/"COMPLETE").write_text("five fixed-split seeds complete; 871 test IDs audited\n")
    print(row,end=""); print(json.dumps(ensemble_metrics,indent=2))

if __name__=="__main__": main()
