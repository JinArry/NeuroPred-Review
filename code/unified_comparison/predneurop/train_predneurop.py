#!/usr/bin/env python3
"""Retrain the released PredNeuroP feature/stacking pipeline on fixed splits."""
import argparse, json, os, sys, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def metrics(y,p):
 q=(p>=.5).astype(int); tn,fp,fn,tp=confusion_matrix(y,q,labels=[0,1]).ravel()
 return {"ACC":accuracy_score(y,q),"SN":recall_score(y,q),"SP":tn/(tn+fp),"Precision":precision_score(y,q),"F1":f1_score(y,q),"MCC":matthews_corrcoef(y,q),"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p)}

def build_models(seed):
 return [ExtraTreesClassifier(n_estimators=100,random_state=seed,n_jobs=8),
         XGBClassifier(n_jobs=8,random_state=seed), MLPClassifier(max_iter=5000,random_state=seed),
         LogisticRegression(solver="liblinear",random_state=seed), MLPClassifier(max_iter=5000,random_state=seed),
         KNeighborsClassifier(n_jobs=8), MLPClassifier(max_iter=5000,random_state=seed), XGBClassifier(n_jobs=8,random_state=seed)]

def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--repo",type=Path,required=True); ap.add_argument("--metadata",type=Path,required=True); ap.add_argument("--split",type=Path,required=True); ap.add_argument("--features",type=Path,required=True); ap.add_argument("--seed",type=int,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args(); started=time.time()
 a.repo=a.repo.resolve(); a.metadata=a.metadata.resolve(); a.split=a.split.resolve(); a.features=a.features.resolve(); a.output=a.output.resolve(); os.chdir(str(a.repo)); sys.path.insert(0,str(a.repo)); from Feature import all_feature
 meta=pd.read_csv(a.metadata,sep="\t")
 if a.features.exists(): z=np.load(a.features,allow_pickle=True); X=z["features"]; ids=z["seq_id"].astype(str)
 else:
  X,groups=all_feature(meta.sequence.tolist()); a.features.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.features,features=X,seq_id=meta.seq_id.to_numpy(),group_lengths=np.array([len(x) for x in groups])); ids=meta.seq_id.astype(str).to_numpy()
 pos={v:i for i,v in enumerate(ids)}; split=pd.read_csv(a.split,sep="\t"); tr=split[split.partition=="train"].seq_id.map(pos).to_numpy(); va=split[split.partition=="validation"].seq_id.map(pos).to_numpy(); te=meta[meta.split=="test"].seq_id.map(pos).to_numpy(); y=meta.label.to_numpy(); lengths=np.load(a.features,allow_pickle=True)["group_lengths"]; bounds=np.r_[0,np.cumsum(lengths)]
 models=build_models(a.seed); cv=StratifiedKFold(n_splits=10,shuffle=True,random_state=a.seed); oof=np.zeros((len(tr),8)); vp=np.zeros((len(va),8)); tp=np.zeros((len(te),8))
 for j,(model,lo,hi) in enumerate(zip(models,bounds[:-1],bounds[1:])):
  for fit,hold in cv.split(X[tr,lo:hi],y[tr]):
   model.fit(X[tr[fit],lo:hi],y[tr[fit]]); oof[hold,j]=model.predict_proba(X[tr[hold],lo:hi])[:,1]
  model.fit(X[tr,lo:hi],y[tr]); vp[:,j]=model.predict_proba(X[va,lo:hi])[:,1]; tp[:,j]=model.predict_proba(X[te,lo:hi])[:,1]
 meta_model=LogisticRegression(solver="liblinear",random_state=a.seed).fit(oof,y[tr]); pv=meta_model.predict_proba(vp)[:,1]; pt=meta_model.predict_proba(tp)[:,1]
 a.output.mkdir(parents=True,exist_ok=True); result={**metrics(y[te],pt),"seed":a.seed,"validation_metrics":metrics(y[va],pv),"elapsed_seconds":time.time()-started}; (a.output/"metrics.json").write_text(json.dumps(result,indent=2)); pd.DataFrame({"seq_id":meta[meta.split=="test"].seq_id,"label":y[te],"probability":pt,"prediction":(pt>=.5).astype(int)}).to_csv(a.output/"test_predictions.tsv",sep="\t",index=False); joblib.dump(meta_model,a.output/"meta_model.joblib"); print(json.dumps(result),flush=True)
if __name__=="__main__": main()
