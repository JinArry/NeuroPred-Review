#!/usr/bin/env python3
"""Remove every training record whose exact sequence occurs in the fixed test set."""
import argparse, csv, json
from pathlib import Path

ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output-dir",type=Path,required=True); a=ap.parse_args()
rows=list(csv.DictReader(a.input.open(),delimiter="\t")); fields=list(rows[0])
test_seq={r["sequence"] for r in rows if r["split"]=="test"}
removed=[r for r in rows if r["split"]=="train" and r["sequence"] in test_seq]
kept=[r for r in rows if not (r["split"]=="train" and r["sequence"] in test_seq)]
a.output_dir.mkdir(parents=True,exist_ok=True)
with (a.output_dir/"metadata.tsv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(kept)
with (a.output_dir/"removed_train_test_exact_duplicates.tsv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(removed)
summary={"input_n":len(rows),"output_n":len(kept),"removed_training_records":len(removed),"removed_unique_sequences":len({r['sequence'] for r in removed}),"train_n":sum(r['split']=='train' for r in kept),"test_n":sum(r['split']=='test' for r in kept),"postfilter_exact_train_test_overlap":len({r['sequence'] for r in kept if r['split']=='train'}&{r['sequence'] for r in kept if r['split']=='test'})}
(a.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary))
