#!/usr/bin/env python3
"""Run a trained NeuroScale reimplementation on any labeled TSV manifest."""

import argparse, csv, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from train_neuroscale import NeuroScale, SequenceDataset, collate, evaluate, read_rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True); ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=102); ap.add_argument("--cache-dir"); args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"], cache_dir=args.cache_dir)
    backbone = AutoModel.from_pretrained(ckpt["model_name"], cache_dir=args.cache_dir)
    model = NeuroScale(backbone, backbone.config.hidden_size); model.load_state_dict(ckpt["model_state"]); model.to(device)
    rows = read_rows(args.manifest); loader = DataLoader(SequenceDataset(rows), batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, tokenizer, args.max_length))
    met, records = evaluate(model, loader, device)
    with open(args.manifest, newline="") as f:
        source_rows = list(csv.DictReader(f, delimiter="\t"))
    source_by_id = {r["seq_id"]: r for r in source_rows}
    source_fields = list(source_rows[0].keys())
    with open(out / "predictions.tsv", "w", newline="") as f:
        fields = source_fields + [x for x in ("prob_np", "pred_label") if x not in source_fields]
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t"); w.writeheader()
        for sid, seq, y, p in records:
            row = dict(source_by_id[sid]); row["prob_np"] = f"{p:.10g}"; row["pred_label"] = int(p >= 0.5); w.writerow(row)
    (out / "metrics.json").write_text(json.dumps(met, indent=2) + "\n"); print(json.dumps(met))


if __name__ == "__main__": main()
