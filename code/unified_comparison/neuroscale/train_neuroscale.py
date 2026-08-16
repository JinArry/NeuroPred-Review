#!/usr/bin/env python3
"""Train a paper-described NeuroScale reimplementation without test-set selection."""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


class SequenceDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class NeuroScale(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone = backbone
        self.drop_in = nn.Dropout(0.1)
        self.drop_branch = nn.Dropout(0.4)
        self.branch1 = nn.Sequential(nn.Linear(hidden_size, 64), nn.ReLU())
        self.branch2 = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.branch3 = nn.Sequential(nn.Linear(hidden_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        self.classifier = nn.Linear(64, 1)

    def forward(self, **tokens):
        x = self.backbone(**tokens).last_hidden_state[:, 0, :]
        x = self.drop_in(x)
        x = sum(self.drop_branch(branch(x)) for branch in (self.branch1, self.branch2, self.branch3))
        return self.classifier(x).squeeze(-1)


def read_rows(path, split=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if split is not None:
        rows = [r for r in rows if r["split"] == split]
    return [{"seq_id": r["seq_id"], "sequence": r["sequence"], "label": int(r["label"])} for r in rows]


def apply_fixed_split(train_all, split_manifest):
    with open(split_manifest, newline="") as f:
        assignments = {r["seq_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    if set(assignments) != {r["seq_id"] for r in train_all}:
        raise ValueError("Fixed split seq_id set does not match metadata training rows")
    train_rows = [r for r in train_all if assignments[r["seq_id"]]["partition"] == "train"]
    val_rows = [r for r in train_all if assignments[r["seq_id"]]["partition"] == "validation"]
    if {r["sequence"] for r in train_rows} & {r["sequence"] for r in val_rows}:
        raise ValueError("Exact sequence crosses fixed train/validation partitions")
    return train_rows, val_rows


def collate(batch, tokenizer, max_length):
    tokens = tokenizer([x["sequence"] for x in batch], padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.float32)
    return tokens, labels, batch


def metrics(y, prob):
    y = np.asarray(y, dtype=int); prob = np.asarray(prob, dtype=float); pred = (prob >= 0.5).astype(int)
    tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
    return {
        "n": len(y), "acc": accuracy_score(y, pred), "sn": recall_score(y, pred, zero_division=0),
        "sp": tn / (tn + fp) if tn + fp else float("nan"),
        "precision": precision_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred), "auroc": roc_auc_score(y, prob),
        "auprc": average_precision_score(y, prob), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, probs, records = [], [], []
    for tokens, labels, batch in loader:
        tokens = {k: v.to(device) for k, v in tokens.items()}
        prob = torch.sigmoid(model(**tokens)).cpu().numpy()
        ys.extend(labels.numpy().tolist()); probs.extend(prob.tolist())
        for row, p in zip(batch, prob): records.append((row["seq_id"], row["sequence"], row["label"], float(p)))
    return metrics(ys, probs), records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--seed", type=int, default=2026); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8); ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--learning-rate", type=float, default=1e-3); ap.add_argument("--max-length", type=int, default=102)
    ap.add_argument("--num-workers", type=int, default=0); ap.add_argument("--cache-dir")
    args = ap.parse_args(); out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    train_all = read_rows(args.metadata, "train"); test_rows = read_rows(args.metadata, "test")
    train_rows, val_rows = apply_fixed_split(train_all, args.split_manifest)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    backbone = AutoModel.from_pretrained(args.model, cache_dir=args.cache_dir)
    model = NeuroScale(backbone, backbone.config.hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    make_collate = lambda b: collate(b, tokenizer, args.max_length)
    gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(SequenceDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                              generator=gen, num_workers=args.num_workers, collate_fn=make_collate)
    val_loader = DataLoader(SequenceDataset(val_rows), batch_size=args.batch_size, shuffle=False, collate_fn=make_collate)
    test_loader = DataLoader(SequenceDataset(test_rows), batch_size=args.batch_size, shuffle=False, collate_fn=make_collate)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss(); best_mcc = -2.0; stale = 0; history = []; started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0
        for tokens, labels_b, _ in train_loader:
            tokens = {k: v.to(device) for k, v in tokens.items()}; labels_b = labels_b.to(device)
            optimizer.zero_grad(set_to_none=True); loss = loss_fn(model(**tokens), labels_b); loss.backward(); optimizer.step()
            total += loss.item() * len(labels_b)
        val_metrics, _ = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / len(train_rows), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row); print(json.dumps(row), flush=True)
        if val_metrics["mcc"] > best_mcc + 1e-8:
            best_mcc = val_metrics["mcc"]; stale = 0
            torch.save({"model_state": model.state_dict(), "model_name": args.model,
                        "hidden_size": backbone.config.hidden_size, "seed": args.seed, "epoch": epoch}, out / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience: break
    checkpoint = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics, records = evaluate(model, test_loader, device)
    with open(out / "test_predictions.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["seq_id", "sequence", "label", "prob_np", "pred_label"])
        for sid, seq, y, p in records: w.writerow([sid, seq, y, f"{p:.10g}", int(p >= 0.5)])
    summary = {"status": "NeuroScale paper-described reimplementation", "config": vars(args),
               "train_n": len(train_rows), "validation_n": len(val_rows), "test_n": len(test_rows),
               "selected_epoch": checkpoint["epoch"], "test_metrics": test_metrics, "elapsed_seconds": time.time() - started}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out / "history.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys(), delimiter="\t"); w.writeheader(); w.writerows(history)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__": main()
