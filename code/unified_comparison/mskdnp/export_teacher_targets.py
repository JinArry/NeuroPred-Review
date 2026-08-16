#!/usr/bin/env python3
"""Export logits and pooled features from a reconstructed MSKDNP teacher."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from torch import nn


class SeqDS(Dataset):
    def __init__(self, frame): self.frame = frame.reset_index(drop=True)
    def __len__(self): return len(self.frame)
    def __getitem__(self, index): return str(self.frame.iloc[index]["seq"])


class Teacher(nn.Module):
    def __init__(self, name, pooling="masked"):
        super().__init__()
        self.pooling = pooling
        self.encoder = AutoModel.from_pretrained(name)
        h = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(nn.Linear(h, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 2))

    def forward(self, tokens):
        hidden = self.encoder(**tokens).last_hidden_state
        if self.pooling == "paper_unmasked": pooled = hidden.mean(1)
        else:
            ids, mask = tokens["input_ids"], tokens["attention_mask"].bool()
            for special in (0, 1, 2, 3): mask &= ids.ne(special)
            pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.classifier(pooled), pooled


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--pooling", choices=("masked", "paper_unmasked"), default="masked")
    parser.add_argument("--max-length", type=int, default=102)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = Teacher(args.model, args.pooling).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state"])
    frame = pd.read_csv(args.csv)
    def collate(sequences):
        padding = "max_length" if args.pooling == "paper_unmasked" else True
        return tokenizer(sequences, padding=padding, truncation=True, max_length=args.max_length, return_tensors="pt")
    loader = DataLoader(SeqDS(frame), batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    logits, features = [], []
    model.eval()
    with torch.inference_mode():
        for tokens in loader:
            tokens = {key: value.to(device) for key, value in tokens.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16): batch_logits, batch_features = model(tokens)
            logits.append(batch_logits.float().cpu().numpy())
            features.append(batch_features.float().cpu().numpy())
    np.save(args.output / "teacher_logits.npy", np.concatenate(logits))
    np.save(args.output / "teacher_features.npy", np.concatenate(features))
    pd.DataFrame({"row_index": np.arange(len(frame)), "seq": frame.seq, "label": frame.label}).to_csv(args.output / "target_index.tsv", sep="\t", index=False)
    print(f"exported {len(frame)} targets from epoch {checkpoint['epoch']}", flush=True)


if __name__ == "__main__": main()
