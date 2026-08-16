#!/usr/bin/env python3
"""Extract the repository-specified ESM-1 t6 layer-6 residue embeddings."""
import argparse
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=100)
    a = p.parse_args()

    df = pd.read_csv(a.metadata, sep="\t")
    model, alphabet = esm.pretrained.esm1_t6_43M_UR50S()
    model = model.eval().cuda()
    convert = alphabet.get_batch_converter()
    feats = np.zeros((len(df), a.max_length, 768), dtype=np.float16)
    with torch.no_grad():
        for start in range(0, len(df), a.batch_size):
            chunk = df.iloc[start:start + a.batch_size]
            batch = [(sid, seq[:a.max_length]) for sid, seq in zip(chunk.seq_id, chunk.sequence)]
            _, _, toks = convert(batch)
            reps = model(toks.cuda(), repr_layers=[6], return_contacts=False)["representations"][6]
            for j, seq in enumerate(chunk.sequence):
                n = min(len(seq), a.max_length)
                feats[start + j, :n] = reps[j, 1:n + 1].cpu().numpy().astype(np.float16)
            print(f"{min(start + a.batch_size, len(df))}/{len(df)}", flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, seq_id=df.seq_id.to_numpy(), label=df.label.to_numpy(), features=feats)


if __name__ == "__main__":
    main()
