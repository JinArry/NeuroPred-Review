#!/usr/bin/env python3
"""Regenerate the released NeuroPred-MTCL ESM-1 t6 residue features.

这是我们为 NeuroPred-MTCL 原始代码复现实验重建的 ESM 特征脚本，
不是作者仓库原始 `extract_esm_features.py` 的逐字副本。

This wrapper preserves the released model, layer, token handling, and separate
train/test padding. The only numerical compatibility fix is keeping padding in
float32; the released script's default float64 zeros silently double the saved
feature size without changing feature values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch


MODEL_NAME = "esm1_t6_43M_UR50S"
LAYER = 6
EMBEDDING_DIM = 768


def extract(
    sequences: list[str],
    labels: np.ndarray,
    model: torch.nn.Module,
    batch_converter,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    max_length = 0
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start : start + batch_size]
            batch = [(f"seq{i}", sequence) for i, sequence in enumerate(batch_sequences)]
            tokens = batch_converter(batch)[2].to(device)
            representations = model(
                tokens, repr_layers=[LAYER], return_contacts=False
            )["representations"][LAYER]
            for index, sequence in enumerate(batch_sequences):
                # Match the released script: remove BOS/EOS and keep one vector per residue.
                residue_features = (
                    representations[index, 1 : len(sequence) + 1]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                features.append(residue_features)
                max_length = max(max_length, residue_features.shape[0])
            print(f"Processed {min(start + batch_size, len(sequences))} / {len(sequences)}")

    padded = np.zeros(
        (len(features), max_length, EMBEDDING_DIM), dtype=np.float32
    )
    for index, feature in enumerate(features):
        padded[index, : feature.shape[0]] = feature
    return padded, labels.astype(np.int64, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    metadata = {
        "method": "NeuroPred-MTCL",
        "source_script": "extract_esm_features.py",
        "model": MODEL_NAME,
        "layer": LAYER,
        "embedding_dim": EMBEDDING_DIM,
        "device": str(device),
        "batch_size": args.batch_size,
        "padding_scope": "separate maximum length for train and test, as released",
        "compatibility_fix": "padding dtype fixed to float32 instead of implicit float64",
        "splits": {},
    }

    for source_name, output_prefix in (
        ("training.csv", "train"),
        ("testing.csv", "val"),
    ):
        frame = pd.read_csv(args.repo / "data" / source_name)
        sequences = frame["Seq"].astype(str).tolist()
        labels = frame["Label"].to_numpy()
        split_features, split_labels = extract(
            sequences,
            labels,
            model,
            batch_converter,
            device,
            args.batch_size,
        )
        np.save(args.output_dir / f"{output_prefix}_esm_seq.npy", split_features)
        np.save(args.output_dir / f"{output_prefix}_labels_seq.npy", split_labels)
        metadata["splits"][output_prefix] = {
            "source": source_name,
            "feature_shape": list(split_features.shape),
            "feature_dtype": str(split_features.dtype),
            "label_counts": {
                str(label): int((split_labels == label).sum())
                for label in np.unique(split_labels)
            },
        }
        print(output_prefix, split_features.shape, split_features.dtype)

    (args.output_dir / "feature_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
