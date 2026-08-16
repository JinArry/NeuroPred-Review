#!/usr/bin/env python3
"""Reproduce NeuroPred-PLM independent-test metrics with original assets.

The original NeuroPred-PLM package downloads its released classifier weights
from Zenodo and uses the packaged ESM configuration in NeuroPredPLM/args.pt.
This script keeps the original repository code unchanged and only wraps
torch.load for compatibility with PyTorch versions where weights_only=True is
the default.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


MODEL_URL = "https://zenodo.org/record/7042286/files/model.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeuroPred-PLM original released model on its test set."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def install_torch_load_compat() -> None:
    original_load = torch.load

    def compat_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = compat_load  # type: ignore[assignment]


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    install_torch_load_compat()

    from NeuroPredPLM.model import EsmModel  # noqa: PLC0415
    from NeuroPredPLM.utils import load_hub_workaround  # noqa: PLC0415

    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    test_df = pd.read_csv(repo / "dataset" / "test.csv")
    peptides = [
        (f"test_{idx:04d}", str(row.seq))
        for idx, row in test_df.reset_index(drop=True).iterrows()
    ]
    labels = test_df["label"].astype(int).to_numpy()

    model = EsmModel()
    model.eval()
    state_dict = load_hub_workaround(MODEL_URL)
    model.load_state_dict(state_dict)
    model = model.to(device)

    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(peptides), args.batch_size):
            batch = peptides[start : start + args.batch_size]
            logits, _att = model(batch, device)
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            probabilities.append(probs)
            predictions.append(probs.argmax(axis=1))

    prob = np.concatenate(probabilities, axis=0)
    pred = np.concatenate(predictions, axis=0)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    metrics = {
        "method": "NeuroPred-PLM",
        "run_type": "original_released_model_independent_test",
        "repository": str(repo),
        "model_url": MODEL_URL,
        "test_csv": str(repo / "dataset" / "test.csv"),
        "device": device,
        "batch_size": args.batch_size,
        "n_test": int(len(labels)),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "acc": float(accuracy_score(labels, pred)),
        "sn": float(recall_score(labels, pred)),
        "sp": float(tn / (tn + fp)),
        "precision": float(precision_score(labels, pred)),
        "f1": float(f1_score(labels, pred)),
        "mcc": float(matthews_corrcoef(labels, pred)),
        "auc": float(roc_auc_score(labels, prob[:, 1])),
        "auprc": float(average_precision_score(labels, prob[:, 1])),
        "torch_version": torch.__version__,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(metrics)

    with (out_dir / "test_predictions.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "sequence", "label", "pred", "prob_non_np", "prob_np"])
        for (sample_id, seq), label, one_pred, one_prob in zip(peptides, labels, pred, prob):
            writer.writerow(
                [
                    sample_id,
                    seq,
                    int(label),
                    int(one_pred),
                    f"{float(one_prob[0]):.10f}",
                    f"{float(one_prob[1]):.10f}",
                ]
            )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
