#!/usr/bin/env python3
"""Reproduce NeuroPred-CLQ independent-test metrics with original assets.

This script uses the original NeuroPred-CLQ repository code, provided
Word2Vec-derived feature matrix, label files, and pretrained FinModel.h5
weights. It does not retrain the model.
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
from tensorflow.keras.utils import to_categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the original pretrained NeuroPred-CLQ independent test."
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Path to the NeuroPred-CLQ working repository.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for metrics and prediction outputs.",
    )
    return parser.parse_args()


def read_fasta_ids(path: Path) -> list[str]:
    ids: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                ids.append(line[1:])
    return ids


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(repo)
    sys.path.insert(0, str(repo))

    from model import ourmodel  # noqa: PLC0415

    data = np.load(repo / "data" / "X.npz")
    x_test = data["x_test"]
    y_test = pd.read_csv(repo / "data" / "Process_data" / "test" / "y_test.csv")[
        "Label"
    ].to_numpy()
    y_test_cat = to_categorical(y_test, dtype=int)

    model = ourmodel()
    model.load_weights(str(repo / "CLQ_model" / "FinModel.h5"))
    probabilities = model.predict([x_test], verbose=0)
    y_pred = probabilities.argmax(axis=1)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "method": "NeuroPred-CLQ",
        "run_type": "original_pretrained_model_independent_test",
        "repository": str(repo),
        "weights": str(repo / "CLQ_model" / "FinModel.h5"),
        "feature_matrix": str(repo / "data" / "X.npz"),
        "n_test": int(len(y_test)),
        "n_positive": int((y_test == 1).sum()),
        "n_negative": int((y_test == 0).sum()),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "acc": float(accuracy_score(y_test, y_pred)),
        "sn": float(recall_score(y_test, y_pred)),
        "sp": float(tn / (tn + fp)),
        "precision": float(precision_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "mcc": float(matthews_corrcoef(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, probabilities[:, 1])),
        "auprc": float(average_precision_score(y_test, probabilities[:, 1])),
        "categorical_label_shape": list(y_test_cat.shape),
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "metrics.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics.keys()), delimiter="\t")
        writer.writeheader()
        writer.writerow(metrics)

    ids = read_fasta_ids(repo / "data" / "Pos_test.txt") + read_fasta_ids(
        repo / "data" / "Neg_test.txt"
    )
    with (out_dir / "test_predictions.tsv").open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "label", "pred", "prob_non_np", "prob_np"])
        for sample_id, label, pred, prob in zip(ids, y_test, y_pred, probabilities):
            writer.writerow(
                [
                    sample_id,
                    int(label),
                    int(pred),
                    f"{float(prob[0]):.10f}",
                    f"{float(prob[1]):.10f}",
                ]
            )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
