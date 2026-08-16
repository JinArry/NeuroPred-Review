#!/usr/bin/env python3
"""Score method prediction TSVs with one fixed binary-classification protocol."""

from __future__ import annotations

import argparse
import json
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


def score(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "threshold": threshold,
        "acc": float(accuracy_score(labels, predictions)),
        "sn": float(recall_score(labels, predictions, zero_division=0)),
        "sp": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, sep="\t")
    required = {"seq_id", "label", "prob_np", "exact_novel", "strict_0_9_novel"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    if frame.seq_id.duplicated().any():
        raise ValueError("seq_id must be unique")
    if frame.prob_np.isna().any() or ((frame.prob_np < 0) | (frame.prob_np > 1)).any():
        raise ValueError("prob_np must contain finite probabilities in [0, 1]")
    results = []
    for subset, selected in (
        ("all_target", frame),
        ("exact_novel", frame[frame.exact_novel.astype(bool)]),
        ("strict_0_9_novel", frame[frame.strict_0_9_novel.astype(bool)]),
    ):
        metrics = score(selected.label.to_numpy(int), selected.prob_np.to_numpy(float), args.threshold)
        results.append({"method": args.method, "direction": args.direction, "subset": subset, **metrics})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    pd.DataFrame(results).to_csv(args.out_dir / "metrics.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
