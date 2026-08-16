#!/usr/bin/env python3
"""Evaluate the released MSKDNP student model on Setmain and Setextra."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class StudentModelWithMapping(nn.Module):
    """Architecture copied from the authors' main.py release script."""

    def __init__(self, student_model: nn.Module, teacher_hidden_size: int = 1280):
        super().__init__()
        self.student_model = student_model
        self.mapping = nn.Linear(student_model.config.hidden_size, teacher_hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.student_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Preserve the authors' unmasked mean pooling exactly.
        student_features = outputs.hidden_states[-1].mean(dim=1)
        return outputs.logits, self.mapping(student_features)


class MLP(nn.Module):
    """Classification head copied from the authors' main.py release script."""

    def __init__(self, input_size: int = 1280, hidden_size: int = 512, num_classes: int = 2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def trusted_torch_load(path: Path):
    """Load trusted weights released in the original MSKDNP repository."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_models(repo: Path, device: torch.device):
    model_dir = repo / "student_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    student_model = StudentModelWithMapping(base_model)
    student_model.load_state_dict(
        trusted_torch_load(model_dir / "student_model_with_mapping.pt")
    )
    classifier = MLP()
    classifier.load_state_dict(trusted_torch_load(model_dir / "Classification_Head.pth"))
    return tokenizer, student_model.to(device).eval(), classifier.to(device).eval()


def evaluate_dataset(
    name: str,
    csv_path: Path,
    tokenizer,
    student_model: nn.Module,
    classifier: nn.Module,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
) -> list[dict]:
    frame = pd.read_csv(csv_path)
    sequences = frame["seq"].astype(str).tolist()
    labels = frame["label"].astype(int).to_numpy()
    encoded = tokenizer(
        sequences,
        padding="max_length",
        truncation=True,
        max_length=100,
        return_tensors="pt",
    )
    loader = DataLoader(
        TensorDataset(encoded["input_ids"], encoded["attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
    )

    base_probabilities = []
    released_mlp_probabilities = []
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for input_ids, attention_mask in loader:
            base_logits, mapped = student_model(
                input_ids.to(device), attention_mask.to(device)
            )
            released_mlp_logits = classifier(mapped)
            base_probabilities.append(
                torch.softmax(base_logits, dim=1)[:, 1].cpu().numpy()
            )
            released_mlp_probabilities.append(
                torch.softmax(released_mlp_logits, dim=1)[:, 1].cpu().numpy()
            )
    synchronize(device)
    elapsed = time.perf_counter() - start

    predictions_frame = frame.copy()
    results = []
    for head, positive_probability in (
        ("base_classifier", np.concatenate(base_probabilities)),
        ("released_mlp", np.concatenate(released_mlp_probabilities)),
    ):
        predictions = (positive_probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        sensitivity = tp / (tp + fn) if tp + fn else float("nan")
        results.append(
            {
                "dataset": name,
                "prediction_head": head,
                "csv_path": str(csv_path),
                "n_samples": int(len(labels)),
                "n_positive": int(labels.sum()),
                "n_negative": int((labels == 0).sum()),
                "accuracy": accuracy_score(labels, predictions),
                "sensitivity_recall": sensitivity,
                "specificity": specificity,
                "precision": precision_score(labels, predictions, zero_division=0),
                "f1": f1_score(labels, predictions, zero_division=0),
                "mcc": matthews_corrcoef(labels, predictions),
                "roc_auc": roc_auc_score(labels, positive_probability),
                "auprc": average_precision_score(labels, positive_probability),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "inference_seconds_both_heads": elapsed,
                "sequences_per_second_both_heads": len(labels) / elapsed,
                "batch_size": batch_size,
                "device": str(device),
            }
        )
        predictions_frame[f"{head}_predicted_label"] = predictions
        predictions_frame[f"{head}_positive_probability"] = positive_probability
    predictions_frame.to_csv(output_dir / f"{name}_predictions.tsv", sep="\t", index=False)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    tokenizer, student_model, classifier = build_models(args.repo, device)

    evaluations = [
        ("setmain_test", args.repo / "dataset" / "test.csv"),
        ("setextra_test", args.repo / "dataset" / "test_extra.csv"),
    ]
    results = [
        result
        for name, csv_path in evaluations
        for result in evaluate_dataset(
            name,
            csv_path,
            tokenizer,
            student_model,
            classifier,
            device,
            args.batch_size,
            args.output_dir,
        )
    ]

    metadata = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model_parameter_count": sum(p.numel() for p in student_model.parameters())
        + sum(p.numel() for p in classifier.parameters()),
    }
    payload = {"metadata": metadata, "results": results}
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(results).to_csv(args.output_dir / "metrics.tsv", sep="\t", index=False)
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
