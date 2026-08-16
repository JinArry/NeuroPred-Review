#!/usr/bin/env python3
"""Train and evaluate the released NeuroPred-MTCL architecture at seed 37."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
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


SEED = 37
PARAMETERS = {
    "hidden_dim": 160,
    "projection_dim": 96,
    "alpha": 0.6,
    "beta": 0.03,
    "temperature": 0.1,
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_heads": 8,
    "dropout_rate": 0.0,
}


def set_determinism() -> None:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_determinism()
    sys.path.insert(0, str(args.repo))
    from model06_v2 import New_Model_06_Attn

    x_train = np.load(args.feature_dir / "train_esm_seq.npy", mmap_mode="r")
    y_train = np.load(args.feature_dir / "train_labels_seq.npy")
    x_test = np.load(args.feature_dir / "val_esm_seq.npy", mmap_mode="r")
    y_test = np.load(args.feature_dir / "val_labels_seq.npy")
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, 2)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, 2)
    # Match the released training entry point, including its 1,024-sample
    # shuffle buffer and tf.data batching/prefetch behavior.
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train_one_hot))
        .shuffle(1024)
        .batch(PARAMETERS["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test_one_hot))
        .batch(PARAMETERS["batch_size"])
        .prefetch(tf.data.AUTOTUNE)
    )

    model = New_Model_06_Attn(
        hidden_dim=PARAMETERS["hidden_dim"],
        projection_dim=PARAMETERS["projection_dim"],
        num_heads=PARAMETERS["num_heads"],
        alpha=PARAMETERS["alpha"],
        beta=PARAMETERS["beta"],
        dropout_rate=PARAMETERS["dropout_rate"],
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(PARAMETERS["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    checkpoint = args.output_dir / "Model06_v2_Attn_seed37_best.weights.h5"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(args.output_dir / "training_history.csv"),
    ]

    start = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    training_seconds = time.perf_counter() - start
    model.load_weights(checkpoint)

    start = time.perf_counter()
    logits = model.predict(x_test, batch_size=PARAMETERS["batch_size"], verbose=0)
    inference_seconds = time.perf_counter() - start
    probabilities = tf.nn.softmax(logits, axis=1).numpy()
    positive_probability = probabilities[:, 1]
    predictions = probabilities.argmax(axis=1)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()

    metrics = {
        "method": "NeuroPred-MTCL",
        "run_type": "original_code_retraining_seed37",
        "seed": SEED,
        "accuracy": accuracy_score(y_test, predictions),
        "sensitivity_recall": recall_score(y_test, predictions),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "mcc": matthews_corrcoef(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, positive_probability),
        "auprc": average_precision_score(y_test, positive_probability),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "best_epoch": int(np.argmax(history.history["val_accuracy"]) + 1),
        "best_validation_accuracy": float(max(history.history["val_accuracy"])),
        "epochs_run": len(history.history["val_accuracy"]),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "parameters": PARAMETERS,
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "gpu_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
        },
        "protocol_note": "Released testing.csv is used as validation data for checkpoint selection, matching the original code.",
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame([metrics]).drop(columns=["parameters", "environment"]).to_csv(
        args.output_dir / "metrics.tsv", sep="\t", index=False
    )
    test_frame = pd.read_csv(args.repo / "data" / "testing.csv")
    test_frame["positive_probability"] = positive_probability
    test_frame["predicted_label"] = predictions
    test_frame.to_csv(args.output_dir / "test_predictions.tsv", sep="\t", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
