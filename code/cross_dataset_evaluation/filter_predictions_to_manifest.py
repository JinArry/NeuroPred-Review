#!/usr/bin/env python3
"""Select predictions by an evaluation manifest with strict one-to-one checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, sep="\t")
    manifest = pd.read_csv(args.manifest, sep="\t")
    for name, frame in (("predictions", predictions), ("manifest", manifest)):
        if "seq_id" not in frame:
            raise ValueError(f"{name} has no seq_id column")
        if frame.seq_id.duplicated().any():
            raise ValueError(f"{name} contains duplicated seq_id values")
    probability_columns = [c for c in predictions if c.startswith("prob_np")]
    if "prob_np" not in probability_columns:
        raise ValueError("predictions has no prob_np column")
    available = predictions.set_index("seq_id")
    missing = manifest.loc[~manifest.seq_id.isin(available.index), "seq_id"]
    if len(missing):
        raise ValueError(f"Missing {len(missing)} predictions; first: {missing.iloc[0]}")
    output = manifest.copy()
    for column in probability_columns:
        output[column] = output.seq_id.map(available[column])
    output["pred"] = (output.prob_np >= 0.5).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
