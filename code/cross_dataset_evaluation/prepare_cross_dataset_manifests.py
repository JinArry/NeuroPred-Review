#!/usr/bin/env python3
"""Prepare leakage-annotated manifests for the A<->B generalization experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data/processed/cross_dataset_generalization"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@njit
def within_edit_threshold(a: np.ndarray, b: np.ndarray, cutoff: int) -> bool:
    """Return whether Levenshtein distance is <= cutoff using a banded DP."""
    n, m = len(a), len(b)
    if abs(n - m) > cutoff:
        return False
    inf = cutoff + 1
    prev = np.full(m + 1, inf, dtype=np.int64)
    curr = np.full(m + 1, inf, dtype=np.int64)
    for j in range(min(m, cutoff) + 1):
        prev[j] = j
    for i in range(1, n + 1):
        curr[:] = inf
        if i <= cutoff:
            curr[0] = i
        lo = max(1, i - cutoff)
        hi = min(m, i + cutoff)
        row_min = inf
        for j in range(lo, hi + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            value = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            curr[j] = value
            row_min = min(row_min, value)
        if row_min > cutoff:
            return False
        prev, curr = curr, prev
    return prev[m] <= cutoff


def encode(sequence: str) -> np.ndarray:
    return np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)


def annotate(source_train: pd.DataFrame, target: pd.DataFrame, threshold: float) -> pd.DataFrame:
    source_by_sequence: dict[str, set[int]] = {}
    length_buckets: dict[int, list[tuple[np.ndarray, int]]] = {}
    for row in source_train.itertuples(index=False):
        seq, label = str(row.sequence), int(row.label)
        source_by_sequence.setdefault(seq, set()).add(label)
        length_buckets.setdefault(len(seq), []).append((encode(seq), label))

    rows = []
    for row in target.itertuples(index=False):
        seq, label = str(row.sequence), int(row.label)
        exact_labels = source_by_sequence.get(seq, set())
        same_09 = bool(label in exact_labels)
        opposite_09 = bool(exact_labels - {label})
        query = encode(seq)
        min_len = int(np.ceil(threshold * len(seq)))
        max_len = int(np.floor(len(seq) / threshold))
        for candidate_len in range(min_len, max_len + 1):
            cutoff = int(np.floor((1.0 - threshold) * max(len(seq), candidate_len) + 1e-12))
            for candidate, candidate_label in length_buckets.get(candidate_len, []):
                if within_edit_threshold(query, candidate, cutoff):
                    if candidate_label == label:
                        same_09 = True
                    else:
                        opposite_09 = True
                if same_09 and opposite_09:
                    break
            if same_09 and opposite_09:
                break
        item = row._asdict()
        item.update(
            exact_source_match=bool(exact_labels),
            exact_same_label=label in exact_labels,
            exact_opposite_label=bool(exact_labels - {label}),
            similar_0_9_same_label=same_09,
            similar_0_9_opposite_label=opposite_09,
            exact_novel=not bool(exact_labels),
            strict_0_9_novel=not (same_09 or opposite_09),
        )
        rows.append(item)
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame, name: str) -> None:
    required = {"seq_id", "sequence", "label", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
    bad = frame[~frame.sequence.astype(str).map(lambda x: bool(x) and set(x) <= STANDARD_AA)]
    if len(bad):
        raise ValueError(f"{name} contains {len(bad)} non-standard amino-acid sequences")


def write_direction(name: str, source: pd.DataFrame, target: pd.DataFrame, out: Path) -> None:
    source_train = source[source["split"] == "train"].copy()
    annotated = annotate(source_train, target.copy(), 0.9)
    direction = out / name
    direction.mkdir(parents=True, exist_ok=True)
    annotated.to_csv(direction / "all_target.tsv", sep="\t", index=False)
    annotated[annotated.exact_novel].to_csv(direction / "exact_novel.tsv", sep="\t", index=False)
    annotated[annotated.strict_0_9_novel].to_csv(
        direction / "strict_0_9_novel.tsv", sep="\t", index=False
    )
    summary = []
    for subset, mask in (
        ("all_target", np.ones(len(annotated), dtype=bool)),
        ("exact_novel", annotated.exact_novel),
        ("strict_0_9_novel", annotated.strict_0_9_novel),
    ):
        selected = annotated[mask]
        summary.append(
            {
                "direction": name,
                "subset": subset,
                "n": len(selected),
                "positive": int(selected.label.sum()),
                "negative": int((selected.label == 0).sum()),
                "source_train_n": len(source_train),
            }
        )
    pd.DataFrame(summary).to_csv(direction / "subset_counts.tsv", sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    a = pd.read_csv(ROOT / "data/processed/dataset_a_predneurop/metadata.tsv", sep="\t")
    b = pd.read_csv(
        ROOT / "data/processed/dataset_b_neuropred_plm_standard20/metadata.tsv", sep="\t"
    )
    validate(a, "Dataset A")
    validate(b, "Dataset B-standard20")
    write_direction("a_train_to_b_standard20", a, b, args.out_dir)
    write_direction("b_train_to_a", b, a, args.out_dir)
    write_direction(
        "a_train_to_b_standard20_test",
        a,
        b[b["split"] == "test"],
        args.out_dir,
    )
    write_direction(
        "b_standard20_train_to_a_test",
        b,
        a[a["split"] == "test"],
        args.out_dir,
    )
    write_direction("a_train_to_a_test", a, a[a["split"] == "test"], args.out_dir)
    write_direction(
        "b_standard20_train_to_b_standard20_test",
        b,
        b[b["split"] == "test"],
        args.out_dir,
    )


if __name__ == "__main__":
    main()
