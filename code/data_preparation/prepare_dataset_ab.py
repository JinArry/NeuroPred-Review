#!/usr/bin/env python3
"""Prepare canonical Dataset A and Dataset B files.

The script converts the selected local source files into a common FASTA +
metadata.tsv layout used by later overlap, similarity, and benchmark scripts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AA20 = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class Record:
    seq_id: str
    sequence: str
    label: int
    dataset: str
    split: str
    source_database: str
    source_method_or_repo: str
    canonical_group: str
    variant: str
    original_id: str = ""

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def has_noncanonical_aa(self) -> bool:
        return bool(set(self.sequence) - AA20)


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header = None
    chunks: list[str] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(chunks).upper()))
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def write_fasta(records: list[Record], path: Path) -> None:
    with path.open("w") as handle:
        for record in records:
            handle.write(f">{record.seq_id}|label={record.label}|split={record.split}\n")
            handle.write(f"{record.sequence}\n")


def write_metadata(records: list[Record], path: Path) -> None:
    fields = [
        "seq_id",
        "sequence",
        "label",
        "dataset",
        "split",
        "source_database",
        "source_method_or_repo",
        "length",
        "has_noncanonical_aa",
        "canonical_group",
        "variant",
        "original_id",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "seq_id": record.seq_id,
                    "sequence": record.sequence,
                    "label": record.label,
                    "dataset": record.dataset,
                    "split": record.split,
                    "source_database": record.source_database,
                    "source_method_or_repo": record.source_method_or_repo,
                    "length": record.length,
                    "has_noncanonical_aa": int(record.has_noncanonical_aa),
                    "canonical_group": record.canonical_group,
                    "variant": record.variant,
                    "original_id": record.original_id,
                }
            )


def prepare_dataset_a() -> list[Record]:
    source = ROOT / "data/raw/source_repos/PredNeuroP/datasets"
    parts = [
        ("Pos_train_fasta.txt", 1, "train", "NeuroPep"),
        ("Neg_train_fasta.txt", 0, "train", "Swiss-Prot"),
        ("Pos_test_fasta.txt", 1, "test", "NeuroPep"),
        ("Neg_test_fasta.txt", 0, "test", "Swiss-Prot"),
    ]
    records: list[Record] = []
    counters = {(1, "train"): 0, (0, "train"): 0, (1, "test"): 0, (0, "test"): 0}
    for filename, label, split, source_database in parts:
        for original_id, sequence in read_fasta(source / filename):
            counters[(label, split)] += 1
            prefix = "pos" if label == 1 else "neg"
            seq_id = f"A_{split}_{prefix}_{counters[(label, split)]:04d}"
            records.append(
                Record(
                    seq_id=seq_id,
                    sequence=sequence,
                    label=label,
                    dataset="dataset_a_predneurop",
                    split=split,
                    source_database=source_database,
                    source_method_or_repo="PredNeuroP",
                    canonical_group="dataset_a_predneurop",
                    variant="canonical",
                    original_id=original_id,
                )
            )
    return records


def prepare_dataset_b() -> list[Record]:
    source = ROOT / "data/raw/source_repos/NeuroPred-PLM/dataset"
    records: list[Record] = []
    counters = {(1, "train"): 0, (0, "train"): 0, (1, "test"): 0, (0, "test"): 0}
    for split in ["train", "test"]:
        with (source / f"{split}.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                label = int(row["label"])
                sequence = row["seq"].strip().upper()
                counters[(label, split)] += 1
                prefix = "pos" if label == 1 else "neg"
                seq_id = f"B_{split}_{prefix}_{counters[(label, split)]:04d}"
                records.append(
                    Record(
                        seq_id=seq_id,
                        sequence=sequence,
                        label=label,
                        dataset="dataset_b_neuropred_plm",
                        split=split,
                        source_database="NeuroPep 2.0" if label == 1 else "UniProt",
                        source_method_or_repo="NeuroPred-PLM",
                        canonical_group="dataset_b_neuropred_plm",
                        variant="original",
                    )
                )
    return records


def export_dataset(records: list[Record], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(records, out_dir / "all.fasta")
    write_fasta([r for r in records if r.label == 1], out_dir / "positives.fasta")
    write_fasta([r for r in records if r.label == 0], out_dir / "negatives.fasta")
    write_fasta([r for r in records if r.split == "train"], out_dir / "train.fasta")
    write_fasta([r for r in records if r.split == "test"], out_dir / "test.fasta")
    write_metadata(records, out_dir / "metadata.tsv")


def summarize(records: list[Record], name: str) -> None:
    print(name)
    for split in ["train", "test"]:
        subset = [r for r in records if r.split == split]
        pos = sum(r.label == 1 for r in subset)
        neg = sum(r.label == 0 for r in subset)
        noncanon = sum(r.has_noncanonical_aa for r in subset)
        unique = len({r.sequence for r in subset})
        print(
            f"  {split}: total={len(subset)} pos={pos} neg={neg} "
            f"unique={unique} noncanonical={noncanon}"
        )
    print(f"  all_unique={len({r.sequence for r in records})} total={len(records)}")


def main() -> None:
    dataset_a = prepare_dataset_a()
    dataset_b = prepare_dataset_b()
    dataset_b_standard20 = [r for r in dataset_b if not r.has_noncanonical_aa]

    export_dataset(dataset_a, ROOT / "data/processed/dataset_a_predneurop")
    export_dataset(dataset_b, ROOT / "data/processed/dataset_b_neuropred_plm_original")
    export_dataset(dataset_b_standard20, ROOT / "data/processed/dataset_b_neuropred_plm_standard20")

    summarize(dataset_a, "dataset_a_predneurop")
    summarize(dataset_b, "dataset_b_neuropred_plm_original")
    summarize(dataset_b_standard20, "dataset_b_neuropred_plm_standard20")


if __name__ == "__main__":
    main()
