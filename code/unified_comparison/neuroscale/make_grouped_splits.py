#!/usr/bin/env python3
"""Create fixed label-stratified train/validation splits grouped by exact sequence."""

import argparse, csv, json, random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True); ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--validation-fraction", type=float, default=0.1); args = ap.parse_args()
    rows = [r for r in csv.DictReader(args.metadata.open(), delimiter="\t") if r["split"] == "train"]
    groups = defaultdict(list)
    for row in rows: groups[row["sequence"]].append(row)
    if any(len({r["label"] for r in members}) != 1 for members in groups.values()):
        raise ValueError("An exact-sequence group contains conflicting labels")
    args.output_dir.mkdir(parents=True, exist_ok=True); summaries = []
    for seed in args.seeds:
        val_sequences = set(); targets = {}
        for label in (0, 1):
            label_groups = [(seq, members) for seq, members in groups.items() if int(members[0]["label"]) == label]
            target = round(sum(len(m) for _, m in label_groups) * args.validation_fraction); targets[label] = target
            random.Random(seed * 1009 + label).shuffle(label_groups)
            selected, count = [], 0
            for seq, members in label_groups:
                if count + len(members) <= target:
                    selected.append(seq); count += len(members)
                if count == target: break
            if count != target:
                raise RuntimeError(f"Could not reach exact validation row target for label {label}: {count}/{target}")
            val_sequences.update(selected)
        output_rows = []
        for row in rows:
            output_rows.append({"seq_id": row["seq_id"], "sequence": row["sequence"], "label": row["label"],
                                "seed": seed, "partition": "validation" if row["sequence"] in val_sequences else "train"})
        path = args.output_dir / f"seed_{seed}.tsv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=output_rows[0].keys(), delimiter="\t"); w.writeheader(); w.writerows(output_rows)
        train_seq = {r["sequence"] for r in output_rows if r["partition"] == "train"}
        val_seq = {r["sequence"] for r in output_rows if r["partition"] == "validation"}
        if train_seq & val_seq: raise AssertionError("Exact sequence crosses train and validation")
        summary = {"seed": seed, "train_n": sum(r["partition"] == "train" for r in output_rows),
                   "validation_n": sum(r["partition"] == "validation" for r in output_rows),
                   "train_positive": sum(r["partition"] == "train" and r["label"] == "1" for r in output_rows),
                   "validation_positive": sum(r["partition"] == "validation" and r["label"] == "1" for r in output_rows),
                   "shared_exact_sequences": 0}
        summaries.append(summary)
    (args.output_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    with (args.output_dir / "summary.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summaries[0].keys(), delimiter="\t"); w.writeheader(); w.writerows(summaries)
    print(json.dumps(summaries))


if __name__ == "__main__": main()
