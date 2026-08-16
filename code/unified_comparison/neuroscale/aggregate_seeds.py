#!/usr/bin/env python3
"""Aggregate NeuroScale test metrics across prespecified random seeds."""

import argparse, csv, json
from pathlib import Path
import numpy as np
from train_neuroscale import metrics


FIELDS = ["acc", "sn", "sp", "precision", "f1", "mcc", "auroc", "auprc"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs-root", type=Path, required=True)
    ap.add_argument("--run-prefix", default="esm2_650m_seed")
    ap.add_argument("--method", default="NeuroScale")
    ap.add_argument("--seeds", nargs="+", type=int, required=True); ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    per_seed, prediction_sets = [], []
    for seed in args.seeds:
        run = args.runs_root / f"{args.run_prefix}{seed}"
        payload = json.loads((run / "metrics.json").read_text()); row = {"seed": seed, "selected_epoch": payload.get("selected_epoch"), **payload["test_metrics"]}
        per_seed.append(row)
        with open(run / "test_predictions.tsv", newline="") as f: prediction_sets.append(list(csv.DictReader(f, delimiter="\t")))
    ids = [[r["seq_id"] for r in rows] for rows in prediction_sets]
    if any(x != ids[0] for x in ids[1:]): raise ValueError("Test seq_id order differs across seeds")
    labels = np.array([int(r["label"]) for r in prediction_sets[0]])
    probs = np.array([[float(r["prob_np"]) for r in rows] for rows in prediction_sets])
    summary = {}
    for field in FIELDS:
        values = np.array([float(r[field]) for r in per_seed])
        summary[field] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1))}
    ensemble_prob = probs.mean(axis=0); ensemble_metrics = metrics(labels, ensemble_prob)
    with open(args.output_dir / "per_seed_metrics.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=per_seed[0].keys(), delimiter="\t"); w.writeheader(); w.writerows(per_seed)
    with open(args.output_dir / "summary_metrics.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["metric", "mean", "sd"])
        for field in FIELDS: w.writerow([field, summary[field]["mean"], summary[field]["sd"]])
    with open(args.output_dir / "ensemble_predictions.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t"); w.writerow(["seq_id", "label", "prob_np", "pred_label"])
        for sid, y, p in zip(ids[0], labels, ensemble_prob): w.writerow([sid, y, f"{p:.10g}", int(p >= 0.5)])
    output = {"method": args.method, "seeds": args.seeds, "mean_sd": summary, "ensemble_metrics": ensemble_metrics}
    (args.output_dir / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    table = " | ".join(f"{100*summary[x]['mean']:.2f} ± {100*summary[x]['sd']:.2f}" for x in FIELDS)
    (args.output_dir / "paper_table_row.md").write_text(f"| {args.method} | " + table + " |\n")
    print(json.dumps(output))


if __name__ == "__main__": main()
