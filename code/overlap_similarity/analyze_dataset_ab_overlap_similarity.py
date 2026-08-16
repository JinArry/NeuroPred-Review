#!/usr/bin/env python3
"""Generate final Dataset A/B overlap and edit-similarity tables/figures."""

from __future__ import annotations

import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dataset_ab_overlap_similarity_final"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
THRESHOLDS = np.array([0.9, 0.8, 0.6, 0.4], dtype=np.float64)
FONT_FAMILY = "Arial"
NPG = {
    "red": "#E64B35",
    "blue": "#4DBBD5",
    "green": "#00A087",
    "navy": "#3C5488",
    "orange": "#F39B7F",
    "purple": "#8491B4",
    "teal": "#91D1C2",
    "darkred": "#DC0000",
    "brown": "#7E6148",
    "gray": "#4D4D4D",
}


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "legend.title_fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def ensure_dirs() -> None:
    for path in [OUT, TABLES, FIGURES, ROOT / ".cache/matplotlib"]:
        path.mkdir(parents=True, exist_ok=True)


def load_records(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def unique_by_sequence(records: list[dict[str, str]]) -> dict[str, str]:
    seq_label: dict[str, str] = {}
    for row in records:
        seq = row["sequence"]
        label = row["label"]
        if seq in seq_label and seq_label[seq] != label:
            raise ValueError(f"Internal label conflict: {seq}")
        seq_label[seq] = label
    return seq_label


def exact_overlap(a: list[dict[str, str]], b: list[dict[str, str]]) -> None:
    a_seq = unique_by_sequence(a)
    b_seq = unique_by_sequence(b)
    rows = []
    for seq in sorted(set(a_seq) & set(b_seq)):
        rows.append(
            {
                "sequence": seq,
                "length": len(seq),
                "dataset_a_label": a_seq[seq],
                "dataset_b_label": b_seq[seq],
                "label_relation": relation(a_seq[seq], b_seq[seq]),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "exact_overlap_sequences.tsv", sep="\t", index=False)

    summary = []
    for label, name in [("1", "positive"), ("0", "negative")]:
        aset = {s for s, l in a_seq.items() if l == label}
        bset = {s for s, l in b_seq.items() if l == label}
        inter = aset & bset
        summary.append(
            {
                "comparison": f"{name}_vs_{name}",
                "dataset_a_unique": len(aset),
                "dataset_b_unique": len(bset),
                "exact_overlap": len(inter),
                "dataset_a_overlap_pct": pct(len(inter), len(aset)),
                "dataset_b_overlap_pct": pct(len(inter), len(bset)),
            }
        )
    aall = set(a_seq)
    ball = set(b_seq)
    inter = aall & ball
    summary.append(
        {
            "comparison": "any_label",
            "dataset_a_unique": len(aall),
            "dataset_b_unique": len(ball),
            "exact_overlap": len(inter),
            "dataset_a_overlap_pct": pct(len(inter), len(aall)),
            "dataset_b_overlap_pct": pct(len(inter), len(ball)),
        }
    )
    pd.DataFrame(summary).to_csv(TABLES / "exact_overlap_summary.tsv", sep="\t", index=False)
    patterns = Counter(row["label_relation"] for row in rows)
    pd.DataFrame([{"label_relation": k, "count": v} for k, v in sorted(patterns.items())]).to_csv(
        TABLES / "exact_overlap_label_patterns.tsv", sep="\t", index=False
    )


def relation(a_label: str, b_label: str) -> str:
    label = {"1": "positive", "0": "negative"}
    return f"A_{label[a_label]}__B_{label[b_label]}"


def pct(n: int, d: int) -> float:
    return round(n / d * 100, 4) if d else 0.0


def prepare_arrays(records: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_label = unique_by_sequence(records)
    seqs = list(seq_label)
    labels = np.array([int(seq_label[s]) for s in seqs], dtype=np.int64)
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    max_len = max(lengths)
    arr = np.zeros((len(seqs), max_len), dtype=np.uint8)
    for i, seq in enumerate(seqs):
        arr[i, : len(seq)] = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return arr, lengths, labels


@njit
def bounded_edit_similarity(a_arr, a_len, b_arr, b_len, cutoff):
    if abs(a_len - b_len) > cutoff:
        return -1.0
    if a_len == 0 or b_len == 0:
        dist = max(a_len, b_len)
        if dist > cutoff:
            return -1.0
        return 1.0 - dist / max(a_len, b_len)

    inf = cutoff + 1
    prev = np.empty(b_len + 1, dtype=np.int64)
    curr = np.empty(b_len + 1, dtype=np.int64)
    for j in range(b_len + 1):
        if j <= cutoff:
            prev[j] = j
        else:
            prev[j] = inf

    for i in range(1, a_len + 1):
        for j in range(b_len + 1):
            curr[j] = inf
        if i <= cutoff:
            curr[0] = i
        start = max(1, i - cutoff)
        end = min(b_len, i + cutoff)
        row_min = inf
        for j in range(start, end + 1):
            cost = 0 if a_arr[i - 1] == b_arr[j - 1] else 1
            deletion = prev[j] + 1
            insertion = curr[j - 1] + 1
            substitution = prev[j - 1] + cost
            val = min(deletion, insertion, substitution)
            curr[j] = val
            if val < row_min:
                row_min = val
        if row_min > cutoff:
            return -1.0
        tmp = prev
        prev = curr
        curr = tmp

    dist = prev[b_len]
    if dist > cutoff:
        return -1.0
    return 1.0 - dist / max(a_len, b_len)


@njit
def compute_match_counts(a_arr, a_lengths, a_labels, b_arr, b_lengths, b_labels, thresholds):
    a_match = np.zeros((2, 2, len(thresholds), len(a_lengths)), dtype=np.uint8)
    b_match = np.zeros((2, 2, len(thresholds), len(b_lengths)), dtype=np.uint8)
    for i in range(len(a_lengths)):
        alen = a_lengths[i]
        alabel = a_labels[i]
        for j in range(len(b_lengths)):
            blen = b_lengths[j]
            max_len = max(alen, blen)
            min_len = min(alen, blen)
            if min_len < int(np.ceil(0.4 * max_len)):
                continue
            cutoff = int(np.floor((1.0 - 0.4) * max_len + 1e-9))
            sim = bounded_edit_similarity(a_arr[i], alen, b_arr[j], blen, cutoff)
            if sim < 0:
                continue
            blabel = b_labels[j]
            for k in range(len(thresholds)):
                if sim + 1e-12 >= thresholds[k]:
                    a_match[alabel, blabel, k, i] = 1
                    b_match[blabel, alabel, k, j] = 1
    a_counts = np.zeros((2, 2, len(thresholds)), dtype=np.int64)
    b_counts = np.zeros((2, 2, len(thresholds)), dtype=np.int64)
    for qlabel in range(2):
        for tlabel in range(2):
            for k in range(len(thresholds)):
                a_counts[qlabel, tlabel, k] = np.sum(a_match[qlabel, tlabel, k])
                b_counts[qlabel, tlabel, k] = np.sum(b_match[qlabel, tlabel, k])
    return a_counts, b_counts


def run_edit_similarity(a: list[dict[str, str]], b: list[dict[str, str]]) -> pd.DataFrame:
    a_arr, a_lengths, a_labels = prepare_arrays(a)
    b_arr, b_lengths, b_labels = prepare_arrays(b)
    a_counts, b_counts = compute_match_counts(a_arr, a_lengths, a_labels, b_arr, b_lengths, b_labels, THRESHOLDS)
    a_totals = {label: int(np.sum(a_labels == label)) for label in [0, 1]}
    b_totals = {label: int(np.sum(b_labels == label)) for label in [0, 1]}
    rows = []
    for k, threshold in enumerate(THRESHOLDS):
        for qlabel in [0, 1]:
            for tlabel in [0, 1]:
                matched = int(a_counts[qlabel, tlabel, k])
                total = a_totals[qlabel]
                rows.append(
                    {
                        "query_dataset": "A",
                        "query_label": qlabel,
                        "target_dataset": "B",
                        "target_label": tlabel,
                        "threshold": threshold,
                        "query_total": total,
                        "query_matched": matched,
                        "query_matched_pct": 100 * matched / total,
                        "metric": "normalized_edit_similarity",
                    }
                )
                matched = int(b_counts[tlabel, qlabel, k])
                total = b_totals[tlabel]
                rows.append(
                    {
                        "query_dataset": "B",
                        "query_label": tlabel,
                        "target_dataset": "A",
                        "target_label": qlabel,
                        "threshold": threshold,
                        "query_total": total,
                        "query_matched": matched,
                        "query_matched_pct": 100 * matched / total,
                        "metric": "normalized_edit_similarity",
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "edit_similarity_summary.tsv", sep="\t", index=False)
    return df


def plot_exact_overlap() -> None:
    summary = pd.read_csv(TABLES / "exact_overlap_summary.tsv", sep="\t")
    plot_df = summary.copy()
    plot_df["comparison_label"] = plot_df["comparison"].map(
        {
            "positive_vs_positive": "Positive",
            "negative_vs_negative": "Negative",
            "any_label": "All labels",
        }
    )
    long_df = plot_df.melt(
        id_vars=["comparison_label"],
        value_vars=["dataset_a_overlap_pct", "dataset_b_overlap_pct"],
        var_name="coverage",
        value_name="overlap_pct",
    )
    long_df["coverage"] = long_df["coverage"].map(
        {
            "dataset_a_overlap_pct": "Dataset A covered by B",
            "dataset_b_overlap_pct": "Dataset B covered by A",
        }
    )
    plt.figure(figsize=(3.5, 2.55))
    ax = sns.barplot(data=long_df, x="comparison_label", y="overlap_pct", hue="coverage", palette=[NPG["blue"], NPG["red"]])
    ax.set_xlabel("")
    ax.set_ylabel("Exact sequence overlap (%)")
    ax.set_ylim(0, 100)
    ax.legend(title="", frameon=False, loc="upper right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=6, padding=1.5)
    sns.despine()
    savefig("figure_1_exact_overlap")


def plot_similarity(df: pd.DataFrame) -> None:
    pairs = [
        ("A", 1, "B", 1, "A positives similar to B positives"),
        ("B", 1, "A", 1, "B positives similar to A positives"),
        ("A", 0, "B", 0, "A negatives similar to B negatives"),
        ("B", 0, "A", 0, "B negatives similar to A negatives"),
    ]
    parts = []
    for qd, ql, td, tl, name in pairs:
        sub = df[(df.query_dataset == qd) & (df.query_label == ql) & (df.target_dataset == td) & (df.target_label == tl)].copy()
        sub["comparison"] = name
        parts.append(sub)
    plot_df = pd.concat(parts, ignore_index=True)
    plt.figure(figsize=(3.8, 2.7))
    ax = sns.lineplot(
        data=plot_df,
        x="threshold",
        y="query_matched_pct",
        hue="comparison",
        marker="o",
        linewidth=1.2,
        markersize=4,
        palette=[NPG["blue"], NPG["red"], NPG["green"], NPG["navy"]],
    )
    ax.set_xlabel("Normalized edit similarity threshold")
    ax.set_ylabel("Query sequences with a match (%)")
    ax.set_xticks([0.4, 0.6, 0.8, 0.9])
    ax.set_ylim(0, 105)
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=1.6)
    sns.despine()
    savefig("figure_2_similarity_coverage")


def plot_cross_label(df: pd.DataFrame) -> None:
    pairs = [
        ("A", 0, "B", 1, "A negatives similar to B positives"),
        ("B", 1, "A", 0, "B positives similar to A negatives"),
        ("A", 1, "B", 0, "A positives similar to B negatives"),
        ("B", 0, "A", 1, "B negatives similar to A positives"),
    ]
    parts = []
    for qd, ql, td, tl, name in pairs:
        sub = df[(df.query_dataset == qd) & (df.query_label == ql) & (df.target_dataset == td) & (df.target_label == tl)].copy()
        sub["comparison"] = name
        parts.append(sub)
    plot_df = pd.concat(parts, ignore_index=True)
    plt.figure(figsize=(3.9, 2.7))
    ax = sns.barplot(
        data=plot_df,
        x="threshold",
        y="query_matched_pct",
        hue="comparison",
        palette=[NPG["red"], NPG["orange"], NPG["green"], NPG["purple"]],
    )
    ax.set_xlabel("Normalized edit similarity threshold")
    ax.set_ylabel("Cross-label similar sequences (%)")
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, handlelength=1.3)
    sns.despine()
    savefig("figure_3_cross_label_similarity")


def plot_heatmap(df: pd.DataFrame) -> None:
    sub = df[df["threshold"].isin([0.9, 0.8, 0.6, 0.4])].copy()
    sub["pair"] = sub.apply(
        lambda r: f"{r.query_dataset}{int(r.query_label)} -> {r.target_dataset}{int(r.target_label)}", axis=1
    )
    pivot = sub.pivot(index="pair", columns="threshold", values="query_matched_pct")
    plt.figure(figsize=(3.4, 2.7))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap=sns.light_palette(NPG["blue"], as_cmap=True),
        cbar_kws={"label": "Matched query (%)"},
        annot_kws={"fontsize": 6},
        linewidths=0.2,
        linecolor="white",
    )
    ax.set_xlabel("Similarity threshold")
    ax.set_ylabel("")
    savefig("figure_4_similarity_heatmap")


def savefig(stem: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES / f"{stem}.png", dpi=600)
    plt.savefig(FIGURES / f"{stem}.pdf")
    plt.close()


def write_report(sim: pd.DataFrame) -> None:
    exact = pd.read_csv(TABLES / "exact_overlap_summary.tsv", sep="\t")
    patterns = pd.read_csv(TABLES / "exact_overlap_label_patterns.tsv", sep="\t")
    report = OUT / "dataset_ab_overlap_similarity_report.md"
    with report.open("w") as handle:
        handle.write("# Dataset A/B Overlap and Similarity Analysis\n\n")
        handle.write("## Similarity Definition\n\n")
        handle.write("Similarity was defined as `1 - Levenshtein distance / max(sequence_length_A, sequence_length_B)`. Exact duplicate sequences were collapsed within each dataset before pairwise comparison. Thresholds were 0.9, 0.8, 0.6, and 0.4.\n\n")
        handle.write("## Exact Overlap Summary\n\n")
        handle.write(exact.to_markdown(index=False))
        handle.write("\n\n## Exact Label Patterns\n\n")
        handle.write(patterns.to_markdown(index=False))
        handle.write("\n\n## Edit-Similarity Summary\n\n")
        handle.write(sim.to_markdown(index=False))
        handle.write("\n\n## Figures\n\n")
        for name in [
            "figure_1_exact_overlap",
            "figure_2_similarity_coverage",
            "figure_3_cross_label_similarity",
            "figure_4_similarity_heatmap",
            "figure_5_venn_exact_overlap_panels",
            "figure_5_venn_positive_exact_overlap",
            "figure_6_venn_negative_exact_overlap",
            "figure_7_venn_all_exact_overlap",
        ]:
            handle.write(f"- `figures/{name}.png`\n")


def main() -> None:
    ensure_dirs()
    set_nature_style()
    sns.set_theme(style="ticks", context="paper", font_scale=1.0, rc=plt.rcParams)
    a = load_records(ROOT / "data/processed/dataset_a_predneurop/metadata.tsv")
    b = load_records(ROOT / "data/processed/dataset_b_neuropred_plm_original/metadata.tsv")
    exact_overlap(a, b)
    sim_path = TABLES / "edit_similarity_summary.tsv"
    if sim_path.exists():
        sim = pd.read_csv(sim_path, sep="\t")
    else:
        sim = run_edit_similarity(a, b)
    plot_exact_overlap()
    plot_similarity(sim)
    plot_cross_label(sim)
    plot_heatmap(sim)
    write_report(sim)
    print(f"Wrote final results to {OUT}")


if __name__ == "__main__":
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    main()
