#!/usr/bin/env python3
"""Draw publication-ready Venn-style overlap figures for Dataset A/B."""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dataset_ab_overlap_similarity_final"
TABLE = OUT / "tables/exact_overlap_summary.tsv"
FIGURES = OUT / "figures"

FONT_FAMILY = "Arial"
COLOR_A = "#356AA0"  # Dataset A blue
COLOR_B = "#D36B3D"  # Dataset B orange
EDGE_A = "#244B70"
EDGE_B = "#93482B"
TEXT = "#222222"


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def circle_intersection_area(r1: float, r2: float, d: float) -> float:
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = r1 * r1 * math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 * r2 * math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    a3 = 0.5 * math.sqrt(
        max(0.0, (-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    )
    return a1 + a2 - a3


def solve_distance(r1: float, r2: float, overlap: float) -> float:
    max_overlap = math.pi * min(r1, r2) ** 2
    if overlap >= max_overlap:
        return abs(r1 - r2)
    if overlap <= 0:
        return r1 + r2
    lo = abs(r1 - r2)
    hi = r1 + r2
    for _ in range(80):
        mid = (lo + hi) / 2
        area = circle_intersection_area(r1, r2, mid)
        if area > overlap:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def venn_geometry(a_total: int, b_total: int, overlap: int) -> tuple[float, float, float]:
    r1 = math.sqrt(a_total / math.pi)
    r2 = math.sqrt(b_total / math.pi)
    d = solve_distance(r1, r2, overlap)
    return r1, r2, d


def draw_venn_on_axis(ax, title: str, panel_label: str, a_total: int, b_total: int, overlap: int) -> None:
    a_only = a_total - overlap
    b_only = b_total - overlap
    r1, r2, d = venn_geometry(a_total, b_total, overlap)

    scale = max(r1, r2)
    r1 /= scale
    r2 /= scale
    d /= scale

    raw_x1 = 0.0
    raw_x2 = d
    union_min = min(raw_x1 - r1, raw_x2 - r2)
    union_max = max(raw_x1 + r1, raw_x2 + r2)
    x_shift = -(union_min + union_max) / 2
    x1, y1 = raw_x1 + x_shift, 0.0
    x2, y2 = raw_x2 + x_shift, 0.0

    ax.add_patch(Circle((x1, y1), r1, facecolor=COLOR_A, edgecolor=EDGE_A, alpha=0.55, linewidth=0.7))
    ax.add_patch(Circle((x2, y2), r2, facecolor=COLOR_B, edgecolor=EDGE_B, alpha=0.50, linewidth=0.7))

    label_positions = label_positions_with_padding(x1, x2, r1, r2)
    ax.text(*label_positions["a_only"], f"{a_only:,}", ha="center", va="center", fontsize=6.5, color=TEXT)
    ax.text(*label_positions["overlap"], f"{overlap:,}", ha="center", va="center", fontsize=6.5, fontweight="bold", color=TEXT)
    ax.text(*label_positions["b_only"], f"{b_only:,}", ha="center", va="center", fontsize=6.5, color=TEXT)

    ax.text(0.5, 1.03, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=TEXT)
    if panel_label:
        ax.text(0.0, 1.03, panel_label, transform=ax.transAxes, ha="left", va="bottom", fontsize=8, fontweight="bold", color=TEXT)

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")


def label_positions_with_padding(x1: float, x2: float, r1: float, r2: float) -> dict[str, tuple[float, float]]:
    """Place labels at high-clearance points inside each Venn region."""
    xs = np.linspace(-1.7, 1.7, 900)
    ys = np.linspace(-1.08, 1.08, 620)
    xx, yy = np.meshgrid(xs, ys)
    da = np.sqrt((xx - x1) ** 2 + yy**2)
    db = np.sqrt((xx - x2) ** 2 + yy**2)
    in_a = da <= r1
    in_b = db <= r2

    specs = {
        "a_only": (in_a & ~in_b, np.minimum(r1 - da, db - r2)),
        "overlap": (in_a & in_b, np.minimum(r1 - da, r2 - db)),
        "b_only": (in_b & ~in_a, np.minimum(r2 - db, da - r1)),
    }
    positions: dict[str, tuple[float, float]] = {}
    for name, (mask, clearance) in specs.items():
        if not np.any(mask):
            positions[name] = ((x1 + x2) / 2, 0.0)
            continue
        score = np.where(mask, clearance - 0.015 * np.abs(yy), -np.inf)
        idx = np.unravel_index(int(np.argmax(score)), score.shape)
        positions[name] = (float(xx[idx]), float(yy[idx]))
    return positions


def draw_venn(title: str, a_total: int, b_total: int, overlap: int, path_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    draw_venn_on_axis(ax, title, "", a_total, b_total, overlap)
    handles = [
        Patch(facecolor=COLOR_A, edgecolor=EDGE_A, alpha=0.55, label="Dataset A"),
        Patch(facecolor=COLOR_B, edgecolor=EDGE_B, alpha=0.50, label="Dataset B"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(FIGURES / f"{path_stem}.png", dpi=300)
    plt.savefig(FIGURES / f"{path_stem}.pdf")
    plt.close()


def draw_combined_venn(df: pd.DataFrame) -> None:
    specs = [
        ("positive_vs_positive", "Positive", "a"),
        ("negative_vs_negative", "Negative", "b"),
        ("any_label", "All sequences", "c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    for ax, (key, title, panel_label) in zip(axes, specs):
        row = df.loc[key]
        draw_venn_on_axis(
            ax,
            title=title,
            panel_label=panel_label,
            a_total=int(row["dataset_a_unique"]),
            b_total=int(row["dataset_b_unique"]),
            overlap=int(row["exact_overlap"]),
        )
    handles = [
        Patch(facecolor=COLOR_A, edgecolor=EDGE_A, alpha=0.55, label="Dataset A"),
        Patch(facecolor=COLOR_B, edgecolor=EDGE_B, alpha=0.50, label="Dataset B"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    plt.subplots_adjust(left=0.015, right=0.995, bottom=0.14, top=0.90, wspace=0.18)
    plt.savefig(FIGURES / "figure_5_venn_exact_overlap_panels.png", dpi=600)
    plt.savefig(FIGURES / "figure_5_venn_exact_overlap_panels.pdf")
    plt.close()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    set_nature_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TABLE, sep="\t").set_index("comparison")

    specs = [
        ("positive_vs_positive", "Positive mature neuropeptide exact overlap", "figure_5_venn_positive_exact_overlap"),
        ("negative_vs_negative", "Negative sample exact overlap", "figure_6_venn_negative_exact_overlap"),
        ("any_label", "All-sequence exact overlap", "figure_7_venn_all_exact_overlap"),
    ]
    for key, title, stem in specs:
        row = df.loc[key]
        draw_venn(
            title=title,
            a_total=int(row["dataset_a_unique"]),
            b_total=int(row["dataset_b_unique"]),
            overlap=int(row["exact_overlap"]),
            path_stem=stem,
        )
    draw_combined_venn(df)

    print(f"Wrote Venn figures to {FIGURES}")


if __name__ == "__main__":
    main()
