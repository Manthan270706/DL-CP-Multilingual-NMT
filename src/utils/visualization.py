"""Publication-Grade Visualization Utility.
Spec Section 2 (TASK-019).
"""

import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metric_comparisons(
    comparison_df: pd.DataFrame,
    save_path: str = "experiments/metric_comparison.png",
) -> None:
    """Plot grouped bar charts for BLEU, METEOR, and COMET across models and directions."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["bleu", "meteor", "comet"]
    titles = ["BLEU Score (0-100)", "METEOR Score (0-100)", "COMET Score (0-100)"]

    for i, (m, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        pivot_df = comparison_df.pivot(index="direction", columns="model", values=m)
        pivot_df.plot(kind="bar", ax=ax, width=0.8, colormap="viridis")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Translation Direction", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.legend(title="Model")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_error_distributions(
    error_summary_dict: Dict[str, Any],
    save_path: str = "experiments/error_distribution.png",
) -> None:
    """Plot error category breakdown as a horizontal bar chart."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    dist = error_summary_dict.get("distribution", {})
    categories = list(dist.keys())
    counts = [dist[cat]["count"] for cat in categories]
    percentages = [dist[cat]["percentage"] for cat in categories]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, counts, color="#3498db", edgecolor="#2980b9")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace("_", " ").title() for c in categories])
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=11)
    ax.set_title("Translation Error Category Distribution", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

    # Annotate bars with percentages
    for bar, pct in zip(bars, percentages):
        width = bar.get_width()
        ax.text(
            width + 0.5, bar.get_y() + bar.get_height() / 2, f"{pct:.1f}%", va="center", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_sentence_length_analysis(
    length_summary_dict: Dict[str, Any],
    save_path: str = "experiments/sentence_length_analysis.png",
) -> None:
    """Plot translation quality across sentence length buckets."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    buckets = length_summary_dict.get("buckets", {})
    bucket_names = list(buckets.keys())
    bleu_scores = [buckets[b]["bleu"] for b in bucket_names]
    meteor_scores = [buckets[b]["meteor"] for b in bucket_names]
    ranges = [buckets[b]["range_words"] for b in bucket_names]

    x = np.arange(len(bucket_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, bleu_scores, width, label="BLEU", color="#2ecc71")
    ax.bar(x + width / 2, meteor_scores, width, label="METEOR", color="#e74c3c")

    ax.set_xlabel("Sentence Length (Words)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Translation Performance vs Sentence Length", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b.title()}\n({r} words)" for b, r in zip(bucket_names, ranges)])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
