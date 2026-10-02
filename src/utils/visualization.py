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


import matplotlib.font_manager as fm


def _get_devanagari_font_prop(size: int = 9) -> fm.FontProperties:
    """Find and return a FontProperties instance that supports Indic/Devanagari glyphs."""
    candidates = ["Nirmala UI", "Nirmala Text", "Mangal", "Aparajita", "Kokila", "Arial Unicode MS"]
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return fm.FontProperties(family=c, size=size)
    return fm.FontProperties(family="sans-serif", size=size)


def plot_attention_heatmap(
    cross_attn_matrix: np.ndarray,
    src_tokens: list,
    tgt_tokens: list,
    title: str = "Decoder Cross-Attention Weights (Source vs Target Tokens)",
    cmap: str = "magma",
    figsize: tuple = (10, 7),
    dark_mode: bool = True,
):
    """Generate a high-contrast publication-grade cross-attention heatmap figure."""
    fig, ax = plt.subplots(figsize=figsize, dpi=180)

    if dark_mode:
        fig.patch.set_facecolor("#151C2C")
        ax.set_facecolor("#0E131F")
        text_color = "#F1F5F9"
        tick_color = "#94A3B8"
    else:
        text_color = "#0F172A"
        tick_color = "#475569"

    cax = ax.matshow(cross_attn_matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))

    font_prop = _get_devanagari_font_prop(size=9)
    ax.set_xticklabels(src_tokens, rotation=45, ha="left", fontproperties=font_prop, color=tick_color)
    ax.set_yticklabels(tgt_tokens, fontproperties=font_prop, color=tick_color)

    ax.set_xlabel("Source Tokens (Encoder)", fontsize=11, fontweight="bold", labelpad=10, color=text_color)
    ax.set_ylabel("Target Tokens (Decoder)", fontsize=11, fontweight="bold", labelpad=10, color=text_color)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=20, color=text_color)

    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention Weight", color=text_color, fontsize=10, fontweight="normal")
    cbar.ax.tick_params(colors=tick_color)

    plt.tight_layout()
    return fig


def create_token_probability_df(tgt_tokens: list, token_probs: list) -> pd.DataFrame:
    """Format token generation probabilities into a clean dataframe."""
    min_len = min(len(tgt_tokens), len(token_probs))
    data = []
    for i in range(min_len):
        tok = tgt_tokens[i]
        prob = token_probs[i]
        data.append({
            "Position": i + 1,
            "Target Token": tok,
            "Generation Probability": prob,
            "Probability (%)": f"{prob * 100:.1f}%",
        })
    return pd.DataFrame(data)

