#!/usr/bin/env python3
"""Generate publication-quality figures from experiment artifacts.

Produces:
  1. Cosine similarity heatmap (safety vs epistemic across layers)
  2. Cross-ablation 2×2 bar chart
  3. Quantization drift plot
  4. Cross-model comparison panel

Usage:
    python scripts/generate_figures.py \
        --artifacts-dir artifacts/ \
        --output-dir artifacts/figures/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.utils.io import configure_logging, ensure_dir, read_json

logger = configure_logging(name="figures")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not installed — run: pip install -e '.[plot]'")


# Publication-quality style
STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def plot_direction_comparison(comparison_path: str, output_dir: str) -> None:
    """Figure 1: Cosine similarity between safety and epistemic directions."""
    data = read_json(comparison_path)
    comps = data["comparisons"]

    layers = [c["layer_index"] for c in comps]
    cos_vals = [c["cosine_similarity"] for c in comps]
    s_sep = [c["safety_separability"] for c in comps]
    e_sep = [c["epistemic_separability"] for c in comps]

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)

        # Top: cosine similarity
        ax1.plot(layers, cos_vals, "o-", color="#2196F3", markersize=3, linewidth=1.2, label="cos(safety, epistemic)")
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax1.axhline(y=0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.7, label="Entanglement threshold")
        ax1.axhline(y=-0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.7)
        ax1.set_ylabel("Cosine similarity")
        ax1.set_title(f"Safety vs. Epistemic Direction Geometry — {data.get('model_id', 'Unknown')}")
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.set_ylim(-1.0, 1.0)

        # Bottom: separability scores
        ax2.plot(layers, s_sep, "s-", color="#F44336", markersize=3, linewidth=1.2, label="Safety separability")
        ax2.plot(layers, e_sep, "^-", color="#4CAF50", markersize=3, linewidth=1.2, label="Epistemic separability")
        ax2.set_xlabel("Layer index")
        ax2.set_ylabel("Separability score")
        ax2.legend(loc="upper right", framealpha=0.9)

        fig.tight_layout()
        out = Path(output_dir) / "fig1_direction_comparison.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved %s", out)

        # Also save PNG for quick viewing
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4.5), sharex=True)
        ax1.plot(layers, cos_vals, "o-", color="#2196F3", markersize=3, linewidth=1.2)
        ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax1.axhline(y=0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.7)
        ax1.axhline(y=-0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.7)
        ax1.set_ylabel("Cosine similarity")
        ax1.set_title(f"Safety vs. Epistemic Direction Geometry — {data.get('model_id', 'Unknown')}")

        ax2.plot(layers, s_sep, "s-", color="#F44336", markersize=3, linewidth=1.2, label="Safety")
        ax2.plot(layers, e_sep, "^-", color="#4CAF50", markersize=3, linewidth=1.2, label="Epistemic")
        ax2.set_xlabel("Layer index")
        ax2.set_ylabel("Separability score")
        ax2.legend()

        fig2.tight_layout()
        fig2.savefig(Path(output_dir) / "fig1_direction_comparison.png")
        plt.close(fig2)


def plot_cross_ablation(results_path: str, output_dir: str) -> None:
    """Figure 2: Cross-ablation 2×3 bar chart."""
    data = read_json(results_path)
    results = data["results"]

    conditions = [r["condition"] for r in results]
    safety_refusal = [r["safety_refusal_rate"] for r in results]
    epistemic_abstention = [r["epistemic_abstention_rate"] for r in results]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5, 3.5))

        x = np.arange(len(conditions))
        width = 0.35

        bars1 = ax.bar(x - width/2, safety_refusal, width, label="Safety refusal rate", color="#F44336", alpha=0.85)
        bars2 = ax.bar(x + width/2, epistemic_abstention, width, label="Epistemic abstention rate", color="#4CAF50", alpha=0.85)

        ax.set_ylabel("Rate")
        ax.set_title("Cross-Ablation: Effect of Removing One Direction on Both Behaviors")
        ax.set_xticks(x)
        labels = ["Baseline", "Ablate Safety\nDirection", "Ablate Epistemic\nDirection"]
        ax.set_xticklabels(labels[:len(conditions)])
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_ylim(0, 1.0)

        # Add value labels
        for bar_group in [bars1, bars2]:
            for bar in bar_group:
                height = bar.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7,
                )

        fig.tight_layout()
        out = Path(output_dir) / "fig2_cross_ablation.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved %s", out)


def plot_quantization_drift(sweep_path: str, output_dir: str) -> None:
    """Figure 3: Quantization drift — cosine similarity to FP16."""
    data = read_json(sweep_path)
    drift_table = data.get("drift_table", [])

    if not drift_table:
        logger.warning("No drift table found — skipping quantization plot")
        return

    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

        precisions = [r["precision"] for r in drift_table]
        s_cos = [r["safety_cosine_vs_fp16"] for r in drift_table]
        e_cos = [r["epistemic_cosine_vs_fp16"] for r in drift_table]
        s_sep = [r["safety_separability"] for r in drift_table]
        e_sep = [r["epistemic_separability"] for r in drift_table]

        x = np.arange(len(precisions))
        sep_rows = [
            row for row in drift_table
            if row.get("safety_separability") is not None and row.get("epistemic_separability") is not None
        ]

        # Left: cosine to FP16
        ax1.bar(x - 0.2, s_cos, 0.35, label="Safety", color="#F44336", alpha=0.85)
        ax1.bar(x + 0.2, e_cos, 0.35, label="Epistemic", color="#4CAF50", alpha=0.85)
        ax1.set_ylabel("Cosine similarity to FP16")
        ax1.set_title("Direction Preservation")
        ax1.set_xticks(x)
        ax1.set_xticklabels(precisions, rotation=30, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1.1)

        # Right: separability
        sep_precisions = [row["precision"] for row in sep_rows]
        sep_safety = [row["safety_separability"] for row in sep_rows]
        sep_epistemic = [row["epistemic_separability"] for row in sep_rows]
        x_sep = np.arange(len(sep_precisions))

        ax2.bar(x_sep - 0.2, sep_safety, 0.35, label="Safety", color="#F44336", alpha=0.85)
        ax2.bar(x_sep + 0.2, sep_epistemic, 0.35, label="Epistemic", color="#4CAF50", alpha=0.85)
        ax2.set_ylabel("Separability score")
        ax2.set_title("Group Separability")
        ax2.set_xticks(x_sep)
        ax2.set_xticklabels(sep_precisions, rotation=30, ha="right")
        ax2.legend()

        fig.suptitle(f"Quantization Effects — {data.get('model_id', 'Unknown')}", y=1.02)
        fig.tight_layout()
        out = Path(output_dir) / "fig3_quantization_drift.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved %s", out)


def plot_cross_model(combined_path: str, output_dir: str) -> None:
    """Figure 4: Cross-model comparison of direction geometry."""
    data = read_json(combined_path)
    results = data["results"]

    with plt.rc_context(STYLE):
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.5), sharey=True)

        if n_models == 1:
            axes = [axes]

        for ax, r in zip(axes, results):
            comps = r["comparisons"]
            layers = [c["layer_index"] for c in comps]
            cos_vals = [c["cosine_similarity"] for c in comps]

            ax.plot(layers, cos_vals, "o-", color="#2196F3", markersize=2, linewidth=1)
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
            ax.axhline(y=0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.5)
            ax.axhline(y=-0.3, color="red", linestyle=":", linewidth=0.5, alpha=0.5)

            model_short = r["model_id"].split("/")[-1]
            ax.set_title(model_short, fontsize=9)
            ax.set_xlabel("Layer")

            # Mark top safety layer
            top_layer_idx = int(r["top_safety_layer"].split(".")[-1])
            top_cos = r["cosine_at_top_safety"]
            ax.axvline(x=top_layer_idx, color="#FF9800", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.annotate(
                f"cos={top_cos:.2f}",
                xy=(top_layer_idx, top_cos),
                xytext=(5, 10), textcoords="offset points",
                fontsize=7, color="#FF9800",
            )

        axes[0].set_ylabel("cos(safety, epistemic)")

        fig.suptitle("Cross-Model: Safety vs. Epistemic Direction Similarity", y=1.02)
        fig.tight_layout()
        out = Path(output_dir) / "fig4_cross_model.pdf"
        fig.savefig(out)
        plt.close(fig)
        logger.info("Saved %s", out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--artifacts-dir", default="artifacts/")
    parser.add_argument("--output-dir", default="artifacts/figures/")
    args = parser.parse_args()

    if not HAS_MPL:
        logger.error("matplotlib is required. Install with: pip install -e '.[plot]'")
        sys.exit(1)

    ensure_dir(args.output_dir)
    base = Path(args.artifacts_dir)

    # Figure 1: Direction comparison
    comparison_files = list(base.glob("directions/*comparison*.json"))
    for f in comparison_files:
        logger.info("Generating Figure 1 from %s", f)
        plot_direction_comparison(str(f), args.output_dir)

    # Figure 2: Cross-ablation
    ablation_files = list(base.glob("cross_ablation/*results*.json"))
    for f in ablation_files:
        logger.info("Generating Figure 2 from %s", f)
        plot_cross_ablation(str(f), args.output_dir)

    # Figure 3: Quantization
    quant_files = list(base.glob("quantization/*sweep*.json"))
    for f in quant_files:
        logger.info("Generating Figure 3 from %s", f)
        plot_quantization_drift(str(f), args.output_dir)

    # Figure 4: Cross-model
    cross_files = list(base.glob("cross_model/combined_results.json"))
    for f in cross_files:
        logger.info("Generating Figure 4 from %s", f)
        plot_cross_model(str(f), args.output_dir)

    logger.info("Done. Figures saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
