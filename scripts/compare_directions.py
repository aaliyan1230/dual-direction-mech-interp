#!/usr/bin/env python3
"""Experiment 1b: Compare safety and epistemic directions across layers.

Loads two direction artifacts (safety + epistemic) and computes:
  - Per-layer cosine similarity between the two directions
  - Angular distance in degrees
  - Combined separability ranking
  - Summary statistics

Usage:
    python scripts/compare_directions.py \
        --safety-artifact artifacts/directions/llama31_8b_safety.json \
        --epistemic-artifact artifacts/directions/llama31_8b_epistemic.json \
        --output artifacts/directions/llama31_8b_comparison.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.editing.directions import angular_distance_degrees, cosine_similarity
from ddmi.evaluation.metrics import DirectionComparisonMetrics
from ddmi.utils.io import configure_logging, read_json, write_json

logger = configure_logging(name="compare_directions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare safety vs epistemic directions")
    parser.add_argument("--safety-artifact", required=True, help="Safety direction JSON")
    parser.add_argument("--epistemic-artifact", required=True, help="Epistemic direction JSON")
    parser.add_argument("--output", required=True, help="Output comparison JSON")
    args = parser.parse_args()

    safety = read_json(args.safety_artifact)
    epistemic = read_json(args.epistemic_artifact)

    # Validate model match
    if safety["model_id"] != epistemic["model_id"]:
        logger.warning(
            "Model mismatch: safety=%s, epistemic=%s",
            safety["model_id"], epistemic["model_id"],
        )

    # Find common layers
    safety_layers = set(safety["directions"].keys())
    epistemic_layers = set(epistemic["directions"].keys())
    common_layers = sorted(
        safety_layers & epistemic_layers,
        key=lambda n: int(n.split(".")[-1]),
    )
    logger.info("Comparing %d common layers", len(common_layers))

    comparisons = []
    for layer_name in common_layers:
        s_dir = safety["directions"][layer_name]["direction"]
        e_dir = epistemic["directions"][layer_name]["direction"]

        cos_sim = cosine_similarity(s_dir, e_dir)
        ang_dist = angular_distance_degrees(s_dir, e_dir)

        s_sep = safety["directions"][layer_name]["separability_score"]
        e_sep = epistemic["directions"][layer_name]["separability_score"]

        layer_idx = int(layer_name.split(".")[-1])
        comp = DirectionComparisonMetrics(
            layer_name=layer_name,
            layer_index=layer_idx,
            cosine_similarity=cos_sim,
            angular_distance_deg=ang_dist,
            safety_separability=s_sep,
            epistemic_separability=e_sep,
        )
        comparisons.append(comp)

        logger.info(
            "  %s: cos=%.4f, angle=%.1f°, sep_s=%.4f, sep_e=%.4f",
            layer_name, cos_sim, ang_dist, s_sep, e_sep,
        )

    # Summary statistics
    cos_values = [c.cosine_similarity for c in comparisons]
    max_cos = max(cos_values)
    min_cos = min(cos_values)
    mean_cos = sum(cos_values) / len(cos_values) if cos_values else 0.0
    max_cos_layer = comparisons[cos_values.index(max_cos)].layer_name
    min_cos_layer = comparisons[cos_values.index(min_cos)].layer_name

    # Find layers where both directions have high separability
    both_high = [
        c for c in comparisons
        if c.safety_separability > 0.1 and c.epistemic_separability > 0.1
    ]
    if both_high:
        logger.info("\nLayers with both safety and epistemic separability > 0.1:")
        for c in both_high:
            logger.info(
                "  %s: cos=%.4f, sep_s=%.4f, sep_e=%.4f",
                c.layer_name, c.cosine_similarity,
                c.safety_separability, c.epistemic_separability,
            )

    logger.info("\n=== SUMMARY ===")
    logger.info("Max cosine: %.4f (at %s)", max_cos, max_cos_layer)
    logger.info("Min cosine: %.4f (at %s)", min_cos, min_cos_layer)
    logger.info("Mean cosine: %.4f", mean_cos)
    logger.info(
        "Interpretation: |cos| > 0.3 → entangled, |cos| < 0.1 → independent"
    )

    # Save artifact
    artifact = {
        "artifact_type": "direction_comparison",
        "model_id": safety["model_id"],
        "safety_artifact": args.safety_artifact,
        "epistemic_artifact": args.epistemic_artifact,
        "num_layers_compared": len(comparisons),
        "summary": {
            "max_cosine": max_cos,
            "max_cosine_layer": max_cos_layer,
            "min_cosine": min_cos,
            "min_cosine_layer": min_cos_layer,
            "mean_cosine": mean_cos,
            "layers_both_high_sep": len(both_high),
        },
        "comparisons": [c.to_dict() for c in comparisons],
    }

    write_json(args.output, artifact)
    logger.info("Comparison artifact saved to %s", args.output)


if __name__ == "__main__":
    main()
