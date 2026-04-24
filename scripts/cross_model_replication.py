#!/usr/bin/env python3
"""Experiment 4: Cross-model replication.

Runs direction extraction + comparison on multiple model families
to test whether the safety↔epistemic relationship generalizes.

Default models (all fit on T4×2 in 4-bit):
  - meta-llama/Llama-3.1-8B-Instruct
    - Qwen/Qwen3-8B
    - microsoft/Phi-4-mini-instruct

Usage:
    python scripts/cross_model_replication.py \
        --output-dir artifacts/cross_model/ \
        --load-in-4bit --prompt-limit 200
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import (
    load_epistemic_prompts,
    load_safety_prompts,
    prompts_to_text_list,
)
from ddmi.editing.directions import (
    angular_distance_degrees,
    cosine_similarity,
    direction_from_contrast,
    rank_layers_by_separability,
    separability_score,
)
from ddmi.models.hooks import collect_activations_batched, get_residual_stream_names
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, ensure_dir, set_seed, write_json

logger = configure_logging(name="cross_model")


DEFAULT_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-8B",
    "microsoft/Phi-4-mini-instruct",
]


def run_single_model(
    model_id: str,
    safety_a: List[str],
    safety_b: List[str],
    epistemic_a: List[str],
    epistemic_b: List[str],
    load_in_4bit: bool = True,
    max_input_length: int = 512,
) -> Dict[str, Any]:
    """Extract directions and compute geometry for a single model."""
    # Phi-4-mini ships custom modeling code
    needs_trust = "phi" in model_id.lower()
    config = ModelLoadConfig(
        model_id=model_id,
        load_in_4bit=load_in_4bit,
        torch_dtype="float16",
        attn_implementation="eager",
        trust_remote_code=needs_trust,
    )

    logger.info("Loading model: %s", model_id)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    layer_names = get_residual_stream_names(model)
    logger.info("Discovered %d layers", len(layer_names))

    # Collect activations
    logger.info("Collecting safety activations...")
    acts_s_a = collect_activations_batched(model, tokenizer, safety_a, layer_names, max_input_length)
    acts_s_b = collect_activations_batched(model, tokenizer, safety_b, layer_names, max_input_length)

    logger.info("Collecting epistemic activations...")
    acts_e_a = collect_activations_batched(model, tokenizer, epistemic_a, layer_names, max_input_length)
    acts_e_b = collect_activations_batched(model, tokenizer, epistemic_b, layer_names, max_input_length)

    # Per-layer analysis
    comparisons = []
    safety_sep_scores = {}
    epistemic_sep_scores = {}

    for layer_name in layer_names:
        s_dir = direction_from_contrast(acts_s_a[layer_name], acts_s_b[layer_name])
        e_dir = direction_from_contrast(acts_e_a[layer_name], acts_e_b[layer_name])
        s_sep = separability_score(acts_s_a[layer_name], acts_s_b[layer_name])
        e_sep = separability_score(acts_e_a[layer_name], acts_e_b[layer_name])
        cos = cosine_similarity(s_dir, e_dir)
        angle = angular_distance_degrees(s_dir, e_dir)

        layer_idx = int(layer_name.split(".")[-1])
        comparisons.append({
            "layer_name": layer_name,
            "layer_index": layer_idx,
            "layer_depth_frac": layer_idx / len(layer_names),
            "cosine_similarity": cos,
            "angular_distance_deg": angle,
            "safety_separability": s_sep,
            "epistemic_separability": e_sep,
        })
        safety_sep_scores[layer_name] = s_sep
        epistemic_sep_scores[layer_name] = e_sep

    # Top layer analysis
    safety_ranked = rank_layers_by_separability(safety_sep_scores)
    epistemic_ranked = rank_layers_by_separability(epistemic_sep_scores)

    top_safety_layer = safety_ranked[0][0]
    top_epistemic_layer = epistemic_ranked[0][0]

    # Cosine at top safety layer
    top_comparison = next(c for c in comparisons if c["layer_name"] == top_safety_layer)

    logger.info("Top safety layer: %s (sep=%.4f)", top_safety_layer, safety_ranked[0][1])
    logger.info("Top epistemic layer: %s (sep=%.4f)", top_epistemic_layer, epistemic_ranked[0][1])
    logger.info(
        "Cosine(safety, epistemic) at top safety layer: %.4f (angle=%.1f°)",
        top_comparison["cosine_similarity"], top_comparison["angular_distance_deg"],
    )

    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "num_layers": len(layer_names),
        "top_safety_layer": top_safety_layer,
        "top_safety_sep": safety_ranked[0][1],
        "top_epistemic_layer": top_epistemic_layer,
        "top_epistemic_sep": epistemic_ranked[0][1],
        "cosine_at_top_safety": top_comparison["cosine_similarity"],
        "angle_at_top_safety": top_comparison["angular_distance_deg"],
        "safety_top5": [{"name": n, "score": s} for n, s in safety_ranked[:5]],
        "epistemic_top5": [{"name": n, "score": s} for n, s in epistemic_ranked[:5]],
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-model replication")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt-limit", type=int, default=200)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    # Load prompts once (shared across all models)
    safety_groups = load_safety_prompts(limit=args.prompt_limit, seed=args.seed)
    epistemic_groups = load_epistemic_prompts(limit=args.prompt_limit, seed=args.seed)

    safety_a = prompts_to_text_list(safety_groups["harmful"])
    safety_b = prompts_to_text_list(safety_groups["benign"])
    epistemic_a = prompts_to_text_list(epistemic_groups["unanswerable"])
    epistemic_b = prompts_to_text_list(epistemic_groups["answerable"])

    all_results = []

    for model_id in args.models:
        logger.info("\n{'='*60}")
        logger.info("Model: %s", model_id)
        logger.info("{'='*60}")

        t0 = time.time()
        result = run_single_model(
            model_id, safety_a, safety_b, epistemic_a, epistemic_b,
            load_in_4bit=args.load_in_4bit,
        )
        result["elapsed_seconds"] = time.time() - t0
        all_results.append(result)

        # Save per-model result
        safe_name = model_id.replace("/", "_").replace("-", "_")
        write_json(
            str(Path(args.output_dir) / f"{safe_name}_comparison.json"),
            result,
        )

    # Summary table
    logger.info("\n=== CROSS-MODEL SUMMARY ===")
    logger.info(
        "%-40s | Top Safety | Top Epist | Cosine | Angle",
        "Model",
    )
    logger.info("-" * 90)
    for r in all_results:
        logger.info(
            "%-40s | %-10s | %-9s | %.4f | %.1f°",
            r["model_id"],
            r["top_safety_layer"],
            r["top_epistemic_layer"],
            r["cosine_at_top_safety"],
            r["angle_at_top_safety"],
        )

    # Save combined
    combined = {
        "artifact_type": "cross_model_replication",
        "models": args.models,
        "prompt_limit": args.prompt_limit,
        "seed": args.seed,
        "results": all_results,
        "summary": [
            {
                "model_id": r["model_id"],
                "top_safety_layer": r["top_safety_layer"],
                "top_epistemic_layer": r["top_epistemic_layer"],
                "cosine_at_top_safety": r["cosine_at_top_safety"],
                "angle_at_top_safety": r["angle_at_top_safety"],
            }
            for r in all_results
        ],
    }

    write_json(str(Path(args.output_dir) / "combined_results.json"), combined)
    logger.info("Combined results saved to %s/combined_results.json", args.output_dir)


if __name__ == "__main__":
    main()
