#!/usr/bin/env python3
"""Experiment 1a: Extract safety or epistemic directions via difference-in-means.

Collects residual-stream activations at every layer, computes the
difference-in-means direction for each layer, and ranks by separability.

Usage:
    python scripts/extract_directions.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --direction-type safety \
        --output artifacts/directions/llama31_8b_safety.json \
        --load-in-4bit --prompt-limit 200
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

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
    direction_from_contrast,
    l2_norm,
    rank_layers_by_separability,
    separability_score,
)
from ddmi.models.hooks import collect_activations_batched, get_residual_stream_names
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, set_seed, write_json

logger = configure_logging(name="extract_directions")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activation directions")
    parser.add_argument("--model-id", required=True, help="HF model ID")
    parser.add_argument(
        "--direction-type",
        required=True,
        choices=["safety", "epistemic"],
        help="Type of direction to extract",
    )
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--prompt-limit", type=int, default=200, help="Total prompts (split 50/50)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load in BnB NF4")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load in BnB INT8")
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load model
    config = ModelLoadConfig(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype="float16",
        attn_implementation="eager",
    )
    logger.info("Loading model: %s", args.model_id)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Discover residual stream layers
    layer_names = get_residual_stream_names(model)
    logger.info("Discovered %d residual stream layers", len(layer_names))

    # Load prompts
    if args.direction_type == "safety":
        prompt_groups = load_safety_prompts(limit=args.prompt_limit, seed=args.seed)
        group_a_key, group_b_key = "harmful", "benign"
    else:
        prompt_groups = load_epistemic_prompts(limit=args.prompt_limit, seed=args.seed)
        group_a_key, group_b_key = "unanswerable", "answerable"

    group_a_texts = prompts_to_text_list(prompt_groups[group_a_key])
    group_b_texts = prompts_to_text_list(prompt_groups[group_b_key])
    logger.info(
        "Group A (%s): %d prompts, Group B (%s): %d prompts",
        group_a_key, len(group_a_texts), group_b_key, len(group_b_texts),
    )

    # Collect activations for both groups
    logger.info("Collecting activations for group A (%s)...", group_a_key)
    t0 = time.time()
    acts_a = collect_activations_batched(
        model, tokenizer, group_a_texts, layer_names,
        max_input_length=args.max_input_length,
    )
    logger.info("Group A activations collected in %.1fs", time.time() - t0)

    logger.info("Collecting activations for group B (%s)...", group_b_key)
    t0 = time.time()
    acts_b = collect_activations_batched(
        model, tokenizer, group_b_texts, layer_names,
        max_input_length=args.max_input_length,
    )
    logger.info("Group B activations collected in %.1fs", time.time() - t0)

    # Compute directions and separability at each layer
    directions = {}
    layer_sep_scores = {}

    for layer_name in layer_names:
        vecs_a = acts_a[layer_name]
        vecs_b = acts_b[layer_name]

        direction = direction_from_contrast(vecs_a, vecs_b)
        sep = separability_score(vecs_a, vecs_b)
        raw_norm = l2_norm(
            [ma - mb for ma, mb in zip(
                [sum(v) / len(v) for v in zip(*vecs_a)],
                [sum(v) / len(v) for v in zip(*vecs_b)],
            )]
        ) if vecs_a and vecs_b else 0.0

        layer_idx = int(layer_name.split(".")[-1])
        directions[layer_name] = {
            "direction": direction,
            "separability_score": sep,
            "raw_diff_norm": raw_norm,
            "num_group_a": len(vecs_a),
            "num_group_b": len(vecs_b),
        }
        layer_sep_scores[layer_name] = sep

        logger.info(
            "  %s: separability=%.4f, raw_norm=%.4f",
            layer_name, sep, raw_norm,
        )

    # Rank layers
    ranked = rank_layers_by_separability(layer_sep_scores)
    logger.info("Top-5 layers by separability:")
    for name, score in ranked[:5]:
        logger.info("  %s: %.4f", name, score)

    # Save artifact
    artifact = {
        "artifact_type": "direction_collection",
        "model_id": args.model_id,
        "direction_type": args.direction_type,
        "source_groups": {"group_a": group_a_key, "group_b": group_b_key},
        "prompt_limit": args.prompt_limit,
        "precision": "bnb_nf4" if args.load_in_4bit else ("bnb_int8" if args.load_in_8bit else "fp16"),
        "seed": args.seed,
        "num_layers": len(layer_names),
        "directions": directions,
        "ranked_layers": [{"name": n, "score": s} for n, s in ranked],
    }

    write_json(args.output, artifact)
    logger.info("Direction artifact saved to %s", args.output)


if __name__ == "__main__":
    main()
