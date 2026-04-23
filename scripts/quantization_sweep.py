#!/usr/bin/env python3
"""Experiment 3: Quantization perturbation sweep.

Measures how different quantization methods perturb the safety and epistemic
directions. For each precision:
  1. Load model at that precision
  2. Extract both directions
  3. Compare to FP16 reference directions (cosine, norm ratio)
  4. Test cross-precision ablation: does FP16-extracted direction still work?

Usage:
    python scripts/quantization_sweep.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --fp16-safety-artifact artifacts/directions/llama31_8b_safety_fp16.json \
        --fp16-epistemic-artifact artifacts/directions/llama31_8b_epistemic_fp16.json \
        --output artifacts/quantization/llama31_8b_sweep.json
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
    cosine_similarity,
    direction_from_contrast,
    l2_norm,
    separability_score,
)
from ddmi.models.hooks import collect_activations_batched, get_residual_stream_names
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, read_json, set_seed, write_json

logger = configure_logging(name="quantization_sweep")


PRECISION_CONFIGS = {
    "fp16": {"load_in_4bit": False, "load_in_8bit": False, "torch_dtype": "float16"},
    "bnb_nf4": {"load_in_4bit": True, "load_in_8bit": False, "torch_dtype": "float16"},
    "bnb_int8": {"load_in_4bit": False, "load_in_8bit": True, "torch_dtype": "float16"},
}


def extract_directions_at_layer(
    model: Any,
    tokenizer: Any,
    layer_name: str,
    safety_a: List[str],
    safety_b: List[str],
    epistemic_a: List[str],
    epistemic_b: List[str],
    max_input_length: int = 512,
) -> Dict[str, Any]:
    """Extract safety and epistemic directions at a specific layer."""
    # Collect activations for all groups at this layer
    acts_safety_a = collect_activations_batched(model, tokenizer, safety_a, [layer_name], max_input_length)
    acts_safety_b = collect_activations_batched(model, tokenizer, safety_b, [layer_name], max_input_length)
    acts_epist_a = collect_activations_batched(model, tokenizer, epistemic_a, [layer_name], max_input_length)
    acts_epist_b = collect_activations_batched(model, tokenizer, epistemic_b, [layer_name], max_input_length)

    s_dir = direction_from_contrast(acts_safety_a[layer_name], acts_safety_b[layer_name])
    e_dir = direction_from_contrast(acts_epist_a[layer_name], acts_epist_b[layer_name])
    s_sep = separability_score(acts_safety_a[layer_name], acts_safety_b[layer_name])
    e_sep = separability_score(acts_epist_a[layer_name], acts_epist_b[layer_name])

    # Raw diff norms (before normalization)
    from ddmi.editing.directions import difference_of_means
    s_raw = l2_norm(difference_of_means(acts_safety_a[layer_name], acts_safety_b[layer_name]))
    e_raw = l2_norm(difference_of_means(acts_epist_a[layer_name], acts_epist_b[layer_name]))

    return {
        "safety_direction": s_dir,
        "epistemic_direction": e_dir,
        "safety_separability": s_sep,
        "epistemic_separability": e_sep,
        "safety_raw_norm": s_raw,
        "epistemic_raw_norm": e_raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantization perturbation sweep")
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--fp16-safety-artifact",
        help="FP16 safety direction artifact (if available, skip FP16 extraction)",
    )
    parser.add_argument(
        "--fp16-epistemic-artifact",
        help="FP16 epistemic direction artifact (if available, skip FP16 extraction)",
    )
    parser.add_argument("--target-layer", help="Specific layer to analyze (default: top safety layer)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load prompts (same across all precisions for fair comparison)
    safety_groups = load_safety_prompts(limit=args.prompt_limit, seed=args.seed)
    epistemic_groups = load_epistemic_prompts(limit=args.prompt_limit, seed=args.seed)

    safety_a = prompts_to_text_list(safety_groups["harmful"])
    safety_b = prompts_to_text_list(safety_groups["benign"])
    epistemic_a = prompts_to_text_list(epistemic_groups["unanswerable"])
    epistemic_b = prompts_to_text_list(epistemic_groups["answerable"])

    # Determine target layer
    target_layer = args.target_layer
    if target_layer is None and args.fp16_safety_artifact:
        fp16_safety = read_json(args.fp16_safety_artifact)
        target_layer = fp16_safety["ranked_layers"][0]["name"]
        logger.info("Using top safety layer from FP16 artifact: %s", target_layer)

    results = {}

    for precision, precision_kwargs in PRECISION_CONFIGS.items():
        logger.info("\n=== Precision: %s ===", precision)

        config = ModelLoadConfig(model_id=args.model_id, **precision_kwargs, attn_implementation="eager")

        t0 = time.time()
        try:
            model, tokenizer = load_model_and_tokenizer(config)
        except Exception as e:
            logger.warning("Failed to load at %s: %s", precision, e)
            results[precision] = {"error": str(e)}
            continue
        logger.info("Model loaded in %.1fs", time.time() - t0)

        # Discover layers if target not set
        if target_layer is None:
            layer_names = get_residual_stream_names(model)
            target_layer = layer_names[len(layer_names) // 2]
            logger.info("Auto-selected target layer: %s", target_layer)

        # Extract directions
        t0 = time.time()
        layer_results = extract_directions_at_layer(
            model, tokenizer, target_layer,
            safety_a, safety_b, epistemic_a, epistemic_b,
        )
        logger.info("Directions extracted in %.1fs", time.time() - t0)

        results[precision] = {
            "safety_direction": layer_results["safety_direction"],
            "epistemic_direction": layer_results["epistemic_direction"],
            "safety_separability": layer_results["safety_separability"],
            "epistemic_separability": layer_results["epistemic_separability"],
            "safety_raw_norm": layer_results["safety_raw_norm"],
            "epistemic_raw_norm": layer_results["epistemic_raw_norm"],
        }

        # Free memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute drift relative to FP16
    if "fp16" in results and "error" not in results.get("fp16", {}):
        fp16_s = results["fp16"]["safety_direction"]
        fp16_e = results["fp16"]["epistemic_direction"]

        drift_table = []
        for precision, res in results.items():
            if "error" in res:
                continue
            s_cos = cosine_similarity(fp16_s, res["safety_direction"])
            e_cos = cosine_similarity(fp16_e, res["epistemic_direction"])
            s_norm_ratio = l2_norm(res["safety_direction"]) / max(l2_norm(fp16_s), 1e-12)
            e_norm_ratio = l2_norm(res["epistemic_direction"]) / max(l2_norm(fp16_e), 1e-12)

            row = {
                "precision": precision,
                "safety_cosine_vs_fp16": s_cos,
                "epistemic_cosine_vs_fp16": e_cos,
                "safety_norm_ratio": s_norm_ratio,
                "epistemic_norm_ratio": e_norm_ratio,
                "safety_separability": res["safety_separability"],
                "epistemic_separability": res["epistemic_separability"],
            }
            drift_table.append(row)

            logger.info(
                "%s: safety_cos=%.4f, epistemic_cos=%.4f, s_norm=%.4f, e_norm=%.4f",
                precision, s_cos, e_cos, s_norm_ratio, e_norm_ratio,
            )
    else:
        drift_table = []
        logger.warning("FP16 baseline not available — skipping drift computation")

    # Save (strip direction vectors from output to keep file manageable)
    for precision in results:
        if "error" not in results[precision]:
            results[precision].pop("safety_direction", None)
            results[precision].pop("epistemic_direction", None)

    artifact = {
        "artifact_type": "quantization_sweep",
        "model_id": args.model_id,
        "target_layer": target_layer,
        "prompt_limit": args.prompt_limit,
        "seed": args.seed,
        "precisions": list(PRECISION_CONFIGS.keys()),
        "drift_table": drift_table,
        "per_precision_results": results,
    }

    write_json(args.output, artifact)
    logger.info("Sweep results saved to %s", args.output)


if __name__ == "__main__":
    main()
