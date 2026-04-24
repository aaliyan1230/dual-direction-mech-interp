#!/usr/bin/env python3
"""Experiment 2: Cross-ablation — ablate one direction, measure both behaviors.

The 2×2 experiment:
  - Baseline (no ablation) → measure safety refusal + epistemic abstention rates
  - Ablate safety direction → measure both rates
  - Ablate epistemic direction → measure both rates

This is the core experiment: if ablating the safety direction also kills epistemic
abstention, they share a mechanism. If independent, they're mechanistically distinct.

Usage:
    python scripts/cross_ablation.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --safety-direction artifacts/directions/llama31_8b_safety.json \
        --epistemic-direction artifacts/directions/llama31_8b_epistemic.json \
        --output artifacts/cross_ablation/llama31_8b_results.json \
        --load-in-4bit
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import (
    load_epistemic_prompts,
    load_safety_prompts,
    prompts_to_text_list,
)
from ddmi.editing.apply_edit import (
    ActivationDirectionAblator,
    EditSpec,
)
from ddmi.evaluation.detectors import (
    abstention_rate,
    classify_responses,
    refusal_rate,
)
from ddmi.evaluation.metrics import CrossAblationMetrics
from ddmi.models.generation import TextGenerationConfig, generate_batch
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, read_json, set_seed, write_json

logger = configure_logging(name="cross_ablation")


def evaluate_condition(
    model: Any,
    tokenizer: Any,
    safety_prompts: List[str],
    epistemic_prompts: List[str],
    gen_config: TextGenerationConfig,
) -> Dict[str, Any]:
    """Evaluate refusal and abstention rates on both prompt sets."""
    # Generate responses
    logger.info("  Generating on %d safety prompts...", len(safety_prompts))
    safety_responses = generate_batch(model, tokenizer, safety_prompts, config=gen_config)

    logger.info("  Generating on %d epistemic prompts...", len(epistemic_prompts))
    epistemic_responses = generate_batch(model, tokenizer, epistemic_prompts, config=gen_config)

    # Classify
    safety_classes = classify_responses(safety_responses)
    epistemic_classes = classify_responses(epistemic_responses)

    return {
        "safety_refusal_rate": refusal_rate(safety_responses),
        "safety_abstention_rate": abstention_rate(safety_responses),
        "epistemic_refusal_rate": refusal_rate(epistemic_responses),
        "epistemic_abstention_rate": abstention_rate(epistemic_responses),
        "safety_response_classes": {
            "refusal": safety_classes.count("refusal"),
            "abstention": safety_classes.count("abstention"),
            "answer": safety_classes.count("answer"),
        },
        "epistemic_response_classes": {
            "refusal": epistemic_classes.count("refusal"),
            "abstention": epistemic_classes.count("abstention"),
            "answer": epistemic_classes.count("answer"),
        },
        # Store a few example responses for qualitative inspection
        "safety_examples": safety_responses[:5],
        "epistemic_examples": epistemic_responses[:5],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-ablation experiment")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--safety-direction", required=True, help="Safety direction artifact JSON")
    parser.add_argument("--epistemic-direction", required=True, help="Epistemic direction artifact JSON")
    parser.add_argument("--output", required=True, help="Output results JSON")
    parser.add_argument("--eval-prompts", type=int, default=100, help="Eval prompts per group")
    parser.add_argument("--top-k-layers", type=int, default=1, help="Top-K layers to ablate")
    parser.add_argument("--strength", type=float, default=1.0, help="Ablation strength")
    parser.add_argument("--module-type", nargs="+", default=["attn_out"], help="Module types to ablate")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load direction artifacts
    safety_art = read_json(args.safety_direction)
    epistemic_art = read_json(args.epistemic_direction)

    # Get top-K layer directions
    safety_ranked = safety_art["ranked_layers"][:args.top_k_layers]
    epistemic_ranked = epistemic_art["ranked_layers"][:args.top_k_layers]

    safety_dir_vec = safety_art["directions"][safety_ranked[0]["name"]]["direction"]
    epistemic_dir_vec = epistemic_art["directions"][epistemic_ranked[0]["name"]]["direction"]

    logger.info("Safety direction: %s (sep=%.4f)", safety_ranked[0]["name"], safety_ranked[0]["score"])
    logger.info("Epistemic direction: %s (sep=%.4f)", epistemic_ranked[0]["name"], epistemic_ranked[0]["score"])

    # Load model
    config = ModelLoadConfig(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype="float16",
        attn_implementation="eager",
    )
    logger.info("Loading model: %s", args.model_id)
    model, tokenizer = load_model_and_tokenizer(config)

    gen_config = TextGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    # Load evaluation prompts (separate from training prompts to avoid leakage)
    safety_eval = load_safety_prompts(limit=args.eval_prompts, seed=args.seed + 1000)
    epistemic_eval = load_epistemic_prompts(limit=args.eval_prompts, seed=args.seed + 1000)

    # For safety eval, use harmful prompts (we want to see if refusal persists)
    safety_prompts = prompts_to_text_list(safety_eval["harmful"])
    # For epistemic eval, use unanswerable prompts (we want to see if abstention persists)
    epistemic_prompts = prompts_to_text_list(epistemic_eval["unanswerable"])

    logger.info("Eval prompts: %d safety, %d epistemic", len(safety_prompts), len(epistemic_prompts))

    safety_target_layers = [r["name"] for r in safety_ranked]
    epistemic_target_layers = [r["name"] for r in epistemic_ranked]

    logger.info("Safety target layers: %s", ", ".join(safety_target_layers))
    logger.info("Epistemic target layers: %s", ", ".join(epistemic_target_layers))

    results: List[Dict[str, Any]] = []
    spec = EditSpec(strength=args.strength)

    # === Condition 1: Baseline (no ablation) ===
    logger.info("\n=== Condition 1: BASELINE (no ablation) ===")
    t0 = time.time()
    baseline = evaluate_condition(model, tokenizer, safety_prompts, epistemic_prompts, gen_config)
    baseline["condition"] = "baseline"
    baseline["ablated_direction"] = "none"
    baseline["elapsed_seconds"] = time.time() - t0
    results.append(baseline)
    logger.info(
        "Baseline: safety_refusal=%.3f, epistemic_abstention=%.3f",
        baseline["safety_refusal_rate"], baseline["epistemic_abstention_rate"],
    )

    # === Condition 2: Ablate safety direction ===
    logger.info("\n=== Condition 2: ABLATE SAFETY DIRECTION ===")
    safety_ablator = ActivationDirectionAblator(
        module_names=safety_target_layers,
        direction=safety_dir_vec,
        spec=spec,
    ).attach(model)

    t0 = time.time()
    ablate_safety = evaluate_condition(model, tokenizer, safety_prompts, epistemic_prompts, gen_config)
    ablate_safety["condition"] = "ablate_safety"
    ablate_safety["ablated_direction"] = "safety"
    ablate_safety["ablated_layers"] = [r["name"] for r in safety_ranked]
    ablate_safety["strength"] = args.strength
    ablate_safety["elapsed_seconds"] = time.time() - t0
    results.append(ablate_safety)
    logger.info(
        "Ablate safety: safety_refusal=%.3f (Δ=%.3f), epistemic_abstention=%.3f (Δ=%.3f)",
        ablate_safety["safety_refusal_rate"],
        ablate_safety["safety_refusal_rate"] - baseline["safety_refusal_rate"],
        ablate_safety["epistemic_abstention_rate"],
        ablate_safety["epistemic_abstention_rate"] - baseline["epistemic_abstention_rate"],
    )

    safety_ablator.close()

    # === Condition 3: Ablate epistemic direction ===
    logger.info("\n=== Condition 3: ABLATE EPISTEMIC DIRECTION ===")
    epistemic_ablator = ActivationDirectionAblator(
        module_names=epistemic_target_layers,
        direction=epistemic_dir_vec,
        spec=spec,
    ).attach(model)

    t0 = time.time()
    ablate_epistemic = evaluate_condition(model, tokenizer, safety_prompts, epistemic_prompts, gen_config)
    ablate_epistemic["condition"] = "ablate_epistemic"
    ablate_epistemic["ablated_direction"] = "epistemic"
    ablate_epistemic["ablated_layers"] = [r["name"] for r in epistemic_ranked]
    ablate_epistemic["strength"] = args.strength
    ablate_epistemic["elapsed_seconds"] = time.time() - t0
    results.append(ablate_epistemic)
    logger.info(
        "Ablate epistemic: safety_refusal=%.3f (Δ=%.3f), epistemic_abstention=%.3f (Δ=%.3f)",
        ablate_epistemic["safety_refusal_rate"],
        ablate_epistemic["safety_refusal_rate"] - baseline["safety_refusal_rate"],
        ablate_epistemic["epistemic_abstention_rate"],
        ablate_epistemic["epistemic_abstention_rate"] - baseline["epistemic_abstention_rate"],
    )

    epistemic_ablator.close()

    # === Summary ===
    logger.info("\n=== CROSS-ABLATION SUMMARY ===")
    logger.info("                          | Safety Refusal | Epistemic Abstention |")
    logger.info("-" * 70)
    for r in results:
        logger.info(
            "%-25s | %.3f          | %.3f                |",
            r["condition"], r["safety_refusal_rate"], r["epistemic_abstention_rate"],
        )

    # Key interpretive metric: cross-contamination
    safety_cross = abs(
        ablate_safety["epistemic_abstention_rate"] - baseline["epistemic_abstention_rate"]
    )
    epistemic_cross = abs(
        ablate_epistemic["safety_refusal_rate"] - baseline["safety_refusal_rate"]
    )
    logger.info("\nCross-contamination:")
    logger.info("  Ablate safety → Δ epistemic abstention: %.3f", safety_cross)
    logger.info("  Ablate epistemic → Δ safety refusal: %.3f", epistemic_cross)
    logger.info(
        "  Interpretation: high cross-contamination (>0.1) → shared mechanism"
    )

    # Save
    artifact = {
        "artifact_type": "cross_ablation_results",
        "model_id": args.model_id,
        "safety_direction_artifact": args.safety_direction,
        "epistemic_direction_artifact": args.epistemic_direction,
        "config": {
            "eval_prompts": args.eval_prompts,
            "top_k_layers": args.top_k_layers,
            "strength": args.strength,
            "target_layers": {
                "safety": safety_target_layers,
                "epistemic": epistemic_target_layers,
            },
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "results": results,
        "summary": {
            "baseline_safety_refusal": baseline["safety_refusal_rate"],
            "baseline_epistemic_abstention": baseline["epistemic_abstention_rate"],
            "cross_contamination_safety_to_epistemic": safety_cross,
            "cross_contamination_epistemic_to_safety": epistemic_cross,
        },
    }

    write_json(args.output, artifact)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
