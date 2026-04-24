#!/usr/bin/env python3
"""Train simple linear probes on activation snapshots.

Usage:
    python scripts/linear_probe.py \
        --model-id meta-llama/Llama-3.1-8B-Instruct \
        --direction-type epistemic \
        --direction-artifact artifacts/directions/meta_llama_Llama_3.1_8B_Instruct_epistemic.json \
        --output artifacts/probes/meta_llama_Llama_3.1_8B_Instruct_epistemic_probe.json \
        --load-in-4bit
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ddmi.data.loaders import load_epistemic_prompts, load_safety_prompts, prompts_to_text_list
from ddmi.evaluation.probes import fit_and_evaluate_binary_probe
from ddmi.models.hooks import collect_activations_batched, get_residual_stream_names
from ddmi.models.loader import ModelLoadConfig, load_model_and_tokenizer
from ddmi.utils.io import configure_logging, read_json, set_seed, write_json

logger = configure_logging(name="linear_probe")


def load_probe_groups(direction_type: str, prompt_limit: int, seed: int) -> Tuple[str, str, List[str], List[str]]:
    """Load prompt groups for a probe task."""
    if direction_type == "safety":
        groups = load_safety_prompts(limit=prompt_limit, seed=seed)
        positive_key, negative_key = "harmful", "benign"
    else:
        groups = load_epistemic_prompts(limit=prompt_limit, seed=seed)
        positive_key, negative_key = "unanswerable", "answerable"

    return (
        positive_key,
        negative_key,
        prompts_to_text_list(groups[positive_key]),
        prompts_to_text_list(groups[negative_key]),
    )


def maybe_selected_layer(direction_artifact_path: str | None) -> str | None:
    """Return the top-ranked layer from a direction artifact when available."""
    if not direction_artifact_path:
        return None

    artifact = read_json(direction_artifact_path)
    ranked_layers = artifact.get("ranked_layers", [])
    if not ranked_layers:
        return None
    return ranked_layers[0]["name"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train linear probes on activation snapshots")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--direction-type", required=True, choices=["safety", "epistemic"])
    parser.add_argument("--direction-artifact", help="Optional direction artifact used to highlight the top layer")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-limit", type=int, default=200)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    selected_layer = maybe_selected_layer(args.direction_artifact)

    positive_key, negative_key, positive_prompts, negative_prompts = load_probe_groups(
        args.direction_type,
        args.prompt_limit,
        args.seed,
    )

    config = ModelLoadConfig(
        model_id=args.model_id,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        torch_dtype="float16",
        attn_implementation="eager",
        trust_remote_code="phi" in args.model_id.lower(),
    )
    logger.info("Loading model: %s", args.model_id)
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(config)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    layer_names = get_residual_stream_names(model)
    logger.info("Discovered %d residual stream layers", len(layer_names))

    logger.info("Collecting %s activations...", positive_key)
    positive_acts = collect_activations_batched(
        model,
        tokenizer,
        positive_prompts,
        layer_names,
        max_input_length=args.max_input_length,
    )
    logger.info("Collecting %s activations...", negative_key)
    negative_acts = collect_activations_batched(
        model,
        tokenizer,
        negative_prompts,
        layer_names,
        max_input_length=args.max_input_length,
    )

    layer_results: List[Dict[str, Any]] = []
    for layer_name in layer_names:
        probe_metrics = fit_and_evaluate_binary_probe(
            positive_acts[layer_name],
            negative_acts[layer_name],
            train_fraction=args.train_fraction,
            seed=args.seed,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        layer_idx = int(layer_name.split(".")[-1])
        layer_results.append(
            {
                "layer_name": layer_name,
                "layer_index": layer_idx,
                **probe_metrics,
            }
        )

    layer_results.sort(key=lambda row: row["test"]["accuracy"], reverse=True)
    best_layer = layer_results[0]
    selected_layer_result = next(
        (row for row in layer_results if row["layer_name"] == selected_layer),
        best_layer,
    )

    artifact = {
        "artifact_type": "linear_probe_results",
        "model_id": args.model_id,
        "direction_type": args.direction_type,
        "positive_group": positive_key,
        "negative_group": negative_key,
        "direction_artifact": args.direction_artifact,
        "config": {
            "prompt_limit": args.prompt_limit,
            "train_fraction": args.train_fraction,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "max_input_length": args.max_input_length,
            "seed": args.seed,
        },
        "summary": {
            "best_test_accuracy": best_layer["test"]["accuracy"],
            "best_layer": best_layer["layer_name"],
            "selected_layer": selected_layer_result["layer_name"],
            "selected_layer_test_accuracy": selected_layer_result["test"]["accuracy"],
            "selected_layer_train_accuracy": selected_layer_result["train"]["accuracy"],
        },
        "layers": layer_results,
    }

    write_json(args.output, artifact)
    logger.info(
        "Best probe accuracy %.3f at %s; selected layer %s accuracy %.3f",
        best_layer["test"]["accuracy"],
        best_layer["layer_name"],
        selected_layer_result["layer_name"],
        selected_layer_result["test"]["accuracy"],
    )

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()