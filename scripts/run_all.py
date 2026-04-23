#!/usr/bin/env python3
"""Run the full experiment pipeline end-to-end.

This is the master orchestration script. Run on Kaggle T4×2 with:

    python scripts/run_all.py --model-id meta-llama/Llama-3.1-8B-Instruct --load-in-4bit

Estimated runtime: ~6-8 hours for single model, ~16-20 hours for all three.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd: list[str], description: str) -> None:
    """Run a command, printing status."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n✗ FAILED: {description} (exit code {result.returncode}, {elapsed:.0f}s)")
        sys.exit(1)
    else:
        print(f"\n✓ DONE: {description} ({elapsed:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--prompt-limit", type=int, default=200)
    parser.add_argument("--eval-prompts", type=int, default=100)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--skip-quant", action="store_true", help="Skip quantization sweep")
    parser.add_argument("--skip-cross-model", action="store_true", help="Skip cross-model replication")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    scripts = root / "scripts"

    quant_flag = ["--load-in-4bit"] if args.load_in_4bit else []
    safe_name = args.model_id.replace("/", "_").replace("-", "_")

    # === Experiment 1a: Extract safety direction ===
    safety_out = f"artifacts/directions/{safe_name}_safety.json"
    run_cmd(
        [sys.executable, str(scripts / "extract_directions.py"),
         "--model-id", args.model_id,
         "--direction-type", "safety",
         "--output", safety_out,
         "--prompt-limit", str(args.prompt_limit),
         "--seed", str(args.seed),
         *quant_flag],
        "Experiment 1a: Extract safety direction",
    )

    # === Experiment 1a: Extract epistemic direction ===
    epistemic_out = f"artifacts/directions/{safe_name}_epistemic.json"
    run_cmd(
        [sys.executable, str(scripts / "extract_directions.py"),
         "--model-id", args.model_id,
         "--direction-type", "epistemic",
         "--output", epistemic_out,
         "--prompt-limit", str(args.prompt_limit),
         "--seed", str(args.seed),
         *quant_flag],
        "Experiment 1a: Extract epistemic direction",
    )

    # === Experiment 1b: Compare directions ===
    comparison_out = f"artifacts/directions/{safe_name}_comparison.json"
    run_cmd(
        [sys.executable, str(scripts / "compare_directions.py"),
         "--safety-artifact", safety_out,
         "--epistemic-artifact", epistemic_out,
         "--output", comparison_out],
        "Experiment 1b: Compare direction geometry",
    )

    # === Experiment 2: Cross-ablation ===
    ablation_out = f"artifacts/cross_ablation/{safe_name}_results.json"
    run_cmd(
        [sys.executable, str(scripts / "cross_ablation.py"),
         "--model-id", args.model_id,
         "--safety-direction", safety_out,
         "--epistemic-direction", epistemic_out,
         "--output", ablation_out,
         "--eval-prompts", str(args.eval_prompts),
         "--seed", str(args.seed),
         *quant_flag],
        "Experiment 2: Cross-ablation",
    )

    # === Experiment 3: Quantization sweep ===
    if not args.skip_quant:
        quant_out = f"artifacts/quantization/{safe_name}_sweep.json"
        run_cmd(
            [sys.executable, str(scripts / "quantization_sweep.py"),
             "--model-id", args.model_id,
             "--fp16-safety-artifact", safety_out,
             "--output", quant_out,
             "--prompt-limit", str(min(args.prompt_limit, 100)),
             "--seed", str(args.seed)],
            "Experiment 3: Quantization perturbation sweep",
        )

    # === Experiment 4: Cross-model replication ===
    if not args.skip_cross_model:
        cross_out = "artifacts/cross_model/"
        run_cmd(
            [sys.executable, str(scripts / "cross_model_replication.py"),
             "--output-dir", cross_out,
             "--prompt-limit", str(args.prompt_limit),
             "--seed", str(args.seed),
             *quant_flag],
            "Experiment 4: Cross-model replication",
        )

    # === Generate figures ===
    run_cmd(
        [sys.executable, str(scripts / "generate_figures.py"),
         "--artifacts-dir", "artifacts/",
         "--output-dir", "artifacts/figures/"],
        "Generate publication figures",
    )

    print(f"\n{'='*60}")
    print("✓ ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    print(f"\nArtifacts: artifacts/")
    print(f"Figures:   artifacts/figures/")
    print(f"Paper:     paper/main.tex")


if __name__ == "__main__":
    main()
