# Dual-Direction Mechanistic Interpretability

**Disentangling safety-refusal and epistemic-abstention directions in LLM activation space.**

Do LLMs refuse harmful requests and abstain on unknown factual questions using the *same* internal mechanism — or are these mechanistically independent behaviors?

This repository provides a reproducible pipeline to:

1. **Extract** both safety-refusal and epistemic-abstention directions via difference-in-means on residual stream activations
2. **Measure** their geometric relationship (cosine similarity, angular separation) across all layers
3. **Cross-ablate** each direction and measure the behavioral impact on both safety refusal and epistemic abstention
4. **Quantify** how deployment-time quantization (NF4, INT8, GPTQ-4bit, AWQ-4bit) perturbs these directions
5. **Replicate** findings across model families (Llama, Qwen, Gemma)

## Key Research Question

> When an instruction-tuned LLM says "I can't help with that" (safety) vs. "I don't know" (epistemic), are these mediated by the same geometric direction in activation space?

## Setup

```bash
# Clone
git clone https://github.com/aaliyan1230/dual-direction-mech-interp.git
cd dual-direction-mech-interp

# Install (core)
pip install -e .

# Install with quantization backends
pip install -e ".[quant]"

# Install with plotting
pip install -e ".[plot]"
```

## Pipeline

### Experiment 1: Direction Extraction & Geometry

```bash
# Extract safety-refusal direction (harmful vs. benign prompts)
python scripts/extract_directions.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --direction-type safety \
    --output artifacts/directions/llama31_8b_safety.json \
    --load-in-4bit --prompt-limit 200

# Extract epistemic-abstention direction (answerable vs. unanswerable)
python scripts/extract_directions.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --direction-type epistemic \
    --output artifacts/directions/llama31_8b_epistemic.json \
    --load-in-4bit --prompt-limit 200

# Compare geometry across layers
python scripts/compare_directions.py \
    --safety-artifact artifacts/directions/llama31_8b_safety.json \
    --epistemic-artifact artifacts/directions/llama31_8b_epistemic.json \
    --output artifacts/directions/llama31_8b_comparison.json
```

### Experiment 2: Cross-Ablation

```bash
python scripts/cross_ablation.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --safety-direction artifacts/directions/llama31_8b_safety.json \
    --epistemic-direction artifacts/directions/llama31_8b_epistemic.json \
    --output artifacts/cross_ablation/llama31_8b_results.json \
    --load-in-4bit
```

### Experiment 3: Quantization Perturbation

```bash
python scripts/quantization_sweep.py \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --output artifacts/quantization/llama31_8b_sweep.json
```

### Experiment 4: Cross-Model Replication

```bash
python scripts/cross_model_replication.py \
    --output artifacts/cross_model/ \
    --load-in-4bit
```

### Generate Figures

```bash
python scripts/generate_figures.py \
    --artifacts-dir artifacts/ \
    --output-dir artifacts/figures/
```

## Project Structure

```
src/ddmi/
    editing/       Direction extraction, projection, ablation
    models/        Model loading, hooks, generation
    data/          Dataset loaders, prompt schemas
    evaluation/    Refusal detection, abstention detection, metrics
    utils/         IO, logging, seeds
scripts/           Runnable experiment scripts
configs/           YAML experiment configs
artifacts/         Generated data (directions, results, figures)
paper/             LaTeX source
tests/             Unit tests
```

## Hardware Requirements

All experiments run on **2× NVIDIA T4 (16 GB each)** — designed for Kaggle free-tier hardware.

- Llama-3.1-8B-Instruct in NF4: ~5 GB VRAM
- Direction extraction (200 prompts): ~1 hour per model
- Cross-ablation evaluation: ~2 hours per model
- Full pipeline (all 4 experiments, 3 models): ~24 GPU-hours

## Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| `ivnle/advbench_harmful_behaviors` | Harmful prompts (safety-refusal positive class) | HF Hub |
| `tatsu-lab/alpaca` | Benign instructions (safety-refusal negative class) | HF Hub |
| `rajpurkar/squad_v2` | Unanswerable questions (epistemic-abstention positive class) | HF Hub |
| `mandarjoshi/trivia_qa` (rc.nocontext) | Answerable questions (epistemic-abstention negative class) | HF Hub |

## Citation

If you find this work useful, please cite:

```bibtex
@misc{shaikh2026dualdirection,
    title={Two Refusals or One? Disentangling Safety and Epistemic Abstention Directions Under Quantization},
    author={Shaikh, Aaliyan},
    year={2026},
    note={Preprint}
}
```

## License

MIT
