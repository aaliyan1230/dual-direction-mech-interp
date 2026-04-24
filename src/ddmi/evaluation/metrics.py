"""Evaluation metrics for cross-ablation experiments."""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CrossAblationMetrics:
    """Metrics for a single cross-ablation condition.

    A cross-ablation condition is: ablate direction X, measure behavior Y.
    """

    condition_name: str
    ablated_direction: str           # 'safety' | 'epistemic' | 'none'
    evaluation_domain: str           # 'safety' | 'epistemic'
    refusal_rate: float              # safety refusal rate on eval prompts
    abstention_rate: float           # epistemic abstention rate on eval prompts
    n_prompts: int
    responses_classified: dict[str, int] = None  # type: ignore

    def to_dict(self) -> dict[str, Any]:
        d = {
            "condition_name": self.condition_name,
            "ablated_direction": self.ablated_direction,
            "evaluation_domain": self.evaluation_domain,
            "refusal_rate": self.refusal_rate,
            "abstention_rate": self.abstention_rate,
            "n_prompts": self.n_prompts,
        }
        if self.responses_classified is not None:
            d["responses_classified"] = self.responses_classified
        return d


@dataclass(frozen=True)
class DirectionComparisonMetrics:
    """Metrics comparing safety and epistemic directions at a given layer."""

    layer_name: str
    layer_index: int
    cosine_similarity: float
    angular_distance_deg: float
    safety_separability: float
    epistemic_separability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_name": self.layer_name,
            "layer_index": self.layer_index,
            "cosine_similarity": self.cosine_similarity,
            "angular_distance_deg": self.angular_distance_deg,
            "safety_separability": self.safety_separability,
            "epistemic_separability": self.epistemic_separability,
        }


@dataclass(frozen=True)
class QuantizationDriftMetrics:
    """Metrics measuring direction drift under quantization."""

    precision_label: str              # 'fp16', 'bnb_nf4', 'bnb_int8', 'gptq_4bit', 'awq_4bit'
    direction_type: str               # 'safety' | 'epistemic'
    cosine_vs_fp16: float             # cosine similarity with FP16-extracted direction
    norm_ratio_vs_fp16: float         # ||d_quant|| / ||d_fp16||
    separability: float               # separability score in this precision
    cross_ablation_asr: float         # ASR when using FP16 direction on quantized model

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision_label": self.precision_label,
            "direction_type": self.direction_type,
            "cosine_vs_fp16": self.cosine_vs_fp16,
            "norm_ratio_vs_fp16": self.norm_ratio_vs_fp16,
            "separability": self.separability,
            "cross_ablation_asr": self.cross_ablation_asr,
        }


def compute_cross_ablation_table(
    results: Sequence[CrossAblationMetrics],
) -> List[Dict[str, Any]]:
    """Format results as a flat table for easy display/export."""
    return [r.to_dict() for r in results]


def compute_direction_comparison_table(
    results: Sequence[DirectionComparisonMetrics],
) -> List[Dict[str, Any]]:
    """Format direction comparison results as a flat table."""
    return [r.to_dict() for r in results]


def bootstrap_rate_ci(
    labels: Sequence[bool],
    *,
    num_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate a bootstrap confidence interval for a Bernoulli rate."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if num_bootstrap <= 0:
        raise ValueError(f"num_bootstrap must be positive, got {num_bootstrap}")

    sample = [1.0 if label else 0.0 for label in labels]
    n = len(sample)
    point_estimate = sum(sample) / n if n else 0.0
    if n == 0:
        return {
            "point_estimate": 0.0,
            "lower": 0.0,
            "upper": 0.0,
            "confidence": confidence,
            "num_bootstrap": num_bootstrap,
            "num_samples": 0,
            "method": "bootstrap_percentile",
        }

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(num_bootstrap):
        draws = [sample[rng.randrange(n)] for _ in range(n)]
        estimates.append(sum(draws) / n)

    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_idx = min(max(int(alpha * num_bootstrap), 0), num_bootstrap - 1)
    upper_idx = min(max(int((1.0 - alpha) * num_bootstrap) - 1, 0), num_bootstrap - 1)

    return {
        "point_estimate": point_estimate,
        "lower": estimates[lower_idx],
        "upper": estimates[upper_idx],
        "confidence": confidence,
        "num_bootstrap": num_bootstrap,
        "num_samples": n,
        "method": "bootstrap_percentile",
    }
