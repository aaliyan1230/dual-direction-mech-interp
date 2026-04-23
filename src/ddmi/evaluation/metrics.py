"""Evaluation metrics for cross-ablation experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence


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
    responses_classified: Dict[str, int] = None  # type: ignore

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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

    def to_dict(self) -> Dict[str, Any]:
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
