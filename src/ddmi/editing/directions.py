"""Direction extraction via difference-in-means and geometric analysis.

Pure-Python vector operations for direction computation, matching the
refusal-suppression project's conventions (List[float] vectors, no numpy
in core library).
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Vector = List[float]


def mean_vector(vectors: Sequence[Sequence[float]]) -> Vector:
    """Element-wise mean of a collection of vectors."""
    if not vectors:
        raise ValueError("Cannot compute mean of empty collection")
    dim = len(vectors[0])
    n = len(vectors)
    result = [0.0] * dim
    for vec in vectors:
        if len(vec) != dim:
            raise ValueError(f"Dimension mismatch: expected {dim}, got {len(vec)}")
        for i in range(dim):
            result[i] += vec[i]
    return [v / n for v in result]


def subtract_vectors(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Element-wise subtraction: left - right."""
    if len(left) != len(right):
        raise ValueError(f"Dimension mismatch: {len(left)} vs {len(right)}")
    return [float(a) - float(b) for a, b in zip(left, right)]


def add_vectors(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Element-wise addition: left + right."""
    if len(left) != len(right):
        raise ValueError(f"Dimension mismatch: {len(left)} vs {len(right)}")
    return [float(a) + float(b) for a, b in zip(left, right)]


def scale_vector(vector: Sequence[float], scalar: float) -> Vector:
    """Scalar multiplication."""
    return [float(v) * scalar for v in vector]


def l2_norm(vector: Sequence[float]) -> float:
    """L2 norm of a vector."""
    return math.sqrt(sum(v * v for v in vector))


def normalize_vector(vector: Sequence[float], eps: float = 1e-12) -> Vector:
    """Normalize to unit length."""
    norm = l2_norm(vector)
    if norm < eps:
        return [0.0] * len(vector)
    return [v / norm for v in vector]


def dot_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product of two vectors."""
    if len(left) != len(right):
        raise ValueError(f"Dimension mismatch: {len(left)} vs {len(right)}")
    return sum(a * b for a, b in zip(left, right))


def cosine_similarity(left: Sequence[float], right: Sequence[float], eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""
    norm_l = l2_norm(left)
    norm_r = l2_norm(right)
    if norm_l < eps or norm_r < eps:
        return 0.0
    return dot_product(left, right) / (norm_l * norm_r)


def angular_distance_degrees(left: Sequence[float], right: Sequence[float]) -> float:
    """Angular distance in degrees between two vectors."""
    cos_sim = cosine_similarity(left, right)
    # Clamp for numerical stability
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return math.degrees(math.acos(cos_sim))


def difference_of_means(group_a: Sequence[Sequence[float]],
                        group_b: Sequence[Sequence[float]]) -> Vector:
    """Raw difference-in-means direction (unnormalized): mean(A) - mean(B)."""
    mean_a = mean_vector(group_a)
    mean_b = mean_vector(group_b)
    return subtract_vectors(mean_a, mean_b)


def direction_from_contrast(group_a: Sequence[Sequence[float]],
                            group_b: Sequence[Sequence[float]]) -> Vector:
    """Normalized difference-in-means direction (unit vector)."""
    raw = difference_of_means(group_a, group_b)
    return normalize_vector(raw)


def separability_score(group_a: Sequence[Sequence[float]],
                       group_b: Sequence[Sequence[float]]) -> float:
    """Separability: distance between centroids / (1 + avg_radius_a + avg_radius_b).

    Higher values indicate the two groups are more separable along the
    difference-in-means direction.
    """
    mean_a = mean_vector(group_a)
    mean_b = mean_vector(group_b)
    centroid_distance = l2_norm(subtract_vectors(mean_a, mean_b))

    def avg_radius(group: Sequence[Sequence[float]], centroid: Vector) -> float:
        if not group:
            return 0.0
        total = sum(l2_norm(subtract_vectors(vec, centroid)) for vec in group)
        return total / len(group)

    radius_a = avg_radius(group_a, mean_a)
    radius_b = avg_radius(group_b, mean_b)
    return centroid_distance / (1.0 + radius_a + radius_b)


def rank_layers_by_separability(
    layer_scores: Mapping[str, float],
) -> List[Tuple[str, float]]:
    """Rank layers by separability score (descending)."""
    return sorted(layer_scores.items(), key=lambda x: x[1], reverse=True)


def pairwise_cosine_matrix(
    directions: Dict[str, Vector],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise cosine similarity matrix for named directions.

    Returns nested dict: result[name_a][name_b] = cosine(a, b).
    """
    keys = sorted(directions.keys())
    result: Dict[str, Dict[str, float]] = {}
    for key_a in keys:
        result[key_a] = {}
        for key_b in keys:
            result[key_a][key_b] = cosine_similarity(
                directions[key_a], directions[key_b]
            )
    return result


def project_onto_direction(vector: Sequence[float], direction: Sequence[float]) -> float:
    """Scalar projection of vector onto direction (direction should be unit-length)."""
    return dot_product(vector, direction)


def batch_project_onto_direction(
    vectors: Sequence[Sequence[float]], direction: Sequence[float],
) -> List[float]:
    """Scalar projection of each vector onto direction."""
    return [project_onto_direction(v, direction) for v in vectors]
