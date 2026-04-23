"""Unit tests for direction extraction and geometry."""
from __future__ import annotations

import math

import pytest

from ddmi.editing.directions import (
    angular_distance_degrees,
    batch_project_onto_direction,
    cosine_similarity,
    difference_of_means,
    direction_from_contrast,
    dot_product,
    l2_norm,
    mean_vector,
    normalize_vector,
    pairwise_cosine_matrix,
    rank_layers_by_separability,
    separability_score,
    subtract_vectors,
)
from ddmi.editing.projection import orthogonalize, remove_direction_component


class TestVectorOps:
    def test_mean_vector(self) -> None:
        vecs = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]
        result = mean_vector(vecs)
        assert result == [2.0, 3.0, 4.0]

    def test_subtract_vectors(self) -> None:
        assert subtract_vectors([3.0, 2.0], [1.0, 1.0]) == [2.0, 1.0]

    def test_l2_norm(self) -> None:
        assert abs(l2_norm([3.0, 4.0]) - 5.0) < 1e-10

    def test_normalize_unit(self) -> None:
        vec = normalize_vector([3.0, 4.0])
        assert abs(l2_norm(vec) - 1.0) < 1e-10

    def test_normalize_zero(self) -> None:
        vec = normalize_vector([0.0, 0.0, 0.0])
        assert vec == [0.0, 0.0, 0.0]

    def test_cosine_parallel(self) -> None:
        assert abs(cosine_similarity([1.0, 0.0], [2.0, 0.0]) - 1.0) < 1e-10

    def test_cosine_orthogonal(self) -> None:
        assert abs(cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-10

    def test_cosine_antiparallel(self) -> None:
        assert abs(cosine_similarity([1.0, 0.0], [-1.0, 0.0]) + 1.0) < 1e-10

    def test_angular_distance_orthogonal(self) -> None:
        angle = angular_distance_degrees([1.0, 0.0], [0.0, 1.0])
        assert abs(angle - 90.0) < 1e-6

    def test_angular_distance_parallel(self) -> None:
        angle = angular_distance_degrees([1.0, 0.0], [1.0, 0.0])
        assert abs(angle) < 1e-6


class TestDirectionExtraction:
    def test_difference_of_means(self) -> None:
        group_a = [[2.0, 4.0], [4.0, 6.0]]  # mean = [3, 5]
        group_b = [[0.0, 0.0], [2.0, 2.0]]  # mean = [1, 1]
        result = difference_of_means(group_a, group_b)
        assert result == [2.0, 4.0]

    def test_direction_from_contrast_normalized(self) -> None:
        group_a = [[10.0, 0.0]]
        group_b = [[0.0, 0.0]]
        direction = direction_from_contrast(group_a, group_b)
        assert abs(l2_norm(direction) - 1.0) < 1e-10

    def test_separability_separated(self) -> None:
        group_a = [[10.0, 0.0], [11.0, 0.0]]
        group_b = [[-10.0, 0.0], [-11.0, 0.0]]
        sep = separability_score(group_a, group_b)
        assert sep > 1.0  # well separated

    def test_separability_overlapping(self) -> None:
        group_a = [[0.0, 0.0], [1.0, 0.0]]
        group_b = [[0.5, 0.0], [1.5, 0.0]]
        sep = separability_score(group_a, group_b)
        assert sep < 1.0  # overlapping

    def test_rank_layers(self) -> None:
        scores = {"layer_0": 0.1, "layer_1": 0.5, "layer_2": 0.3}
        ranked = rank_layers_by_separability(scores)
        assert ranked[0] == ("layer_1", 0.5)

    def test_pairwise_matrix(self) -> None:
        dirs = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
        matrix = pairwise_cosine_matrix(dirs)
        assert abs(matrix["a"]["a"] - 1.0) < 1e-10
        assert abs(matrix["a"]["b"]) < 1e-10


class TestProjection:
    def test_orthogonalize(self) -> None:
        candidate = [1.0, 1.0]
        reference = [1.0, 0.0]
        result = orthogonalize(candidate, reference)
        # Should remove x-component, leaving [0, 1]
        assert abs(result[0]) < 1e-10
        assert abs(result[1] - 1.0) < 1e-10

    def test_remove_direction_full(self) -> None:
        vec = [3.0, 4.0]
        direction = [1.0, 0.0]
        result = remove_direction_component(vec, direction, strength=1.0)
        assert abs(result[0]) < 1e-10
        assert abs(result[1] - 4.0) < 1e-10

    def test_remove_direction_half(self) -> None:
        vec = [4.0, 0.0]
        direction = [1.0, 0.0]
        result = remove_direction_component(vec, direction, strength=0.5)
        assert abs(result[0] - 2.0) < 1e-10

    def test_batch_project(self) -> None:
        vecs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
        direction = normalize_vector([1.0, 0.0])
        projections = batch_project_onto_direction(vecs, direction)
        assert abs(projections[0] - 1.0) < 1e-10
        assert abs(projections[1]) < 1e-10
        assert abs(projections[2] - 1.0) < 1e-10
