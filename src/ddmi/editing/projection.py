"""Vector projection and orthogonalization utilities."""
from __future__ import annotations

from typing import Sequence

from ddmi.editing.directions import (
    Vector,
    dot_product,
    normalize_vector,
    scale_vector,
    subtract_vectors,
)


def project_vector(vector: Sequence[float], onto: Sequence[float]) -> Vector:
    """Project vector onto direction `onto` (assumes `onto` is unit-length)."""
    coeff = dot_product(vector, onto)
    return scale_vector(list(onto), coeff)


def orthogonalize(candidate: Sequence[float], reference: Sequence[float]) -> Vector:
    """Remove the component of candidate along reference (Gram-Schmidt step).

    Reference should be unit-length for correct projection.
    """
    ref_unit = normalize_vector(reference)
    projection = project_vector(candidate, ref_unit)
    return subtract_vectors(list(candidate), projection)


def remove_direction_component(
    vector: Sequence[float], direction: Sequence[float], strength: float = 1.0,
) -> Vector:
    """Remove `strength` fraction of the direction component from vector.

    strength=1.0 fully removes the component (orthogonal projection).
    strength=0.0 leaves the vector unchanged.
    """
    dir_unit = normalize_vector(direction)
    proj = project_vector(vector, dir_unit)
    scaled_proj = scale_vector(proj, strength)
    return subtract_vectors(list(vector), scaled_proj)
