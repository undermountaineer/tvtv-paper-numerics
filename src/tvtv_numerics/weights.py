"""Utilities for positive, discontinuous space-time weight maps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.floating]


def validate_weight_map(
    weight: Array,
    expected_shape: tuple[int, ...],
    name: str,
) -> Array:
    """Return a finite positive weight map of the requested shape."""
    result = np.asarray(weight, dtype=float)

    if result.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}"
        )

    if (
        not np.all(np.isfinite(result))
        or np.any(result <= 0.0)
    ):
        raise ValueError(
            f"{name} must contain finite positive values"
        )

    return result


def forward_stencil_lower_envelope(
    weight: Array,
) -> Array:
    """Lower envelope associated with the forward gradient stencil.

    At a spatial interface, a forward difference receives the
    minimum of the weights in the cells touched by the outgoing
    x- and y-differences. Constant weights remain unchanged.
    """
    weight = np.asarray(weight, dtype=float)

    if weight.ndim != 3:
        raise ValueError(
            "weight must have shape (nt, ny, nx)"
        )

    if (
        not np.all(np.isfinite(weight))
        or np.any(weight <= 0.0)
    ):
        raise ValueError(
            "weight must contain finite positive values"
        )

    result = weight.copy()

    if weight.shape[-1] > 1:
        result[..., :-1] = np.minimum(
            result[..., :-1],
            weight[..., 1:],
        )

    if weight.shape[-2] > 1:
        result[..., :-1, :] = np.minimum(
            result[..., :-1, :],
            weight[..., 1:, :],
        )

    return result