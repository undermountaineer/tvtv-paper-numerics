"""Weighted discrete energy for the manuscript-aligned TV-TV model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .operators import (
    spatial_gradient,
    transport_operator,
    vector_field_gradient,
)
from .weights import (
    forward_stencil_lower_envelope,
    validate_weight_map,
)


Array = NDArray[np.floating]


def l32_fidelity(
    u: Array,
    f: Array,
) -> float:
    """Return (2/3) * sum |u-f|^(3/2)."""
    u = np.asarray(u, dtype=float)
    f = np.asarray(f, dtype=float)

    if u.shape != f.shape:
        raise ValueError(
            "u and f must have the same shape"
        )

    return float(
        (2.0 / 3.0)
        * np.sum(np.abs(u - f) ** 1.5)
    )


def weighted_image_tv(
    u: Array,
    alpha_weight: Array,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Weighted isotropic spatial TV of the image sequence."""
    u = np.asarray(u, dtype=float)

    alpha = validate_weight_map(
        alpha_weight,
        u.shape,
        "alpha_weight",
    )

    alpha = forward_stencil_lower_envelope(alpha)

    gradient_norm = np.linalg.norm(
        spatial_gradient(u, hx, hy),
        axis=-1,
    )

    return float(
        np.sum(alpha * gradient_norm)
    )


def weighted_velocity_tv(
    v: Array,
    beta_weight: Array,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Weighted vectorial TV using the Frobenius norm."""
    v = np.asarray(v, dtype=float)

    if v.ndim != 4 or v.shape[-1] != 2:
        raise ValueError(
            "v must have shape (nt-1, ny, nx, 2)"
        )

    beta = validate_weight_map(
        beta_weight,
        v.shape[:-1],
        "beta_weight",
    )

    beta = forward_stencil_lower_envelope(beta)

    gradient_norm = np.linalg.norm(
        vector_field_gradient(v, hx, hy),
        axis=(-2, -1),
    )

    return float(
        np.sum(beta * gradient_norm)
    )


def weighted_u_block_objective(
    u: Array,
    f: Array,
    v: Array,
    alpha_weight: Array,
    kappa: float,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Objective of the convex image block for fixed velocity.

    The common cell-volume factor is retained so that this value is directly
    comparable with :func:`manuscript_energy`.
    """
    if min(kappa, ht, hx, hy) <= 0.0:
        raise ValueError("kappa and grid spacings must be positive")

    cell_volume = float(ht * hx * hy)
    transport = float(kappa) * float(
        np.sum(np.abs(transport_operator(u, v, ht, hx, hy)))
    )
    return float(
        cell_volume
        * (
            l32_fidelity(u, f)
            + weighted_image_tv(u, alpha_weight, hx, hy)
            + transport
        )
    )


def weighted_v_block_objective(
    v: Array,
    u: Array,
    beta_weight: Array,
    kappa: float,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Objective of the convex velocity block for fixed image."""
    if min(kappa, ht, hx, hy) <= 0.0:
        raise ValueError("kappa and grid spacings must be positive")

    cell_volume = float(ht * hx * hy)
    transport = float(kappa) * float(
        np.sum(np.abs(transport_operator(u, v, ht, hx, hy)))
    )
    return float(
        cell_volume
        * (
            weighted_velocity_tv(v, beta_weight, hx, hy)
            + transport
        )
    )


def manuscript_energy(
    u: Array,
    v: Array,
    f: Array,
    alpha_weight: Array,
    beta_weight: Array,
    kappa: float,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> dict[str, float]:
    """Return all terms of the penalized manuscript energy."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    f = np.asarray(f, dtype=float)

    if u.ndim != 3 or u.shape != f.shape:
        raise ValueError(
            "u and f must have the same shape (nt, ny, nx)"
        )

    expected_v_shape = (
        u.shape[0] - 1,
        *u.shape[1:],
        2,
    )

    if v.shape != expected_v_shape:
        raise ValueError(
            "v must have shape (nt-1, ny, nx, 2)"
        )

    if min(kappa, ht, hx, hy) <= 0.0:
        raise ValueError(
            "kappa and grid spacings must be positive"
        )

    cell_volume = float(ht * hx * hy)

    fidelity = (
        l32_fidelity(u, f)
        * cell_volume
    )

    tv_u = (
        weighted_image_tv(
            u,
            alpha_weight,
            hx,
            hy,
        )
        * cell_volume
    )

    tv_v = (
        weighted_velocity_tv(
            v,
            beta_weight,
            hx,
            hy,
        )
        * cell_volume
    )

    transport = (
        float(kappa)
        * float(
            np.sum(
                np.abs(
                    transport_operator(
                        u,
                        v,
                        ht,
                        hx,
                        hy,
                    )
                )
            )
        )
        * cell_volume
    )

    return {
        "fidelity": fidelity,
        "tv_u": tv_u,
        "tv_v": tv_v,
        "transport": transport,
        "total": fidelity + tv_u + tv_v + transport,
    }
