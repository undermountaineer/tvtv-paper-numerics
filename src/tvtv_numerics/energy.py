"""Discrete energies and diagnostics for the BDS reference model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .operators import (
    spatial_gradient,
    transport_operator,
    vector_field_gradient,
)


Array = NDArray[np.floating]


def scalar_spatial_tv(
    u: Array,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Unweighted isotropic spatial TV summed over all frames."""
    gradient = spatial_gradient(u, hx=hx, hy=hy)
    return float(np.sum(np.linalg.norm(gradient, axis=-1)))


def componentwise_velocity_tv(
    v: Array,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Sum of the isotropic spatial TVs of v_x and v_y."""
    gradient = vector_field_gradient(v, hx=hx, hy=hy)
    return float(np.sum(np.linalg.norm(gradient, axis=-1)))


def transport_residual(
    u: Array,
    v: Array,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> float:
    """Mean absolute residual of the optical-flow equation."""
    residual = transport_operator(
        u,
        v,
        ht=ht,
        hx=hx,
        hy=hy,
    )
    return float(np.mean(np.abs(residual)))


def relative_change(new: Array, old: Array) -> float:
    """Relative Euclidean change with denominator bounded below by one."""
    new = np.asarray(new)
    old = np.asarray(old)

    if new.shape != old.shape:
        raise ValueError("new and old must have the same shape")

    return float(
        np.linalg.norm(new - old)
        / max(1.0, np.linalg.norm(old))
    )


def bds_energy(
    u: Array,
    v: Array,
    f: Array,
    alpha: float,
    beta: float,
    kappa: float,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> dict[str, float]:
    """Return all components of the constant-parameter BDS energy."""
    u = np.asarray(u)
    v = np.asarray(v)
    f = np.asarray(f)

    if u.shape != f.shape:
        raise ValueError("u and f must have the same shape")

    expected_v_shape = (u.shape[0] - 1, *u.shape[1:], 2)
    if v.shape != expected_v_shape:
        raise ValueError(
            "v must have shape (nt-1, ny, nx, 2)"
        )

    if min(alpha, beta, kappa, ht, hx, hy) <= 0.0:
        raise ValueError(
            "parameters and grid spacings must be positive"
        )

    cell_volume = float(hx * hy * ht)

    fidelity = (
        0.5
        * float(np.sum((u - f) ** 2))
        * cell_volume
    )

    tv_u = (
        float(alpha)
        * scalar_spatial_tv(u, hx, hy)
        * cell_volume
    )

    tv_v = (
        float(beta)
        * componentwise_velocity_tv(v, hx, hy)
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