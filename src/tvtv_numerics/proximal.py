"""Pointwise proximal mappings and dual projections."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


Array = NDArray[np.floating]


def _positive_radius(radius: ArrayLike) -> Array:
    radius_array = np.asarray(radius)

    if np.any(radius_array <= 0.0):
        raise ValueError("radius must be strictly positive")

    return radius_array


def project_scalar_tv_ball(
    p: Array,
    radius: ArrayLike,
) -> Array:
    """Project a two-component field onto pointwise Euclidean balls."""
    p = np.asarray(p)

    if p.shape[-1] != 2:
        raise ValueError("the last axis of p must have length 2")

    radius_array = _positive_radius(radius)
    norm = np.sqrt(p[..., 0] * p[..., 0] + p[..., 1] * p[..., 1])
    scale = np.maximum(1.0, norm / radius_array)
    return p / scale[..., None]


def project_frobenius_tv_ball(
    p: Array,
    radius: ArrayLike,
) -> Array:
    """Project a 2x2 matrix field onto pointwise Frobenius balls."""
    p = np.asarray(p)

    if p.shape[-2:] != (2, 2):
        raise ValueError(
            "the final two axes of p must have shape (2, 2)"
        )

    radius_array = _positive_radius(radius)
    norm = np.sqrt(
        p[..., 0, 0] * p[..., 0, 0]
        + p[..., 0, 1] * p[..., 0, 1]
        + p[..., 1, 0] * p[..., 1, 0]
        + p[..., 1, 1] * p[..., 1, 1]
    )
    scale = np.maximum(1.0, norm / radius_array)
    return p / scale[..., None, None]


def prox_l32_fidelity(
    z: Array,
    f: Array,
    tau: float,
) -> Array:
    r"""Proximal map of tau * (2/3) * |u-f|^(3/2)."""
    tau = float(tau)

    if tau <= 0.0:
        raise ValueError("tau must be positive")

    z = np.asarray(z)
    f = np.asarray(f)

    if z.shape != f.shape:
        raise ValueError("z and f must have the same shape")

    difference = z - f
    root = 0.5 * (
        np.sqrt(tau * tau + 4.0 * np.abs(difference))
        - tau
    )

    return f + np.sign(difference) * root * root


def prox_optical_flow_affine(
    v_tilde: Array,
    u_t: Array,
    grad_u: Array,
    tau: float,
    *,
    gradient_norm_squared: Array | None = None,
) -> Array:
    r"""Proximal map of tau * |u_t + grad_u dot v|.

    A motion block may supply the cached squared norm of its fixed image
    gradient. The clipped scalar formula includes both saturated branches
    and the zero-gradient case without boolean indexing into vector arrays.
    """
    tau = float(tau)

    if tau <= 0.0:
        raise ValueError("tau must be positive")

    v_tilde = np.asarray(v_tilde)
    u_t = np.asarray(u_t)
    grad_u = np.asarray(grad_u)

    if (
        v_tilde.shape != grad_u.shape
        or v_tilde.shape[-1] != 2
    ):
        raise ValueError(
            "v_tilde and grad_u must have the same shape ending in 2"
        )

    if u_t.shape != v_tilde.shape[:-1]:
        raise ValueError(
            "u_t must have shape v_tilde.shape[:-1]"
        )

    gx, gy = grad_u[..., 0], grad_u[..., 1]
    rho = gx * v_tilde[..., 0] + gy * v_tilde[..., 1]
    rho += u_t
    if gradient_norm_squared is None:
        grad_norm_sq = gx * gx + gy * gy
    else:
        grad_norm_sq = np.asarray(gradient_norm_squared)
        if grad_norm_sq.shape != u_t.shape:
            raise ValueError("gradient_norm_squared must have the shape of u_t")

    correction = np.divide(
        rho, grad_norm_sq, out=np.zeros_like(rho), where=grad_norm_sq > 0.0,
    )
    np.clip(correction, -tau, tau, out=correction)
    result = np.empty_like(v_tilde)
    result[..., 0] = v_tilde[..., 0] - correction * gx
    result[..., 1] = v_tilde[..., 1] - correction * gy

    return result

def project_componentwise_tv_ball(
    p: Array, radius: ArrayLike
) -> Array:
    """Project each velocity component onto its own spatial TV ball."""
    p = np.asarray(p)

    if p.shape[-2:] != (2, 2):
        raise ValueError(
            "the final axes of p must be (component, derivative)"
        )

    radius_array = _positive_radius(radius)
    norm = np.linalg.norm(p, axis=-1)

    denominator = (
        radius_array
        if radius_array.ndim == 0
        else radius_array[..., None]
    )

    scale = np.maximum(1.0, norm / denominator)
    return p / scale[..., None]
