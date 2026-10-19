"""Finite-difference operators used by the TV-TV model."""

from __future__ import annotations
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.floating]


def _positive_spacing(value: float, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def dx_forward(u: Array, hx: float = 1.0) -> Array:
    """Forward x-difference with zero value at the right boundary."""
    hx = _positive_spacing(hx, "hx")
    u = np.asarray(u)
    result = np.zeros_like(u)
    result[..., :-1] = (u[..., 1:] - u[..., :-1]) / hx
    return result


def dx_forward_adjoint(p: Array, hx: float = 1.0) -> Array:
    """Euclidean adjoint of dx_forward."""
    hx = _positive_spacing(hx, "hx")
    p = np.asarray(p)
    result = np.zeros_like(p)
    result[..., :-1] -= p[..., :-1] / hx
    result[..., 1:] += p[..., :-1] / hx
    return result


def dy_forward(u: Array, hy: float = 1.0) -> Array:
    """Forward y-difference with zero value at the lower boundary."""
    hy = _positive_spacing(hy, "hy")
    u = np.asarray(u)
    result = np.zeros_like(u)
    result[..., :-1, :] = (u[..., 1:, :] - u[..., :-1, :]) / hy
    return result


def dy_forward_adjoint(p: Array, hy: float = 1.0) -> Array:
    """Euclidean adjoint of dy_forward."""
    hy = _positive_spacing(hy, "hy")
    p = np.asarray(p)
    result = np.zeros_like(p)
    result[..., :-1, :] -= p[..., :-1, :] / hy
    result[..., 1:, :] += p[..., :-1, :] / hy
    return result


def dt_forward(u: Array, ht: float = 1.0) -> Array:
    """Forward time-difference on the nt-1 temporal intervals."""
    ht = _positive_spacing(ht, "ht")
    u = np.asarray(u)

    if u.ndim < 1 or u.shape[0] < 2:
        raise ValueError("u must contain at least two time frames")

    return (u[1:] - u[:-1]) / ht


def dt_forward_adjoint(p: Array, nt: int, ht: float = 1.0) -> Array:
    """Euclidean adjoint of dt_forward."""
    ht = _positive_spacing(ht, "ht")
    p = np.asarray(p)

    if nt < 2 or p.shape[0] != nt - 1:
        raise ValueError("p must have nt-1 temporal intervals")

    result = np.zeros((nt, *p.shape[1:]), dtype=p.dtype)
    result[:-1] -= p / ht
    result[1:] += p / ht
    return result


def dx_central(u: Array, hx: float = 1.0) -> Array:
    """Central x-difference with zero values at both x-boundaries."""
    hx = _positive_spacing(hx, "hx")
    u = np.asarray(u)
    result = np.zeros_like(u)

    if u.shape[-1] >= 3:
        result[..., 1:-1] = (
            u[..., 2:] - u[..., :-2]
        ) / (2.0 * hx)

    return result


def dx_central_adjoint(p: Array, hx: float = 1.0) -> Array:
    """Euclidean adjoint of dx_central."""
    hx = _positive_spacing(hx, "hx")
    p = np.asarray(p)
    result = np.zeros_like(p)

    if p.shape[-1] >= 3:
        result[..., 2:] += p[..., 1:-1] / (2.0 * hx)
        result[..., :-2] -= p[..., 1:-1] / (2.0 * hx)

    return result


def dy_central(u: Array, hy: float = 1.0) -> Array:
    """Central y-difference with zero values at both y-boundaries."""
    hy = _positive_spacing(hy, "hy")
    u = np.asarray(u)
    result = np.zeros_like(u)

    if u.shape[-2] >= 3:
        result[..., 1:-1, :] = (
            u[..., 2:, :] - u[..., :-2, :]
        ) / (2.0 * hy)

    return result


def dy_central_adjoint(p: Array, hy: float = 1.0) -> Array:
    """Euclidean adjoint of dy_central."""
    hy = _positive_spacing(hy, "hy")
    p = np.asarray(p)
    result = np.zeros_like(p)

    if p.shape[-2] >= 3:
        result[..., 2:, :] += p[..., 1:-1, :] / (2.0 * hy)
        result[..., :-2, :] -= p[..., 1:-1, :] / (2.0 * hy)

    return result


def spatial_gradient(
    u: Array,
    hx: float = 1.0,
    hy: float = 1.0,
) -> Array:
    """Return the forward spatial gradient in component order (x, y)."""
    return np.stack(
        (
            dx_forward(u, hx),
            dy_forward(u, hy),
        ),
        axis=-1,
    )


def spatial_gradient_adjoint(
    p: Array,
    hx: float = 1.0,
    hy: float = 1.0,
) -> Array:
    """Euclidean adjoint of spatial_gradient."""
    p = np.asarray(p)

    if p.shape[-1] != 2:
        raise ValueError(
            "the last axis of p must contain the x and y components"
        )

    return (
        dx_forward_adjoint(p[..., 0], hx)
        + dy_forward_adjoint(p[..., 1], hy)
    )


def transport_operator(
    u: Array,
    v: Array,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> Array:
    """Apply D_t u + grad_c(u) dot v on the temporal intervals."""
    u = np.asarray(u)
    v = np.asarray(v)

    if u.ndim != 3:
        raise ValueError("u must have shape (nt, ny, nx)")

    expected_shape = (u.shape[0] - 1, *u.shape[1:], 2)

    if v.shape != expected_shape:
        raise ValueError(f"v must have shape {expected_shape}")

    ux = dx_central(u[:-1], hx)
    uy = dy_central(u[:-1], hy)

    return (
        dt_forward(u, ht)
        + v[..., 0] * ux
        + v[..., 1] * uy
    )


def transport_adjoint(
    p: Array,
    v: Array,
    nt: int,
    ht: float = 1.0,
    hx: float = 1.0,
    hy: float = 1.0,
) -> Array:
    """Adjoint in u of transport_operator for fixed v."""
    p = np.asarray(p)
    v = np.asarray(v)

    if v.shape != (*p.shape, 2):
        raise ValueError("v must have shape (*p.shape, 2)")

    result = dt_forward_adjoint(p, nt=nt, ht=ht)
    result[:-1] += dx_central_adjoint(v[..., 0] * p, hx)
    result[:-1] += dy_central_adjoint(v[..., 1] * p, hy)
    return result

def vector_field_gradient(
    v: Array, hx: float = 1.0, hy: float = 1.0
) -> Array:
    """Spatial gradient of v with axes (..., component, derivative)."""
    v = np.asarray(v)
    if v.ndim != 4 or v.shape[-1] != 2:
        raise ValueError("v must have shape (nt-1, ny, nx, 2)")

    return np.stack(
        (
            spatial_gradient(v[..., 0], hx, hy),
            spatial_gradient(v[..., 1], hx, hy),
        ),
        axis=-2,
    )


def vector_field_gradient_adjoint(
    p: Array, hx: float = 1.0, hy: float = 1.0
) -> Array:
    """Euclidean adjoint of :func:`vector_field_gradient`."""
    p = np.asarray(p)
    if p.ndim != 5 or p.shape[-2:] != (2, 2):
        raise ValueError("p must have shape (nt-1, ny, nx, 2, 2)")

    return np.stack(
        (
            spatial_gradient_adjoint(p[..., 0, :], hx, hy),
            spatial_gradient_adjoint(p[..., 1, :], hx, hy),
        ),
        axis=-1,
    )


def _output_norm_squared(value: Any) -> float:
    """Squared Euclidean norm for arrays or tuples of arrays."""
    if isinstance(value, (tuple, list)):
        return float(sum(_output_norm_squared(item) for item in value))

    array = np.asarray(value)
    return float(np.vdot(array, array).real)


def estimate_operator_norm_squared(
    forward: Callable[[Array], Any],
    adjoint: Callable[[Any], Array],
    domain_shape: tuple[int, ...],
    iterations: int = 60,
    seed: int = 20260907,
) -> float:
    """Estimate ||K||^2 by power iteration applied to K* K."""
    if iterations < 1:
        raise ValueError("iterations must be positive")

    rng = np.random.default_rng(seed)
    x = rng.normal(size=domain_shape)
    x /= np.linalg.norm(x)

    for _ in range(iterations):
        z = np.asarray(adjoint(forward(x)))
        z_norm = np.linalg.norm(z)

        if z_norm == 0.0:
            return 0.0

        x = z / z_norm

    return _output_norm_squared(forward(x)) / float(
        np.vdot(x, x).real
    )