"""Metrics used by the supervised TV-TV experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.floating]


def rmse(
    value: Array,
    reference: Array,
) -> float:
    value = np.asarray(
        value,
        dtype=float,
    )

    reference = np.asarray(
        reference,
        dtype=float,
    )

    if value.shape != reference.shape:
        raise ValueError(
            "value and reference must have the same shape"
        )

    return float(
        np.sqrt(
            np.mean(
                (value - reference) ** 2
            )
        )
    )


def psnr(
    value: Array,
    reference: Array,
    data_range: float = 1.0,
) -> float:
    if data_range <= 0.0:
        raise ValueError(
            "data_range must be positive"
        )

    error = rmse(
        value,
        reference,
    )

    if error == 0.0:
        return float("inf")

    return float(
        20.0
        * np.log10(data_range / error)
    )


def endpoint_error(
    value: Array,
    reference: Array,
    weights: Array | None = None,
) -> float:
    """Mean or weighted mean Euclidean flow error."""
    value = np.asarray(
        value,
        dtype=float,
    )

    reference = np.asarray(
        reference,
        dtype=float,
    )

    if (
        value.shape != reference.shape
        or value.shape[-1] != 2
    ):
        raise ValueError(
            "flow arrays must have the same shape ending in 2"
        )

    error = np.linalg.norm(
        value - reference,
        axis=-1,
    )

    if weights is None:
        return float(
            np.mean(error)
        )

    weights = np.asarray(
        weights,
        dtype=float,
    )

    if (
        weights.shape != error.shape
        or np.any(weights < 0.0)
    ):
        raise ValueError(
            "weights must be nonnegative and "
            "match the flow grid"
        )

    total_weight = float(
        np.sum(weights)
    )

    if total_weight <= 0.0:
        raise ValueError(
            "weights must have positive sum"
        )

    return float(
        np.sum(weights * error)
        / total_weight
    )


def upper_level_loss(
    u: Array,
    v: Array,
    clean_u: Array,
    clean_v: Array | None,
    gamma_u: float = 1.0,
    gamma_v: float = 1.0,
    exponent: float = 1.5,
    image_time_weights: Array | None = None,
    cell_volume: float = 1.0,
) -> dict[str, float]:
    """Return the discrete summed upper-level loss."""
    u = np.asarray(
        u,
        dtype=float,
    )

    v = np.asarray(
        v,
        dtype=float,
    )

    clean_u = np.asarray(
        clean_u,
        dtype=float,
    )

    clean_v = None if clean_v is None else np.asarray(clean_v, dtype=float)

    if u.shape != clean_u.shape:
        raise ValueError(
            "u and clean_u must have the same shape"
        )

    expected_v_shape = (
        u.shape[0] - 1,
        *u.shape[1:],
        2,
    )

    if (
        v.shape != expected_v_shape
        or (clean_v is not None and v.shape != clean_v.shape)
    ):
        raise ValueError(
            "the velocity arrays have incompatible shapes"
        )

    if clean_v is None and gamma_v != 0.0:
        raise ValueError("reference motion is required when gamma_v is positive")

    if (
        min(gamma_u, gamma_v) < 0.0
        or exponent <= 1.0
        or cell_volume <= 0.0
    ):
        raise ValueError(
            "loss weights must be nonnegative "
            "and scales must be positive"
        )

    image_error = (
        np.abs(u - clean_u) ** exponent
    )

    if image_time_weights is not None:
        time_weights = np.asarray(
            image_time_weights,
            dtype=float,
        )

        if (
            time_weights.shape != (u.shape[0],)
            or np.any(time_weights < 0.0)
        ):
            raise ValueError(
                "image_time_weights must be nonnegative "
                "with shape (nt,)"
            )

        image_error = (
            image_error
            * time_weights[:, None, None]
        )

    velocity_error = 0.0 if clean_v is None or gamma_v == 0.0 else (
        np.linalg.norm(
            v - clean_v,
            axis=-1,
        )
        ** exponent
    )

    image = float(
        gamma_u
        * np.sum(image_error)
        * cell_volume
    )

    velocity = float(
        gamma_v
        * np.sum(velocity_error)
        * cell_volume
    )

    return {
        "image": image,
        "velocity": velocity,
        "total": image + velocity,
    }

def relative_l2_error(value: Array, reference: Array) -> float:
    """Relative Euclidean error with explicit zero-reference handling."""
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if value.shape != reference.shape:
        raise ValueError("value and reference must have the same shape")
    numerator = float(np.linalg.norm(value - reference))
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def evaluation_metrics(
    u: Array,
    v: Array,
    clean_u: Array,
    clean_v: Array | None,
    gamma_u: float = 1.0,
    gamma_v: float = 1.0,
    exponent: float = 1.5,
    image_time_weights: Array | None = None,
    cell_volume: float = 1.0,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Collect the manuscript loss and familiar image/flow metrics."""
    loss = upper_level_loss(
        u,
        v,
        clean_u,
        clean_v,
        gamma_u=gamma_u,
        gamma_v=gamma_v,
        exponent=exponent,
        image_time_weights=image_time_weights,
        cell_volume=cell_volume,
    )
    metrics = {
        "upper_loss": loss["total"],
        "image_loss": loss["image"],
        "velocity_loss": loss["velocity"],
        "image_rmse": rmse(u, clean_u),
        "relative_image_l2": relative_l2_error(u, clean_u),
        "psnr": psnr(u, clean_u, data_range=data_range),
    }
    if clean_v is not None:
        metrics["endpoint_error"] = endpoint_error(v, clean_v)
    return metrics
