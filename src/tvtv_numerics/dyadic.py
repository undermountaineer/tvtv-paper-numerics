"""Index-based dyadic space-time cells and weight-map assembly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.floating]


@dataclass(frozen=True, order=True)
class DyadicCell:
    """An index-based dyadic space-time cell.

    Temporal bounds index intervals. The image slice contains
    frames t0 through t1, while the velocity slice contains
    intervals t0 through t1-1.

    Spatial bounds are half-open pixel-index ranges.
    """

    level: int

    t0: int
    t1: int

    y0: int
    y1: int

    x0: int
    x1: int

    @classmethod
    def root(
        cls,
        data_shape: tuple[int, int, int],
    ) -> "DyadicCell":
        nt, ny, nx = data_shape

        if nt < 2 or ny < 1 or nx < 1:
            raise ValueError(
                "data_shape must be "
                "(nt>=2, ny>=1, nx>=1)"
            )

        return cls(
            level=0,
            t0=0,
            t1=nt - 1,
            y0=0,
            y1=ny,
            x0=0,
            x1=nx,
        )

    @property
    def temporal_intervals(
        self,
    ) -> int:
        return self.t1 - self.t0

    @property
    def u_shape(
        self,
    ) -> tuple[int, int, int]:
        return (
            self.temporal_intervals + 1,
            self.y1 - self.y0,
            self.x1 - self.x0,
        )

    @property
    def v_shape(
        self,
    ) -> tuple[int, int, int, int]:
        return (
            self.temporal_intervals,
            self.y1 - self.y0,
            self.x1 - self.x0,
            2,
        )

    @property
    def u_slices(
        self,
    ) -> tuple[slice, slice, slice]:
        return (
            slice(self.t0, self.t1 + 1),
            slice(self.y0, self.y1),
            slice(self.x0, self.x1),
        )

    @property
    def v_slices(
        self,
    ) -> tuple[
        slice,
        slice,
        slice,
        slice,
    ]:
        return (
            slice(self.t0, self.t1),
            slice(self.y0, self.y1),
            slice(self.x0, self.x1),
            slice(None),
        )

    def image_time_weights(
        self,
    ) -> Array:
        """Trapezoidal weights additive under temporal splitting."""
        weights = np.ones(
            self.temporal_intervals + 1,
            dtype=float,
        )

        weights[0] = 0.5
        weights[-1] = 0.5

        return weights

    def children(
        self,
    ) -> tuple["DyadicCell", ...]:
        """Return the eight children in two space dimensions."""
        nt_intervals = self.temporal_intervals
        ny = self.y1 - self.y0
        nx = self.x1 - self.x0

        if (
            nt_intervals < 2
            or nt_intervals % 2 != 0
        ):
            raise ValueError(
                "the cell needs an even number of "
                "at least two time intervals"
            )

        if (
            ny < 2
            or ny % 2 != 0
            or nx < 2
            or nx % 2 != 0
        ):
            raise ValueError(
                "both spatial side lengths must be "
                "even and at least two"
            )

        tm = (self.t0 + self.t1) // 2
        ym = (self.y0 + self.y1) // 2
        xm = (self.x0 + self.x1) // 2

        time_bounds = (
            (self.t0, tm),
            (tm, self.t1),
        )

        y_bounds = (
            (self.y0, ym),
            (ym, self.y1),
        )

        x_bounds = (
            (self.x0, xm),
            (xm, self.x1),
        )

        return tuple(
            DyadicCell(
                level=self.level + 1,
                t0=ta,
                t1=tb,
                y0=ya,
                y1=yb,
                x0=xa,
                x1=xb,
            )
            for ta, tb in time_bounds
            for ya, yb in y_bounds
            for xa, xb in x_bounds
        )


def assemble_weight_maps(
    data_shape: tuple[int, int, int],
    cell_parameters: dict[
        DyadicCell,
        tuple[float, float],
    ],
) -> tuple[Array, Array]:
    """Assemble image and velocity weight maps.

    Temporal boundary frames are shared by adjacent cells.
    Taking the minimum there implements the lower
    semicontinuous interface convention.
    """
    nt, ny, nx = data_shape

    alpha = np.full(
        (nt, ny, nx),
        np.inf,
        dtype=float,
    )

    beta = np.full(
        (nt - 1, ny, nx),
        np.inf,
        dtype=float,
    )

    for cell, parameter in cell_parameters.items():
        if len(parameter) != 2:
            raise ValueError(
                "every cell parameter must be "
                "an (alpha, beta) pair"
            )

        alpha_value, beta_value = map(
            float,
            parameter,
        )

        if min(alpha_value, beta_value) <= 0.0:
            raise ValueError(
                "cell parameters must be positive"
            )

        image_slice = cell.u_slices
        velocity_slice = cell.v_slices[:-1]

        alpha[image_slice] = np.minimum(
            alpha[image_slice],
            alpha_value,
        )

        beta[velocity_slice] = np.minimum(
            beta[velocity_slice],
            beta_value,
        )

    if (
        not np.all(np.isfinite(alpha))
        or not np.all(np.isfinite(beta))
    ):
        raise ValueError(
            "the supplied cells do not cover "
            "the full data domain"
        )

    return alpha, beta