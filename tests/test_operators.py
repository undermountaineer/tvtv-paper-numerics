import unittest

import numpy as np

from tvtv_numerics.operators import (
    dt_forward,
    dt_forward_adjoint,
    dx_central,
    dx_central_adjoint,
    dx_forward,
    dx_forward_adjoint,
    dy_central,
    dy_central_adjoint,
    dy_forward,
    dy_forward_adjoint,
    spatial_gradient,
    spatial_gradient_adjoint,
    transport_adjoint,
    transport_operator,
)


class TestOperators(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260907)
        self.shape = (5, 7, 8)
        self.hx = 0.17
        self.hy = 0.23
        self.ht = 0.31

    def assert_adjoint(
        self,
        forward,
        adjoint,
        domain_shape,
        range_shape,
    ):
        x = self.rng.normal(size=domain_shape)
        y = self.rng.normal(size=range_shape)

        left = np.vdot(forward(x), y)
        right = np.vdot(x, adjoint(y))

        np.testing.assert_allclose(
            left,
            right,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_forward_difference_adjointness(self):
        self.assert_adjoint(
            lambda x: dx_forward(x, self.hx),
            lambda y: dx_forward_adjoint(y, self.hx),
            self.shape,
            self.shape,
        )

        self.assert_adjoint(
            lambda x: dy_forward(x, self.hy),
            lambda y: dy_forward_adjoint(y, self.hy),
            self.shape,
            self.shape,
        )

        self.assert_adjoint(
            lambda x: dt_forward(x, self.ht),
            lambda y: dt_forward_adjoint(
                y,
                nt=self.shape[0],
                ht=self.ht,
            ),
            self.shape,
            (self.shape[0] - 1, *self.shape[1:]),
        )

    def test_central_difference_adjointness(self):
        self.assert_adjoint(
            lambda x: dx_central(x, self.hx),
            lambda y: dx_central_adjoint(y, self.hx),
            self.shape,
            self.shape,
        )

        self.assert_adjoint(
            lambda x: dy_central(x, self.hy),
            lambda y: dy_central_adjoint(y, self.hy),
            self.shape,
            self.shape,
        )

    def test_spatial_gradient_adjointness(self):
        self.assert_adjoint(
            lambda x: spatial_gradient(
                x,
                self.hx,
                self.hy,
            ),
            lambda y: spatial_gradient_adjoint(
                y,
                self.hx,
                self.hy,
            ),
            self.shape,
            (*self.shape, 2),
        )

    def test_transport_adjointness_for_fixed_velocity(self):
        v = self.rng.normal(
            size=(
                self.shape[0] - 1,
                *self.shape[1:],
                2,
            )
        )

        self.assert_adjoint(
            lambda x: transport_operator(
                x,
                v,
                self.ht,
                self.hx,
                self.hy,
            ),
            lambda y: transport_adjoint(
                y,
                v,
                nt=self.shape[0],
                ht=self.ht,
                hx=self.hx,
                hy=self.hy,
            ),
            self.shape,
            (self.shape[0] - 1, *self.shape[1:]),
        )

    def test_constant_array_has_zero_derivatives(self):
        u = np.ones(self.shape)

        np.testing.assert_array_equal(
            dx_forward(u),
            0.0,
        )
        np.testing.assert_array_equal(
            dy_forward(u),
            0.0,
        )
        np.testing.assert_array_equal(
            dt_forward(u),
            0.0,
        )
        np.testing.assert_array_equal(
            dx_central(u),
            0.0,
        )
        np.testing.assert_array_equal(
            dy_central(u),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()