import unittest

import numpy as np

from tvtv_numerics.proximal import (
    project_frobenius_tv_ball,
    project_scalar_tv_ball,
    prox_l32_fidelity,
    prox_optical_flow_affine,
)


class TestProximalMappings(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20260907)

    def test_scalar_tv_projection_respects_radius(self):
        p = self.rng.normal(size=(3, 5, 6, 2))
        radius = self.rng.uniform(
            0.2,
            1.0,
            size=(3, 5, 6),
        )

        projected = project_scalar_tv_ball(p, radius)
        norm = np.linalg.norm(projected, axis=-1)

        self.assertTrue(
            np.all(norm <= radius + 1e-12)
        )

    def test_scalar_tv_projection_leaves_interior_unchanged(self):
        p = 0.01 * self.rng.normal(
            size=(3, 5, 6, 2)
        )

        projected = project_scalar_tv_ball(
            p,
            radius=1.0,
        )

        np.testing.assert_allclose(projected, p)

    def test_frobenius_projection_respects_radius(self):
        p = self.rng.normal(
            size=(3, 5, 6, 2, 2)
        )
        radius = self.rng.uniform(
            0.2,
            1.0,
            size=(3, 5, 6),
        )

        projected = project_frobenius_tv_ball(
            p,
            radius,
        )
        norm = np.linalg.norm(
            projected,
            axis=(-2, -1),
        )

        self.assertTrue(
            np.all(norm <= radius + 1e-12)
        )

    def test_l32_proximal_satisfies_optimality_equation(self):
        z = self.rng.normal(size=(4, 5))
        f = self.rng.normal(size=(4, 5))
        tau = 0.37

        result = prox_l32_fidelity(z, f, tau)
        difference = result - f

        derivative = (
            np.sign(difference)
            * np.sqrt(np.abs(difference))
        )

        residual = (
            result
            - z
            + tau * derivative
        )

        np.testing.assert_allclose(
            residual,
            0.0,
            rtol=1e-11,
            atol=1e-11,
        )

    def test_affine_flow_proximal_eliminates_middle_residual(self):
        v_tilde = np.zeros((2, 3, 4, 2))
        grad_u = np.zeros_like(v_tilde)
        grad_u[..., 0] = 2.0
        u_t = np.full(
            v_tilde.shape[:-1],
            0.2,
        )

        result = prox_optical_flow_affine(
            v_tilde,
            u_t,
            grad_u,
            tau=0.1,
        )

        residual = (
            u_t
            + np.sum(
                grad_u * result,
                axis=-1,
            )
        )

        np.testing.assert_allclose(
            residual,
            0.0,
            atol=1e-12,
        )

    def test_affine_flow_proximal_does_nothing_when_gradient_is_zero(
        self,
    ):
        v_tilde = self.rng.normal(
            size=(2, 3, 4, 2)
        )
        grad_u = np.zeros_like(v_tilde)
        u_t = self.rng.normal(
            size=v_tilde.shape[:-1]
        )

        result = prox_optical_flow_affine(
            v_tilde,
            u_t,
            grad_u,
            tau=0.1,
        )

        np.testing.assert_allclose(
            result,
            v_tilde,
        )


if __name__ == "__main__":
    unittest.main()