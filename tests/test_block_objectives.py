"""Focused checks for the two fixes used by the minimal paper experiment."""

import unittest

import numpy as np

from tvtv_numerics.weighted_energy import (
    weighted_u_block_objective,
    weighted_v_block_objective,
)
from tvtv_numerics.weighted_solver import (
    WeightedTVTVConfig,
    solve_weighted_u_block,
    solve_weighted_v_block,
)
from tvtv_numerics.weights import forward_stencil_lower_envelope


class TestBlockObjectiveConsistency(unittest.TestCase):
    def setUp(self):
        self.config = WeightedTVTVConfig(
            max_inner_u_iterations=4,
            max_inner_v_iterations=4,
            min_inner_iterations=1,
            norm_iterations=6,
            hx=0.2,
            hy=0.25,
            ht=0.5,
        )
        self.u = np.broadcast_to(
            np.array([0.2, 0.3, 0.5, 0.7, 0.8]), (3, 4, 5)
        ).copy()
        self.f = np.full(self.u.shape, 0.4)
        self.v = np.zeros((2, 4, 5, 2))
        self.v[..., 0] = np.array([0.0, 0.1, 0.3, 0.5, 0.6])

    def test_image_block_reports_the_same_objective_as_raw_weights(self):
        alpha = np.full(self.u.shape, 0.2)
        alpha[..., 3:] = 0.02
        expected = weighted_u_block_objective(
            self.u, self.f, self.v, alpha, self.config.kappa,
            self.config.ht, self.config.hx, self.config.hy,
        )
        double_transformed = weighted_u_block_objective(
            self.u, self.f, self.v, forward_stencil_lower_envelope(alpha),
            self.config.kappa, self.config.ht, self.config.hx, self.config.hy,
        )
        self.assertNotAlmostEqual(expected, double_transformed, places=10)
        result = solve_weighted_u_block(
            self.f, self.v, alpha, self.u, self.config,
        )
        self.assertAlmostEqual(result.objective_before, expected, places=12)
        self.assertAlmostEqual(
            result.objective_after,
            weighted_u_block_objective(
                result.value, self.f, self.v, alpha, self.config.kappa,
                self.config.ht, self.config.hx, self.config.hy,
            ),
            places=12,
        )

    def test_motion_block_reports_the_same_objective_as_raw_weights(self):
        beta = np.full(self.v.shape[:-1], 0.3)
        beta[..., 3:] = 0.03
        expected = weighted_v_block_objective(
            self.v, self.u, beta, self.config.kappa,
            self.config.ht, self.config.hx, self.config.hy,
        )
        double_transformed = weighted_v_block_objective(
            self.v, self.u, forward_stencil_lower_envelope(beta),
            self.config.kappa, self.config.ht, self.config.hx, self.config.hy,
        )
        self.assertNotAlmostEqual(expected, double_transformed, places=10)
        result = solve_weighted_v_block(self.u, beta, self.v, self.config)
        self.assertAlmostEqual(result.objective_before, expected, places=12)
        self.assertAlmostEqual(
            result.objective_after,
            weighted_v_block_objective(
                result.value, self.u, beta, self.config.kappa,
                self.config.ht, self.config.hx, self.config.hy,
            ),
            places=12,
        )
