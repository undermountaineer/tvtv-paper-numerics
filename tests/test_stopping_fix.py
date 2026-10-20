"""A residual-qualified block may need objective damping without extra PDHG."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from tvtv_numerics.operators import spatial_gradient_adjoint, transport_adjoint
from tvtv_numerics.proximal import prox_l32_fidelity
from tvtv_numerics.weighted_solver import (
    WeightedTVTVConfig, solve_weighted_u_block, solve_weighted_v_block,
)


class TestStoppingFix(unittest.TestCase):
    def setUp(self):
        self.f = np.full((3, 8, 8), .4)
        self.v = np.zeros((2, 8, 8, 2))
        self.config = WeightedTVTVConfig(hx=1/8, hy=1/8, step_policy="balanced",
            residual_mode="scaled", min_inner_iterations=10,
            max_inner_u_iterations=30, max_inner_v_iterations=30)

    def test_image_transient_above_optimal_objective_does_not_exhaust_cap(self):
        # The initial constant image is already a minimizer. Small feasible
        # warm duals produce a transient with a positive objective. The old
        # objective gate consumes the entire cap despite small residuals.
        rng = np.random.default_rng(308)
        dual = (1e-5*rng.normal(size=(*self.f.shape, 2)), np.zeros(self.v.shape[:-1]))
        result = solve_weighted_u_block(self.f, self.v, np.full(self.f.shape, .001),
                                        self.f, self.config, dual)
        self.assertTrue(result.converged)
        self.assertTrue(result.returned_state_checked)
        self.assertEqual(result.iterations, 10)
        self.assertLess(result.accepted_step_scale, 1.)
        self.assertLessEqual(result.objective_after-result.objective_before, 1.01e-8)
        # Check the reported primal residual at the returned damped image.
        gradient = spatial_gradient_adjoint(result.dual[0], 1/8, 1/8)
        gradient += transport_adjoint(result.dual[1], self.v, nt=3, ht=1., hx=1/8, hy=1/8)
        probe = prox_l32_fidelity(result.value-result.primal_step_size*gradient,
                                  self.f, result.primal_step_size)
        expected = np.sqrt(np.mean((result.value-probe)**2))/.4
        self.assertAlmostEqual(result.primal_residual, expected, places=14)

    def test_motion_transient_above_optimal_objective_does_not_exhaust_cap(self):
        rng = np.random.default_rng(308)
        dual = 1e-7*rng.normal(size=(*self.v.shape, 2))
        result = solve_weighted_v_block(self.f, np.full(self.v.shape[:-1], .0125),
                                        self.v, self.config, dual)
        self.assertTrue(result.converged)
        self.assertTrue(result.returned_state_checked)
        self.assertEqual(result.iterations, 10)
        self.assertLess(result.accepted_step_scale, 1.)
        self.assertLessEqual(result.objective_after-result.objective_before, 1.01e-8)

    def test_bad_returned_image_cannot_pass_using_raw_iterate_residual(self):
        # Simulate an invalid safeguard result: raw PDHG residuals are zero,
        # but the returned state is not a fixed point of the data proximal.
        def invalid(initial, candidate, objective, config):
            return initial+.01, objective(initial), objective(initial+.01), 0., True
        with patch("tvtv_numerics.weighted_solver._objective_safeguard", invalid):
            result = solve_weighted_u_block(self.f, self.v,
                np.full(self.f.shape, .001), self.f, self.config)
        self.assertFalse(result.converged)
        self.assertGreater(result.primal_residual, result.primal_residual_limit)

    def test_below_minimum_inner_cap_does_not_become_converged(self):
        config = replace(self.config, max_inner_u_iterations=3)
        result = solve_weighted_u_block(self.f, self.v,
            np.full(self.f.shape, .001), self.f, config)
        self.assertFalse(result.converged)

    def test_bad_damping_certificate_causes_more_inner_work(self):
        import tvtv_numerics.weighted_solver as module
        real_safeguard = module._objective_safeguard
        for block in ("image", "motion"):
            with self.subTest(block=block):
                calls = []
                def transient_bad_safeguard(initial, candidate, objective, config):
                    calls.append(1)
                    if len(calls) == 1:
                        bad = initial+.01
                        return bad, objective(initial), objective(bad), .5, True
                    return real_safeguard(initial, candidate, objective, config)
                with patch.object(module, "_objective_safeguard", transient_bad_safeguard):
                    if block == "image":
                        result = solve_weighted_u_block(self.f, self.v,
                            np.full(self.f.shape, .001), self.f, self.config)
                    else:
                        result = solve_weighted_v_block(self.f,
                            np.full(self.v.shape[:-1], .0125), self.v, self.config)
                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 11)


if __name__ == "__main__":
    unittest.main()
