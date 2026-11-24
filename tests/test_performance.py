"""Check the numerical contracts affected by the small performance pass."""
from dataclasses import replace
import tempfile
import unittest

import numpy as np

from tvtv_numerics.checkpoint_cache import ResumableSolverCache
from tvtv_numerics.operators import (
    spatial_gradient, spatial_gradient_adjoint, transport_adjoint, transport_operator,
)
from tvtv_numerics.proximal import project_scalar_tv_ball, prox_l32_fidelity, prox_optical_flow_affine
from tvtv_numerics.weighted_solver import (
    WeightedTVTVConfig, balanced_u_steps, solve_weighted_tvtv, solve_weighted_u_block,
    solve_weighted_v_block,
)
from tvtv_numerics.weights import forward_stencil_lower_envelope


class TestPerformanceChanges(unittest.TestCase):
    def test_affine_proximal_optimality_including_zero_and_saturated_branches(self):
        rng = np.random.default_rng(184)
        w = rng.normal(size=(2, 7, 9, 2))
        gradient = rng.normal(size=w.shape)
        gradient[0, 0] = 0.
        offset = 3. * rng.normal(size=w.shape[:-1])
        tau = .37
        norm_squared = np.sum(gradient*gradient, axis=-1)
        original_inputs = [x.copy() for x in (w, gradient, offset)]
        result = prox_optical_flow_affine(w, offset, gradient, tau)
        cached = prox_optical_flow_affine(w, offset, gradient, tau,
                                         gradient_norm_squared=norm_squared)
        np.testing.assert_array_equal(result, cached)
        residual = offset + np.sum(gradient*result, axis=-1)
        subgradient = np.sign(residual)
        zero = np.abs(residual) < 1e-12
        dual = np.sum((w-result)*gradient, axis=-1)
        subgradient[zero] = np.divide(dual, tau*norm_squared,
                                     out=np.zeros_like(dual), where=norm_squared>0)[zero]
        self.assertTrue(np.all(np.abs(subgradient) <= 1.+1e-12))
        np.testing.assert_allclose(result-w+tau*subgradient[..., None]*gradient,
                                   0., atol=2e-12)
        np.testing.assert_array_equal(result[0, 0], w[0, 0])
        for array, original in zip((w, gradient, offset), original_inputs):
            np.testing.assert_array_equal(array, original)

    def test_check_interval_requires_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                WeightedTVTVConfig(inner_check_interval=value)

    def test_sparse_checks_preserve_fixed_work_and_check_final_iterate(self):
        rng = np.random.default_rng(185)
        f = .4+.05*rng.normal(size=(3, 5, 7))
        initial_u = f.copy()
        initial_v = .01*rng.normal(size=(2, 5, 7, 2))
        alpha = np.full(f.shape, .02); alpha[..., 3:] = .005
        beta = np.full(initial_v.shape[:-1], .03)
        config = WeightedTVTVConfig(hx=.2, hy=.3, ht=.5, max_inner_u_iterations=23, max_inner_v_iterations=23,
            min_inner_iterations=2, inner_tolerance=1e-25, inner_absolute_tolerance=1e-25)
        sparse = replace(config, inner_check_interval=5)
        image_dense = solve_weighted_u_block(f, initial_v, alpha, initial_u, config)
        image_sparse = solve_weighted_u_block(f, initial_v, alpha, initial_u, sparse)
        motion_dense = solve_weighted_v_block(initial_u, beta, initial_v, config)
        motion_sparse = solve_weighted_v_block(initial_u, beta, initial_v, sparse)
        for dense, thin in ((image_dense, image_sparse), (motion_dense, motion_sparse)):
            self.assertEqual(thin.iterations, 23)
            self.assertEqual(thin.residual_evaluations, 6)  # 2,7,12,17,22,23
            self.assertEqual(dense.residual_evaluations, 22)
            self.assertFalse(thin.converged)
            np.testing.assert_array_equal(dense.value, thin.value)
            self.assertEqual(dense.raw_primal_residual, thin.raw_primal_residual)
            self.assertEqual(dense.raw_dual_residual, thin.raw_dual_residual)
        np.testing.assert_array_equal(f, initial_u)

    def test_carried_operator_values_match_explicit_pdhg(self):
        # The solver forms K ubar from carried K u values; explicit PDHG must give the same duals.
        rng = np.random.default_rng(187)
        f = .4+.05*rng.normal(size=(3, 5, 7))
        v = .01*rng.normal(size=(2, 5, 7, 2))
        alpha = np.full(f.shape, .02); alpha[..., 3:] = .005
        config = WeightedTVTVConfig(hx=.2, hy=.3, ht=.5, max_inner_u_iterations=17, min_inner_iterations=2,
                                    inner_tolerance=1e-25, inner_absolute_tolerance=1e-25)
        result = solve_weighted_u_block(f, v, alpha, f, config)
        tau, sigma_tv, sigma_transport, _ = balanced_u_steps(v, f.shape, config)
        radius = forward_stencil_lower_envelope(alpha)
        h = (config.ht, config.hx, config.hy)
        u = u_bar = f
        p, q = np.zeros((*f.shape, 2)), np.zeros(v.shape[:-1])
        for _ in range(17):
            p = project_scalar_tv_ball(p + sigma_tv*spatial_gradient(u_bar, config.hx, config.hy), radius)
            q = np.clip(q + sigma_transport*transport_operator(u_bar, v, *h), -config.kappa, config.kappa)
            gradient = spatial_gradient_adjoint(p, config.hx, config.hy) + transport_adjoint(q, v, 3, *h)
            u, u_old = prox_l32_fidelity(u - tau*gradient, f, tau), u
            u_bar = u + config.theta*(u - u_old)
        self.assertEqual(result.iterations, 17)
        np.testing.assert_allclose(result.dual[0], p, rtol=1e-10, atol=1e-13)
        np.testing.assert_allclose(result.dual[1], q, rtol=1e-10, atol=1e-13)

    def test_check_at_cap_below_minimum_is_reported_without_false_convergence(self):
        config = WeightedTVTVConfig(min_inner_iterations=10, max_inner_u_iterations=3,
                                    inner_check_interval=5)
        f = np.full((3, 4, 4), .4)
        result = solve_weighted_u_block(f, np.zeros((2, 4, 4, 2)), np.full(f.shape, .02), f, config)
        self.assertEqual(result.residual_evaluations, 1)
        self.assertTrue(np.isfinite(result.residual))
        self.assertFalse(result.converged)

    def test_sparse_check_checkpoint_resume_matches_uninterrupted_run(self):
        rng = np.random.default_rng(186)
        f = .4+.05*rng.normal(size=(3, 4, 6))
        alpha, beta = np.full(f.shape, .02), np.full((2, 4, 6), .03)
        config = WeightedTVTVConfig(update_order="u_then_v",
            inner_check_interval=5, max_outer_iterations=5, max_inner_u_iterations=23,
            max_inner_v_iterations=23, min_inner_iterations=2, inner_tolerance=1e-12, outer_tolerance=1e-12)
        whole = solve_weighted_tvtv(f, alpha, beta, config)
        with tempfile.TemporaryDirectory() as directory:
            cache = ResumableSolverCache(directory)
            cache.run(f, alpha, beta, config, outer_iteration_budget=2)
            resumed = cache.run(f, alpha, beta, config)
        np.testing.assert_array_equal(whole.u, resumed.u)
        np.testing.assert_array_equal(whole.v, resumed.v)
        self.assertEqual(whole.history, resumed.history)


if __name__ == "__main__":
    unittest.main()
