import unittest

import numpy as np

from tvtv_numerics.energy import scalar_spatial_tv
from tvtv_numerics.operators import (
    vector_field_gradient,
)
from tvtv_numerics.weighted_energy import (
    l32_fidelity,
    manuscript_energy,
    weighted_image_tv,
    weighted_u_block_objective,
    weighted_v_block_objective,
    weighted_velocity_tv,
)


class TestWeightedEnergy(unittest.TestCase):
    def test_l32_fidelity(self):
        u = np.array(
            [[[0.0, 1.0]]]
        )

        f = np.zeros_like(u)

        self.assertAlmostEqual(
            l32_fidelity(u, f),
            2.0 / 3.0,
        )

    def test_constant_image_weight_scales_tv(self):
        rng = np.random.default_rng(7)

        u = rng.normal(
            size=(3, 5, 6)
        )

        alpha = 0.17

        self.assertAlmostEqual(
            weighted_image_tv(
                u,
                np.full(u.shape, alpha),
            ),
            alpha * scalar_spatial_tv(u),
            places=12,
        )

    def test_constant_velocity_weight_scales_tv(self):
        rng = np.random.default_rng(8)

        v = rng.normal(
            size=(2, 5, 6, 2)
        )

        beta = 0.23

        unweighted = float(
            np.sum(
                np.linalg.norm(
                    vector_field_gradient(v),
                    axis=(-2, -1),
                )
            )
        )

        self.assertAlmostEqual(
            weighted_velocity_tv(
                v,
                np.full(
                    v.shape[:-1],
                    beta,
                ),
            ),
            beta * unweighted,
            places=12,
        )

    def test_block_objectives_match_full_energy_up_to_constants(self):
        rng = np.random.default_rng(14)
        u = rng.normal(size=(3, 4, 5))
        f = rng.normal(size=u.shape)
        v = rng.normal(size=(2, 4, 5, 2))
        alpha = np.full(u.shape, 0.02)
        beta = np.full(v.shape[:-1], 0.04)

        energy = manuscript_energy(u, v, f, alpha, beta, 1.3)
        u_objective = weighted_u_block_objective(
            u, f, v, alpha, 1.3
        )
        v_objective = weighted_v_block_objective(
            v, u, beta, 1.3
        )

        self.assertAlmostEqual(
            u_objective + energy["tv_v"], energy["total"]
        )
        self.assertAlmostEqual(
            v_objective + energy["fidelity"] + energy["tv_u"],
            energy["total"],
        )


if __name__ == "__main__":
    unittest.main()
