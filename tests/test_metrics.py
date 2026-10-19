import unittest

import numpy as np

from tvtv_numerics.dyadic import DyadicCell
from tvtv_numerics.metrics import (
    evaluation_metrics,
    psnr,
    relative_l2_error,
    rmse,
    upper_level_loss,
)


class TestMetrics(unittest.TestCase):
    def test_rmse_and_psnr(self):
        reference = np.zeros((2, 3, 4))
        value = np.full_like(reference, 0.1)
        self.assertAlmostEqual(rmse(value, reference), 0.1)
        self.assertAlmostEqual(psnr(value, reference), 20.0)

    def test_upper_loss_is_additive_over_children(self):
        rng = np.random.default_rng(9)
        shape = (5, 4, 4)
        u = rng.normal(size=shape)
        clean_u = rng.normal(size=shape)
        v = rng.normal(size=(4, 4, 4, 2))
        clean_v = rng.normal(size=v.shape)
        root = DyadicCell.root(shape)

        parent_loss = upper_level_loss(
            u,
            v,
            clean_u,
            clean_v,
            gamma_u=1.0,
            gamma_v=0.3,
            image_time_weights=root.image_time_weights(),
        )["total"]

        child_loss = 0.0
        for child in root.children():
            child_loss += upper_level_loss(
                u[child.u_slices],
                v[child.v_slices],
                clean_u[child.u_slices],
                clean_v[child.v_slices],
                gamma_u=1.0,
                gamma_v=0.3,
                image_time_weights=child.image_time_weights(),
            )["total"]

        self.assertAlmostEqual(
            parent_loss,
            child_loss,
            places=12,
        )

    def test_relative_l2_error(self):
        reference = np.ones((2, 2))
        value = 2.0 * reference

        self.assertAlmostEqual(
            relative_l2_error(value, reference),
            1.0,
        )
        self.assertEqual(
            relative_l2_error(reference, reference),
            0.0,
        )

    def test_evaluation_metrics_are_exact_on_ground_truth(self):
        u = np.full((3, 2, 2), 0.4)
        v = np.zeros((2, 2, 2, 2))

        metrics = evaluation_metrics(u, v, u, v)

        self.assertEqual(metrics["upper_loss"], 0.0)
        self.assertEqual(metrics["image_rmse"], 0.0)
        self.assertEqual(metrics["relative_image_l2"], 0.0)
        self.assertEqual(metrics["endpoint_error"], 0.0)
        self.assertTrue(np.isinf(metrics["psnr"]))


if __name__ == "__main__":
    unittest.main()