import unittest

import numpy as np

from tvtv_numerics.dyadic import DyadicCell
from tvtv_numerics.metrics import (
    psnr,
    rmse,
    upper_level_loss,
)
from tvtv_numerics.original_1024 import children


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
        for child in children(root):
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


if __name__ == "__main__":
    unittest.main()
