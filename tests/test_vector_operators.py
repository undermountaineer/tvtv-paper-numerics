import unittest

import numpy as np

from tvtv_numerics.operators import (
    estimate_operator_norm_squared,
    vector_field_gradient,
    vector_field_gradient_adjoint,
)


class TestVectorFieldOperators(unittest.TestCase):
    def test_vector_field_gradient_adjointness(self):
        rng = np.random.default_rng(20260907)

        v = rng.normal(size=(3, 6, 7, 2))
        p = rng.normal(size=(3, 6, 7, 2, 2))

        left = np.vdot(
            vector_field_gradient(v),
            p,
        )

        right = np.vdot(
            v,
            vector_field_gradient_adjoint(p),
        )

        np.testing.assert_allclose(
            left,
            right,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    def test_operator_norm_estimate_is_positive(self):
        shape = (3, 6, 7, 2)

        norm_squared = estimate_operator_norm_squared(
            vector_field_gradient,
            vector_field_gradient_adjoint,
            shape,
            iterations=30,
        )

        self.assertGreater(norm_squared, 0.0)
        self.assertTrue(np.isfinite(norm_squared))


if __name__ == "__main__":
    unittest.main()