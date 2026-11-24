import unittest

import numpy as np

from tvtv_numerics.operators import (
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


if __name__ == "__main__":
    unittest.main()
