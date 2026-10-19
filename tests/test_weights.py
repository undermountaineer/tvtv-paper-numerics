import unittest

import numpy as np

from tvtv_numerics.weights import (
    forward_stencil_lower_envelope,
)


class TestWeights(unittest.TestCase):
    def test_constant_weight_is_unchanged(self):
        weight = np.full(
            (3, 4, 5),
            0.2,
        )

        np.testing.assert_array_equal(
            forward_stencil_lower_envelope(weight),
            weight,
        )

    def test_interface_uses_smaller_weight(self):
        forward = np.array(
            [[[1.0, 4.0]]]
        )

        reversed_weight = np.array(
            [[[4.0, 1.0]]]
        )

        self.assertEqual(
            forward_stencil_lower_envelope(
                forward
            )[0, 0, 0],
            1.0,
        )

        self.assertEqual(
            forward_stencil_lower_envelope(
                reversed_weight
            )[0, 0, 0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()