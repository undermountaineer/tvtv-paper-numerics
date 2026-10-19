"""Check sampling, complete searches, endpoint guards and quadtree decisions."""
import unittest
from unittest.mock import patch
import numpy as np

from tvtv_numerics.dyadic import DyadicCell
from tvtv_numerics.original_1024 import Original1024, best, block_average, children, split_accepted


def experiment(loss, unresolved=None):
    obj = Original1024.__new__(Original1024)
    obj.settings = {
        "data": {"resolution": 32, "design_seed": 1},
        "search": {"initial_alphas": [1., 2., 4.], "bounds": [.125, 32.],
                   "expansion_factor": 2., "max_expansions": 3, "midpoint_rounds": 1},
        "refinement": {"max_depth": 2, "rho": 1.}}
    def row(job):
        eligible = unresolved is None or job["alpha"] != unresolved
        return dict(job, eligible=eligible, upper_loss=loss(job) if eligible else None)
    obj.row = row
    return obj


class TestOriginal1024(unittest.TestCase):
    def test_average_before_noise_and_flux_conservation(self):
        a = np.arange(2*8*8).reshape(2, 8, 8)
        coarse = block_average(a, 4)
        np.testing.assert_array_equal(coarse[0], [[4.5, 6.5, 8.5, 10.5], [20.5, 22.5, 24.5, 26.5],
                                                 [36.5, 38.5, 40.5, 42.5], [52.5, 54.5, 56.5, 58.5]])
        np.testing.assert_array_equal(coarse.mean((1, 2)), a.mean((1, 2)))
        with self.assertRaises(ValueError):
            block_average(a, 3)

    def test_spatial_children_are_a_disjoint_cover_keep_time(self):
        cell = DyadicCell.root((5, 32, 32))
        counts = np.zeros(cell.u_shape)
        for c in children(cell):
            self.assertEqual((c.t0, c.t1), (0, 4))
            counts[c.u_slices] += 1
        np.testing.assert_array_equal(counts, 1)

    def test_unfinished_candidates_block_selection(self):
        self.assertIsNone(best([dict(alpha=1., upper_loss=1., eligible=True),
                                dict(alpha=2., upper_loss=None, eligible=False)]))
        obj = experiment(lambda j: (j["alpha"]-2)**2, unresolved=4.)
        state = obj.search(obj.root_cell)
        self.assertFalse(state["complete"])
        self.assertEqual(state["pending"][0]["alpha"], 4.)

    def test_search_expands_and_refines_in_log_space(self):
        obj = experiment(lambda j: np.log(j["alpha"]/8.)**2)
        state = obj.search(obj.root_cell, inherited=3.)
        self.assertTrue(state["complete"])
        self.assertFalse(state["boundary_limited"])
        self.assertEqual(state["selected"]["alpha"], 8.)
        alphas = {r["alpha"] for r in state["rows"]}
        self.assertTrue({3., 8., 16.}.issubset(alphas))
        self.assertIn(float(np.sqrt(8.*16.)), alphas)

    def test_monotone_endpoint_is_reported_and_blocks_tree(self):
        obj = experiment(lambda j: 1/j["alpha"])
        state = obj.search(obj.root_cell)
        self.assertTrue(state["boundary_limited"])
        self.assertFalse(obj.tree()["complete"])
        self.assertIn("boundary", obj.tree()["blocked"])

    def test_equal_costs_reject_and_rho_margin_is_respected(self):
        self.assertFalse(split_accepted(4., [1., 1., 1., 1.], 1.))
        self.assertTrue(split_accepted(4., [.9]*4, 1.))
        self.assertFalse(split_accepted(4., [.99]*4, .98))

    def test_rejection_keeps_parent_and_acceptance_recurses_to_depth(self):
        obj = experiment(lambda j: 1.)
        def learned(cell, inherited=None):
            loss = (4. if cell.level == 0 else 1.)
            return dict(complete=True, boundary_limited=False, pending=[], rows=[],
                        selected=dict(alpha=2., upper_loss=loss))
        obj.search = learned
        state = obj.tree()
        self.assertTrue(state["complete"])
        self.assertEqual(len(state["leaves"]), 1)
        self.assertFalse(state["decisions"][0]["accepted"])
        def decreasing(cell, inherited=None):
            return dict(complete=True, boundary_limited=False, pending=[], rows=[],
                        selected=dict(alpha=2., upper_loss=10.**(-cell.level)))
        obj.search = decreasing
        state = obj.tree()
        self.assertTrue(state["complete"])
        self.assertEqual(len(state["leaves"]), 16)
        self.assertEqual(len(state["decisions"]), 5)
        self.assertTrue(all(r["accepted"] for r in state["decisions"]))


if __name__ == "__main__":
    unittest.main()
