"""Regression checks for selection, crop geometry and exact checkpoint resume."""
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from tvtv_numerics.checkpoint_cache import ResumableSolverCache, SOURCE_FILES
from tvtv_numerics.handoff import verified_source
from tvtv_numerics.workflow import CELL_NAMES, select_alphas, spatial_children
from tvtv_numerics.weighted_solver import WeightedTVTVConfig, solve_weighted_tvtv


class TestWorkflow(unittest.TestCase):
    def test_spatial_children_keep_the_full_time_interval(self):
        shape = (5, 8, 12)
        coverage = np.zeros(shape, dtype=int)
        for cell in spatial_children(shape).values():
            self.assertEqual((cell.t0, cell.t1), (0, 4))
            coverage[cell.u_slices] += 1
        np.testing.assert_array_equal(coverage, 1)

    def test_unfinished_low_loss_candidate_blocks_selection(self):
        rows = [dict(cell=name, alpha=a, upper_loss=loss, eligible=True)
                for name in CELL_NAMES for a, loss in ((.001, .02), (.002, .03))]
        self.assertEqual(select_alphas(rows), dict.fromkeys(CELL_NAMES, .001))
        rows[-1].update(upper_loss=None, partial_upper_loss=0., eligible=False)
        self.assertIsNone(select_alphas(rows))

    def test_exact_resume_matches_uninterrupted_alternation(self):
        rng = np.random.default_rng(12)
        f = .4 + .05*rng.normal(size=(3, 4, 6))
        a, b = np.full(f.shape, .02), np.full((2, 4, 6), .03)
        cfg = WeightedTVTVConfig(step_policy="balanced", residual_mode="scaled", update_order="u_then_v",
                                max_outer_iterations=5, max_inner_u_iterations=30, max_inner_v_iterations=30,
                                min_inner_iterations=2, norm_iterations=6, inner_tolerance=1e-9, outer_tolerance=1e-9)
        whole = solve_weighted_tvtv(f, a, b, cfg)
        with tempfile.TemporaryDirectory() as directory:
            cache = ResumableSolverCache(directory)
            partial = cache.run(f, a, b, cfg, outer_iteration_budget=2)
            self.assertFalse(partial.converged)
            resumed = cache.run(f, a, b, cfg)
        np.testing.assert_array_equal(whole.u, resumed.u)
        np.testing.assert_array_equal(whole.v, resumed.v)
        self.assertEqual(whole.history, resumed.history)

    def test_import_rejects_changed_scientific_code(self):
        import tvtv_numerics.checkpoint_cache as module
        directory = Path(module.__file__).parent
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as archive:
            for name in SOURCE_FILES:
                data = (directory/name).read_bytes()
                if name == "weights.py":
                    data += b"\n# changed scientific source\n"
                archive.writestr("src/tvtv_numerics/"+name, data)
        with zipfile.ZipFile(io.BytesIO(content.getvalue())) as archive:
            with self.assertRaisesRegex(ValueError, "Scientific source mismatch"):
                verified_source(archive)


if __name__ == "__main__":
    unittest.main()
