#!/usr/bin/env python3
"""Can the v-block find the motion when u is right? Diagnostic on a toy scene (default 'split').

Solves only the velocity subproblem, tightly, for a fixed image u:
  clean/fit     u = clean reference, v starts at the global fit (0.53 px off on the shapes)
  clean/true    u = clean reference, v starts at the true motion
  stuck/stuck   u, v = where the joint solve started from the fit stops
  stuck/true    u = that stuck image, v starts at the true motion
Velocity errors are split into edge pixels (nonzero central gradient of the clean frame),
where the transport term sees the motion, and shape interiors, which only TV(v) can fill in.

    python -u scripts/toy_v_block.py
    python -u scripts/toy_v_block.py --motion split-slow
"""
from __future__ import annotations

# First: sets one BLAS thread and puts src/ on sys.path.
from toy_heterogeneous_noise import MOTIONS, ROOT, render, solve

import argparse
from dataclasses import replace
import json
import time

import numpy as np

from tvtv_numerics.data import add_noise, estimate_translation
from tvtv_numerics.operators import dx_central, dy_central
from tvtv_numerics.weighted_solver import WeightedTVTVConfig, solve_weighted_v_block

SIGMA = (.05, .2)  # toy defaults, left / right


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", choices=MOTIONS, default="split")
    p.add_argument("--alpha", type=float, default=2.83e-3, help="image TV weight of the joint solve")
    p.add_argument("--max-inner", type=int, default=20000, help="v-block iteration cap")
    p.add_argument("--tolerance", type=float, default=1e-6, help="scaled v-block residual limit")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()

    n = a.resolution
    h = 1 / n
    settings = json.loads((ROOT / "configs/synthetic_256.json").read_text())
    cfg = WeightedTVTVConfig(**dict(settings["solver"]["local"], hx=h, hy=h))
    tight = replace(cfg, max_inner_v_iterations=a.max_inner, inner_tolerance=a.tolerance,
                    inner_absolute_tolerance=a.tolerance * 1e-3)
    beta = settings["model"]["beta"]

    shifts = np.array(MOTIONS[a.motion])
    ref = render(n, shifts)
    left = np.arange(n) < n // 2
    f = add_noise(ref, np.where(left, *SIGMA) * np.ones((n, 1)), a.seed)
    v_fit, _ = estimate_translation(f, h, h, 1.)
    v_true = np.broadcast_to(np.where(left[:, None], shifts[0], shifts[1])[None] * h, v_fit.shape).copy()
    edges = np.hypot(dx_central(ref[:-1], h), dy_central(ref[:-1], h)) > 1e-9
    interior = (np.abs(ref[:-1] - .2) > 1e-9) & ~edges

    def epe(v, mask):
        return float(np.linalg.norm(v - v_true, axis=-1)[mask].mean() / h)

    print(f"{a.motion} scene {n}x{n}, alpha {a.alpha:.3e}; joint solve from the global fit...", flush=True)
    u_stuck, v_stuck, info = solve(a.alpha, f, v_fit, beta, cfg)
    print(f"  outer={info['outer']} converged={info['converged']} energy={info['energy']:.7e} "
          f"EPE edges {epe(v_stuck, edges):.3f} px, interior {epe(v_stuck, interior):.3f} px\n")

    b = np.full(v_fit.shape[:-1], beta)
    cases = {"clean/fit": (ref, v_fit), "clean/true": (ref, v_true),
             "stuck/stuck": (u_stuck, v_stuck), "stuck/true": (u_stuck, v_true)}
    print(f"{'case':<12} {'EPE edges px':>14} {'EPE interior px':>16} "
          f"{'v-block objective before -> after':>34} {'iters':>6} {'conv':>5} {'sec':>6}")
    for name, (u, v_start) in cases.items():
        start = time.perf_counter()
        r = solve_weighted_v_block(u, b, v_start, tight)
        print(f"{name:<12} {epe(v_start, edges):6.3f} -> {epe(r.value, edges):5.3f} "
              f"{epe(v_start, interior):7.3f} -> {epe(r.value, interior):5.3f} "
              f"{r.objective_before:15.9e} -> {r.objective_after:15.9e} "
              f"{r.iterations:6d} {str(r.converged):>5} {time.perf_counter() - start:6.1f}", flush=True)


if __name__ == "__main__":
    main()
