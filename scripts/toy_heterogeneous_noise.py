#!/usr/bin/env python3
"""Toy TV-TV run: the same square and disk on each half, low noise left, high noise right.

Scans a constant alpha, picks the best alpha for each half from those same
full-domain solves (crops of the loss, not independent crop solves as in the
protocol), then solves once with the resulting two-valued alpha map. Beta and
the solver tolerances are the "local" block of configs/synthetic_256.json.
Nothing is cached or resumed.

--motion chooses how each pair of shapes moves (see MOTIONS); outputs go to
results/toy_heterogeneous_noise/<motion>/ (or <motion>_v0-true/ with --v0 true).

    python -u scripts/toy_heterogeneous_noise.py                                 # 64x64, translation
    python -u scripts/toy_heterogeneous_noise.py --motion split
    python -u scripts/toy_heterogeneous_noise.py --resolution 32 --max-outer 5   # smoke test
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from itertools import repeat
import json
import os
from pathlib import Path
import sys
import time

# One BLAS thread per process (as in run_original_1024.py); set before numpy loads.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # no editable install needed, as in benchmark_performance.py

from tvtv_numerics.data import add_noise, estimate_translation
from tvtv_numerics.dyadic import DyadicCell
from tvtv_numerics.metrics import upper_level_loss
from tvtv_numerics.weighted_solver import WeightedTVTVConfig, solve_weighted_tvtv

FRAMES = 5
# Pixels per frame (x, y) of the (left, right) pair of shapes. The flat background never shows motion.
MOTIONS = {
    "translation": ((.75, .375), (.75, .375)),  # as in the 256 protocol; the global initial fit v0 is already right
    "split": ((.75, 0.), (0., .75)),            # left pair moves right, right pair moves down; v0 is wrong
    "split-slow": ((.25, 0.), (0., .25)),       # split at a third of the speed: linearisation error ~d^2
    "static": ((0., 0.), (0., 0.)),             # no motion: separates TV effects from the transport term
}
# Ratio sqrt(2). Alpha=5.3e-4 was best at 256, and alpha scales like h (fidelity vs alpha*|grad u|).
ALPHAS = np.geomspace(5e-4, 3.2e-2, 13)


def render(n, shifts, ss=8):
    """Square (0.8) and disk (0.5) on a 0.2 background, once per half, each pair moving by its own
    shift (pixels per frame); block-averaged from ss*n samples."""
    c = (np.arange(n * ss) + .5) / (n * ss)
    frames = []
    for t in range(FRAMES):
        z = np.full((n * ss, n * ss), .2)
        for x0, (sx, sy) in zip((.25, .75), shifts):  # shapes stay inside their half for 4 intervals at n>=32
            x = c[None, :] - sx * t / n
            y = c[:, None] - sy * t / n
            z[(np.abs(x - x0) <= .1) & (np.abs(y - .3) <= .1)] = .8
            z[np.hypot(x - x0, y - .7) <= .1] = .5
        frames.append(z.reshape(n, ss, n, ss).mean((1, 3)))
    return np.stack(frames)


def solve(alpha, f, v0, beta, cfg):
    """alpha: scalar or (ny, nx) map, constant in time. Starts from u=f and the observation-only v0."""
    start = time.perf_counter()
    r = solve_weighted_tvtv(f, np.broadcast_to(alpha, f.shape).copy(), np.full(v0.shape[:-1], beta), cfg,
                            initial_v=v0)
    return r.u, r.v, dict(outer=r.outer_iterations, converged=r.converged, energy=float(r.history[-1]["total"]),
                          seconds=time.perf_counter() - start)


def losses(u, v, ref, h):
    """Protocol training loss (image only, trapezoidal in time) on the whole grid and on each half."""
    weights = DyadicCell.root(ref.shape).image_time_weights()
    half = ref.shape[-1] // 2

    def loss(cols):
        return upper_level_loss(u[..., cols], v[..., cols, :], ref[..., cols], None, gamma_v=0.,
                                image_time_weights=weights, cell_volume=h * h)["total"]  # ht = 1

    out = dict(J=loss(slice(None)), J_left=loss(slice(None, half)), J_right=loss(slice(half, None)))
    assert np.isclose(out["J_left"] + out["J_right"], out["J"]), "half losses must add up to the full loss"
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--motion", choices=MOTIONS, default="translation")
    p.add_argument("--v0", choices=("fit", "true"), default="fit",
                   help="initial velocity: the observation-only global fit, or the true motion (diagnostic)")
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--sigma-left", type=float, default=.05)
    p.add_argument("--sigma-right", type=float, default=.2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-outer", type=int, help="override the outer-iteration cap (smoke tests)")
    p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    p.add_argument("--output", type=Path, default=ROOT / "results/toy_heterogeneous_noise")
    a = p.parse_args()
    n = a.resolution
    if n < 32 or n % 2:
        p.error("resolution must be even and at least 32")
    if min(a.sigma_left, a.sigma_right) < 0:
        p.error("noise levels must be nonnegative")

    h = 1 / n
    settings = json.loads((ROOT / "configs/synthetic_256.json").read_text())
    cfg = WeightedTVTVConfig(**dict(settings["solver"]["local"], hx=h, hy=h))
    if a.max_outer:
        cfg = replace(cfg, max_outer_iterations=a.max_outer)
    beta = settings["model"]["beta"]

    shifts = np.array(MOTIONS[a.motion])
    ref = render(n, shifts)
    left = np.arange(n) < n // 2
    sigma = np.where(left, a.sigma_left, a.sigma_right) * np.ones((n, 1))
    f = add_noise(ref, sigma, a.seed)
    v0, init = estimate_translation(f, h, h, 1.)
    v_true = np.where(left[:, None], shifts[0], shifts[1])[None] * h  # (1, nx, 2), physical units, ht = 1
    on_shapes = np.abs(ref[:-1] - .2) > 1e-9  # pixels a shape covers at the start of each interval
    if a.v0 == "true":  # flat background takes its half's velocity; any value is consistent there
        v0 = np.broadcast_to(v_true, v0.shape).copy()

    def epe(v):
        """Mean endpoint error (pixels per frame) on the shapes: the background has no defined motion."""
        return float(np.linalg.norm(v - v_true, axis=-1)[on_shapes].mean() / h)

    print(f"{n}x{n}, {FRAMES} frames, motion '{a.motion}', sigma {a.sigma_left} (left) / {a.sigma_right} (right); "
          f"global fit {np.round(init['displacement_xy_pixels_per_interval'], 3)} px, "
          f"true {shifts[0]} (left) / {shifts[1]} (right) px; starting from the {a.v0} motion")
    print(f"{len(ALPHAS)} constant-alpha solves on {a.workers} workers...", flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        scan = list(pool.map(solve, ALPHAS, repeat(f), repeat(v0), repeat(beta), repeat(cfg)))

    rows = [dict(alpha=float(alpha), **losses(u, v, ref, h), epe_px=epe(v), **info)
            for alpha, (u, v, info) in zip(ALPHAS, scan)]
    print(f"{'alpha':>10} {'J':>11} {'J_left':>11} {'J_right':>11} {'energy E':>13} {'outer':>5} {'conv':>5} "
          f"{'EPE px':>7} {'sec':>6}")
    for r in rows:
        print(f"{r['alpha']:10.3e} {r['J']:11.5e} {r['J_left']:11.5e} {r['J_right']:11.5e} {r['energy']:13.7e} "
              f"{r['outer']:5d} {str(r['converged']):>5} {r['epe_px']:7.3f} {r['seconds']:6.1f}")

    best = {k: int(np.argmin([r[k] for r in rows])) for k in ("J", "J_left", "J_right")}
    for k, i in best.items():
        if i in (0, len(ALPHAS) - 1):
            print(f"WARNING: the best alpha for {k} is at the end of the scan; widen ALPHAS.")
    if not all(r["converged"] for r in rows):
        print("WARNING: some solves hit the outer cap; their losses are partial.")

    const = rows[best["J"]]
    a_left, a_right = ALPHAS[best["J_left"]], ALPHAS[best["J_right"]]
    alpha_map = np.where(left, a_left, a_right) * np.ones((n, 1))
    u_map, v_map, info = solve(alpha_map, f, v0, beta, cfg)
    spatial = dict(alpha_left=float(a_left), alpha_right=float(a_right), **losses(u_map, v_map, ref, h),
                   epe_px=epe(v_map), **info)
    gain = 1 - spatial["J"] / const["J"]
    print(f"\nbest constant alpha {const['alpha']:.3e}: J = {const['J']:.5e}, E = {const['energy']:.7e}")
    print(f"alpha map {a_left:.3e} (left) / {a_right:.3e} (right): J = {spatial['J']:.5e}, "
          f"E = {spatial['energy']:.7e}, gain {100 * gain:.2f}% "
          f"(outer={spatial['outer']}, converged={spatial['converged']})")
    print(f"sum of the per-half scan minima: {rows[best['J_left']]['J_left'] + rows[best['J_right']]['J_right']:.5e}")

    m = FRAMES // 2
    u_c, v_c, _ = scan[best["J"]]
    err_max = float(np.percentile(np.abs(np.stack((u_c, u_map)) - ref), 99.5))
    v_px = (v_map / h).mean(0)  # pixels per frame, mean over the intervals
    fig, ax = plt.subplots(3, 4, figsize=(16, 11.5), constrained_layout=True)
    panels = [
        (f[m], f"f, frame {m}", "gray", 0, 1),
        (ref[m], "clean reference", "gray", 0, 1),
        (sigma, "noise std", "viridis", None, None),
        (alpha_map, "alpha map (best per half)", "viridis", None, None),
        (u_c[m], f"u, constant alpha {const['alpha']:.2e}", "gray", 0, 1),
        (u_map[m], "u, alpha map", "gray", 0, 1),
        (np.abs(u_c - ref)[m], "|u - ref|, constant", "magma", 0, err_max),
        (np.abs(u_map - ref)[m], "|u - ref|, map", "magma", 0, err_max),
        (None, None, None, None, None), (None, None, None, None, None),
        (v_px[..., 0], "v_x (px/frame), map", "RdBu_r", -1, 1),
        (v_px[..., 1], "v_y (px/frame), map", "RdBu_r", -1, 1),
    ]
    for axis, (img, title, cmap, lo, hi) in zip(ax.flat, panels):
        if img is not None:
            im = axis.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            axis.set_title(title)
            axis.axis("off")
            fig.colorbar(im, ax=axis, shrink=.8)
    for k, color in (("J", "k"), ("J_left", "tab:blue"), ("J_right", "tab:red")):
        values = np.array([r[k] for r in rows])
        ax[2, 0].semilogx(ALPHAS, values / values.min(), "-o", color=color, ms=3, label=k)
        ax[2, 0].axvline(ALPHAS[best[k]], color=color, ls=":")
    ax[2, 0].set(xlabel="alpha", ylabel="loss / its minimum", ylim=(.99, 1.5), title="training loss, constant alpha")
    ax[2, 0].legend()
    ax[2, 1].semilogx(ALPHAS, [r["outer"] for r in rows], "k-o", ms=3)
    ax[2, 1].set(xlabel="alpha", ylabel="outer iterations", title="solver cost")

    out = a.output / (a.motion if a.v0 == "fit" else f"{a.motion}_v0-{a.v0}")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "toy.png", dpi=110)
    plt.close(fig)
    np.savez_compressed(out / "arrays.npz", f=f, reference=ref, sigma=sigma, alphas=ALPHAS, alpha_map=alpha_map,
                        v0=v0, u_constant=u_c, v_constant=v_c, u_map=u_map, v_map=v_map)
    (out / "summary.json").write_text(json.dumps(dict(
        resolution=n, frames=FRAMES, motion=a.motion, v0_mode=a.v0, shifts_px=shifts.tolist(), sigma=[a.sigma_left, a.sigma_right],
        seed=a.seed, beta=beta, solver=asdict(cfg), initialization=init, scan=rows, constant=const, spatial=spatial,
        relative_gain=gain), indent=2))
    print(f"outputs in {out}")


if __name__ == "__main__":
    main()
