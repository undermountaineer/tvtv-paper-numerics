#!/usr/bin/env python3
"""Time one declared TV-TV solve; partial runs are never labelled converged.

Run each case in a fresh process on the same machine. This measures solver
work including lightweight progress reporting, not the workflow's full-array
checkpoint writes, cache lookups, or parameter learning.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
from types import ModuleType

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, choices=(128, 256, 512), required=True)
    parser.add_argument("--case", choices=("original", "optimized", "interval5", "interval10"), required=True)
    parser.add_argument("--baseline-dir", type=Path, help="Directory containing original proximal.py and weighted_solver.py, or the original repo root.")
    parser.add_argument("--mode", choices=("local", "tighter"), default="local")
    parser.add_argument("--alpha", type=float, default=0.000875)
    parser.add_argument("--seconds", type=float, default=1200.)
    parser.add_argument("--fixed-inner", type=int, help="Fixed-work diagnostic: one outer update, this many iterations in each block.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite([args.alpha, args.seconds]).all() or min(args.alpha, args.seconds) <= 0:
        parser.error("alpha and seconds must be positive and finite")
    if args.fixed_inner is not None and args.fixed_inner < 1:
        parser.error("fixed-inner must be positive")

    root = Path(__file__).resolve().parents[1]
    current_source = root / "src/tvtv_numerics"
    sys.path.insert(0, str(root / "src"))
    from tvtv_numerics.data import add_noise, estimate_translation
    from tvtv_numerics.original_1024 import block_average

    if args.case == "original":
        if args.baseline_dir is None:
            parser.error("original requires --baseline-dir")
        baseline = args.baseline_dir.resolve()
        if (baseline / "src/tvtv_numerics").is_dir():
            baseline = baseline / "src/tvtv_numerics"
        if not all((baseline / n).is_file() for n in ("proximal.py", "weighted_solver.py")):
            parser.error("baseline-dir must contain both original source files")
        package = ModuleType("_performance_original")
        package.__path__ = [str(baseline), str(current_source)]
        sys.modules[package.__name__] = package
        solver = importlib.import_module(package.__name__ + ".weighted_solver")
        source = baseline
    else:
        import tvtv_numerics.weighted_solver as solver
        source = current_source

    settings = json.loads((root / "configs/synthetic_original_1024.json").read_text())
    setup_start = time.perf_counter()
    master_path = root / settings["data"]["master"]
    if hashlib.sha256(master_path.read_bytes()).hexdigest() != settings["data"]["master_sha256"]:
        raise ValueError("Master checksum does not match the declared experiment")
    with np.load(master_path, allow_pickle=False) as saved:
        reference = block_average(saved["reference"], args.resolution)
        true_velocity = saved["velocity_xy"].copy()
    noisy = add_noise(reference, settings["data"]["noise_std"], settings["data"]["design_seed"])
    h = 1. / args.resolution
    initial_v, initialization = estimate_translation(noisy, h, h, 1.)
    alpha = np.full(noisy.shape, args.alpha)
    beta = np.full(initial_v.shape[:-1], settings["model"]["beta"])
    config = solver.WeightedTVTVConfig(**settings["solver"][args.mode])
    replacements = dict(hx=h, hy=h)
    if args.case != "original":
        replacements["inner_check_interval"] = {"optimized": 1, "interval5": 5, "interval10": 10}[args.case]
    if args.fixed_inner is not None:
        # Keep dense checks dense even in this diagnostic. Tiny tolerances
        # prevent a noise-bearing test problem from stopping before the cap.
        replacements.update(max_outer_iterations=1, min_inner_iterations=1,
                            max_inner_u_iterations=args.fixed_inner, max_inner_v_iterations=args.fixed_inner,
                            inner_tolerance=1e-25, inner_absolute_tolerance=1e-25)
    config = replace(config, **replacements)

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "full" if args.fixed_inner is None else f"fixed{args.fixed_inner}"
    name = f"{args.resolution}-{args.mode}-{args.case}-{suffix}"
    progress_path = args.output / (name + ".progress.json")
    metadata_path = args.output / (name + ".json")
    arrays_path = args.output / (name + ".npz")
    setup_seconds = time.perf_counter() - setup_start
    started = time.perf_counter()

    def progress(state):
        history = state["history"]
        payload = dict(outer_iterations=state["outer_iterations"], elapsed_seconds=time.perf_counter()-started,
                       last=history[-1])
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2))
        temporary.replace(progress_path)
        if state["outer_iterations"] % 10 == 0:
            print(f"{name}: outer={state['outer_iterations']}, elapsed={payload['elapsed_seconds']:.1f}s", flush=True)

    result = solver.solve_weighted_tvtv(noisy, alpha, beta, config, initial_u=noisy, initial_v=initial_v,
        checkpoint_callback=progress, should_stop=lambda: time.perf_counter()-started >= args.seconds)
    elapsed = time.perf_counter() - started
    weights = np.ones(noisy.shape[0]); weights[[0, -1]] = .5
    loss = float(h*h*np.sum(weights[:, None, None]*np.abs(result.u-reference)**1.5))
    data = dict(case=args.case, resolution=args.resolution, frames=noisy.shape[0], mode=args.mode,
                diagnostic_fixed_work=args.fixed_inner is not None, elapsed_seconds=elapsed,
                setup_seconds=setup_seconds, converged=result.converged, stop_reason=result.stop_reason,
                outer_iterations=result.outer_iterations,
                u_inner_iterations=sum(int(r["u_inner_iterations"]) for r in result.history),
                v_inner_iterations=sum(int(r["v_inner_iterations"]) for r in result.history),
                config=asdict(config), image_loss=loss,
                rmse=float(np.sqrt(np.mean((result.u-reference)**2))),
                motion_endpoint_error_pixels=float(np.mean(np.linalg.norm((result.v-true_velocity)/h, axis=-1))),
                source_hashes={n:hashlib.sha256((source/n).read_bytes()).hexdigest()
                               for n in ("proximal.py", "weighted_solver.py")},
                noisy_sha256=hashlib.sha256(noisy.tobytes()).hexdigest(),
                initial_v_sha256=hashlib.sha256(initial_v.tobytes()).hexdigest(),
                master_sha256=settings["data"]["master_sha256"], initialization=initialization,
                environment=dict(python=platform.python_version(), numpy=np.__version__,
                                 platform=platform.platform(), cpu_count=os.cpu_count(), threads=1),
                checkpoint_policy="lightweight progress only; no full-array disk checkpoints",
                history=result.history)
    np.savez(arrays_path, u=result.u, v=result.v)
    metadata_path.write_text(json.dumps(data, indent=2, allow_nan=False))
    print(json.dumps({k:v for k,v in data.items() if k not in ("history", "config", "environment", "initialization")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
