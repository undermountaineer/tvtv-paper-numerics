"""One experiment: local alpha selection, global comparison, fresh-noise checks.

All heavy work is called by scripts/run.py in isolated processes. Reporting
reads saved arrays; it never starts a solve or ranks an unfinished candidate.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .checkpoint_cache import ResumableSolverCache, scientific_code_id, solve_identity
from .data import Sequence, add_noise, estimate_translation, load_sequence, make_sequence, save_sequence
from .dyadic import DyadicCell, assemble_weight_maps
from .metrics import endpoint_error, psnr, rmse, upper_level_loss
from .weighted_solver import WeightedTVTVConfig, WeightedTVTVResult

CELL_NAMES = ("top_left", "top_right", "bottom_left", "bottom_right")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(path)


def spatial_children(shape):
    """Four spatial children, each containing the entire time interval."""
    cell = DyadicCell.root(shape)
    if shape[1] % 2 or shape[2] % 2:
        raise ValueError("Spatial dimensions must be even.")
    ym, xm = shape[1]//2, shape[2]//2
    bounds = ((0, ym, 0, xm), (0, ym, xm, shape[2]),
              (ym, shape[1], 0, xm), (ym, shape[1], xm, shape[2]))
    return {name: DyadicCell(1, 0, cell.t1, *bound) for name, bound in zip(CELL_NAMES, bounds)}


def image_loss(data, u, v):
    return upper_level_loss(u, v, data.reference, None, gamma_u=1., gamma_v=0.,
        image_time_weights=DyadicCell.root(data.noisy.shape).image_time_weights(),
        cell_volume=data.hx*data.hy*data.ht)["total"]


def select_alphas(rows):
    """Resolve the whole declared shortlist before selecting any child."""
    if not rows or not all(row["eligible"] for row in rows):
        return None
    return {name: min((r for r in rows if r["cell"] == name),
                      key=lambda r: (r["upper_loss"], r["alpha"]))["alpha"] for name in CELL_NAMES}


class Experiment:
    def __init__(self, root, config_path=None):
        self.root = Path(root).resolve()
        self.config_path = Path(config_path or self.root/"configs/proof_of_concept.json").resolve()
        self.settings = json.loads(self.config_path.read_text(encoding="utf-8"))
        model = self.settings["model"]
        if set(model["frozen_map"]) != set(CELL_NAMES) or not model["alpha_shortlist"]:
            raise ValueError("Declare all four map cells and a nonempty alpha shortlist.")
        if model["parent_alpha"] not in model["alpha_shortlist"]:
            raise ValueError("Include the parent control in the local alpha shortlist.")
        seeds = self.settings["data"]["validation_seeds"]
        if not seeds or len(set(seeds)) != len(seeds) or self.settings["data"]["design_seed"] in seeds:
            raise ValueError("Validation seeds must be nonempty, distinct and separate from the design seed.")
        if any(not np.isfinite(a) or a <= 0 for a in
               [*model["alpha_shortlist"], *model["frozen_map"].values(), model["beta"], model["parent_alpha"]]):
            raise ValueError("All declared parameters must be positive and finite.")
        for settings in self.settings["solver"].values():
            WeightedTVTVConfig(**settings)
            if settings["kappa"] != model["kappa"]:
                raise ValueError("Model kappa and solver kappa must agree.")
        source = Path(__file__).parent
        identity = dict(protocol=1, settings=self.settings, solver_source=scientific_code_id(),
                        data_source=hashlib.sha256((source/"data.py").read_bytes() +
                                                  (source/"minimal_phantom.py").read_bytes()).hexdigest(),
                        scoring_source=hashlib.sha256((source/"metrics.py").read_bytes()).hexdigest())
        self.token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.out = self.root/"results/proof_of_concept"/self.token
        self.cache = self.root/"results/proof_of_concept/cache"
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(self.out/"protocol.json", identity)

    def input_path(self, seed):
        return self.out/"inputs"/f"seed-{seed}.npz"

    def prepare(self):
        settings = self.settings["data"]
        seed = settings["design_seed"]
        design_path = self.input_path(seed)
        if not design_path.is_file():
            data = make_sequence(settings["resolution"], settings["frames"], settings["supersample"],
                                 settings["noise_std"], seed)
            v0, _ = estimate_translation(data.noisy, data.hx, data.hy, data.ht)
            save_sequence(design_path, data, data.noisy.copy(), v0)
        design, _, _ = load_sequence(design_path)
        for seed in settings["validation_seeds"]:
            path = self.input_path(seed)
            if not path.is_file():
                noisy = add_noise(design.reference, settings["noise_std"], seed)
                data = replace(design, noisy=noisy)
                v0, _ = estimate_translation(noisy, data.hx, data.hy, data.ht)
                save_sequence(path, data, noisy.copy(), v0)
        return self.report(plots=False)

    def jobs(self, stage):
        model = self.settings["model"]
        seed = self.settings["data"]["design_seed"]
        if stage == "learn":
            return [dict(stage=stage, seed=seed, cell=None, alpha=model["parent_alpha"], mode="nominal")] + [
                dict(stage=stage, seed=seed, cell=name, alpha=a, mode="local")
                for name in CELL_NAMES for a in model["alpha_shortlist"]]
        if stage == "compare":
            return [dict(stage=stage, seed=seed, cell=None, alpha=a, mode="tighter")
                    for a in (model["parent_alpha"], None)]
        if stage == "validate":
            return [dict(stage=stage, seed=seed, cell=None, alpha=a, mode="tighter")
                    for seed in self.settings["data"]["validation_seeds"]
                    for a in (*model["alpha_shortlist"], None)]
        raise ValueError("Unknown solve stage.")

    def problem(self, job):
        data, u0, v0 = load_sequence(self.input_path(job["seed"]))
        cell_name = job["cell"]
        if cell_name:
            cell = spatial_children(data.noisy.shape)[cell_name]
            sl, vs = cell.u_slices, cell.v_slices
            data = replace(data, reference=data.reference[sl], noisy=data.noisy[sl],
                           true_velocity=data.true_velocity[vs],
                           region_masks=data.region_masks[:, :, cell.y0:cell.y1, cell.x0:cell.x1])
            u0, v0 = u0[sl], v0[vs]
        beta = self.settings["model"]["beta"]
        if job["alpha"] is None:
            cells = spatial_children(data.noisy.shape)
            alpha, beta_map = assemble_weight_maps(data.noisy.shape,
                {cell: (self.settings["model"]["frozen_map"][name], beta) for name, cell in cells.items()})
        else:
            alpha = np.full(data.noisy.shape, job["alpha"])
            beta_map = np.full(v0.shape[:-1], beta)
        cfg = WeightedTVTVConfig(**dict(self.settings["solver"][job["mode"]],
                                        hx=data.hx, hy=data.hy, ht=data.ht))
        return data, u0, v0, alpha, beta_map, cfg

    def key(self, job):
        data, u0, v0, a, b, cfg = self.problem(job)
        return solve_identity(data.noisy, a, b, cfg, u0, v0)

    def inspect(self, job, with_arrays=False):
        key, identity = self.key(job)
        row = dict(job, key=key, eligible=False, status="not_run", upper_loss=None)
        entry = self.cache/key
        path = entry/"result.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as z:
                meta = json.loads(str(z["metadata_json"]))
                complete = meta["result"]["converged"]
            if not complete and (entry/"checkpoint.npz").is_file():
                path = entry/"checkpoint.npz"
        else:
            path = entry/"checkpoint.npz"
        if not path.is_file():
            return (row, None) if with_arrays else row
        with np.load(path, allow_pickle=False) as z:
            u, v = z["u"].copy(), z["v"].copy()
            meta = json.loads(str(z["metadata_json"]))
        if meta["identity"] != identity:
            raise ValueError("Saved candidate has incompatible provenance.")
        data, _, _, _, _, cfg = self.problem(job)
        if path.name == "checkpoint.npz":
            fields = meta["state"]
            eligible = False
            stop = "maximum_outer_iterations" if fields["outer_iterations"] >= cfg.max_outer_iterations else "checkpoint_stop"
        else:
            fields = meta["result"]
            eligible = bool(fields["converged"] and fields["stop_reason"] == "converged")
            stop = fields["stop_reason"]
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError("Nonfinite reconstruction cannot be reported.")
        loss = image_loss(data, u, v)
        final = fields["history"][-1]
        eligible = eligible and np.isfinite(loss) and np.isfinite(final["total"])
        row.update(eligible=bool(eligible), status="completed" if eligible else stop,
                   upper_loss=loss if eligible else None, partial_upper_loss=None if eligible else loss,
                   RMSE=rmse(u, data.reference), PSNR=psnr(u, data.reference),
                   velocity_endpoint_error_pixels=endpoint_error(v*data.ht/np.array([data.hx, data.hy]),
                                                                 data.true_velocity*data.ht/np.array([data.hx, data.hy])),
                   outer_iterations=fields["outer_iterations"], lower_energy=final["total"],
                   cumulative_solver_seconds=meta["cumulative_solver_seconds"], final_diagnostics=final)
        texture = data.region_masks[:, 2]
        if np.any(texture):
            row["texture_RMSE"] = rmse(u[texture], data.reference[texture])
        return (row, (u, v)) if with_arrays else row

    def solve(self, job, seconds):
        data, u0, v0, a, b, cfg = self.problem(job)
        cache = ResumableSolverCache(self.cache)
        cache.run(data.noisy, a, b, cfg, u0, v0, seconds=seconds)
        return self.inspect(job)

    def report(self, plots=True):
        all_jobs = [job for stage in ("learn", "compare", "validate") for job in self.jobs(stage)]
        # A report before preparation gives a helpful error rather than an implicit solve.
        if any(not self.input_path(j["seed"]).is_file() for j in all_jobs):
            raise FileNotFoundError("Inputs are missing. Run prepare or import-handoff first.")
        rows = [self.inspect(job) for job in all_jobs]
        local = [r for r in rows if r["stage"] == "learn" and r["cell"]]
        parent = next(r for r in rows if r["stage"] == "learn" and not r["cell"])
        selected = select_alphas(local)
        learning = dict(complete=selected is not None and parent["eligible"], selected_alphas=selected,
                        strict_split_accepted=None, frozen_map=self.settings["model"]["frozen_map"],
                        frozen_map_matches_selection=selected == self.settings["model"]["frozen_map"] if selected else None)
        if learning["complete"]:
            selected_cost = sum(next(r["upper_loss"] for r in local if r["cell"] == name and r["alpha"] == alpha)
                                for name, alpha in selected.items())
            anchor_cost = sum(r["upper_loss"] for r in local if r["alpha"] == self.settings["model"]["parent_alpha"])
            learning.update(parent_global_loss=parent["upper_loss"], children_parent_alpha_loss=anchor_cost,
                            children_selected_loss=selected_cost,
                            strict_split_accepted=selected_cost < parent["upper_loss"],
                            note="Independent crop costs include crop-boundary effects. Global gain is assessed separately.")
        comparisons = []
        for stage in ("compare", "validate"):
            group = [r for r in rows if r["stage"] == stage]
            for seed in sorted({r["seed"] for r in group}):
                current = [r for r in group if r["seed"] == seed]
                pair = dict(stage=stage, seed=seed, complete=all(r["eligible"] for r in current))
                if pair["complete"]:
                    constant = min((r for r in current if r["alpha"] is not None), key=lambda r: r["upper_loss"])
                    spatial = next(r for r in current if r["alpha"] is None)
                    pair.update(constant_alpha=constant["alpha"], constant_loss=constant["upper_loss"],
                                spatial_loss=spatial["upper_loss"], relative_gain=1-spatial["upper_loss"]/constant["upper_loss"],
                                RMSE_relative_gain=1-spatial["RMSE"]/constant["RMSE"])
                comparisons.append(pair)
        validation = [p for p in comparisons if p["stage"] == "validate"]
        result = dict(output_directory=str(self.out), token=self.token, rows=rows, learning=learning,
                      comparisons=comparisons, completed_solves=sum(r["eligible"] for r in rows), planned_solves=len(rows),
                      validation_complete=all(p["complete"] for p in validation))
        if validation and result["validation_complete"]:
            result.update(spatial_wins=sum(p["relative_gain"] > 0 for p in validation),
                          mean_relative_gain=float(np.mean([p["relative_gain"] for p in validation])))
        write_json(self.out/"summary.json", result)
        write_json(self.root/"results/proof_of_concept/latest.json", dict(output_directory=str(self.out)))
        columns = ("stage", "seed", "cell", "alpha", "mode", "eligible", "status", "upper_loss", "partial_upper_loss",
                   "RMSE", "PSNR", "texture_RMSE", "outer_iterations", "cumulative_solver_seconds", "key")
        with (self.out/"metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        if plots:
            from .figures import save_figures
            save_figures(self, result)
        return result

    def plan(self, stage):
        result = self.prepare()
        return dict(output_directory=str(self.out), jobs=[dict(job=job, **self.inspect(job)) for job in self.jobs(stage)],
                    completed_solves=result["completed_solves"], planned_solves=result["planned_solves"])

    def export(self):
        result = self.report()
        path = self.out/"POC_RESULTS.zip"
        files = [p for p in self.out.rglob("*") if p.is_file() and p.suffix not in (".zip", ".tmp")]
        files.extend(p for p in (self.root/"src/tvtv_numerics").glob("*.py"))
        files.extend((self.config_path, self.root/"scripts/run.py"))
        for row in result["rows"]:
            files.extend(p for p in (self.cache/row["key"]).glob("*.npz"))
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(set(files)):
                archive.write(file, file.relative_to(self.root))
        return dict(archive=str(path), output_directory=str(self.out))


def smoke(root):
    """Small constant-sequence fixed-point and checkpoint/resume check."""
    f = np.full((3, 7, 8), .4)
    cfg = WeightedTVTVConfig(step_policy="balanced", residual_mode="scaled", update_order="u_then_v",
                            max_outer_iterations=4, max_inner_u_iterations=40, max_inner_v_iterations=40,
                            min_inner_iterations=2, norm_iterations=6)
    a, b = np.full(f.shape, .001), np.full((2, 7, 8), .0125)
    cache = ResumableSolverCache(Path(root)/"results/proof_of_concept/smoke-cache")
    cache.run(f, a, b, cfg, outer_iteration_budget=1)
    result = cache.run(f, a, b, cfg)
    if not result.converged or not np.allclose(result.u, f, atol=1e-12) or not np.allclose(result.v, 0, atol=1e-12):
        raise RuntimeError("Smoke check failed.")
    return dict(smoke_passed=True, converged=result.converged)
