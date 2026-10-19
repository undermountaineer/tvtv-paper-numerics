"""Original high-resolution phantom: bounded alpha learning and spatial quadtree.

The solver and loss are reused unchanged. Only converged candidates may select
an alpha or accept a split. All cells retain all frames and the original spacing.
"""
from dataclasses import replace
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from .checkpoint_cache import scientific_code_id
from .data import Sequence, add_noise, estimate_translation, load_sequence, save_sequence
from .dyadic import DyadicCell, assemble_weight_maps
from .minimal_phantom import region_masks
from .weighted_solver import WeightedTVTVConfig
from .workflow import Experiment, write_json


def block_average(reference, resolution):
    """Average clean pixels first; observation noise is added afterwards."""
    a = np.asarray(reference, dtype=float)
    if a.ndim != 3 or a.shape[1] != a.shape[2] or a.shape[1] % resolution:
        raise ValueError("Working resolution must divide the square master grid.")
    if not np.isfinite(a).all():
        raise ValueError("The clean master must be finite.")
    k = a.shape[1] // resolution
    return a.reshape(a.shape[0], resolution, k, resolution, k).mean((2, 4))


def children(cell):
    """Four spatial children, with no split of the time interval."""
    if min(cell.y1-cell.y0, cell.x1-cell.x0) < 2 or (cell.y1-cell.y0) % 2 or (cell.x1-cell.x0) % 2:
        raise ValueError("Cannot split these spatial bounds.")
    ym, xm = (cell.y0+cell.y1)//2, (cell.x0+cell.x1)//2
    return [DyadicCell(cell.level+1, cell.t0, cell.t1, ya, yb, xa, xb)
            for ya, yb in ((cell.y0, ym), (ym, cell.y1))
            for xa, xb in ((cell.x0, xm), (xm, cell.x1))]


def best(rows):
    if not rows or not all(r["eligible"] for r in rows):
        return None
    return min(rows, key=lambda r: (r["upper_loss"], r["alpha"]))


def split_accepted(parent_loss, child_losses, rho):
    return sum(child_losses) < rho * parent_loss


class Original1024(Experiment):
    """Reuse the reviewed cache, solve and inspection methods, with new inputs."""
    def __init__(self, root, config_path=None):
        self.root = Path(root).resolve()
        self.config_path = Path(config_path or self.root/"configs/synthetic_original_1024.json").resolve()
        self.settings = json.loads(self.config_path.read_text())
        ds, model, search, tree = (self.settings[k] for k in ("data", "model", "search", "refinement"))
        n = ds["resolution"]
        if type(n) is not int or n < 32 or n > 1024 or 1024 % n:
            raise ValueError("Use a working resolution dividing 1024, at least 32.")
        if type(tree["max_depth"]) is not int or not 0 <= tree["max_depth"] <= 4 or n//2**tree["max_depth"] < 8:
            raise ValueError("Use depth <=4 and cells at least 8 pixels wide.")
        if not 0 < tree["rho"] <= 1:
            raise ValueError("rho must be in (0,1].")
        if not np.isfinite(ds["noise_std"]) or ds["noise_std"] <= 0:
            raise ValueError("Noise standard deviation must be positive and finite.")
        seeds = [ds["design_seed"], *ds["validation_seeds"]]
        if len(seeds) < 2 or len(set(seeds)) != len(seeds):
            raise ValueError("Design and validation seeds must be distinct.")
        lo, hi = search["bounds"]
        if not 0 < lo < hi or not np.isfinite([lo, hi, model["beta"], model["kappa"]]).all() or min(model.values()) <= 0:
            raise ValueError("Use positive finite model parameters and alpha bounds.")
        if len(search["initial_alphas"]) < 3 or any(not lo <= a <= hi for a in search["initial_alphas"]):
            raise ValueError("Declare at least three initial alphas inside the bounds.")
        if not np.isfinite(search["expansion_factor"]) or search["expansion_factor"] <= 1:
            raise ValueError("Expansion factor must exceed one.")
        for key in ("max_expansions", "midpoint_rounds"):
            if type(search[key]) is not int or not 0 <= search[key] <= 4:
                raise ValueError("Search rounds must be integers between zero and four.")
        for cfg in self.settings["solver"].values():
            WeightedTVTVConfig(**cfg)
            if cfg["kappa"] != model["kappa"]:
                raise ValueError("Solver and model kappa must agree.")
        self.master = self.root/ds["master"]
        master_hash = hashlib.sha256(self.master.read_bytes()).hexdigest()
        if master_hash != ds["master_sha256"]:
            raise ValueError("Master checksum differs: use the original 1024 bundle or explicitly change the protocol.")
        source = Path(__file__).parent
        identity = dict(protocol="original_1024_quadtree_v1", settings=self.settings,
                        master_sha256=master_hash, solver_source=scientific_code_id(),
                        workflow_source=hashlib.sha256(b"".join((source/name).read_bytes() for name in
                            ("original_1024.py", "data.py", "minimal_phantom.py", "metrics.py", "workflow.py", "dyadic.py"))).hexdigest())
        self.token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
        self.base = self.root/"results/synthetic_original_1024"
        self.out = self.base/self.token
        self.cache = self.base/"cache"
        self.out.mkdir(parents=True, exist_ok=True)
        write_json(self.out/"protocol.json", identity)
        self._rows = {}

    @property
    def root_cell(self):
        return DyadicCell.root((5, self.settings["data"]["resolution"], self.settings["data"]["resolution"]))

    def prepare(self):
        ds = self.settings["data"]
        with np.load(self.master, allow_pickle=False) as z:
            master, times = z["reference"], z["times"]
            velocity = z["velocity_xy"]
            hx, hy, ht = (float(z[k]) for k in ("hx", "hy", "ht"))
        if master.shape != (5, 1024, 1024) or not np.array_equal(times, np.arange(5.)) or velocity.shape != (2,) or not np.isfinite(velocity).all():
            raise ValueError("Expected the original five-frame, 1024-square master.")
        if (hx, hy, ht) != (1/1024, 1/1024, 1.):
            raise ValueError("Master spacings differ from the original scene.")
        n = ds["resolution"]
        reference = block_average(master, n)
        c = (np.arange(n)+.5)/n
        masks = np.stack([region_masks(c[None, :]-velocity[0]*t, c[:, None]-velocity[1]*t) for t in times])
        for seed in (ds["design_seed"], *ds["validation_seeds"]):
            path = self.input_path(seed)
            if not path.exists():
                noisy = add_noise(reference, ds["noise_std"], seed)
                data = Sequence(reference, noisy, times, 1/n, 1/n, ht,
                                np.broadcast_to(velocity, (4, n, n, 2)).copy(), masks)
                v0, diagnostic = estimate_translation(noisy, data.hx, data.hy, ht)
                save_sequence(path, data, noisy.copy(), v0)
                write_json(path.with_suffix(".initialization.json"), diagnostic)
        return self.report()

    def job(self, cell, alpha, mode="local", seed=None, leaves=None):
        return dict(stage="original_1024", seed=seed if seed is not None else self.settings["data"]["design_seed"],
                    cell=list(vars(cell).values()), alpha=alpha, mode=mode, leaves=leaves)

    def problem(self, job):
        data, u0, v0 = load_sequence(self.input_path(job["seed"]))
        cell = DyadicCell(*job["cell"])
        sl, vs = cell.u_slices, cell.v_slices
        data = replace(data, reference=data.reference[sl], noisy=data.noisy[sl],
                       true_velocity=data.true_velocity[vs],
                       region_masks=data.region_masks[:, :, cell.y0:cell.y1, cell.x0:cell.x1])
        u0, v0 = u0[sl], v0[vs]
        beta = self.settings["model"]["beta"]
        if job["alpha"] is None:
            alpha, b = assemble_weight_maps(data.noisy.shape,
                {DyadicCell(*leaf["cell"]): (leaf["alpha"], beta) for leaf in job["leaves"]})
        else:
            alpha, b = np.full(data.noisy.shape, job["alpha"]), np.full(v0.shape[:-1], beta)
        cfg = WeightedTVTVConfig(**dict(self.settings["solver"][job["mode"]], hx=data.hx, hy=data.hy, ht=data.ht))
        return data, u0, v0, alpha, b, cfg

    def row(self, job):
        key = json.dumps(job, sort_keys=True)
        if key not in self._rows:
            self._rows[key] = self.inspect(job)
        return self._rows[key]

    def search(self, cell, inherited=None):
        cfg = self.settings["search"]
        alphas = set(cfg["initial_alphas"])
        if inherited is not None:
            alphas.add(inherited)
        rows = []
        def evaluate():
            nonlocal rows
            rows = [self.row(self.job(cell, a)) for a in sorted(alphas)]
            return best(rows)
        chosen = None
        expansions = 0
        while True:
            chosen = evaluate()
            if chosen is None:
                return dict(complete=False, rows=rows, pending=[r for r in rows if not r["eligible"]])
            ordered = sorted(alphas)
            a = chosen["alpha"]
            if a not in (ordered[0], ordered[-1]):
                break
            proposal = max(cfg["bounds"][0], a/cfg["expansion_factor"]) if a == ordered[0] else min(cfg["bounds"][1], a*cfg["expansion_factor"])
            if expansions >= cfg["max_expansions"] or proposal in alphas:
                return dict(complete=True, boundary_limited=True, selected=chosen, rows=rows, pending=[])
            alphas.add(proposal)
            expansions += 1
        for _ in range(cfg["midpoint_rounds"]):
            ordered = sorted(alphas)
            i = ordered.index(chosen["alpha"])
            alphas.update((float(np.sqrt(ordered[i-1]*ordered[i])), float(np.sqrt(ordered[i]*ordered[i+1]))))
            chosen = evaluate()
            if chosen is None:
                return dict(complete=False, rows=rows, pending=[r for r in rows if not r["eligible"]])
        return dict(complete=True, boundary_limited=False, selected=chosen, rows=rows, pending=[])

    def tree(self):
        decisions, leaves = [], []
        root = self.search(self.root_cell)
        if not root["complete"]:
            return dict(complete=False, pending=root["pending"], root=root, leaves=[], decisions=[])
        if root["boundary_limited"]:
            return dict(complete=False, pending=[], blocked="Root alpha is at the search boundary; extend the range before refinement.", root=root, leaves=[], decisions=[])
        cfg = self.settings["refinement"]
        def visit(cell, learned):
            chosen = learned["selected"]
            leaf = dict(cell=list(vars(cell).values()), alpha=chosen["alpha"])
            if cell.level >= cfg["max_depth"]:
                leaves.append(leaf)
                return None
            child_cells = children(cell)
            learned_children = [self.search(c, chosen["alpha"]) for c in child_cells]
            pending = [r for s in learned_children for r in s["pending"]]
            if pending:
                return dict(pending=pending)
            if any(s["boundary_limited"] for s in learned_children):
                return dict(pending=[], blocked="A child alpha is at the search boundary; extend the range before accepting a split.")
            costs = [s["selected"]["upper_loss"] for s in learned_children]
            accepted = split_accepted(chosen["upper_loss"], costs, cfg["rho"])
            decisions.append(dict(cell=leaf["cell"], parent_alpha=chosen["alpha"], parent_loss=chosen["upper_loss"],
                                  children_loss=sum(costs), accepted=accepted,
                                  children=[dict(cell=list(vars(c).values()), alpha=s["selected"]["alpha"]) for c, s in zip(child_cells, learned_children)]))
            if not accepted:
                leaves.append(leaf)
                return None
            for c, s in zip(child_cells, learned_children):
                stopped = visit(c, s)
                if stopped:
                    return stopped
            return None
        stopped = visit(self.root_cell, root)
        return dict(complete=stopped is None, pending=(stopped or {}).get("pending", []),
                    blocked=(stopped or {}).get("blocked"), root=root, leaves=leaves, decisions=decisions)

    def plan(self, stage):
        if not self.input_path(self.settings["data"]["design_seed"]).exists():
            raise FileNotFoundError("Run prepare first.")
        self._rows = {}
        if stage == "baseline":
            state = self.search(self.root_cell)
            pending = state["pending"]
        else:
            state = self.tree()
            if stage == "refine":
                pending = state["pending"]
            elif stage in ("compare", "validate"):
                if not state["complete"]:
                    raise ValueError("Complete baseline and refine before comparing or validating.")
                alpha = state["root"]["selected"]["alpha"]
                seeds = [self.settings["data"]["design_seed"]] if stage == "compare" else self.settings["data"]["validation_seeds"]
                jobs = [self.job(self.root_cell, a, "tighter", seed, state["leaves"] if a is None else None)
                        for seed in seeds for a in (alpha, None)]
                # A rejected first split can still be checked globally as a labelled diagnostic.
                if stage == "compare" and state["decisions"] and not state["decisions"][0]["accepted"]:
                    jobs.append(self.job(self.root_cell, None, "tighter", seeds[0], state["decisions"][0]["children"]))
                pending = [self.row(j) for j in jobs if not self.row(j)["eligible"]]
            else:
                raise ValueError("Unknown stage.")
        write_json(self.out/"learning.json", state)
        return dict(output_directory=str(self.out), complete=not pending and not state.get("blocked"),
                    blocked=state.get("blocked"), jobs=pending)

    def report(self, plots=False):
        self._rows = {}
        if not self.input_path(self.settings["data"]["design_seed"]).exists():
            raise FileNotFoundError("Run prepare first.")
        state = self.tree()
        comparisons = []
        if state["complete"]:
            alpha = state["root"]["selected"]["alpha"]
            for seed in (self.settings["data"]["design_seed"], *self.settings["data"]["validation_seeds"]):
                constant = self.row(self.job(self.root_cell, alpha, "tighter", seed))
                spatial = self.row(self.job(self.root_cell, None, "tighter", seed, state["leaves"]))
                pair = dict(seed=seed, complete=constant["eligible"] and spatial["eligible"], constant=constant, spatial=spatial)
                if pair["complete"]:
                    pair.update(relative_gain=1-spatial["upper_loss"]/constant["upper_loss"],
                                RMSE_relative_gain=1-spatial["RMSE"]/constant["RMSE"])
                comparisons.append(pair)
        result = dict(output_directory=str(self.out), token=self.token, learning=state, comparisons=comparisons,
                      rows=list(self._rows.values()), note="Alpha is selected on the design seed only. Fresh-noise comparisons use the frozen constant and partition. A finite search does not certify a global alpha optimum.")
        if state["complete"] and state["decisions"] and not state["decisions"][0]["accepted"]:
            row = self.row(self.job(self.root_cell, None, "tighter", leaves=state["decisions"][0]["children"]))
            result["rejected_split_global_diagnostic"] = row
            result["rows"] = list(self._rows.values())
        write_json(self.out/"summary.json", result)
        write_json(self.out/"learning.json", state)
        write_json(self.base/"latest.json", dict(output_directory=str(self.out)))
        with (self.out/"candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("seed", "cell", "alpha", "mode", "eligible", "status", "upper_loss", "RMSE", "texture_RMSE", "outer_iterations", "cumulative_solver_seconds", "key"), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(result["rows"])
        return result

    def export(self):
        report = self.report()
        path = self.out/"ORIGINAL_1024_RESULTS.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in self.out.rglob("*"):
                if p.is_file() and p.suffix not in (".zip", ".tmp"):
                    z.write(p, str(p.relative_to(self.out)))
            for key in {r["key"] for r in report["rows"]}:
                for name in ("result.npz", "checkpoint.npz", "progress.json"):
                    p = self.cache/key/name
                    if p.exists():
                        z.write(p, f"cache/{key}/{name}")
            z.write(self.config_path, "config.json")
            for p in (self.root/"src/tvtv_numerics").glob("*.py"):
                z.write(p, "source/"+p.name)
        return dict(archive=str(path))
