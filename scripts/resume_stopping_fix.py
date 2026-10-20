"""Explicitly continue verified pre-fix checkpoints under the corrected policy.

Old outputs are retained. Imported checkpoints are never marked converged;
two new successful outer iterations are required by the corrected solver.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from tvtv_numerics.checkpoint_cache import (
    _entry_lock, _read_npz, _save_npz, scientific_code_id,
)
from tvtv_numerics.original_1024 import Original1024
from tvtv_numerics.weighted_solver import _checkpoint_problem_id
from tvtv_numerics.weighted_energy import manuscript_energy
from tvtv_numerics.weights import forward_stencil_lower_envelope
from tvtv_numerics.workflow import write_json

OLD_SOURCE = "93dd98ba7858110582c7cb0bbc25c126e0092a9134bdf45897a30addc6d65f9d"
NEW_SOURCE = "c497ffd9c6f0483235219f59f9dd14123616c55b63471d8b2d2da197367b8dfb"
JOB_FIELDS = ("stage", "seed", "cell", "alpha", "mode", "leaves")


def same_npz(left, right):
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        return set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)


def continue_checkpoints(root, config_path):
    root, config_path = Path(root).resolve(), Path(config_path).resolve()
    if scientific_code_id() != NEW_SOURCE:
        raise ValueError("This continuation is only supported for the exact reviewed stopping-fix source.")
    original = json.loads(config_path.read_text())
    base = root / "results/synthetic_original_1024"
    candidates = []
    for path in base.glob("*/protocol.json"):
        protocol = json.loads(path.read_text())
        if protocol.get("settings") == original and protocol.get("solver_source") == OLD_SOURCE:
            candidates.append((path.parent, protocol))
    if len(candidates) != 1:
        raise ValueError("Expected exactly one old performance-pass protocol matching this config. No checkpoint imported.")
    old_out, old_protocol = candidates[0]
    settings = copy.deepcopy(original)
    for cfg in settings["solver"].values():
        # This is execution headroom for continuing a checkpoint already at
        # outer=128. Stopping accuracies and update parameters are unchanged.
        cfg["max_outer_iterations"] = max(256, cfg["max_outer_iterations"])
    new_config = config_path.with_name(config_path.stem + "_stopping_fixed.json")
    if new_config.exists() and json.loads(new_config.read_text()) != settings:
        raise ValueError(f"Refusing to replace a different configuration: {new_config}")
    write_json(new_config, settings)
    new = Original1024(root, new_config)
    new_protocol = json.loads((new.out / "protocol.json").read_text())
    for field in ("workflow_source", "master_sha256"):
        if new_protocol[field] != old_protocol[field]:
            raise ValueError(f"Incompatible {field}; no checkpoint imported.")
    seeds = [settings["data"]["design_seed"], *settings["data"]["validation_seeds"]]
    for seed in seeds:
        source = old_out / "inputs" / f"seed-{seed}.npz"
        target = new.input_path(seed)
        if not source.is_file():
            raise ValueError(f"Missing exact old observations: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not same_npz(source, target):
                raise ValueError("Existing new inputs differ from the original observations; refusing migration.")
        else:
            temp = target.with_suffix(".migration.tmp")
            shutil.copy2(source, temp)
            temp.replace(target)
        diagnostic = source.with_suffix(".initialization.json")
        if diagnostic.is_file() and not target.with_suffix(".initialization.json").exists():
            shutil.copy2(diagnostic, target.with_suffix(".initialization.json"))
    summary = json.loads((old_out / "summary.json").read_text())
    if Path(summary["output_directory"]).name != old_out.name:
        raise ValueError("Old summary does not identify the selected protocol.")
    manifest_path = new.out / "STOPPING_FIX_CONTINUATION.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "old_output": str(old_out), "old_solver_source": OLD_SOURCE,
        "new_solver_source": scientific_code_id(), "new_config": str(new_config),
        "policy": "Old primal/dual states continued explicitly; old convergence counters reset. Completed results are not imported. Tighter global comparisons start fresh.",
        "imports": [],
    }
    seen = {r["new_key"] for r in manifest["imports"]}
    for row in summary["rows"]:
        # Continue only local training solves. Final tighter comparisons must
        # use the same corrected policy from the original initialization.
        if row["mode"] != "local" or row["status"] == "not_run":
            continue
        job = {key: row[key] for key in JOB_FIELDS}
        key, identity = new.key(job)
        expected_old = copy.deepcopy(identity)
        expected_old["source"] = OLD_SOURCE
        old_key = hashlib.sha256(json.dumps(expected_old, sort_keys=True).encode()).hexdigest()
        if old_key != row["key"]:
            raise ValueError("Old candidate/input identity differs; refusing checkpoint import.")
        path = new.cache / old_key / "checkpoint.npz"
        if not path.exists():
            print("No complete primal/dual checkpoint; this candidate will run fresh:", old_key[:12])
            continue
        arrays, metadata = _read_npz(path)
        if metadata["identity"] != expected_old:
            raise ValueError("Old checkpoint source/settings/input identity differs.")
        state = metadata["state"]
        data, u0, v0, alpha, beta, cfg = new.problem(job)
        if state["schema"] != 1 or state["problem_id"] != _checkpoint_problem_id(data.noisy, alpha, beta, u0, v0, cfg):
            raise ValueError("Old checkpoint problem/state protocol differs.")
        outer = state["outer_iterations"]
        if (type(outer) is not int or outer < 1 or len(state["history"]) != outer + 1
                or int(state["history"][-1]["iteration"]) != outer or not metadata["has_duals"]):
            raise ValueError("Incomplete or inconsistent old primal/dual checkpoint.")
        shapes = dict(u=u0.shape, v=v0.shape, u_dual_tv=(*u0.shape, 2),
                      u_dual_transport=v0.shape[:-1], v_dual_tv=(*v0.shape, 2))
        if set(arrays) != set(shapes) or any(arrays[k].shape != s or not np.isfinite(arrays[k]).all() for k, s in shapes.items()):
            raise ValueError("Checkpoint arrays have invalid shapes or nonfinite values.")
        a_radius = forward_stencil_lower_envelope(alpha)
        b_radius = forward_stencil_lower_envelope(beta)/cfg.kappa
        if (np.any(np.linalg.norm(arrays["u_dual_tv"], axis=-1) > a_radius*(1+1e-10)+1e-14)
                or np.any(np.abs(arrays["u_dual_transport"]) > cfg.kappa*(1+1e-10)+1e-14)
                or np.any(np.linalg.norm(arrays["v_dual_tv"], axis=(-2,-1)) > b_radius*(1+1e-10)+1e-14)):
            raise ValueError("Old dual checkpoint is outside the unchanged feasible sets.")
        energy = manuscript_energy(arrays["u"], arrays["v"], data.noisy,
            alpha, beta, cfg.kappa, cfg.ht, cfg.hx, cfg.hy)["total"]
        if not np.isclose(energy, state["previous_total"], rtol=1e-10, atol=1e-12):
            raise ValueError("Old checkpoint arrays and recorded energy disagree.")
        if outer + cfg.required_outer_successes > cfg.max_outer_iterations:
            raise ValueError("Not enough new outer-iteration headroom to recheck this checkpoint.")
        target = new.cache / key
        target.mkdir(parents=True, exist_ok=True)
        with _entry_lock(target / "entry.lock"):
            if (target / "checkpoint.npz").exists() or (target / "result.npz").exists():
                for name in ("checkpoint.npz", "result.npz"):
                    if (target / name).exists() and _read_npz(target / name)[1]["identity"] != identity:
                        raise ValueError("Existing corrected-policy cache identity differs.")
                print("Retained existing corrected-policy progress:", key[:12])
                continue
            changed = copy.deepcopy(metadata)
            changed["identity"] = identity
            changed["state"]["consecutive_successes"] = 0
            changed["provenance"] = {"old_key": old_key, "old_solver_source": OLD_SOURCE,
                                      "policy_changed_after_outer": outer}
            _save_npz(target / "checkpoint.npz", arrays, changed)
            write_json(target / "progress.json", dict(key=key, outer_iterations=outer,
                converged=False, cumulative_solver_seconds=metadata["cumulative_solver_seconds"]))
        if key not in seen:
            manifest["imports"].append(dict(old_key=old_key, new_key=key, outer=outer,
                alpha=job["alpha"], seed=job["seed"], cell=job["cell"],
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                old_completed=bool(row["eligible"])))
            seen.add(key)
            write_json(manifest_path, manifest)
        print(f"Continued alpha={job['alpha']}: outer={outer}; fresh convergence checks required.")
    write_json(manifest_path, manifest)
    new.report()
    print("\nResume configuration:", new_config)
    print("Previous outputs retained:", old_out)
    print("Corrected-policy output:", new.out)
    return new_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/meeting_256.json"))
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    continue_checkpoints(root, args.config.resolve())


if __name__ == "__main__":
    main()
