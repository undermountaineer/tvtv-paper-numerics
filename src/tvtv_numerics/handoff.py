"""Import completed solves from the supplied October 7 handoff without solving.

Only numerical inputs and compatible completed results are copied. Old scripts,
paths and checkpoints are unnecessary. Original bytes stay in the user's ZIP.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from .checkpoint_cache import SOURCE_FILES, ResumableSolverCache
from .data import Sequence, estimate_translation, make_sequence, save_sequence
from .weighted_solver import WeightedTVTVResult
from .workflow import write_json


def unpack_npz(content):
    with np.load(io.BytesIO(content), allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files if name != "metadata_json"}
        metadata = json.loads(str(z["metadata_json"])) if "metadata_json" in z.files else None
    return arrays, metadata


def unique_member(archive, suffix):
    names = [n for n in archive.namelist() if n.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"Expected one handoff member ending in {suffix}; found {len(names)}.")
    return names[0]


def verified_source(archive):
    """Require identical scientific code, allowing only LF/CRLF differences."""
    source = Path(__file__).parent
    raw = []
    for name in SOURCE_FILES:
        old = archive.read(unique_member(archive, "/tvtv_numerics/"+name))
        current = (source/name).read_bytes()
        if old.replace(b"\r\n", b"\n") != current.replace(b"\r\n", b"\n"):
            raise ValueError(f"Scientific source mismatch: {name}. No results adopted.")
        raw.append(old)
    return hashlib.sha256(b"".join(raw)).hexdigest()


def save_input_if_compatible(experiment, seed, data, u0, v0):
    """Prefer exact archived input bytes after checking the observation-only fit."""
    estimate, _ = estimate_translation(data.noisy, data.hx, data.hy, data.ht)
    if not np.array_equal(u0, data.noisy) or not np.allclose(v0, estimate, rtol=1e-11, atol=1e-14):
        raise ValueError("Handoff initialization is not the declared observation-only fit.")
    path = experiment.input_path(seed)
    if path.is_file():
        from .data import load_sequence
        existing, old_u0, old_v0 = load_sequence(path)
        if all(np.array_equal(a, b) for a, b in ((existing.noisy, data.noisy), (existing.reference, data.reference),
                                                (old_u0, u0), (old_v0, v0))):
            return
        # Never overwrite input arrays supporting another run's saved results.
        if experiment.cache.is_dir() and any(experiment.cache.glob("*/result.npz")):
            raise ValueError("Existing prepared inputs differ from archived bytes and the cache is populated. "
                             "Import into a fresh checkout or separate copy of the project.")
    save_sequence(path, data, u0, v0)


def import_handoff(experiment, path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with zipfile.ZipFile(path) as outer:
        expected = ("DFFI_STEP3_REPAIR_RESULTS.zip", "SPATIAL_NOISE_ROBUSTNESS_RESULTS.zip")
        archives = [zipfile.ZipFile(io.BytesIO(outer.read(name))) for name in expected]
    try:
        sources = [verified_source(archive) for archive in archives]
        fresh = archives[1]
        arrays, _ = unpack_npz(fresh.read(unique_member(fresh, "/phantom_v2.npz")))
        data = Sequence(**{k: arrays[k] for k in Sequence.__dataclass_fields__})
        for name in ("hx", "hy", "ht"):
            setattr(data, name, float(getattr(data, name)))
        settings = experiment.settings["data"]
        generated = make_sequence(settings["resolution"], settings["frames"], settings["supersample"],
                                  settings["noise_std"], settings["design_seed"])
        for name in ("reference", "noisy", "region_masks", "true_velocity", "times"):
            if not np.allclose(getattr(data, name), getattr(generated, name), rtol=1e-12, atol=1e-14):
                raise ValueError(f"The handoff {name} differs from the configured experiment.")
        state, _ = unpack_npz(fresh.read(unique_member(fresh, "/initial_state.npz")))
        save_input_if_compatible(experiment, settings["design_seed"], data, state["u0"], state["v0"])
        for seed in settings["validation_seeds"]:
            saved, _ = unpack_npz(fresh.read(unique_member(fresh, f"/data/seed-{seed}.npz")))
            from .data import add_noise
            if not np.allclose(saved["noisy"], add_noise(data.reference, settings["noise_std"], seed), rtol=1e-12, atol=1e-14):
                raise ValueError("Fresh-noise data do not match the declared seed.")
            save_input_if_compatible(experiment, seed, replace(data, noisy=saved["noisy"]), saved["u0"], saved["v0"])
        jobs = [job for stage in ("learn", "compare", "validate") for job in experiment.jobs(stage)]
        identities = {}
        for job in jobs:
            _, identity = experiment.key(job)
            match = json.dumps({k: v for k, v in identity.items() if k != "source"}, sort_keys=True)
            identities[match] = job
        cache = ResumableSolverCache(experiment.cache)
        adopted = []
        seen = set()
        for archive, source in zip(archives, sources):
            for name in archive.namelist():
                if "/resumable_solves/" not in name or not name.endswith("/result.npz"):
                    continue
                arrays, meta = unpack_npz(archive.read(name))
                if meta["identity"]["source"] != source:
                    raise ValueError("Archived result scientific identity does not match its supplied source.")
                match = json.dumps({k: v for k, v in meta["identity"].items() if k != "source"}, sort_keys=True)
                if match not in identities or match in seen:
                    continue
                fields = meta["result"]
                if not fields["converged"] or fields["stop_reason"] != "converged":
                    continue
                job = identities[match]
                data, u0, v0, a, b, cfg = experiment.problem(job)
                if arrays["u"].shape != data.noisy.shape or arrays["v"].shape != v0.shape:
                    raise ValueError("Archived result shapes are incompatible.")
                if len(fields["history"]) != fields["outer_iterations"]+1:
                    raise ValueError("Archived iteration history is inconsistent.")
                if not all(np.isfinite(row["total"]) for row in fields["history"]):
                    raise ValueError("Archived energy history is nonfinite.")
                result = WeightedTVTVResult(u=arrays["u"], v=arrays["v"], **fields)
                key = cache.adopt_completed(data.noisy, a, b, cfg, u0, v0, result,
                    meta["cumulative_solver_seconds"],
                    dict(imported_from=path.name, original_member=name, original_identity=meta["identity"],
                         original_result_sha256=hashlib.sha256(archive.read(name)).hexdigest(),
                         migration="identical code with LF/CRLF normalization; exact saved inputs"))
                seen.add(match)
                adopted.append(dict(job=job, key=key))
        write_json(experiment.out/"import.json", dict(archive=path.name, imported_completed=len(adopted),
                                                     adopted=adopted, new_solves=0))
        report = experiment.report()
        return dict(output_directory=str(experiment.out), imported_completed=len(adopted), new_solves=0,
                    completed_solves=report["completed_solves"], planned_solves=report["planned_solves"])
    finally:
        for archive in archives:
            archive.close()
