"""The reviewed phantom, reproducible noise and observation-only initialization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import expit

from . import minimal_phantom as phantom


@dataclass
class Sequence:
    reference: np.ndarray
    noisy: np.ndarray
    times: np.ndarray
    hx: float
    hy: float
    ht: float
    true_velocity: np.ndarray
    region_masks: np.ndarray


def make_sequence(resolution=128, frames=5, supersample=4, noise_std=.1, seed=20260930):
    """Render exactly the resolved phantom used in the October 7 handoff.

    Crops retain these spacings; they are never rescaled to a new unit square.
    Gaussian observations are not clipped. Ground truth is only for evaluation.
    """
    if resolution < 32 or resolution % 2 or frames < 3 or supersample < 1 or noise_std < 0:
        raise ValueError("Use an even resolution >=32, >=3 frames and valid sampling/noise.")
    n, ss = int(resolution), int(supersample)
    h = 1. / n
    times = np.arange(frames, dtype=float)
    velocity = phantom.VELOCITY_XY
    centres = (np.arange(n * ss) + .5) * h / ss
    reference = []
    masks = []
    for t in times:
        x = centres[None, :] - velocity[0] * t
        y = centres[:, None] - velocity[1] * t
        p, q = phantom.rotated(x, y, .480, .694, -15)
        dx, dy = np.abs(p) - (.347 - .026), np.abs(q) - (.126 - .026)
        slab = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)) + np.minimum(np.maximum(dx, dy), 0) - .026 <= 0
        slits = np.zeros(np.broadcast_shapes(x.shape, y.shape), bool)
        for c, w, length in ((-.247, .004, .057), (-.202, .006, .073), (-.151, .009, .087)):
            slits |= (np.abs(p-c) <= w) & (np.abs(q) <= length)
        envelope = expit((p-.035)/.013) * expit((.308-p)/.012) * expit((.103-np.abs(q))/.008)
        original = .047*np.cos(2*np.pi*(34*p+10*q+5*q*q)) + .033*np.cos(2*np.pi*(-13*p+39*q))
        resolved = .08*np.cos(2*np.pi*(14*p+4*q+2*q*q)) + .06*np.cos(2*np.pi*(-5*p+16*q))
        change = np.where(slab & ~slits, envelope * (resolved-original), 0)
        field = phantom.scene(x, y) + change
        reference.append(field.reshape(n, ss, n, ss).mean((1, 3)))
        c = (np.arange(n) + .5) * h
        masks.append(phantom.region_masks(c[None, :]-velocity[0]*t, c[:, None]-velocity[1]*t))
    reference = np.stack(reference)
    return Sequence(reference, add_noise(reference, noise_std, seed), times, h, h, 1.,
                    np.broadcast_to(velocity, (frames-1, n, n, 2)).copy(), np.stack(masks))


def add_noise(reference, sigma, seed):
    return np.asarray(reference, dtype=float) + sigma * np.random.default_rng(seed).normal(size=reference.shape)


def estimate_translation(noisy, hx, hy, ht):
    """Pool adjacent-frame linear translation fits using only the observation."""
    f = np.asarray(noisy, dtype=float)
    if f.ndim != 3 or min(f.shape) < 3 or not np.isfinite(f).all():
        raise ValueError("Expected a finite sequence with >=3 pixels/frames.")
    smooth = gaussian_filter(f, sigma=(0, 2., 2.), mode="reflect")
    midpoint = (smooth[:-1]+smooth[1:])/2
    gx = (midpoint[..., 1:-1, 2:]-midpoint[..., 1:-1, :-2])/2
    gy = (midpoint[..., 2:, 1:-1]-midpoint[..., :-2, 1:-1])/2
    dt = (smooth[1:]-smooth[:-1])[..., 1:-1, 1:-1]
    design = np.stack((gx, gy), axis=-1).reshape(-1, 2)
    displacement, _, rank, singular = np.linalg.lstsq(design, -dt.ravel(), rcond=None)
    if rank != 2 or singular[-1] <= 0 or singular[0]/singular[-1] > 1e6:
        raise ValueError("The observation does not identify both translation components.")
    velocity = displacement*np.array([hx, hy])/ht
    v0 = np.broadcast_to(velocity, (f.shape[0]-1, *f.shape[1:], 2)).copy()
    return v0, dict(method="pooled_midpoint_translation_v1", blur_sigma_pixels=2.,
                    displacement_xy_pixels_per_interval=displacement.tolist(), uses_ground_truth=False)


def save_sequence(path: Path, sequence: Sequence, u0, v0):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **vars(sequence), u0=u0, v0=v0)
    temporary.replace(path)


def load_sequence(path: Path):
    with np.load(path, allow_pickle=False) as saved:
        fields = {name: saved[name].copy() for name in Sequence.__dataclass_fields__}
        for name in ("hx", "hy", "ht"):
            fields[name] = float(fields[name])
        return Sequence(**fields), saved["u0"].copy(), saved["v0"].copy()
