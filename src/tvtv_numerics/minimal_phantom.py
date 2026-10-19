#!/usr/bin/env python3
"""Continuous base scene and descriptive region masks for the moving phantom.

The active, resolved-texture version is rendered by data.make_sequence.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit

VELOCITY_XY = np.array([0.003, 0.0015], dtype=np.float64)
REGION_NAMES = ("soft_shading", "sharp_ring", "fine_texture", "flat_slab")


def rotated(x, y, cx, cy, degrees):
    angle = np.deg2rad(degrees)
    dx, dy = x - cx, y - cy
    return np.cos(angle) * dx + np.sin(angle) * dy, -np.sin(angle) * dx + np.cos(angle) * dy


def scene(x, y):
    """Continuous field on R^2, with edges interpreted almost everywhere.

    x increases rightward, y downward. This function can be evaluated outside
    [0,1]^2; translation does not require wrapping or extrapolating image data.
    """
    z = np.full(np.broadcast_shapes(x.shape, y.shape), 0.16, dtype=np.float64)

    # Soft elliptical form: broad shading and an intrinsically diffuse edge.
    p, q = rotated(x, y, 0.325, 0.305, -23.0)
    rho = np.sqrt((p / 0.232) ** 2 + (q / 0.151) ** 2)
    soft = expit((1.0 - rho) / 0.065)
    shade = 0.46 + 0.19 * expit(-p / 0.075) + 0.08 * np.exp(-((q + 0.05) / 0.07) ** 2)
    z = (1.0 - soft) * z + soft * shade

    # Crisp annulus: constant material with curved sharp interfaces.
    ring_r = np.hypot(x - 0.738, y - 0.294)
    ring = (ring_r <= 0.119) & (ring_r >= 0.064)
    z = np.where(ring, 0.83, z)

    # One long, tilted slab. Its boundary is sharp; the right side has a
    # smoothly introduced, low-contrast, two-directional woven texture.
    p, q = rotated(x, y, 0.480, 0.694, -15.0)
    dx = np.abs(p) - (0.347 - 0.026)
    dy = np.abs(q) - (0.126 - 0.026)
    sdf = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)) + np.minimum(np.maximum(dx, dy), 0.0) - 0.026
    slab = sdf <= 0.0
    envelope = expit((p - 0.035) / 0.013) * expit((0.308 - p) / 0.012) * expit((0.103 - np.abs(q)) / 0.008)
    # Units are cycles per unit domain length. The faint crossed carriers
    # provide two-dimensional detail, rather than a single aperture pattern.
    weave = 0.047 * np.cos(2.0 * np.pi * (34.0 * p + 10.0 * q + 5.0 * q * q))
    weave += 0.033 * np.cos(2.0 * np.pi * (-13.0 * p + 39.0 * q))
    material = 0.62 - 0.075 * expit((p - 0.01) / 0.025) + envelope * weave
    z = np.where(slab, material, z)

    # Three fine, high-contrast slits on the otherwise flat left part.
    slits = np.zeros(z.shape, dtype=bool)
    for centre, half_width, half_length in ((-0.247, 0.004, 0.057), (-0.202, 0.006, 0.073), (-0.151, 0.009, 0.087)):
        slits |= (np.abs(p - centre) <= half_width) & (np.abs(q) <= half_length)
    z = np.where(slab & slits, 0.22, z)
    return z


def region_masks(x, y):
    """Interior masks for descriptive inspection; not learned partitions."""
    p1, q1 = rotated(x, y, 0.325, 0.305, -23.0)
    smooth = (p1 / 0.17) ** 2 + (q1 / 0.09) ** 2 <= 1.0
    r = np.hypot(x - 0.738, y - 0.294)
    sharp = (r >= 0.04) & (r <= 0.14)
    p, q = rotated(x, y, 0.480, 0.694, -15.0)
    texture = (p >= 0.075) & (p <= 0.265) & (np.abs(q) <= 0.075)
    flat = (p >= -0.11) & (p <= -0.04) & (np.abs(q) <= 0.08)
    return np.stack((smooth, sharp, texture, flat), axis=0)
