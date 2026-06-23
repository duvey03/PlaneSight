"""Canny edge detector (pure numpy/scipy) - the v1 trace front-end.

The Phase 1 experiment (planesight-0pb / c3r) picked a Canny-style gradient
operator on the top DEM bands over the structure tensor: it gave the best recall
at competitive linearity, and Canny adds two things plain gradient magnitude
lacks - non-maximum suppression (thins edges to 1 px natively) and hysteresis
(links weak edges to strong ones, beating a blunt top-k threshold).

scikit-image is unavailable (decision D11), so the classic pipeline is implemented
here: Gaussian gradient -> magnitude/orientation -> non-maximum suppression ->
double-threshold -> hysteresis. Thresholds are given as QUANTILES of the gradient
magnitude (robust across the arbitrary scales of different derivative bands), not
absolute values. NaN/invalid pixels are excluded from the thresholds and the
output. Returns a boolean edge map ready for ``vectorize.polylines_from_mask``.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, label, map_coordinates

__all__ = ["gaussian_gradient", "non_max_suppression", "hysteresis", "canny"]

_EIGHT = np.ones((3, 3), dtype=int)


def gaussian_gradient(img, sigma: float = 1.0):
    """Gaussian first-derivative gradient ``(gx, gy)``; NaN-filled for the filter."""
    a = np.asarray(img, dtype=float)
    finite = np.isfinite(a)
    fill = float(np.nanmedian(a)) if finite.any() else 0.0
    a = np.where(finite, a, fill)
    gx = gaussian_filter(a, sigma, order=(0, 1))  # d/dx (cols)
    gy = gaussian_filter(a, sigma, order=(1, 0))  # d/dy (rows)
    return gx, gy


def non_max_suppression(mag, gx, gy):
    """Thin edges by keeping only gradient-magnitude maxima along the gradient.

    Samples the magnitude one step forward and back along the (unit) gradient via
    bilinear interpolation and keeps a pixel only if it is >= both neighbours.
    """
    mag = np.asarray(mag, dtype=float)
    eps = 1e-12
    norm = mag + eps
    ux, uy = gx / norm, gy / norm  # unit gradient (points across the edge)
    rows, cols = np.indices(mag.shape, dtype=float)
    fwd = map_coordinates(mag, [rows + uy, cols + ux], order=1, mode="nearest")
    bwd = map_coordinates(mag, [rows - uy, cols - ux], order=1, mode="nearest")
    keep = (mag >= fwd) & (mag >= bwd)
    return np.where(keep, mag, 0.0)


def hysteresis(nms, low: float, high: float):
    """Keep weak edges (>= ``low``) only if 8-connected to a strong edge (>= ``high``)."""
    # weak candidates must be NMS survivors (nms > 0): suppressed pixels are exact
    # zeros and must never qualify, even when low == 0 on a sparse-edge image.
    strong = (nms >= high) & (nms > 0.0)
    weak = (nms >= low) & (nms > 0.0)
    lab, n = label(weak, structure=_EIGHT)
    if n == 0:
        return np.zeros(nms.shape, dtype=bool)
    keep = np.unique(lab[strong])
    keep = keep[keep > 0]
    return np.isin(lab, keep)


def canny(response, sigma: float = 1.0, low_quantile: float = 0.85,
          high_quantile: float = 0.95, valid_mask=None):
    """Full Canny edge map of a response image. Returns a boolean mask.

    Args:
        sigma: Gaussian derivative scale (px).
        low_quantile, high_quantile: hysteresis thresholds as quantiles of the
            gradient magnitude over valid pixels.
        valid_mask: optional mask; pixels outside it are excluded from the
            thresholds and never returned as edges.
    """
    if not 0.0 <= low_quantile <= high_quantile <= 1.0:
        raise ValueError("need 0 <= low_quantile <= high_quantile <= 1")
    resp = np.asarray(response, dtype=float)
    finite = np.isfinite(resp)
    valid = finite if valid_mask is None else finite & np.asarray(valid_mask, dtype=bool)
    gx, gy = gaussian_gradient(resp, sigma)
    mag = np.hypot(gx, gy)
    mag = np.where(valid, mag, 0.0)
    nms = non_max_suppression(mag, gx, gy)
    nms = np.where(valid, nms, 0.0)
    # thresholds are quantiles of the full gradient-magnitude distribution over
    # valid pixels - NOT of the nonzero NMS values (on a clean image the edge is
    # nearly all the nonzero values, so that would slice away the true edge).
    vals = mag[valid]
    if vals.size == 0 or vals.max() == 0.0:
        return np.zeros(resp.shape, dtype=bool)
    low = float(np.quantile(vals, low_quantile))
    high = float(np.quantile(vals, high_quantile))
    edges = hysteresis(nms, low, high)
    return edges & valid
