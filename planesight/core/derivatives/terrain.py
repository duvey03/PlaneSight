"""DEM-derived terrain layers for the detection input stack (pure numpy/scipy).

Computes the topographic derivatives that expose bedding/contact traces - slope,
aspect, (multi-azimuth) hillshade, Topographic Position Index, and curvature -
from a 2D elevation array plus its pixel size. No GDAL here: a caller reads the
DEM into an array (GDAL stays at the I/O edge, as in ``attitude/sample.py``) and
passes it in, which keeps every derivative CI-testable against analytic surfaces.

Which of these are most diagnostic is the open question in Phase 1 (issue
planesight-bc1, ARCHITECTURE.md S7.2); this module provides the candidates the
experiment harness sweeps. All functions return float arrays the same shape as
the input. NaN (e.g. DEM nodata) propagates through the 3x3 neighbourhood, so a
nodata pixel nulls its immediate neighbours' derivatives; array edges use
replicated borders rather than going NaN.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gradients",
    "slope",
    "aspect",
    "hillshade",
    "multi_hillshade",
    "tpi",
    "curvature",
]


def _neighbors(dem):
    """Return the nine 3x3 neighbour arrays (north-up) for vectorised stencils.

    Numbering matches Zevenbergen-Thorne / ESRI, row 0 = north::

        z1 z2 z3      NW N NE
        z4 z5 z6   =  W  C E
        z7 z8 z9      SW S SE

    Borders are replicated (edge mode) so the output keeps the input shape; the
    centre ``z5`` is the array itself.
    """
    a = np.asarray(dem, dtype=float)
    if a.ndim != 2:
        raise ValueError("dem must be a 2D array")
    p = np.pad(a, 1, mode="edge")
    z1, z2, z3 = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    z4, z5, z6 = p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:]
    z7, z8, z9 = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    return z1, z2, z3, z4, z5, z6, z7, z8, z9


def gradients(dem, px: float, py: float | None = None):
    """East/north partial derivatives (dz/dx, dz/dy) by Horn's 3x3 method.

    Args:
        dem: 2D elevation array, row 0 = north.
        px: pixel size in the x (east) direction, metres.
        py: pixel size in the y (north) direction; defaults to ``px``.

    Returns:
        ``(fx, fy)`` arrays. ``fx`` is positive toward east, ``fy`` positive
        toward north (both in metres-per-metre, i.e. rise/run).
    """
    if py is None:
        py = px
    if px <= 0 or py <= 0:
        raise ValueError("pixel sizes must be positive")
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = _neighbors(dem)
    fx = ((z3 + 2 * z6 + z9) - (z1 + 2 * z4 + z7)) / (8.0 * px)
    fy = ((z1 + 2 * z2 + z3) - (z7 + 2 * z8 + z9)) / (8.0 * py)
    # Horn's stencil omits the centre cell, so a nodata cell would otherwise get
    # a finite gradient from its neighbours; keep nodata cells nodata.
    nodata = ~np.isfinite(z5)
    fx = np.where(nodata, np.nan, fx)
    fy = np.where(nodata, np.nan, fy)
    return fx, fy


def slope(dem, px: float, py: float | None = None, degrees: bool = True):
    """Steepest-descent slope magnitude. Zero on a flat surface, exact on a plane."""
    fx, fy = gradients(dem, px, py)
    s = np.arctan(np.hypot(fx, fy))
    return np.degrees(s) if degrees else s


def aspect(dem, px: float, py: float | None = None):
    """Downslope azimuth (degrees clockwise from north, 0-360).

    The direction water would flow. Flat cells (zero gradient) return NaN.
    """
    fx, fy = gradients(dem, px, py)
    az = np.degrees(np.arctan2(-fx, -fy)) % 360.0  # downhill = -gradient
    az = np.where(np.hypot(fx, fy) == 0.0, np.nan, az)
    return az


def hillshade(dem, px: float, py: float | None = None,
              azimuth: float = 315.0, altitude: float = 45.0):
    """Shaded relief in [0, 1] for one illumination direction.

    Computed as the clamped dot product of the upward surface normal with the
    sun vector, so it is exact and convention-stable (no aspect round-trip).

    Args:
        azimuth: sun azimuth, degrees clockwise from north.
        altitude: sun elevation above the horizon, degrees.
    """
    fx, fy = gradients(dem, px, py)
    az, alt = np.radians(azimuth), np.radians(altitude)
    # sun unit vector in (east, north, up)
    se, sn, su = np.cos(alt) * np.sin(az), np.cos(alt) * np.cos(az), np.sin(alt)
    # upward surface normal is (-fx, -fy, 1) before normalisation
    dot = (-fx * se - fy * sn + su) / np.sqrt(fx * fx + fy * fy + 1.0)
    return np.clip(dot, 0.0, 1.0)


def multi_hillshade(dem, px: float, py: float | None = None,
                    azimuths=(315.0, 45.0, 135.0, 225.0), altitude: float = 45.0):
    """Mean of hillshades from several azimuths - removes single-light directional
    bias so traces are lit regardless of their orientation (a common pre-detection
    step). Returns a [0, 1] array."""
    if len(azimuths) == 0:
        raise ValueError("need at least one azimuth")
    acc = np.zeros_like(np.asarray(dem, dtype=float))
    for a in azimuths:
        acc += hillshade(dem, px, py, azimuth=a, altitude=altitude)
    return acc / len(azimuths)


def tpi(dem, radius: int = 5):
    """Topographic Position Index: elevation minus the mean of a square window.

    Positive on ridges/crests, negative in valleys, ~0 on planar slopes - which
    makes bedding/contact breaks stand out (it was already useful in the 2020
    prototype, ARCHITECTURE.md S7.2). NaN-aware: cells whose window is entirely
    NaN return NaN. ``radius`` is in pixels (window is ``2*radius+1`` square).
    """
    a = np.asarray(dem, dtype=float)
    if a.ndim != 2:
        raise ValueError("dem must be a 2D array")
    if radius < 1:
        raise ValueError("radius must be >= 1")
    w = 2 * radius + 1
    finite = np.isfinite(a)
    filled = np.where(finite, a, 0.0)
    # box sum via separable cumulative trick is overkill; uniform window mean
    # ignoring NaNs = (sum of finite) / (count of finite) over the window.
    from scipy.ndimage import uniform_filter

    ssum = uniform_filter(filled, size=w, mode="nearest") * (w * w)
    scnt = uniform_filter(finite.astype(float), size=w, mode="nearest") * (w * w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(scnt > 0, ssum / scnt, np.nan)
    return a - mean


def curvature(dem, px: float, py: float | None = None, kind: str = "total"):
    """Surface curvature by the Zevenbergen-Thorne (1987) quadratic fit.

    Args:
        kind: ``"total"`` (negative Laplacian, -2*(d2z/dx2 + d2z/dy2)/2; convex up
            positive), ``"plan"`` (curvature across the slope - across-contour),
            or ``"profile"`` (curvature down the slope).

    Assumes near-square pixels; uses the mean pixel size when px != py. Plan and
    profile are undefined on flat cells (zero gradient) -> 0 there.
    """
    if py is None:
        py = px
    if px <= 0 or py <= 0:
        raise ValueError("pixel sizes must be positive")
    L = 0.5 * (px + py)
    z1, z2, z3, z4, z5, z6, z7, z8, z9 = _neighbors(dem)
    d = ((z4 + z6) / 2.0 - z5) / (L * L)   # ~ 1/2 d2z/dx2
    e = ((z2 + z8) / 2.0 - z5) / (L * L)   # ~ 1/2 d2z/dy2
    f = (-z1 + z3 + z7 - z9) / (4.0 * L * L)
    g = (-z4 + z6) / (2.0 * L)
    h = (z2 - z8) / (2.0 * L)

    if kind == "total":
        return -2.0 * (d + e)
    denom = g * g + h * h
    with np.errstate(invalid="ignore", divide="ignore"):
        if kind == "plan":
            cur = -2.0 * (d * h * h + e * g * g - f * g * h) / denom
        elif kind == "profile":
            cur = 2.0 * (d * g * g + e * h * h + f * g * h) / denom
        else:
            raise ValueError("kind must be 'total', 'plan', or 'profile'")
    return np.where(denom == 0.0, 0.0, cur)
