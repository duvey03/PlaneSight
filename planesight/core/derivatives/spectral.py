"""Sentinel-2-derived spectral layers for the detection input stack (pure numpy).

Sentinel-2 is a first-class detection input, not just a human aid: at 10-20 m it
can out-resolve GLO-30 (30 m) and exposes lithological/colour contacts the
topography does not (complementary to the terrain derivatives - DEM catches
relief-expressed bedding, S2 catches spectral contacts). Which bands/ratios are
most diagnostic is part of Phase 1 issue planesight-bc1 (ARCHITECTURE.md S7.2);
this module supplies the candidates the experiment harness sweeps.

Inputs are surface-reflectance band arrays (already co-registered to a common
grid - resampling lives at the GDAL I/O edge, not here). Band keys follow
``data/sources.py`` S2_BANDS. All functions are pure numpy and divide-safe.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ratio",
    "normalized_ratio",
    "ndvi",
    "clay_index",
    "iron_oxide_index",
    "ferrous_index",
    "percentile_stretch",
]


def _f(a):
    return np.asarray(a, dtype=float)


def ratio(a, b):
    """Simple band ratio a / b, returning NaN where b == 0 (divide-safe)."""
    a, b = _f(a), _f(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = a / b
    return np.where(b == 0.0, np.nan, out)


def normalized_ratio(a, b):
    """Normalised difference (a - b) / (a + b) in [-1, 1]; NaN where a + b == 0.

    The NDVI-style form used for most S2 geology indices - bounded and robust to
    overall brightness, so it highlights spectral *contrast* (i.e. contacts).
    """
    a, b = _f(a), _f(b)
    s = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a - b) / s
    return np.where(s == 0.0, np.nan, out)


def ndvi(nir, red):
    """Vegetation index (NIR-red)/(NIR+red). High = vegetation, which obscures
    geology - useful both as a mask and because vegetation edges are confounders
    a detector should learn to down-weight."""
    return normalized_ratio(nir, red)


def clay_index(swir16, swir22):
    """Clay / hydroxyl-mineral ratio B11/B12 (swir16/swir22). Tracks argillic
    alteration and clay-rich lithologies - a classic lithological discriminator."""
    return ratio(swir16, swir22)


def iron_oxide_index(red, blue):
    """Iron-oxide ratio B4/B2 (red/blue). Highlights gossans / oxidised,
    iron-stained surfaces that often mark lithological boundaries."""
    return ratio(red, blue)


def ferrous_index(swir16, nir):
    """Ferrous-iron ratio B11/B8 (swir16/nir). Complements the iron-oxide ratio
    for mafic vs felsic discrimination."""
    return ratio(swir16, nir)


def percentile_stretch(band, lo: float = 2.0, hi: float = 98.0):
    """Linearly stretch a band to [0, 1] between its lo/hi percentiles (NaN-safe).

    Detectors and false-colour composites both want each channel on a common
    scale; percentile clipping ignores reflectance outliers (snow, water, cloud
    shadow). NaNs are preserved.
    """
    a = _f(band)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return a
    p_lo, p_hi = np.percentile(finite, [lo, hi])
    if p_hi <= p_lo:
        return np.where(np.isfinite(a), 0.0, np.nan)
    return np.clip((a - p_lo) / (p_hi - p_lo), 0.0, 1.0)
