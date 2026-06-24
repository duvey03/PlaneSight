"""Named multi-band input stack assembly for trace detection (pure numpy/scipy).

The detector consumes a *stack* of co-registered bands, not one greyscale image
(ARCHITECTURE.md S7.2). This module names the candidate bands - DEM-derived
(``terrain``) and Sentinel-2-derived (``spectral``) - behind small registries so
the Phase 1 experiment harness (planesight-bc1 / planesight-c3r) can request any
combination by name and sweep them.

Co-registration (resampling S2 onto the DEM grid / a common CRS) happens upstream
at the GDAL I/O edge; everything here assumes the input arrays already share a
grid and shape.
"""

from __future__ import annotations

import numpy as np

from . import spectral, terrain

__all__ = [
    "TERRAIN_BANDS",
    "SPECTRAL_BANDS",
    "DEFAULT_TERRAIN",
    "DEFAULT_SPECTRAL",
    "build_terrain_stack",
    "build_spectral_stack",
    "normalize01",
    "assemble",
]

#: DEM-derived band builders: name -> f(dem, px) -> 2D array. These are the
#: starting-hypothesis candidates from S7.2; Phase 1 decides which survive.
TERRAIN_BANDS = {
    "elevation": lambda dem, px: np.asarray(dem, dtype=float),
    "slope": lambda dem, px: terrain.slope(dem, px),
    "hillshade": lambda dem, px: terrain.hillshade(dem, px),
    "multi_hillshade": lambda dem, px: terrain.multi_hillshade(dem, px),
    "tpi": lambda dem, px: terrain.tpi(dem, radius=5),
    "curvature": lambda dem, px: terrain.curvature(dem, px, kind="total"),
    "plan_curvature": lambda dem, px: terrain.curvature(dem, px, kind="plan"),
    "profile_curvature": lambda dem, px: terrain.curvature(dem, px, kind="profile"),
}

#: Sentinel-2-derived band builders: name -> f(bands) -> 2D array, where ``bands``
#: maps S2 band keys (data/sources.py S2_BANDS) to co-registered arrays.
SPECTRAL_BANDS = {
    "ndvi": lambda b: spectral.ndvi(b["nir"], b["red"]),
    "clay": lambda b: spectral.clay_index(b["swir16"], b["swir22"]),
    "iron_oxide": lambda b: spectral.iron_oxide_index(b["red"], b["blue"]),
    "ferrous": lambda b: spectral.ferrous_index(b["swir16"], b["nir"]),
    # geology SWIR false-colour channels (B12/B11/B2), each stretched to [0, 1]
    "swir22_s": lambda b: spectral.percentile_stretch(b["swir22"]),
    "swir16_s": lambda b: spectral.percentile_stretch(b["swir16"]),
}

#: Curated defaults so a first-time user gets a result with no configuration.
#: From the Phase 1 cross-region experiment (scripts/detector_experiment.py) run on
#: Nepal + Pakistan + Canada with COVERAGE-CORRECTED Sentinel-2 mosaics. Across all
#: three regions, curvature and slope are the robust, dominant contact exposers
#: (the best band in every region is a DEM derivative). Sentinel-2 is a secondary
#: contributor: iron_oxide is the most consistent spectral band (~0.16-0.30 recall
#: everywhere); SWIR/clay/ndvi are weak or region/season-sensitive. The default
#: still carries both families, but spectral is provisional - an earlier "SWIR
#: dominates in arid Pakistan" result turned out to be a scene-coverage artifact.
DEFAULT_TERRAIN = ("profile_curvature", "curvature", "slope", "multi_hillshade")
DEFAULT_SPECTRAL = ("iron_oxide", "ferrous", "swir16_s")


def build_terrain_stack(dem, px: float, names=DEFAULT_TERRAIN):
    """Compute the named DEM-derived bands. Returns ``(names, {name: array})``."""
    names = tuple(names)
    out = {}
    for n in names:
        if n not in TERRAIN_BANDS:
            raise KeyError(f"unknown terrain band {n!r}; have {sorted(TERRAIN_BANDS)}")
        out[n] = TERRAIN_BANDS[n](dem, px)
    return names, out


def build_spectral_stack(bands, names=DEFAULT_SPECTRAL):
    """Compute the named S2-derived bands from a dict of co-registered band arrays.

    Returns ``(names, {name: array})``. Raises ``KeyError`` if a requested index
    needs an S2 band that was not supplied.
    """
    names = tuple(names)
    out = {}
    for n in names:
        if n not in SPECTRAL_BANDS:
            raise KeyError(f"unknown spectral band {n!r}; have {sorted(SPECTRAL_BANDS)}")
        out[n] = SPECTRAL_BANDS[n](bands)
    return names, out


def normalize01(arr, lo: float = 2.0, hi: float = 98.0):
    """Percentile-stretch any band to [0, 1] for detector input (NaN-preserving).

    A thin alias to ``spectral.percentile_stretch`` so terrain and spectral bands
    land on a common scale before stacking.
    """
    return spectral.percentile_stretch(arr, lo=lo, hi=hi)


def assemble(layers, normalize: bool = True):
    """Stack named 2D bands into an ``(H, W, B)`` array with a shared NaN mask.

    Args:
        layers: mapping ``{name: 2D array}``, all the same shape.
        normalize: if True, percentile-stretch each band to [0, 1] so no single
            high-range band (e.g. elevation) dominates a detector.

    Returns:
        ``(names, cube)`` where ``names`` is the band order and ``cube`` is
        ``(H, W, B)`` float. A pixel that is NaN in any band is set NaN across all
        bands, so detectors see a single consistent valid-data footprint.
    """
    names = list(layers)
    if not names:
        raise ValueError("no layers to assemble")
    arrays = [np.asarray(layers[n], dtype=float) for n in names]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("all layers must share the same shape (co-register first)")
    if normalize:
        arrays = [normalize01(a) for a in arrays]
    cube = np.stack(arrays, axis=-1)
    invalid = ~np.isfinite(cube).all(axis=-1)
    cube[invalid] = np.nan
    return names, cube
