"""Cost surface for live-wire tracing: low cost = contact-like (assisted tracing, T0).

Builds the float cost grid the live-wire solver walks (planesight-fe7). The primary,
reliable signal is DEM curvature magnitude (contacts are curvature anomalies - the same
signal the classical detector and the ml snap-to-edge use). The builder is composable:
T0 wires only the curvature ("edge") term; T1 adds detector-response, drainage and
orientation-incoherence terms via the same weighted sum (spec S4) with NO signature
change, so the GUI and solver never move. Pure numpy (+ terrain derivatives); no GDAL/Qt.
"""

from __future__ import annotations

import numpy as np

from planesight.core.derivatives.terrain import curvature

# Above the max in-bounds cost (floor + sum of weights), so paths route around nodata
# rather than through it without forbidding it outright (concealed-segment fallback).
_NODATA_PENALTY = 1e3


def _norm01(a: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Robust normalise to [0, 1] by the 2nd/98th percentiles of the valid pixels."""
    a = np.nan_to_num(np.asarray(a, dtype=float), nan=0.0)
    pool = a[valid] if valid is not None else a.ravel()
    if pool.size:
        lo, hi = np.percentile(pool, [2, 98])
    else:
        lo, hi = 0.0, 1.0
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def curvature_magnitude(dem: np.ndarray, px: float, py: float | None = None) -> np.ndarray:
    """Contact-exposing curvature magnitude = |profile curvature| + |total curvature|.

    Profile catches the down-slope break at a contact; total catches the overall
    convexity. Their summed magnitude is high on contacts and low on smooth slopes - the
    raw "edge" signal for :func:`build_cost_surface` (which normalises it).
    """
    prof = curvature(dem, px, py, kind="profile")
    tot = curvature(dem, px, py, kind="total")
    return np.abs(prof) + np.abs(tot)


def build_cost_surface(
    curv_mag: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    detector_resp: np.ndarray | None = None,
    drainage_penalty: np.ndarray | None = None,
    orient_incoherence: np.ndarray | None = None,
    w_edge: float = 1.0,
    w_det: float = 0.0,
    w_drain: float = 0.0,
    w_orient: float = 0.0,
    floor: float = 0.1,
) -> np.ndarray:
    """Weighted cost grid (low = contact-like) from the available signals (spec S4).

    ``cost = floor + w_edge*(1 - n(curv_mag)) + w_det*(1 - n(detector_resp))
             + w_drain*n(drainage_penalty) + w_orient*n(orient_incoherence)``

    where ``n`` robust-normalises to [0, 1]. ``floor`` (> 0) keeps every step positive
    for Dijkstra and bounds how cheap an ideal contact pixel gets. Terms whose input is
    ``None`` (or weight 0) drop out - T0 passes only ``curv_mag``; T1 adds the rest.
    Invalid pixels (``valid is False``) are pushed to a large cost so the wire avoids
    nodata without being forbidden from crossing a concealed gap.

    Args:
        curv_mag: curvature magnitude (see :func:`curvature_magnitude`); high = contact.
        valid: finite-data mask; ``None`` treats all pixels as valid.
        detector_resp: edge/contact response (e.g. Canny or ML probability); high = contact.
        drainage_penalty: per-pixel creek penalty in [0, 1+]; high = avoid.
        orient_incoherence: orientation-incoherence in [0, 1+]; high = avoid.
        w_edge, w_det, w_drain, w_orient: term weights.
        floor: minimum (most contact-like) cost; must be > 0.

    Returns:
        Float cost grid, same shape as ``curv_mag``, all entries > 0.
    """
    if floor <= 0:
        raise ValueError("floor must be positive (Dijkstra needs non-negative weights)")
    cost = np.full(np.asarray(curv_mag).shape, float(floor))
    cost += w_edge * (1.0 - _norm01(curv_mag, valid))
    if detector_resp is not None and w_det:
        cost += w_det * (1.0 - _norm01(detector_resp, valid))
    if drainage_penalty is not None and w_drain:
        cost += w_drain * _norm01(drainage_penalty, valid)
    if orient_incoherence is not None and w_orient:
        cost += w_orient * _norm01(orient_incoherence, valid)
    if valid is not None:
        cost = np.where(valid, cost, floor + _NODATA_PENALTY)
    return cost
