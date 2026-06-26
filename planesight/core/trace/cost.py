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

from planesight.core.derivatives.terrain import curvature, slope, tpi
from planesight.core.detect.canny import canny

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

    where ``n`` robust-normalises the raw edge signals (curvature, detector response) to
    [0, 1]; the penalty terms (drainage, orientation-incoherence) are already calibrated
    to [0, 1] by their builders and are only clipped (re-normalising a mostly-zero
    penalty field would blow it up). ``floor`` (> 0) keeps every step positive for
    Dijkstra and bounds how cheap an ideal contact pixel gets. Terms whose input is
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
        cost += w_drain * np.clip(np.nan_to_num(drainage_penalty, nan=0.0), 0.0, 1.0)
    if orient_incoherence is not None and w_orient:
        cost += w_orient * np.clip(np.nan_to_num(orient_incoherence, nan=0.0), 0.0, 1.0)
    if valid is not None:
        cost = np.where(valid, cost, floor + _NODATA_PENALTY)
    return cost


def contact_strength(
    dem: np.ndarray,
    px: float,
    py: float | None = None,
    *,
    valid: np.ndarray | None = None,
    w_curv: float = 0.7,
    w_slope: float = 0.15,
    w_tpi: float = 0.15,
    tpi_radius: int = 5,
    canny_sigma: float = 1.0,
    canny_low: float = 0.80,
    canny_high: float = 0.92,
    canny_level: float = 0.9,
):
    """Blended contact-likeness in [0, 1] (high = contact) + the Canny rail mask.

    Combines the complementary DEM signals so the live wire has a strong, *continuous*
    cheap network to follow (a curvature-only surface is too fragmented to auto-trace):

    - a soft, continuous guidance term = weighted curvature + slope + |TPI| (each robust-
      normalised), which pulls the wire toward breaks-in-slope/position and bridges gaps;
    - crisp **Canny rails** on the curvature response (thin, connected edges), boosted to
      ``canny_level`` so the wire rides them - this is what makes it auto-trace.

    Args:
        dem: 2D elevation array (metric CRS, NaN nodata).
        px, py: pixel size(s) in metres.
        valid: finite-data mask; ``None`` treats all pixels as valid.
        w_curv, w_slope, w_tpi: weights of the soft guidance signals.
        tpi_radius: TPI window radius (px).
        canny_sigma, canny_low, canny_high: Canny scale + hysteresis quantiles.
        canny_level: strength assigned to Canny-rail pixels (near 1 = near floor cost).

    Returns:
        ``(strength, rails)`` - the [0, 1] strength field and the boolean Canny mask
        (the latter doubles as a viewable "edges" product).
    """
    prof = np.abs(curvature(dem, px, py, kind="profile"))
    total = np.abs(curvature(dem, px, py, kind="total"))
    slp = slope(dem, px, py)
    tp = np.abs(tpi(dem, radius=tpi_radius))
    wsum = w_curv + w_slope + w_tpi
    soft = (w_curv * _norm01(prof, valid)
            + w_slope * _norm01(slp, valid)
            + w_tpi * _norm01(tp, valid)) / max(wsum, 1e-6)
    rails = canny(prof + total, sigma=canny_sigma, low_quantile=canny_low,
                  high_quantile=canny_high, valid_mask=valid)
    strength = np.where(rails, np.maximum(soft, canny_level), soft)
    if valid is not None:
        strength = np.where(valid, strength, 0.0)
    return np.clip(strength, 0.0, 1.0), rails


def trace_cost_surface(
    dem: np.ndarray,
    px: float,
    py: float | None = None,
    *,
    valid: np.ndarray | None = None,
    drainage: np.ndarray | None = None,
    w_edge: float = 1.0,
    w_drain: float = 3.0,
    floor: float = 0.1,
    **strength_kwargs,
):
    """The live-wire cost surface from the blended contact strength (a NON-default option).

    ``cost = floor + w_edge*(1 - contact_strength) + w_drain*drainage`` - low on the
    crisp/continuous contact network, high on creeks and structureless ground.

    NOTE: the cross-AOI sensitivity study (debug/trace_sensitivity_study.py) found this
    blend consistently *loses* to plain curvature+drainage on trace adherence for
    topographically-expressed contacts (extra cheap pixels let the wire drift onto
    parallel edges). BuildCostTask therefore uses curvature+drainage. This entry point is
    kept for the curvature-invisible case the ML probability map (T3) will target, where
    the extra signals may earn their keep. ``strength``/``rails`` are also Data-tab
    preview products (via :func:`contact_strength`).

    Returns ``(cost, strength, rails)``.
    """
    strength, rails = contact_strength(dem, px, py, valid=valid, **strength_kwargs)
    cost = floor + w_edge * (1.0 - strength)
    if drainage is not None and w_drain:
        cost = cost + w_drain * np.clip(np.nan_to_num(drainage, nan=0.0), 0.0, 1.0)
    if valid is not None:
        cost = np.where(valid, cost, floor + _NODATA_PENALTY)
    return cost, strength, rails
